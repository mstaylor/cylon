
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Hash partitioning operations for tables
//!
//! Ported from cpp/src/cylon/arrow/arrow_partition_kernels.cpp Partition()

use std::sync::Arc;
use arrow::array::{ArrayRef, Int64Array, RecordBatch};
use arrow::compute::take;
use arrow::datatypes::Schema;
use crate::arrow::arrow_partition_kernels::create_hash_partition_kernel;
use crate::error::{CylonError, CylonResult, Code};
use crate::table::Table;

/// Partition table rows by hash of specified columns
/// Returns: A vector of RecordBatches, where each batch corresponds to a partition.
///
/// # Arguments
/// * `table` - The table to partition
/// * `hash_columns` - Indices of columns to use for hashing
/// * `num_partitions` - The number of partitions to create
///
/// Corresponds to C++ Partition function in cpp/src/cylon/arrow/arrow_partition_kernels.cpp (lines 442-522)
pub fn hash_partition_table(
    table: &Table,
    hash_columns: &[usize],
    num_partitions: usize,
) -> CylonResult<Vec<RecordBatch>> {
    if num_partitions == 0 {
        return Err(CylonError::new(
            Code::Invalid,
            "Number of partitions cannot be zero".to_string(),
        ));
    }

    if hash_columns.is_empty() {
        return Err(CylonError::new(
            Code::Invalid,
            "Hash columns cannot be empty".to_string(),
        ));
    }

    // 1. Combine batches if necessary
    let combined_batch = if table.num_batches() > 1 {
        let schema = table.schema().ok_or_else(|| {
            CylonError::new(Code::Invalid, "Table has no schema".to_string())
        })?;
        arrow::compute::concat_batches(&schema, table.batches())
            .map_err(|e| CylonError::new(Code::ExecutionError, e.to_string()))?
    } else if let Some(batch) = table.batch(0) {
        batch.clone()
    } else {
        // Empty table, return empty partitions
        let schema = table.schema().unwrap_or_else(|| Arc::new(Schema::empty()));
        let empty_batch = RecordBatch::new_empty(schema);
        return Ok(vec![empty_batch; num_partitions]);
    };

    let num_rows = combined_batch.num_rows();
    if num_rows == 0 {
        let schema = table.schema().unwrap_or_else(|| Arc::new(Schema::empty()));
        let empty_batch = RecordBatch::new_empty(schema);
        return Ok(vec![empty_batch; num_partitions]);
    }

    // 2. Compute composite hash for each row
    let mut hashes = vec![0u32; num_rows];
    for &col_idx in hash_columns {
        let column = combined_batch.column(col_idx);
        let kernel = create_hash_partition_kernel(column.data_type())?;
        kernel.update_hash(column, &mut hashes)?;
    }

    // 3. Determine partition for each row and group row indices by partition
    let mut partitions: Vec<Vec<i64>> = vec![Vec::new(); num_partitions];
    for (i, &hash) in hashes.iter().enumerate() {
        let partition_idx = (hash as usize) % num_partitions;
        partitions[partition_idx].push(i as i64);
    }

    // 4. Create new RecordBatches for each partition using `take` kernel
    let mut partitioned_batches = Vec::with_capacity(num_partitions);
    for indices in partitions {
        if indices.is_empty() {
            // Create an empty batch with the same schema
            partitioned_batches.push(RecordBatch::new_empty(combined_batch.schema()));
            continue;
        }

        let indices_array = Int64Array::from(indices);
        let mut new_columns: Vec<ArrayRef> = Vec::with_capacity(combined_batch.num_columns());
        for col in combined_batch.columns() {
            let new_col = take(col.as_ref(), &indices_array, None)
                .map_err(|e| CylonError::new(Code::ExecutionError, e.to_string()))?;
            new_columns.push(new_col);
        }
        let new_batch = RecordBatch::try_new(combined_batch.schema(), new_columns)
            .map_err(|e| CylonError::new(Code::ExecutionError, e.to_string()))?;
        partitioned_batches.push(new_batch);
    }

    Ok(partitioned_batches)
}
