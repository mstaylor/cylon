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

//! Hash-based Partitioning implementation
//!
//! Ported from cpp/src/cylon/partition/partition.hpp and partition.cpp

use crate::error::{CylonResult, CylonError, Code};
use crate::table::Table;
use crate::arrow::arrow_partition_kernels::create_hash_partition_kernel;
use arrow::compute::concat_batches;
use std::collections::HashMap;

/// Hash Partition - Partition table into multiple tables based on hash of columns
/// Corresponds to C++ HashPartition function (table.cpp:1112-1130)
///
/// Partitions a table into multiple tables based on hash values of specified columns.
/// Each row is assigned to a partition based on: hash(row) % num_partitions
///
/// # Arguments
/// * `table` - Input table to partition
/// * `hash_columns` - Column indices to hash for partitioning
/// * `num_partitions` - Number of partitions to create
///
/// # Returns
/// A HashMap mapping partition index to Table for that partition
///
/// # Example
/// ```ignore
/// use cylon::partition::hash_partition;
///
/// // Partition table into 4 partitions based on column 0
/// let partitions = hash_partition(&table, &[0], 4)?;
/// // Access partition 2
/// let partition_2 = &partitions[&2];
/// ```
pub fn hash_partition(
    table: &Table,
    hash_columns: &[usize],
    num_partitions: usize,
) -> CylonResult<HashMap<usize, Table>> {
    if hash_columns.is_empty() {
        return Err(CylonError::new(
            Code::Invalid,
            "hash_columns cannot be empty".to_string()
        ));
    }

    if num_partitions == 0 {
        return Err(CylonError::new(
            Code::Invalid,
            "num_partitions must be > 0".to_string()
        ));
    }

    // Combine batches (C++ does this implicitly)
    let schema = table.schema().ok_or_else(|| {
        CylonError::new(Code::Invalid, "Table has no schema".to_string())
    })?;

    let combined_batch = if table.batches().len() > 1 {
        concat_batches(&schema, table.batches())
            .map_err(|e| CylonError::new(Code::ExecutionError,
                format!("Failed to combine batches: {}", e)))?
    } else if table.batches().len() == 1 {
        table.batches()[0].clone()
    } else {
        return Err(CylonError::new(Code::Invalid, "Table has no batches".to_string()));
    };

    let num_rows = combined_batch.num_rows();

    // MapToHashPartitions: compute target partition for each row
    // (C++ partition.cpp:71-107)
    let mut target_partitions = vec![0u32; num_rows];
    let mut partition_hist = vec![0u32; num_partitions];

    // For multi-column hashing, we compute a composite hash
    if hash_columns.len() == 1 {
        // Single column hash (simpler case)
        let col_idx = hash_columns[0];
        if col_idx >= combined_batch.num_columns() {
            return Err(CylonError::new(
                Code::Invalid,
                format!("Column index {} out of range", col_idx)
            ));
        }

        let column = combined_batch.column(col_idx);
        let kernel = create_hash_partition_kernel(column.data_type())?;

        // Compute hash for each row and assign to partition
        for i in 0..num_rows {
            let hash = kernel.to_hash(column, i as i64);
            let partition = (hash as usize) % num_partitions;
            target_partitions[i] = partition as u32;
            partition_hist[partition] += 1;
        }
    } else {
        // Multi-column hash (composite hash)
        let mut partial_hashes = vec![0u32; num_rows];

        for &col_idx in hash_columns {
            if col_idx >= combined_batch.num_columns() {
                return Err(CylonError::new(
                    Code::Invalid,
                    format!("Column index {} out of range", col_idx)
                ));
            }

            let column = combined_batch.column(col_idx);
            let kernel = create_hash_partition_kernel(column.data_type())?;

            // Update partial hashes with this column's contribution
            kernel.update_hash(column, &mut partial_hashes)?;
        }

        // Assign rows to partitions based on final hash
        for i in 0..num_rows {
            let partition = (partial_hashes[i] as usize) % num_partitions;
            target_partitions[i] = partition as u32;
            partition_hist[partition] += 1;
        }
    }

    // Split: Create partitioned tables (C++ partition.cpp:132-211)
    split_table(&combined_batch, table, num_partitions, &target_partitions, &partition_hist)
}

/// Split table into partitions based on target partition assignments
/// Corresponds to C++ Split function (partition.cpp:132-157)
///
/// Internal function used by hash_partition to create the actual partitioned tables
fn split_table(
    combined_batch: &arrow::record_batch::RecordBatch,
    table: &Table,
    num_partitions: usize,
    target_partitions: &[u32],
    partition_hist: &[u32],
) -> CylonResult<HashMap<usize, Table>> {
    let schema = combined_batch.schema();

    // Build indices for each partition
    let mut partition_indices: Vec<Vec<usize>> = vec![Vec::new(); num_partitions];
    for part_idx in 0..num_partitions {
        partition_indices[part_idx].reserve(partition_hist[part_idx] as usize);
    }

    for (row_idx, &partition) in target_partitions.iter().enumerate() {
        partition_indices[partition as usize].push(row_idx);
    }

    // Create tables for each partition using take kernel
    let mut result = HashMap::new();

    for partition_idx in 0..num_partitions {
        let indices = &partition_indices[partition_idx];

        if indices.is_empty() {
            // Create empty table with same schema
            let empty_batch = arrow::record_batch::RecordBatch::new_empty(schema.clone());
            let empty_table = Table::from_record_batch(table.get_context(), empty_batch)?;
            result.insert(partition_idx, empty_table);
        } else {
            // Use Arrow's take kernel to extract rows for this partition
            let indices_array = arrow::array::UInt64Array::from(
                indices.iter().map(|&i| i as u64).collect::<Vec<_>>()
            );

            let mut partition_columns = Vec::new();
            for col_idx in 0..combined_batch.num_columns() {
                let column = combined_batch.column(col_idx);
                let taken = arrow::compute::take(column.as_ref(), &indices_array, None)
                    .map_err(|e| CylonError::new(Code::ExecutionError,
                        format!("Failed to take rows for partition {}: {}", partition_idx, e)))?;
                partition_columns.push(taken);
            }

            let partition_batch = arrow::record_batch::RecordBatch::try_new(
                schema.clone(),
                partition_columns
            ).map_err(|e| CylonError::new(Code::ExecutionError,
                format!("Failed to create batch for partition {}: {}", partition_idx, e)))?;

            let partition_table = Table::from_record_batch(table.get_context(), partition_batch)?;
            result.insert(partition_idx, partition_table);
        }
    }

    Ok(result)
}
