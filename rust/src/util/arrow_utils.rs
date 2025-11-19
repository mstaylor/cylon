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

//! Utility functions for working with Arrow arrays and data
//!
//! Ported from cpp/src/cylon/util/arrow_utils.hpp and arrow_utils.cpp

use arrow::array::{Array, ArrayRef};
use arrow::record_batch::RecordBatch;
use arrow::datatypes::Schema;

/// Get the size in bytes of an array
pub fn array_size_bytes(array: &dyn Array) -> usize {
    array.get_array_memory_size()
}

/// Convert Arrow RecordBatch to Vec<ArrayRef>
pub fn record_batch_to_arrays(batch: &RecordBatch) -> Vec<ArrayRef> {
    batch.columns().to_vec()
}

/// Get column names from schema
pub fn schema_column_names(schema: &Schema) -> Vec<String> {
    schema.fields().iter().map(|f| f.name().clone()).collect()
}

/// Check if schema has column
pub fn schema_has_column(schema: &Schema, name: &str) -> bool {
    schema.field_with_name(name).is_ok()
}

use crate::{CylonResult, CylonError, table::Table};
use arrow::compute;
use arrow::array::{Int64Array, UInt64Array};
use std::sync::Arc;

/// Sample a table uniformly by selecting rows at evenly-spaced intervals
/// Corresponds to C++ util::SampleTableUniform (arrow_utils.cpp:230-263)
///
/// # Arguments
/// * `table` - The table to sample from
/// * `num_samples` - Number of samples to take
/// * `sort_columns` - Column indices to include in the sample (pass all columns if None)
///
/// # Returns
/// A new table with uniformly sampled rows
///
/// # Algorithm (matches C++ implementation):
/// 1. Select specified columns (or all if sort_columns is None)
/// 2. Handle empty table/zero samples case
/// 3. Calculate step size: num_rows / (num_samples + 1.0)
/// 4. Build indices array with uniform spacing
/// 5. Use arrow::compute::take() to select rows by indices
///
/// # Example
/// ```ignore
/// let sampled = sample_table_uniform(&table, 100, Some(&[0, 1]))?;
/// ```
pub fn sample_table_uniform(
    table: &Table,
    num_samples: usize,
    sort_columns: Option<&[usize]>,
) -> CylonResult<Table> {
    let ctx = table.get_context();
    let num_rows = table.rows();

    // Handle empty cases (C++ line 238-241)
    if num_rows == 0 || num_samples == 0 {
        // Return empty table with same schema
        let empty_batches = vec![];
        return Table::from_record_batches(ctx, empty_batches);
    }

    // Calculate step size (C++ line 243-244)
    let step = num_rows as f32 / (num_samples as f32 + 1.0);
    let mut acc = step;

    // Build indices array (C++ lines 245-256)
    let mut indices_vec = Vec::with_capacity(num_samples);
    for _ in 0..num_samples {
        indices_vec.push(acc as i64);
        acc += step;
    }
    let indices = Arc::new(Int64Array::from(indices_vec));

    // Select columns if specified, otherwise use all columns
    let result_table = if let Some(cols) = sort_columns {
        // Select only the specified columns first (use project for column indices)
        let selected = table.project(cols)?;

        // Get all batches and apply take to each
        let mut result_batches = Vec::new();
        for batch_idx in 0..selected.num_batches() {
            let batch = selected.batch(batch_idx).ok_or_else(|| {
                CylonError::new(crate::error::Code::Invalid, format!("Batch {} not found", batch_idx))
            })?;

            // Apply take to each column in the batch
            let mut new_columns = Vec::new();
            for col in batch.columns() {
                // Arrow compute::take expects &dyn Array, indices is Arc<Int64Array>
                let taken = compute::take(col.as_ref(), indices.as_ref(), None)?;
                new_columns.push(taken);
            }

            let new_batch = RecordBatch::try_new(batch.schema(), new_columns)?;
            result_batches.push(new_batch);
        }

        Table::from_record_batches(ctx, result_batches)?
    } else {
        // Apply take to all columns in the table
        let mut result_batches = Vec::new();
        for batch_idx in 0..table.num_batches() {
            let batch = table.batch(batch_idx).ok_or_else(|| {
                CylonError::new(crate::error::Code::Invalid, format!("Batch {} not found", batch_idx))
            })?;

            // Apply take to each column in the batch (C++ lines 258-261)
            let mut new_columns = Vec::new();
            for col in batch.columns() {
                // Arrow compute::take expects &dyn Array, indices is Arc<Int64Array>
                let taken = compute::take(col.as_ref(), indices.as_ref(), None)?;
                new_columns.push(taken);
            }

            let new_batch = RecordBatch::try_new(batch.schema(), new_columns)?;
            result_batches.push(new_batch);
        }

        Table::from_record_batches(ctx, result_batches)?
    };

    Ok(result_table)
}

/// Take rows from a table by indices
/// Corresponds to the pattern used in C++ SortTable (arrow_utils.cpp:62-69)
///
/// # Arguments
/// * `table` - The table to take rows from
/// * `indices` - Array of row indices to select
///
/// # Returns
/// A new table with only the selected rows
///
/// # Example
/// ```ignore
/// let indices = Int64Array::from(vec![0, 2, 4, 6]);
/// let selected = take_rows(&table, &indices)?;
/// ```
pub fn take_rows(table: &Table, indices: &Int64Array) -> CylonResult<Table> {
    let ctx = table.get_context();

    // Apply take to all columns in all batches
    let mut result_batches = Vec::new();
    for batch_idx in 0..table.num_batches() {
        let batch = table.batch(batch_idx).ok_or_else(|| {
            CylonError::new(crate::error::Code::Invalid, format!("Batch {} not found", batch_idx))
        })?;

        // Apply take to each column in the batch
        let mut new_columns = Vec::new();
        for col in batch.columns() {
            // Arrow compute::take expects &dyn Array
            let taken = compute::take(col.as_ref(), indices as &dyn Array, None)?;
            new_columns.push(taken);
        }

        let new_batch = RecordBatch::try_new(batch.schema(), new_columns)?;
        result_batches.push(new_batch);
    }

    Table::from_record_batches(ctx, result_batches)
}

// TODO: Port additional functions from cpp/src/cylon/util/arrow_utils.cpp:
// - SortTable
// - SortTableMultiColumns
// - copy_array_by_indices
// - free_table
// - Duplicate (ChunkedArray and Table)
// - SampleArray
// - SampleTableUniform
// - GetChunkOrEmptyArray
// - GetNumberSplitsToFitInCache
// - GetBytesAndElements
// - CreateEmptyTable
// - MakeEmptyArrowTable
// - CheckArrowTableContainsChunks
// - MakeDummyArray
// - WrapNumericVector
