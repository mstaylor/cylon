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

//! Utilities for join operations
//!
//! Ported from cpp/src/cylon/join/join_utils.cpp

use std::sync::Arc;
use arrow::array::{ArrayRef, RecordBatch, UInt64Array};
use arrow::compute::take;
use arrow::datatypes::{Schema, Field};

use crate::table::Table;
use crate::error::{CylonResult, CylonError, Code};

/// Get array from RecordBatch column
///
/// Corresponds to C++ cylon::util::GetChunkOrEmptyArray (arrow_utils.cpp:265-272)
///
/// In C++, arrow::Table columns are ChunkedArrays (multiple chunks), and this utility
/// extracts the first chunk as a single Array. In Rust, RecordBatch columns are already
/// single ArrayRef, so this is essentially a clone, but we provide it for structural
/// parity with the C++ codebase.
fn get_chunk_or_empty_array(batch: &RecordBatch, col_idx: usize) -> ArrayRef {
    batch.column(col_idx).clone()
}

/// Build the schema for the final joined table
///
/// Corresponds to C++ build_final_table_schema function (join_utils.cpp:30-62)
fn build_final_table_schema(
    left_batch: &RecordBatch,
    right_batch: &RecordBatch,
    left_table_prefix: &str,
    right_table_prefix: &str,
) -> Arc<Schema> {
    use std::collections::HashMap;

    // creating joined schema (join_utils.cpp:35)
    let mut fields: Vec<Field> = Vec::new();

    // (join_utils.cpp:37)
    let mut column_name_index: HashMap<String, usize> = HashMap::new();

    // adding left table (join_utils.cpp:40-43)
    for field in left_batch.schema().fields() {
        column_name_index.insert(field.name().clone(), fields.len());
        fields.push(Field::new(
            field.name(),
            field.data_type().clone(),
            true, // Always nullable in join results
        ));
    }

    // adding right table (join_utils.cpp:46-58)
    for field in right_batch.schema().fields() {
        if let Some(&idx) = column_name_index.get(field.name()) {
            // same column name exists in the left table (join_utils.cpp:48-54)
            // make the existing column name prefixed with left column prefix
            let left_field = &fields[idx];
            fields[idx] = Field::new(
                &format!("{}{}", left_table_prefix, left_field.name()),
                left_field.data_type().clone(),
                true,
            );

            // new field will be prefixed with the right table
            let new_field_name = format!("{}{}", right_table_prefix, field.name());
            column_name_index.insert(new_field_name.clone(), fields.len());
            fields.push(Field::new(
                &new_field_name,
                field.data_type().clone(),
                true,
            ));
        } else {
            // this is a unique column name (join_utils.cpp:57-58)
            column_name_index.insert(field.name().clone(), fields.len());
            fields.push(Field::new(
                field.name(),
                field.data_type().clone(),
                true,
            ));
        }
    }

    Arc::new(Schema::new(fields))
}

/// Build the final joined table from row indices
///
/// Corresponds to C++ build_final_table function
pub fn build_final_table(
    left_indices: &[i64],
    right_indices: &[i64],
    left_table: &Table,
    right_table: &Table,
    left_suffix: &str,
    right_suffix: &str,
) -> CylonResult<Table> {
    if left_indices.len() != right_indices.len() {
        return Err(CylonError::new(
            Code::Invalid,
            "left and right indices must have the same length".to_string(),
        ));
    }

    let left_batch = left_table.batch(0).ok_or_else(|| {
        CylonError::new(Code::Invalid, "left table has no batches".to_string())
    })?;

    let right_batch = right_table.batch(0).ok_or_else(|| {
        CylonError::new(Code::Invalid, "right table has no batches".to_string())
    })?;

    // Build schema for joined table
    let schema = build_final_table_schema(left_batch, right_batch, left_suffix, right_suffix);

    // Build output columns by taking from left and right tables
    let mut output_columns: Vec<ArrayRef> = Vec::new();

    // Convert i64 indices to Option<u64> for take function
    // -1 means null/no match
    let left_take_indices: Vec<Option<u64>> = left_indices.iter()
        .map(|&idx| if idx >= 0 { Some(idx as u64) } else { None })
        .collect();

    let right_take_indices: Vec<Option<u64>> = right_indices.iter()
        .map(|&idx| if idx >= 0 { Some(idx as u64) } else { None })
        .collect();

    // Create index arrays with nulls
    let left_index_array = UInt64Array::from(left_take_indices);
    let right_index_array = UInt64Array::from(right_take_indices);

    // Take from left table columns
    for i in 0..left_batch.num_columns() {
        let column = left_batch.column(i);
        let taken = take(column, &left_index_array, None)
            .map_err(|e| CylonError::new(
                Code::ExecutionError,
                format!("Failed to take from left column {}: {}", i, e),
            ))?;
        output_columns.push(taken);
    }

    // Take from right table columns
    for i in 0..right_batch.num_columns() {
        let column = right_batch.column(i);
        let taken = take(column, &right_index_array, None)
            .map_err(|e| CylonError::new(
                Code::ExecutionError,
                format!("Failed to take from right column {}: {}", i, e),
            ))?;
        output_columns.push(taken);
    }

    // Create the final record batch
    let result_batch = RecordBatch::try_new(schema, output_columns)
        .map_err(|e| CylonError::new(
            Code::ExecutionError,
            format!("Failed to create result batch: {}", e),
        ))?;

    Table::from_record_batch(left_table.get_context(), result_batch)
}

/// Build the final joined table from row indices for inplace sort join
///
/// Corresponds to C++ build_final_table_inplace_index function (join_utils.cpp:64-136)
///
/// This version handles the mapping from sorted positions to original indices:
/// - For join columns: use sorted positions directly (data is already sorted)
/// - For other columns: map sorted positions back to original indices
///
/// Note: C++ uses SortIndicesInPlace which modifies the table in place.
/// In Rust, we pass the sorted join columns separately since we don't modify the original tables.
pub fn build_final_table_inplace_index(
    left_inplace_column: usize,
    right_inplace_column: usize,
    left_indices: &[i64],
    right_indices: &[i64],
    left_index_sorted_column: &UInt64Array,  // (join_utils.cpp:67)
    right_index_sorted_column: &UInt64Array, // (join_utils.cpp:68)
    left_sorted_join_column: &ArrayRef,      // Sorted join column (not in C++ signature - C++ sorts in place)
    right_sorted_join_column: &ArrayRef,     // Sorted join column (not in C++ signature - C++ sorts in place)
    left_table: &Table,
    right_table: &Table,
    left_table_prefix: &str,
    right_table_prefix: &str,
) -> CylonResult<Table> {
    let left_batch = left_table.batch(0).ok_or_else(|| {
        CylonError::new(Code::Invalid, "left table has no batches".to_string())
    })?;

    let right_batch = right_table.batch(0).ok_or_else(|| {
        CylonError::new(Code::Invalid, "right table has no batches".to_string())
    })?;

    // (join_utils.cpp:75-76)
    let schema = build_final_table_schema(left_batch, right_batch, left_table_prefix, right_table_prefix);

    // (join_utils.cpp:78-79)
    let mut indices_indexed: Vec<i64> = Vec::with_capacity(left_indices.len());

    // (join_utils.cpp:81-87)
    for &v in left_indices {
        if v < 0 {
            indices_indexed.push(v);
        } else {
            indices_indexed.push(left_index_sorted_column.value(v as usize) as i64);
        }
    }

    let mut data_arrays: Vec<ArrayRef> = Vec::new();

    // build arrays for left tab (join_utils.cpp:90-105)
    for i in 0..left_batch.num_columns() {
        // For join column, use sorted positions directly (join_utils.cpp:95-98)
        // C++ sorts in place, so ca is already sorted. In Rust, we pass sorted column separately.
        // For other columns, use mapped indices (join_utils.cpp:100-102)
        // C++ uses: copy_array_by_indices with GetChunkOrEmptyArray
        let destination_col_array = if i == left_inplace_column {
            // C++ uses: cylon::util::GetChunkOrEmptyArray(ca, 0) where ca is sorted in place
            // In Rust, we use the separately passed sorted column
            let take_indices: Vec<Option<u64>> = left_indices.iter()
                .map(|&idx| if idx >= 0 { Some(idx as u64) } else { None })
                .collect();
            let index_array = UInt64Array::from(take_indices);
            take(left_sorted_join_column.as_ref(), &index_array, None)
                .map_err(|e| CylonError::new(
                    Code::ExecutionError,
                    format!("Failed to copy left column {}: {}", i, e),
                ))?
        } else {
            // C++ uses: cylon::util::GetChunkOrEmptyArray(ca, 0) (join_utils.cpp:101)
            let ca = get_chunk_or_empty_array(left_batch, i);
            let take_indices: Vec<Option<u64>> = indices_indexed.iter()
                .map(|&idx| if idx >= 0 { Some(idx as u64) } else { None })
                .collect();
            let index_array = UInt64Array::from(take_indices);
            take(&ca, &index_array, None)
                .map_err(|e| CylonError::new(
                    Code::ExecutionError,
                    format!("Failed to copy left column {}: {}", i, e),
                ))?
        };
        data_arrays.push(destination_col_array);
    }

    // (join_utils.cpp:107-108)
    indices_indexed.clear();
    indices_indexed.reserve(right_indices.len());

    // (join_utils.cpp:109-115)
    for &v in right_indices {
        if v < 0 {
            indices_indexed.push(v);
        } else {
            indices_indexed.push(right_index_sorted_column.value(v as usize) as i64);
        }
    }

    // build arrays for right tab (join_utils.cpp:117-133)
    for i in 0..right_batch.num_columns() {
        // For join column, use sorted positions directly (join_utils.cpp:122-125)
        // C++ sorts in place, so ca is already sorted. In Rust, we pass sorted column separately.
        // For other columns, use mapped indices (join_utils.cpp:127-129)
        let destination_col_array = if i == right_inplace_column {
            // C++ uses: cylon::util::GetChunkOrEmptyArray(ca, 0) where ca is sorted in place
            // In Rust, we use the separately passed sorted column
            let take_indices: Vec<Option<u64>> = right_indices.iter()
                .map(|&idx| if idx >= 0 { Some(idx as u64) } else { None })
                .collect();
            let index_array = UInt64Array::from(take_indices);
            take(right_sorted_join_column.as_ref(), &index_array, None)
                .map_err(|e| CylonError::new(
                    Code::ExecutionError,
                    format!("Failed to copy right column {}: {}", i, e),
                ))?
        } else {
            // C++ uses: cylon::util::GetChunkOrEmptyArray(ca, 0) (join_utils.cpp:128)
            let ca = get_chunk_or_empty_array(right_batch, i);
            let take_indices: Vec<Option<u64>> = indices_indexed.iter()
                .map(|&idx| if idx >= 0 { Some(idx as u64) } else { None })
                .collect();
            let index_array = UInt64Array::from(take_indices);
            take(&ca, &index_array, None)
                .map_err(|e| CylonError::new(
                    Code::ExecutionError,
                    format!("Failed to copy right column {}: {}", i, e),
                ))?
        };
        data_arrays.push(destination_col_array);
    }

    // (join_utils.cpp:134)
    let result_batch = RecordBatch::try_new(schema, data_arrays)
        .map_err(|e| CylonError::new(
            Code::ExecutionError,
            format!("Failed to create result batch: {}", e),
        ))?;

    Table::from_record_batch(left_table.get_context(), result_batch)
}
