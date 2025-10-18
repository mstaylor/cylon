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
    let mut fields = Vec::new();

    // Add left table columns
    // Note: In outer joins, all columns can have nulls
    for field in left_batch.schema().fields() {
        let field_name = if !left_suffix.is_empty() &&
            right_batch.schema().field_with_name(field.name()).is_ok() {
            format!("{}{}", field.name(), left_suffix)
        } else {
            field.name().clone()
        };

        fields.push(Field::new(
            &field_name,
            field.data_type().clone(),
            true, // Always nullable in join results
        ));
    }

    // Add right table columns
    for field in right_batch.schema().fields() {
        let field_name = if !right_suffix.is_empty() &&
            left_batch.schema().field_with_name(field.name()).is_ok() {
            format!("{}{}", field.name(), right_suffix)
        } else {
            field.name().clone()
        };

        fields.push(Field::new(
            &field_name,
            field.data_type().clone(),
            true, // Always nullable in join results
        ));
    }

    let schema = Arc::new(Schema::new(fields));

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
