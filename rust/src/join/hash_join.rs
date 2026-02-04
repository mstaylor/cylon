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

//! Hash join implementation
//!
//! Ported from cpp/src/cylon/join/hash_join.cpp
//!
//! This module provides both:
//! - RecordBatch-level functions (WASM-compatible, always available)
//! - Table-level functions (requires `full` feature)

use std::sync::Arc;
use std::collections::HashMap as StdHashMap;
use hashbrown::HashMap;
use arrow::array::{ArrayRef, RecordBatch, UInt64Array};
use arrow::compute::take;
use arrow::datatypes::{Schema, Field};
use arrow_row::{RowConverter, SortField};

use crate::table::Table;
use crate::join::config::{JoinConfig, JoinType};
use crate::join::utils::build_final_table;
use crate::error::{CylonResult, CylonError, Code};

// =============================================================================
// Core Probing Functions (WASM-compatible)
// =============================================================================

/// Probe the hash map for inner join (no fill for unmatched rows)
/// Corresponds to C++ probe_hash_map_no_fill
fn probe_hash_map_inner(
    hash_map: &HashMap<i64, Vec<i64>>,
    probe_size: i64,
    build_indices: &mut Vec<i64>,
    probe_indices: &mut Vec<i64>,
) {
    for i in 0..probe_size {
        if let Some(matching_rows) = hash_map.get(&i) {
            for &build_idx in matching_rows {
                build_indices.push(build_idx);
                probe_indices.push(i);
            }
        }
    }
}

/// Probe the hash map for left/right join (fill with -1 for unmatched rows)
/// Corresponds to C++ probe_hash_map_with_fill
fn probe_hash_map_with_fill(
    hash_map: &HashMap<i64, Vec<i64>>,
    probe_size: i64,
    build_indices: &mut Vec<i64>,
    probe_indices: &mut Vec<i64>,
) {
    for i in 0..probe_size {
        if let Some(matching_rows) = hash_map.get(&i) {
            for &build_idx in matching_rows {
                build_indices.push(build_idx);
                probe_indices.push(i);
            }
        } else {
            // No match - add null entry
            build_indices.push(-1);
            probe_indices.push(i);
        }
    }
}

/// Probe the hash map for full outer join
/// Corresponds to C++ probe_hash_map_outer
fn probe_hash_map_outer(
    hash_map: &HashMap<i64, Vec<i64>>,
    build_size: i64,
    probe_size: i64,
    build_indices: &mut Vec<i64>,
    probe_indices: &mut Vec<i64>,
) {
    let mut build_survivors = vec![true; build_size as usize];
    let mut probe_survivors = vec![true; probe_size as usize];

    // Probe and find matches
    for i in 0..probe_size {
        if let Some(matching_rows) = hash_map.get(&i) {
            probe_survivors[i as usize] = false; // This probe row matched
            for &build_idx in matching_rows {
                build_indices.push(build_idx);
                probe_indices.push(i);
                build_survivors[build_idx as usize] = false; // Mark build row as matched
            }
        }
    }

    // Add unmatched build rows
    for (i, &survivor) in build_survivors.iter().enumerate() {
        if survivor {
            build_indices.push(i as i64);
            probe_indices.push(-1);
        }
    }

    // Add unmatched probe rows
    for (i, &survivor) in probe_survivors.iter().enumerate() {
        if survivor {
            build_indices.push(-1);
            probe_indices.push(i as i64);
        }
    }
}

/// Dispatch to appropriate probe function based on join type
/// Corresponds to C++ do_probe
fn do_probe(
    join_type: JoinType,
    hash_map: &HashMap<i64, Vec<i64>>,
    build_size: i64,
    probe_size: i64,
    build_indices: &mut Vec<i64>,
    probe_indices: &mut Vec<i64>,
) {
    match join_type {
        JoinType::Inner => {
            probe_hash_map_inner(hash_map, probe_size, build_indices, probe_indices);
        }
        JoinType::Left | JoinType::Right => {
            probe_hash_map_with_fill(hash_map, probe_size, build_indices, probe_indices);
        }
        JoinType::FullOuter => {
            probe_hash_map_outer(hash_map, build_size, probe_size, build_indices, probe_indices);
        }
    }
}

/// Calculate join metadata: which table to build from and initial capacity
/// Corresponds to C++ calculate_metadata
fn calculate_join_metadata(
    join_type: JoinType,
    left_size: i64,
    right_size: i64,
) -> (bool, i64) {
    match join_type {
        JoinType::Left => {
            // Build from right, probe from left
            (true, left_size)
        }
        JoinType::Right => {
            // Build from left, probe from right
            (false, right_size)
        }
        JoinType::Inner => {
            let init_size = std::cmp::min(left_size, right_size);
            let build_from_right = left_size <= right_size;
            (!build_from_right, init_size)
        }
        JoinType::FullOuter => {
            let init_size = left_size + right_size;
            let build_from_right = left_size <= right_size;
            (!build_from_right, init_size)
        }
    }
}

// =============================================================================
// RecordBatch-level Functions (WASM-compatible)
// =============================================================================

/// Single-column array index hash join (WASM-compatible)
/// Returns (left_indices, right_indices) for building the final table
///
/// Corresponds to C++ ArrayIndexHashJoin
pub fn array_index_hash_join(
    left_array: &ArrayRef,
    right_array: &ArrayRef,
    join_type: JoinType,
) -> CylonResult<(Vec<i64>, Vec<i64>)> {
    if left_array.data_type() != right_array.data_type() {
        return Err(CylonError::new(
            Code::Invalid,
            "left and right array types are not equal".to_string(),
        ));
    }

    let left_size = left_array.len() as i64;
    let right_size = right_array.len() as i64;

    // Determine build/probe strategy
    let (build_from_right, init_capacity) = calculate_join_metadata(
        join_type,
        left_size,
        right_size,
    );

    let (build_array, probe_array) = if build_from_right {
        (right_array, left_array)
    } else {
        (left_array, right_array)
    };

    let build_size = build_array.len() as i64;
    let probe_size = probe_array.len() as i64;

    // Create row converter for hashing
    let fields = vec![SortField::new(build_array.data_type().clone())];
    let converter = RowConverter::new(fields)
        .map_err(|e| CylonError::new(
            Code::ExecutionError,
            format!("Failed to create row converter: {}", e),
        ))?;

    // Convert build array to rows
    let build_rows = converter.convert_columns(&[build_array.clone()])
        .map_err(|e| CylonError::new(
            Code::ExecutionError,
            format!("Failed to convert build array: {}", e),
        ))?;

    // Convert probe array to rows
    let probe_rows = converter.convert_columns(&[probe_array.clone()])
        .map_err(|e| CylonError::new(
            Code::ExecutionError,
            format!("Failed to convert probe array: {}", e),
        ))?;

    // Build hash map: row_bytes -> list of row indices
    let mut row_to_index: HashMap<Vec<u8>, Vec<i64>> = HashMap::with_capacity(build_size as usize);

    for i in 0..build_size {
        let row_bytes = build_rows.row(i as usize).as_ref().to_vec();
        row_to_index.entry(row_bytes)
            .or_insert_with(Vec::new)
            .push(i);
    }

    // Convert to index-based map for probing
    let mut index_map: HashMap<i64, Vec<i64>> = HashMap::with_capacity(probe_size as usize);

    for i in 0..probe_size {
        let probe_row_bytes = probe_rows.row(i as usize).as_ref().to_vec();
        if let Some(build_indices) = row_to_index.get(&probe_row_bytes) {
            index_map.insert(i, build_indices.clone());
        }
    }

    // Probe the hash map
    let mut build_indices = Vec::with_capacity(init_capacity as usize);
    let mut probe_indices = Vec::with_capacity(init_capacity as usize);

    do_probe(
        join_type,
        &index_map,
        build_size,
        probe_size,
        &mut build_indices,
        &mut probe_indices,
    );

    // Map indices back to left/right
    let (left_indices, right_indices) = if build_from_right {
        (probe_indices, build_indices)
    } else {
        (build_indices, probe_indices)
    };

    Ok((left_indices, right_indices))
}

/// Build the schema for the final joined table (WASM-compatible)
fn build_final_batch_schema(
    left_batch: &RecordBatch,
    right_batch: &RecordBatch,
    left_suffix: &str,
    right_suffix: &str,
) -> Arc<Schema> {
    let mut fields: Vec<Field> = Vec::new();
    let mut column_name_index: StdHashMap<String, usize> = StdHashMap::new();

    // Add left table columns
    for field in left_batch.schema().fields() {
        column_name_index.insert(field.name().clone(), fields.len());
        fields.push(Field::new(
            field.name(),
            field.data_type().clone(),
            true,
        ));
    }

    // Add right table columns (with suffix for duplicates)
    for field in right_batch.schema().fields() {
        if let Some(&idx) = column_name_index.get(field.name()) {
            // Same column name exists - add suffixes
            let left_field = &fields[idx];
            fields[idx] = Field::new(
                &format!("{}{}", left_field.name(), left_suffix),
                left_field.data_type().clone(),
                true,
            );

            let new_field_name = format!("{}{}", field.name(), right_suffix);
            column_name_index.insert(new_field_name.clone(), fields.len());
            fields.push(Field::new(
                &new_field_name,
                field.data_type().clone(),
                true,
            ));
        } else {
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

/// Hash join on RecordBatches (WASM-compatible)
///
/// This is the core join function that works directly on Arrow RecordBatches
/// without requiring CylonContext or Table wrappers.
///
/// # Arguments
/// * `left_batch` - Left input RecordBatch
/// * `right_batch` - Right input RecordBatch
/// * `left_on` - Column indices from left batch to join on
/// * `right_on` - Column indices from right batch to join on
/// * `join_type` - Type of join (Inner, Left, Right, FullOuter)
/// * `left_suffix` - Suffix for duplicate column names from left
/// * `right_suffix` - Suffix for duplicate column names from right
///
/// # Returns
/// Joined RecordBatch
pub fn hash_join_batches(
    left_batch: &RecordBatch,
    right_batch: &RecordBatch,
    left_on: &[usize],
    right_on: &[usize],
    join_type: JoinType,
    left_suffix: &str,
    right_suffix: &str,
) -> CylonResult<RecordBatch> {
    if left_on.len() != right_on.len() {
        return Err(CylonError::new(
            Code::Invalid,
            "left_on and right_on must have same length".to_string(),
        ));
    }

    if left_on.is_empty() {
        return Err(CylonError::new(
            Code::Invalid,
            "Must specify at least one join column".to_string(),
        ));
    }

    let left_size = left_batch.num_rows() as i64;
    let right_size = right_batch.num_rows() as i64;

    // Determine build/probe strategy
    let (build_from_right, init_capacity) = calculate_join_metadata(
        join_type,
        left_size,
        right_size,
    );

    let (build_batch, probe_batch) = if build_from_right {
        (right_batch, left_batch)
    } else {
        (left_batch, right_batch)
    };

    let (build_col_indices, probe_col_indices) = if build_from_right {
        (right_on, left_on)
    } else {
        (left_on, right_on)
    };

    let build_size = build_batch.num_rows() as i64;
    let probe_size = probe_batch.num_rows() as i64;

    // Extract key columns
    let build_arrays: Vec<ArrayRef> = build_col_indices.iter()
        .map(|&i| build_batch.column(i).clone())
        .collect();

    let probe_arrays: Vec<ArrayRef> = probe_col_indices.iter()
        .map(|&i| probe_batch.column(i).clone())
        .collect();

    // Create row converters
    let build_fields: Vec<SortField> = build_arrays.iter()
        .map(|arr| SortField::new(arr.data_type().clone()))
        .collect();

    let build_converter = RowConverter::new(build_fields)
        .map_err(|e| CylonError::new(
            Code::ExecutionError,
            format!("Failed to create build row converter: {}", e),
        ))?;

    let probe_fields: Vec<SortField> = probe_arrays.iter()
        .map(|arr| SortField::new(arr.data_type().clone()))
        .collect();

    let probe_converter = RowConverter::new(probe_fields)
        .map_err(|e| CylonError::new(
            Code::ExecutionError,
            format!("Failed to create probe row converter: {}", e),
        ))?;

    // Convert to rows
    let build_rows = build_converter.convert_columns(&build_arrays)
        .map_err(|e| CylonError::new(
            Code::ExecutionError,
            format!("Failed to convert build columns: {}", e),
        ))?;

    let probe_rows = probe_converter.convert_columns(&probe_arrays)
        .map_err(|e| CylonError::new(
            Code::ExecutionError,
            format!("Failed to convert probe columns: {}", e),
        ))?;

    // Build hash map
    let mut row_to_index: HashMap<Vec<u8>, Vec<i64>> = HashMap::with_capacity(build_size as usize);
    for i in 0..build_size {
        let row_bytes = build_rows.row(i as usize).as_ref().to_vec();
        row_to_index.entry(row_bytes).or_default().push(i);
    }

    // Create index map for probing
    let mut index_map: HashMap<i64, Vec<i64>> = HashMap::with_capacity(probe_size as usize);
    for i in 0..probe_size {
        let probe_row_bytes = probe_rows.row(i as usize).as_ref().to_vec();
        if let Some(build_indices) = row_to_index.get(&probe_row_bytes) {
            index_map.insert(i, build_indices.clone());
        }
    }

    // Probe
    let mut build_indices = Vec::with_capacity(init_capacity as usize);
    let mut probe_indices = Vec::with_capacity(init_capacity as usize);

    do_probe(
        join_type,
        &index_map,
        build_size,
        probe_size,
        &mut build_indices,
        &mut probe_indices,
    );

    // Map indices back to left/right
    let (left_indices, right_indices) = if build_from_right {
        (probe_indices, build_indices)
    } else {
        (build_indices, probe_indices)
    };

    // Build output schema
    let schema = build_final_batch_schema(left_batch, right_batch, left_suffix, right_suffix);

    // Build output columns
    let mut output_columns: Vec<ArrayRef> = Vec::new();

    // Convert indices for take
    let left_take_indices: Vec<Option<u64>> = left_indices.iter()
        .map(|&idx| if idx >= 0 { Some(idx as u64) } else { None })
        .collect();
    let right_take_indices: Vec<Option<u64>> = right_indices.iter()
        .map(|&idx| if idx >= 0 { Some(idx as u64) } else { None })
        .collect();

    let left_index_array = UInt64Array::from(left_take_indices);
    let right_index_array = UInt64Array::from(right_take_indices);

    // Take from left
    for i in 0..left_batch.num_columns() {
        let taken = take(left_batch.column(i), &left_index_array, None)
            .map_err(|e| CylonError::new(
                Code::ExecutionError,
                format!("Failed to take from left column {}: {}", i, e),
            ))?;
        output_columns.push(taken);
    }

    // Take from right
    for i in 0..right_batch.num_columns() {
        let taken = take(right_batch.column(i), &right_index_array, None)
            .map_err(|e| CylonError::new(
                Code::ExecutionError,
                format!("Failed to take from right column {}: {}", i, e),
            ))?;
        output_columns.push(taken);
    }

    RecordBatch::try_new(schema, output_columns)
        .map_err(|e| CylonError::new(
            Code::ExecutionError,
            format!("Failed to create result batch: {}", e),
        ))
}

// =============================================================================
// Table-level Functions
// =============================================================================

/// Multi-column hash join using row converter
/// Corresponds to C++ multi_index_hash_join
pub fn multi_index_hash_join(
    left_table: &Table,
    right_table: &Table,
    config: &JoinConfig,
) -> CylonResult<Table> {
    let left_batch = left_table.batch(0).ok_or_else(|| {
        CylonError::new(Code::Invalid, "left table has no batches".to_string())
    })?;

    let right_batch = right_table.batch(0).ok_or_else(|| {
        CylonError::new(Code::Invalid, "right table has no batches".to_string())
    })?;

    let left_size = left_batch.num_rows() as i64;
    let right_size = right_batch.num_rows() as i64;

    // Determine build/probe strategy
    let (build_from_right, init_capacity) = calculate_join_metadata(
        config.join_type(),
        left_size,
        right_size,
    );

    let (build_table, probe_table) = if build_from_right {
        (right_table, left_table)
    } else {
        (left_table, right_table)
    };

    let (build_col_indices, probe_col_indices) = if build_from_right {
        (config.right_column_indices(), config.left_column_indices())
    } else {
        (config.left_column_indices(), config.right_column_indices())
    };

    let build_batch = build_table.batch(0).unwrap();
    let probe_batch = probe_table.batch(0).unwrap();

    let build_size = build_batch.num_rows() as i64;
    let probe_size = probe_batch.num_rows() as i64;

    // Extract build columns
    let mut build_arrays = Vec::new();
    for &col_idx in build_col_indices {
        if col_idx >= build_batch.num_columns() {
            return Err(CylonError::new(
                Code::Invalid,
                format!("build column index {} out of bounds", col_idx),
            ));
        }
        build_arrays.push(build_batch.column(col_idx).clone());
    }

    // Extract probe columns
    let mut probe_arrays = Vec::new();
    for &col_idx in probe_col_indices {
        if col_idx >= probe_batch.num_columns() {
            return Err(CylonError::new(
                Code::Invalid,
                format!("probe column index {} out of bounds", col_idx),
            ));
        }
        probe_arrays.push(probe_batch.column(col_idx).clone());
    }

    // Create row converter for build table
    let build_fields: Vec<SortField> = build_arrays
        .iter()
        .map(|arr| SortField::new(arr.data_type().clone()))
        .collect();

    let build_converter = RowConverter::new(build_fields)
        .map_err(|e| CylonError::new(
            Code::ExecutionError,
            format!("Failed to create build row converter: {}", e),
        ))?;

    let build_rows = build_converter.convert_columns(&build_arrays)
        .map_err(|e| CylonError::new(
            Code::ExecutionError,
            format!("Failed to convert build columns: {}", e),
        ))?;

    // Create row converter for probe table
    let probe_fields: Vec<SortField> = probe_arrays
        .iter()
        .map(|arr| SortField::new(arr.data_type().clone()))
        .collect();

    let probe_converter = RowConverter::new(probe_fields)
        .map_err(|e| CylonError::new(
            Code::ExecutionError,
            format!("Failed to create probe row converter: {}", e),
        ))?;

    let probe_rows = probe_converter.convert_columns(&probe_arrays)
        .map_err(|e| CylonError::new(
            Code::ExecutionError,
            format!("Failed to convert probe columns: {}", e),
        ))?;

    // Build hash map: row_bytes -> list of row indices
    let mut row_to_index: HashMap<Vec<u8>, Vec<i64>> = HashMap::with_capacity(build_size as usize);

    for i in 0..build_size {
        let row_bytes = build_rows.row(i as usize).as_ref().to_vec();
        row_to_index.entry(row_bytes)
            .or_insert_with(Vec::new)
            .push(i);
    }

    // Create index map for probing
    let mut index_map: HashMap<i64, Vec<i64>> = HashMap::with_capacity(probe_size as usize);

    for i in 0..probe_size {
        let probe_row_bytes = probe_rows.row(i as usize).as_ref().to_vec();
        if let Some(build_indices) = row_to_index.get(&probe_row_bytes) {
            index_map.insert(i, build_indices.clone());
        }
    }

    // Probe the hash map
    let mut build_indices = Vec::with_capacity(init_capacity as usize);
    let mut probe_indices = Vec::with_capacity(init_capacity as usize);

    do_probe(
        config.join_type(),
        &index_map,
        build_size,
        probe_size,
        &mut build_indices,
        &mut probe_indices,
    );

    // Map indices back to left/right
    let (left_indices, right_indices) = if build_from_right {
        (probe_indices, build_indices)
    } else {
        (build_indices, probe_indices)
    };

    // Build final table
    build_final_table(
        &left_indices,
        &right_indices,
        left_table,
        right_table,
        config.left_table_suffix(),
        config.right_table_suffix(),
    )
}

/// Main hash join entry point for Tables
/// Corresponds to C++ HashJoin
pub fn hash_join(
    left_table: &Table,
    right_table: &Table,
    config: &JoinConfig,
) -> CylonResult<Table> {
    // Validate column indices
    if config.left_column_indices().len() != config.right_column_indices().len() {
        return Err(CylonError::new(
            Code::Invalid,
            "left and right index vector sizes should be the same".to_string(),
        ));
    }

    // Check if single-column or multi-column join
    if config.left_column_indices().len() == 1 {
        // Single column join - use optimized path
        let left_batch = left_table.batch(0).ok_or_else(|| {
            CylonError::new(Code::Invalid, "left table has no batches".to_string())
        })?;

        let right_batch = right_table.batch(0).ok_or_else(|| {
            CylonError::new(Code::Invalid, "right table has no batches".to_string())
        })?;

        let left_col_idx = config.left_column_indices()[0];
        let right_col_idx = config.right_column_indices()[0];

        if left_col_idx >= left_batch.num_columns() {
            return Err(CylonError::new(
                Code::Invalid,
                format!("left column index {} out of bounds", left_col_idx),
            ));
        }

        if right_col_idx >= right_batch.num_columns() {
            return Err(CylonError::new(
                Code::Invalid,
                format!("right column index {} out of bounds", right_col_idx),
            ));
        }

        let left_array = left_batch.column(left_col_idx);
        let right_array = right_batch.column(right_col_idx);

        let (left_indices, right_indices) = array_index_hash_join(
            left_array,
            right_array,
            config.join_type(),
        )?;

        build_final_table(
            &left_indices,
            &right_indices,
            left_table,
            right_table,
            config.left_table_suffix(),
            config.right_table_suffix(),
        )
    } else {
        // Multi-column join
        multi_index_hash_join(left_table, right_table, config)
    }
}
