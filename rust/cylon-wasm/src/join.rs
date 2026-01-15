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

//! Hash join implementation for WASM
//!
//! Two-phase algorithm:
//! 1. Build: Create hash map from smaller table
//! 2. Probe: Scan larger table and lookup matches
//!
//! Probe methods handle different join semantics:
//! - probe_inner: Only matched rows (Inner)
//! - probe_with_fill: Keep unmatched probe rows with NULL (Left/Right)
//! - probe_outer: Keep ALL unmatched from both sides (Full Outer)

use std::sync::Arc;
use hashbrown::HashMap;
use arrow::array::{ArrayRef, Int32Builder, Int64Builder, Float32Builder, Float64Builder,
                   StringBuilder, BooleanBuilder, Array, Int32Array, Int64Array,
                   Float32Array, Float64Array, StringArray, BooleanArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use arrow_row::{RowConverter, SortField};
use serde::{Deserialize, Serialize};

use crate::table::Table;
use crate::error::{WasmError, WasmResult};

/// Join type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JoinType {
    Inner,
    Left,
    Right,
    FullOuter,
}

/// Join configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JoinConfig {
    pub join_type: JoinType,
    pub left_on: Vec<usize>,
    pub right_on: Vec<usize>,
    #[serde(default = "default_left_suffix")]
    pub left_suffix: String,
    #[serde(default = "default_right_suffix")]
    pub right_suffix: String,
}

fn default_left_suffix() -> String { "_l".to_string() }
fn default_right_suffix() -> String { "_r".to_string() }

impl JoinConfig {
    pub fn new(join_type: JoinType, left_on: Vec<usize>, right_on: Vec<usize>) -> Self {
        Self {
            join_type,
            left_on,
            right_on,
            left_suffix: default_left_suffix(),
            right_suffix: default_right_suffix(),
        }
    }
}

// =============================================================================
// Probe Functions - Handle different join semantics
// =============================================================================

/// Inner join: Only emit rows with matches
fn probe_inner(
    index_map: &HashMap<i64, Vec<i64>>,
    probe_size: i64,
    build_indices: &mut Vec<i64>,
    probe_indices: &mut Vec<i64>,
) {
    for i in 0..probe_size {
        if let Some(matching) = index_map.get(&i) {
            for &build_idx in matching {
                build_indices.push(build_idx);
                probe_indices.push(i);
            }
        }
        // No match = skip (inner join discards unmatched)
    }
}

/// Left/Right join: Keep unmatched probe rows with NULL (-1) on build side
fn probe_with_fill(
    index_map: &HashMap<i64, Vec<i64>>,
    probe_size: i64,
    build_indices: &mut Vec<i64>,
    probe_indices: &mut Vec<i64>,
) {
    for i in 0..probe_size {
        if let Some(matching) = index_map.get(&i) {
            for &build_idx in matching {
                build_indices.push(build_idx);
                probe_indices.push(i);
            }
        } else {
            // No match - keep probe row, NULL for build side
            build_indices.push(-1);
            probe_indices.push(i);
        }
    }
}

/// Full outer join: Keep ALL unmatched rows from both sides
fn probe_outer(
    index_map: &HashMap<i64, Vec<i64>>,
    build_size: i64,
    probe_size: i64,
    build_indices: &mut Vec<i64>,
    probe_indices: &mut Vec<i64>,
) {
    let mut build_matched = vec![false; build_size as usize];
    let mut probe_matched = vec![false; probe_size as usize];

    // Find all matches
    for i in 0..probe_size {
        if let Some(matching) = index_map.get(&i) {
            probe_matched[i as usize] = true;
            for &build_idx in matching {
                build_indices.push(build_idx);
                probe_indices.push(i);
                build_matched[build_idx as usize] = true;
            }
        }
    }

    // Add unmatched build rows (NULL on probe side)
    for (i, &matched) in build_matched.iter().enumerate() {
        if !matched {
            build_indices.push(i as i64);
            probe_indices.push(-1);
        }
    }

    // Add unmatched probe rows (NULL on build side)
    for (i, &matched) in probe_matched.iter().enumerate() {
        if !matched {
            build_indices.push(-1);
            probe_indices.push(i as i64);
        }
    }
}

/// Dispatch to appropriate probe function based on join type
fn do_probe(
    join_type: JoinType,
    index_map: &HashMap<i64, Vec<i64>>,
    build_size: i64,
    probe_size: i64,
    build_indices: &mut Vec<i64>,
    probe_indices: &mut Vec<i64>,
) {
    match join_type {
        JoinType::Inner => probe_inner(index_map, probe_size, build_indices, probe_indices),
        JoinType::Left | JoinType::Right => probe_with_fill(index_map, probe_size, build_indices, probe_indices),
        JoinType::FullOuter => probe_outer(index_map, build_size, probe_size, build_indices, probe_indices),
    }
}

// =============================================================================
// Join Strategy
// =============================================================================

/// Determine which table to build from and initial result capacity
fn calculate_metadata(join_type: JoinType, left_size: i64, right_size: i64) -> (bool, i64) {
    match join_type {
        JoinType::Left => (true, left_size),      // Build from right, probe left
        JoinType::Right => (false, right_size),   // Build from left, probe right
        JoinType::Inner => {
            // Build from smaller table for efficiency
            let build_right = left_size > right_size;
            (build_right, std::cmp::min(left_size, right_size))
        }
        JoinType::FullOuter => {
            let build_right = left_size > right_size;
            (build_right, left_size + right_size)
        }
    }
}

// =============================================================================
// Array Operations
// =============================================================================

/// Take values from array by indices, -1 becomes NULL
fn take_array_by_indices(array: &ArrayRef, indices: &[i64]) -> WasmResult<ArrayRef> {
    match array.data_type() {
        DataType::Int32 => {
            let arr = array.as_any().downcast_ref::<Int32Array>().unwrap();
            let mut builder = Int32Builder::with_capacity(indices.len());
            for &idx in indices {
                if idx < 0 || arr.is_null(idx as usize) {
                    builder.append_null();
                } else {
                    builder.append_value(arr.value(idx as usize));
                }
            }
            Ok(Arc::new(builder.finish()) as ArrayRef)
        }
        DataType::Int64 => {
            let arr = array.as_any().downcast_ref::<Int64Array>().unwrap();
            let mut builder = Int64Builder::with_capacity(indices.len());
            for &idx in indices {
                if idx < 0 || arr.is_null(idx as usize) {
                    builder.append_null();
                } else {
                    builder.append_value(arr.value(idx as usize));
                }
            }
            Ok(Arc::new(builder.finish()) as ArrayRef)
        }
        DataType::Float32 => {
            let arr = array.as_any().downcast_ref::<Float32Array>().unwrap();
            let mut builder = Float32Builder::with_capacity(indices.len());
            for &idx in indices {
                if idx < 0 || arr.is_null(idx as usize) {
                    builder.append_null();
                } else {
                    builder.append_value(arr.value(idx as usize));
                }
            }
            Ok(Arc::new(builder.finish()) as ArrayRef)
        }
        DataType::Float64 => {
            let arr = array.as_any().downcast_ref::<Float64Array>().unwrap();
            let mut builder = Float64Builder::with_capacity(indices.len());
            for &idx in indices {
                if idx < 0 || arr.is_null(idx as usize) {
                    builder.append_null();
                } else {
                    builder.append_value(arr.value(idx as usize));
                }
            }
            Ok(Arc::new(builder.finish()) as ArrayRef)
        }
        DataType::Utf8 => {
            let arr = array.as_any().downcast_ref::<StringArray>().unwrap();
            let mut builder = StringBuilder::with_capacity(indices.len(), indices.len() * 32);
            for &idx in indices {
                if idx < 0 || arr.is_null(idx as usize) {
                    builder.append_null();
                } else {
                    builder.append_value(arr.value(idx as usize));
                }
            }
            Ok(Arc::new(builder.finish()) as ArrayRef)
        }
        DataType::Boolean => {
            let arr = array.as_any().downcast_ref::<BooleanArray>().unwrap();
            let mut builder = BooleanBuilder::with_capacity(indices.len());
            for &idx in indices {
                if idx < 0 || arr.is_null(idx as usize) {
                    builder.append_null();
                } else {
                    builder.append_value(arr.value(idx as usize));
                }
            }
            Ok(Arc::new(builder.finish()) as ArrayRef)
        }
        dt => Err(WasmError::unsupported(format!("Unsupported type for join: {:?}", dt))),
    }
}

/// Build final joined table from matched indices
fn build_final_table(
    left_indices: &[i64],
    right_indices: &[i64],
    left: &Table,
    right: &Table,
    left_suffix: &str,
    right_suffix: &str,
) -> WasmResult<Table> {
    let left_batch = left.batch();
    let right_batch = right.batch();

    let mut fields = Vec::new();
    let mut arrays = Vec::new();

    // Detect duplicate column names
    let left_schema = left_batch.schema();
    let right_schema = right_batch.schema();
    let left_names: Vec<String> = left_schema.fields().iter()
        .map(|f| f.name().clone()).collect();
    let right_names: Vec<String> = right_schema.fields().iter()
        .map(|f| f.name().clone()).collect();

    // Add left columns (with suffix if name collision)
    for (i, field) in left_schema.fields().iter().enumerate() {
        let name = if right_names.contains(field.name()) {
            format!("{}{}", field.name(), left_suffix)
        } else {
            field.name().clone()
        };
        fields.push(Field::new(&name, field.data_type().clone(), true));
        arrays.push(take_array_by_indices(left_batch.column(i), left_indices)?);
    }

    // Add right columns (with suffix if name collision)
    for (i, field) in right_schema.fields().iter().enumerate() {
        let name = if left_names.contains(field.name()) {
            format!("{}{}", field.name(), right_suffix)
        } else {
            field.name().clone()
        };
        fields.push(Field::new(&name, field.data_type().clone(), true));
        arrays.push(take_array_by_indices(right_batch.column(i), right_indices)?);
    }

    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema, arrays)
        .map_err(|e| WasmError::arrow_error(e.to_string()))?;

    Ok(Table::new(batch))
}

// =============================================================================
// Public API
// =============================================================================

/// Perform hash join operation
pub fn hash_join(left: &Table, right: &Table, config: &JoinConfig) -> WasmResult<Table> {
    if config.left_on.len() != config.right_on.len() {
        return Err(WasmError::invalid("left_on and right_on must have same length"));
    }
    if config.left_on.is_empty() {
        return Err(WasmError::invalid("Must specify at least one join column"));
    }

    let left_batch = left.batch();
    let right_batch = right.batch();
    let left_size = left_batch.num_rows() as i64;
    let right_size = right_batch.num_rows() as i64;

    // Determine build/probe strategy
    let (build_from_right, capacity) = calculate_metadata(config.join_type, left_size, right_size);

    let (build_table, probe_table) = if build_from_right {
        (right, left)
    } else {
        (left, right)
    };

    let (build_cols, probe_cols) = if build_from_right {
        (&config.right_on, &config.left_on)
    } else {
        (&config.left_on, &config.right_on)
    };

    let build_batch = build_table.batch();
    let probe_batch = probe_table.batch();
    let build_size = build_batch.num_rows() as i64;
    let probe_size = probe_batch.num_rows() as i64;

    // Extract key columns
    let build_arrays: Vec<ArrayRef> = build_cols.iter()
        .map(|&i| build_batch.column(i).clone())
        .collect();
    let probe_arrays: Vec<ArrayRef> = probe_cols.iter()
        .map(|&i| probe_batch.column(i).clone())
        .collect();

    // Create row converters for hashing
    let build_fields: Vec<SortField> = build_arrays.iter()
        .map(|arr| SortField::new(arr.data_type().clone()))
        .collect();
    let build_converter = RowConverter::new(build_fields)
        .map_err(|e| WasmError::execution_error(e.to_string()))?;

    let probe_fields: Vec<SortField> = probe_arrays.iter()
        .map(|arr| SortField::new(arr.data_type().clone()))
        .collect();
    let probe_converter = RowConverter::new(probe_fields)
        .map_err(|e| WasmError::execution_error(e.to_string()))?;

    // Convert to comparable row format
    let build_rows = build_converter.convert_columns(&build_arrays)
        .map_err(|e| WasmError::execution_error(e.to_string()))?;
    let probe_rows = probe_converter.convert_columns(&probe_arrays)
        .map_err(|e| WasmError::execution_error(e.to_string()))?;

    // BUILD PHASE: Create hash map from build table
    let mut row_to_indices: HashMap<Vec<u8>, Vec<i64>> = HashMap::with_capacity(build_size as usize);
    for i in 0..build_size {
        let row_bytes = build_rows.row(i as usize).as_ref().to_vec();
        row_to_indices.entry(row_bytes).or_default().push(i);
    }

    // Create probe index → build indices map
    let mut index_map: HashMap<i64, Vec<i64>> = HashMap::with_capacity(probe_size as usize);
    for i in 0..probe_size {
        let probe_bytes = probe_rows.row(i as usize).as_ref().to_vec();
        if let Some(build_indices) = row_to_indices.get(&probe_bytes) {
            index_map.insert(i, build_indices.clone());
        }
    }

    // PROBE PHASE: Find matches based on join type
    let mut build_indices = Vec::with_capacity(capacity as usize);
    let mut probe_indices = Vec::with_capacity(capacity as usize);
    do_probe(
        config.join_type,
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

    build_final_table(
        &left_indices,
        &right_indices,
        left,
        right,
        &config.left_suffix,
        &config.right_suffix,
    )
}
