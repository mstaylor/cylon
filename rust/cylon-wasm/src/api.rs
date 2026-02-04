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

//! WASM API layer
//!
//! Arrow IPC (binary) for table data, JSON for configs.
//! Designed for efficient processing of large tables.

use wasm_bindgen::prelude::*;

use crate::table::Table;
use crate::join::{hash_join, JoinConfig};
use crate::groupby::{hash_groupby, GroupByConfig};
use crate::filter::{filter, FilterConfig};
use crate::ops::{self, SortConfig, GroupByConfig as CylonGroupByConfig};
use crate::error::WasmResult;

// =============================================================================
// Internal helpers
// =============================================================================

fn parse_table(data: &[u8]) -> WasmResult<Table> {
    Table::from_arrow_ipc(data)
}

fn table_to_ipc(table: &Table) -> WasmResult<Vec<u8>> {
    table.to_arrow_ipc()
}

// =============================================================================
// Table Info & Conversion
// =============================================================================

/// Get table info from Arrow IPC binary
/// Returns JSON: { num_rows, num_columns, columns: [{ name, type }] }
#[wasm_bindgen]
pub fn table_info(data: &[u8]) -> Result<String, JsValue> {
    let table = parse_table(data)?;
    let batch = table.batch();

    let columns: Vec<serde_json::Value> = batch.schema()
        .fields()
        .iter()
        .map(|f| serde_json::json!({
            "name": f.name(),
            "type": format!("{:?}", f.data_type())
        }))
        .collect();

    let info = serde_json::json!({
        "num_rows": batch.num_rows(),
        "num_columns": batch.num_columns(),
        "columns": columns
    });

    serde_json::to_string(&info)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Convert JSON table to Arrow IPC (for initial data loading)
#[wasm_bindgen]
pub fn json_to_ipc(json: &str) -> Result<Vec<u8>, JsValue> {
    let data = crate::table::TableData::from_json(json)?;
    let table = Table::from_table_data(&data)?;
    Ok(table_to_ipc(&table)?)
}

/// Convert Arrow IPC to JSON (for debugging/display only - not for large tables)
#[wasm_bindgen]
pub fn ipc_to_json(data: &[u8]) -> Result<String, JsValue> {
    let table = parse_table(data)?;
    table.to_table_data()?.to_json().map_err(|e| JsValue::from_str(&e.to_string()))
}

// =============================================================================
// Join Operations
// =============================================================================

/// Join two tables
/// config_json: { join_type: "inner"|"left"|"right"|"full_outer", left_on: [0], right_on: [0] }
#[wasm_bindgen]
pub fn join_tables(left: &[u8], right: &[u8], config_json: &str) -> Result<Vec<u8>, JsValue> {
    let left_table = parse_table(left)?;
    let right_table = parse_table(right)?;
    let config: JoinConfig = serde_json::from_str(config_json)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
    let result = hash_join(&left_table, &right_table, &config)?;
    Ok(table_to_ipc(&result)?)
}

// =============================================================================
// Filter Operations
// =============================================================================

/// Filter table rows
/// config_json: { predicates: [{ column, op, value }], logic: "and"|"or" }
#[wasm_bindgen]
pub fn filter_table(data: &[u8], config_json: &str) -> Result<Vec<u8>, JsValue> {
    let table = parse_table(data)?;
    let config: FilterConfig = serde_json::from_str(config_json)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
    let result = filter(&table, &config)?;
    Ok(table_to_ipc(&result)?)
}

// =============================================================================
// GroupBy Operations
// =============================================================================

/// GroupBy with aggregations (WASM-native implementation)
/// config_json: { keys: [0], aggregations: [{ column, op, alias? }] }
#[wasm_bindgen]
pub fn groupby_table(data: &[u8], config_json: &str) -> Result<Vec<u8>, JsValue> {
    let table = parse_table(data)?;
    let config: GroupByConfig = serde_json::from_str(config_json)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
    let result = hash_groupby(&table, &config)?;
    Ok(table_to_ipc(&result)?)
}

/// GroupBy using cylon's hash_groupby
/// config_json: { keys: [0], aggregations: [{ column: 1, op: "sum" }] }
/// Supported ops: sum, min, max, count, mean, var, stddev, nunique
#[wasm_bindgen]
pub fn cylon_groupby(data: &[u8], config_json: &str) -> Result<Vec<u8>, JsValue> {
    let table = parse_table(data)?;
    let config: CylonGroupByConfig = serde_json::from_str(config_json)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;
    let result = ops::groupby(&table, &config)?;
    Ok(table_to_ipc(&result)?)
}

// =============================================================================
// Table Operations (using cylon)
// =============================================================================

/// Project (select) specific columns by index
/// columns_json: [0, 2, 3]
#[wasm_bindgen]
pub fn project_table(data: &[u8], columns_json: &str) -> Result<Vec<u8>, JsValue> {
    let table = parse_table(data)?;
    let columns: Vec<usize> = serde_json::from_str(columns_json)
        .map_err(|e| JsValue::from_str(&format!("Invalid columns: {}", e)))?;
    let result = ops::project(&table, &columns)?;
    Ok(table_to_ipc(&result)?)
}

/// Slice table - get rows from offset to offset+length
#[wasm_bindgen]
pub fn slice_table(data: &[u8], offset: usize, length: usize) -> Result<Vec<u8>, JsValue> {
    let table = parse_table(data)?;
    let result = ops::slice(&table, offset, length)?;
    Ok(table_to_ipc(&result)?)
}

/// Get first n rows
#[wasm_bindgen]
pub fn head_table(data: &[u8], n: usize) -> Result<Vec<u8>, JsValue> {
    let table = parse_table(data)?;
    let result = ops::head(&table, n)?;
    Ok(table_to_ipc(&result)?)
}

/// Get last n rows
#[wasm_bindgen]
pub fn tail_table(data: &[u8], n: usize) -> Result<Vec<u8>, JsValue> {
    let table = parse_table(data)?;
    let result = ops::tail(&table, n)?;
    Ok(table_to_ipc(&result)?)
}

/// Sort table by single column
#[wasm_bindgen]
pub fn sort_table(data: &[u8], column: usize, ascending: bool) -> Result<Vec<u8>, JsValue> {
    let table = parse_table(data)?;
    let result = ops::sort(&table, column, ascending)?;
    Ok(table_to_ipc(&result)?)
}

/// Sort table by multiple columns
/// config_json: { columns: [{ column: 0, ascending: true }, ...] }
#[wasm_bindgen]
pub fn sort_table_multi(data: &[u8], config_json: &str) -> Result<Vec<u8>, JsValue> {
    let table = parse_table(data)?;
    let config: SortConfig = serde_json::from_str(config_json)
        .map_err(|e| JsValue::from_str(&format!("Invalid sort config: {}", e)))?;
    let result = ops::sort_multi(&table, &config)?;
    Ok(table_to_ipc(&result)?)
}

/// Merge tables vertically (concatenate rows)
/// Takes array of Arrow IPC tables
#[wasm_bindgen]
pub fn merge_tables(tables: js_sys::Array) -> Result<Vec<u8>, JsValue> {
    let parsed: Result<Vec<Table>, _> = tables.iter()
        .map(|v| {
            let arr = js_sys::Uint8Array::from(v);
            let data = arr.to_vec();
            parse_table(&data).map_err(|e| JsValue::from_str(&e.to_string()))
        })
        .collect();
    let parsed = parsed?;
    let table_refs: Vec<&Table> = parsed.iter().collect();
    let result = ops::merge(&table_refs)?;
    Ok(table_to_ipc(&result)?)
}

// =============================================================================
// Set Operations (using cylon)
// =============================================================================

/// Union - rows from both tables (removes duplicates)
#[wasm_bindgen]
pub fn union_tables(left: &[u8], right: &[u8]) -> Result<Vec<u8>, JsValue> {
    let left_table = parse_table(left)?;
    let right_table = parse_table(right)?;
    let result = ops::union(&left_table, &right_table)?;
    Ok(table_to_ipc(&result)?)
}

/// Subtract - rows in left that are not in right
#[wasm_bindgen]
pub fn subtract_tables(left: &[u8], right: &[u8]) -> Result<Vec<u8>, JsValue> {
    let left_table = parse_table(left)?;
    let right_table = parse_table(right)?;
    let result = ops::subtract(&left_table, &right_table)?;
    Ok(table_to_ipc(&result)?)
}

/// Intersect - rows that exist in both tables
#[wasm_bindgen]
pub fn intersect_tables(left: &[u8], right: &[u8]) -> Result<Vec<u8>, JsValue> {
    let left_table = parse_table(left)?;
    let right_table = parse_table(right)?;
    let result = ops::intersect(&left_table, &right_table)?;
    Ok(table_to_ipc(&result)?)
}

/// Unique - remove duplicate rows
/// columns_json: [0, 1] - column indices to consider for uniqueness
#[wasm_bindgen]
pub fn unique_table(data: &[u8], columns_json: &str, keep_first: bool) -> Result<Vec<u8>, JsValue> {
    let table = parse_table(data)?;
    let columns: Vec<usize> = serde_json::from_str(columns_json)
        .map_err(|e| JsValue::from_str(&format!("Invalid columns: {}", e)))?;
    let result = ops::unique(&table, &columns, keep_first)?;
    Ok(table_to_ipc(&result)?)
}

// =============================================================================
// Compute Aggregates (using cylon)
// =============================================================================

/// Compute sum of a column
#[wasm_bindgen]
pub fn compute_sum(data: &[u8], column: usize) -> Result<f64, JsValue> {
    let table = parse_table(data)?;
    Ok(ops::compute_sum(&table, column)?)
}

/// Compute min of a column
#[wasm_bindgen]
pub fn compute_min(data: &[u8], column: usize) -> Result<f64, JsValue> {
    let table = parse_table(data)?;
    Ok(ops::compute_min(&table, column)?)
}

/// Compute max of a column
#[wasm_bindgen]
pub fn compute_max(data: &[u8], column: usize) -> Result<f64, JsValue> {
    let table = parse_table(data)?;
    Ok(ops::compute_max(&table, column)?)
}

/// Compute count of a column (non-null values)
#[wasm_bindgen]
pub fn compute_count(data: &[u8], column: usize) -> Result<i64, JsValue> {
    let table = parse_table(data)?;
    Ok(ops::compute_count(&table, column)?)
}

/// Compute mean of a column
#[wasm_bindgen]
pub fn compute_mean(data: &[u8], column: usize) -> Result<f64, JsValue> {
    let table = parse_table(data)?;
    Ok(ops::compute_mean(&table, column)?)
}

/// Compute variance of a column
/// ddof: delta degrees of freedom (0 for population, 1 for sample)
#[wasm_bindgen]
pub fn compute_variance(data: &[u8], column: usize, ddof: i32) -> Result<f64, JsValue> {
    let table = parse_table(data)?;
    Ok(ops::compute_variance(&table, column, ddof)?)
}

/// Compute standard deviation of a column
/// ddof: delta degrees of freedom (0 for population, 1 for sample)
#[wasm_bindgen]
pub fn compute_stddev(data: &[u8], column: usize, ddof: i32) -> Result<f64, JsValue> {
    let table = parse_table(data)?;
    Ok(ops::compute_stddev(&table, column, ddof)?)
}

// =============================================================================
// Partitioning (for host-orchestrated distributed operations)
// =============================================================================

/// Hash partition table into multiple partitions
/// This is the key primitive for distributed operations - the host orchestrates:
/// 1. Call hash_partition() to split data by hash key
/// 2. Use native communication (FMI/MPI/UCX) for all-to-all shuffle
/// 3. Call join_tables()/union_tables()/etc. for local compute
///
/// Returns array of Arrow IPC tables, one per partition
#[wasm_bindgen]
pub fn hash_partition(data: &[u8], columns_json: &str, num_partitions: usize) -> Result<js_sys::Array, JsValue> {
    let table = parse_table(data)?;
    let columns: Vec<usize> = serde_json::from_str(columns_json)
        .map_err(|e| JsValue::from_str(&format!("Invalid columns: {}", e)))?;
    let partitions = ops::hash_partition(&table, &columns, num_partitions)?;

    let result = js_sys::Array::new();
    for partition in partitions {
        let ipc_data = table_to_ipc(&partition)?;
        let uint8_array = js_sys::Uint8Array::from(ipc_data.as_slice());
        result.push(&uint8_array);
    }
    Ok(result)
}