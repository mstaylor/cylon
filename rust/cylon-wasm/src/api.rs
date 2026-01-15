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
//! JSON-based API for JavaScript/Python interop.
//! All functions accept JSON strings and return JSON strings.

use wasm_bindgen::prelude::*;

use crate::table::{Table, TableData};
use crate::join::{hash_join, JoinConfig};
use crate::groupby::{hash_groupby, aggregate_column, GroupByConfig, AggregationOp};
use crate::filter::{filter, FilterConfig};
use crate::error::WasmResult;

// =============================================================================
// Internal helpers
// =============================================================================

fn parse_table(json: &str) -> WasmResult<Table> {
    let data = TableData::from_json(json)?;
    Table::from_table_data(&data)
}

fn table_to_json(table: &Table) -> WasmResult<String> {
    table.to_table_data()?.to_json()
}

// =============================================================================
// WASM Exports - Table Operations
// =============================================================================

/// Join two tables
///
/// # Arguments
/// * `left_json` - Left table as JSON
/// * `right_json` - Right table as JSON
/// * `config_json` - JoinConfig as JSON: { join_type, left_on, right_on, left_suffix?, right_suffix? }
///
/// # Returns
/// Joined table as JSON
///
/// # Example (JavaScript)
/// ```javascript
/// const result = join_tables(
///     JSON.stringify(leftTable),
///     JSON.stringify(rightTable),
///     JSON.stringify({
///         join_type: "inner",
///         left_on: [0],
///         right_on: [0]
///     })
/// );
/// ```
#[wasm_bindgen]
pub fn join_tables(left_json: &str, right_json: &str, config_json: &str) -> Result<String, JsValue> {
    let left = parse_table(left_json)?;
    let right = parse_table(right_json)?;
    let config: JoinConfig = serde_json::from_str(config_json)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let result = hash_join(&left, &right, &config)?;
    Ok(table_to_json(&result)?)
}

/// Group table by keys and compute aggregations
///
/// # Arguments
/// * `table_json` - Input table as JSON
/// * `config_json` - GroupByConfig as JSON: { keys, aggregations: [{ column, op, alias? }] }
///
/// # Returns
/// Grouped table as JSON
///
/// # Example (JavaScript)
/// ```javascript
/// const result = groupby_table(
///     JSON.stringify(table),
///     JSON.stringify({
///         keys: [0],
///         aggregations: [
///             { column: 1, op: "sum" },
///             { column: 1, op: "mean", alias: "avg_value" }
///         ]
///     })
/// );
/// ```
#[wasm_bindgen]
pub fn groupby_table(table_json: &str, config_json: &str) -> Result<String, JsValue> {
    let table = parse_table(table_json)?;
    let config: GroupByConfig = serde_json::from_str(config_json)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let result = hash_groupby(&table, &config)?;
    Ok(table_to_json(&result)?)
}

/// Filter table rows based on predicates
///
/// # Arguments
/// * `table_json` - Input table as JSON
/// * `config_json` - FilterConfig as JSON: { predicates: [{ column, op, value }], logic? }
///
/// # Returns
/// Filtered table as JSON
///
/// # Example (JavaScript)
/// ```javascript
/// const result = filter_table(
///     JSON.stringify(table),
///     JSON.stringify({
///         predicates: [
///             { column: 1, op: "gt", value: 50 },
///             { column: 2, op: "eq", value: "active" }
///         ],
///         logic: "and"
///     })
/// );
/// ```
#[wasm_bindgen]
pub fn filter_table(table_json: &str, config_json: &str) -> Result<String, JsValue> {
    let table = parse_table(table_json)?;
    let config: FilterConfig = serde_json::from_str(config_json)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let result = filter(&table, &config)?;
    Ok(table_to_json(&result)?)
}

/// Compute single aggregation over a column
///
/// # Arguments
/// * `table_json` - Input table as JSON
/// * `column` - Column index
/// * `op` - Aggregation operation: "sum", "mean", "min", "max", "count"
///
/// # Returns
/// Aggregation result as f64
#[wasm_bindgen]
pub fn aggregate(table_json: &str, column: usize, op: &str) -> Result<f64, JsValue> {
    let table = parse_table(table_json)?;
    let agg_op = match op {
        "sum" => AggregationOp::Sum,
        "mean" => AggregationOp::Mean,
        "min" => AggregationOp::Min,
        "max" => AggregationOp::Max,
        "count" => AggregationOp::Count,
        _ => return Err(JsValue::from_str(&format!("Unknown aggregation: {}", op))),
    };

    Ok(aggregate_column(&table, column, agg_op)?)
}

// =============================================================================
// WASM Exports - Table Utilities
// =============================================================================

/// Get table info (rows, columns, schema)
///
/// # Returns
/// JSON object: { num_rows, num_columns, columns: [{ name, type }] }
#[wasm_bindgen]
pub fn table_info(table_json: &str) -> Result<String, JsValue> {
    let table = parse_table(table_json)?;
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

/// Select specific columns from table
///
/// # Arguments
/// * `table_json` - Input table as JSON
/// * `columns_json` - Array of column indices as JSON, e.g., "[0, 2, 3]"
///
/// # Returns
/// Table with selected columns as JSON
#[wasm_bindgen]
pub fn select_columns(table_json: &str, columns_json: &str) -> Result<String, JsValue> {
    let table_data = TableData::from_json(table_json)?;
    let columns: Vec<usize> = serde_json::from_str(columns_json)
        .map_err(|e| JsValue::from_str(&format!("Invalid columns: {}", e)))?;

    let mut result = TableData::new();
    for &col in &columns {
        if col >= table_data.num_columns() {
            return Err(JsValue::from_str(&format!("Column {} out of bounds", col)));
        }
        result.add_column(
            table_data.columns[col].clone(),
            table_data.data[col].clone(),
        ).map_err(|e| JsValue::from_str(&e.to_string()))?;
    }

    result.to_json().map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Concatenate two tables vertically (union)
///
/// # Arguments
/// * `top_json` - First table as JSON
/// * `bottom_json` - Second table as JSON
///
/// # Returns
/// Combined table as JSON (schemas must match)
#[wasm_bindgen]
pub fn concat_tables(top_json: &str, bottom_json: &str) -> Result<String, JsValue> {
    let top = TableData::from_json(top_json)?;
    let bottom = TableData::from_json(bottom_json)?;

    if top.columns != bottom.columns {
        return Err(JsValue::from_str("Tables must have matching column names"));
    }

    if top.num_columns() != bottom.num_columns() {
        return Err(JsValue::from_str("Tables must have same number of columns"));
    }

    // Concatenate each column
    let mut result = TableData::new();
    for (i, col_name) in top.columns.iter().enumerate() {
        use crate::table::ColumnData;

        let combined = match (&top.data[i], &bottom.data[i]) {
            (ColumnData::Int32(a), ColumnData::Int32(b)) => {
                let mut v = a.clone();
                v.extend(b.iter().cloned());
                ColumnData::Int32(v)
            }
            (ColumnData::Int64(a), ColumnData::Int64(b)) => {
                let mut v = a.clone();
                v.extend(b.iter().cloned());
                ColumnData::Int64(v)
            }
            (ColumnData::Float32(a), ColumnData::Float32(b)) => {
                let mut v = a.clone();
                v.extend(b.iter().cloned());
                ColumnData::Float32(v)
            }
            (ColumnData::Float64(a), ColumnData::Float64(b)) => {
                let mut v = a.clone();
                v.extend(b.iter().cloned());
                ColumnData::Float64(v)
            }
            (ColumnData::String(a), ColumnData::String(b)) => {
                let mut v = a.clone();
                v.extend(b.iter().cloned());
                ColumnData::String(v)
            }
            (ColumnData::Boolean(a), ColumnData::Boolean(b)) => {
                let mut v = a.clone();
                v.extend(b.iter().cloned());
                ColumnData::Boolean(v)
            }
            _ => return Err(JsValue::from_str(&format!(
                "Column '{}' has mismatched types", col_name
            ))),
        };

        result.add_column(col_name.clone(), combined)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }

    result.to_json().map_err(|e| JsValue::from_str(&e.to_string()))
}

// =============================================================================
// WASM Exports - Batch Operations (for Lambda pipelines)
// =============================================================================

/// Execute a pipeline of operations
///
/// # Arguments
/// * `table_json` - Input table as JSON
/// * `pipeline_json` - Array of operations as JSON
///
/// # Example pipeline
/// ```json
/// [
///     { "op": "filter", "config": { "predicates": [...] } },
///     { "op": "groupby", "config": { "keys": [...], "aggregations": [...] } }
/// ]
/// ```
#[wasm_bindgen]
pub fn execute_pipeline(table_json: &str, pipeline_json: &str) -> Result<String, JsValue> {
    let mut current = parse_table(table_json)?;

    let pipeline: Vec<serde_json::Value> = serde_json::from_str(pipeline_json)
        .map_err(|e| JsValue::from_str(&format!("Invalid pipeline: {}", e)))?;

    for step in pipeline {
        let op = step.get("op")
            .and_then(|v| v.as_str())
            .ok_or_else(|| JsValue::from_str("Each step must have 'op' field"))?;

        let config = step.get("config")
            .ok_or_else(|| JsValue::from_str("Each step must have 'config' field"))?;

        current = match op {
            "filter" => {
                let cfg: FilterConfig = serde_json::from_value(config.clone())
                    .map_err(|e| JsValue::from_str(&format!("Invalid filter config: {}", e)))?;
                filter(&current, &cfg)?
            }
            "groupby" => {
                let cfg: GroupByConfig = serde_json::from_value(config.clone())
                    .map_err(|e| JsValue::from_str(&format!("Invalid groupby config: {}", e)))?;
                hash_groupby(&current, &cfg)?
            }
            _ => return Err(JsValue::from_str(&format!("Unknown operation: {}", op))),
        };
    }

    Ok(table_to_json(&current)?)
}
