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

//! Hash-based GroupBy implementation for WASM
//!
//! Groups rows by key columns and computes aggregations on value columns.
//! Uses SIMD-optimized aggregation functions where available.

use std::sync::Arc;
use hashbrown::HashMap;
use arrow::array::{ArrayRef, Int32Array, Int64Array, Float32Array, Float64Array, StringArray, Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use arrow_row::{RowConverter, SortField};
use serde::{Deserialize, Serialize};

use crate::table::Table;
use crate::error::{WasmError, WasmResult};
use crate::simd::{simd_sum_f32, simd_sum_f64, simd_min_f32, simd_max_f32};

/// Aggregation operation types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AggregationOp {
    Sum,
    Mean,
    Min,
    Max,
    Count,
}

/// Single aggregation specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Aggregation {
    pub column: usize,
    pub op: AggregationOp,
    #[serde(default)]
    pub alias: Option<String>,
}

/// GroupBy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupByConfig {
    pub keys: Vec<usize>,
    pub aggregations: Vec<Aggregation>,
}

impl GroupByConfig {
    pub fn new(keys: Vec<usize>, aggregations: Vec<Aggregation>) -> Self {
        Self { keys, aggregations }
    }
}

/// Accumulator for aggregations within a group
#[derive(Debug, Clone)]
struct GroupAccumulator {
    sum: f64,
    count: i64,
    min: f64,
    max: f64,
}

impl GroupAccumulator {
    fn new() -> Self {
        Self {
            sum: 0.0,
            count: 0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    fn add(&mut self, value: f64) {
        self.sum += value;
        self.count += 1;
        self.min = self.min.min(value);
        self.max = self.max.max(value);
    }

    fn result(&self, op: AggregationOp) -> f64 {
        match op {
            AggregationOp::Sum => self.sum,
            AggregationOp::Mean => {
                if self.count > 0 { self.sum / self.count as f64 } else { 0.0 }
            }
            AggregationOp::Min => self.min,
            AggregationOp::Max => self.max,
            AggregationOp::Count => self.count as f64,
        }
    }
}

/// Extract numeric value from array at index
fn get_numeric_value(array: &ArrayRef, idx: usize) -> Option<f64> {
    if array.is_null(idx) {
        return None;
    }

    match array.data_type() {
        DataType::Int32 => {
            let arr = array.as_any().downcast_ref::<Int32Array>()?;
            Some(arr.value(idx) as f64)
        }
        DataType::Int64 => {
            let arr = array.as_any().downcast_ref::<Int64Array>()?;
            Some(arr.value(idx) as f64)
        }
        DataType::Float32 => {
            let arr = array.as_any().downcast_ref::<Float32Array>()?;
            Some(arr.value(idx) as f64)
        }
        DataType::Float64 => {
            let arr = array.as_any().downcast_ref::<Float64Array>()?;
            Some(arr.value(idx))
        }
        _ => None,
    }
}

/// Build key columns for output from unique keys
fn build_key_columns(
    unique_keys: &[Vec<u8>],
    key_arrays: &[ArrayRef],
    row_converter: &mut RowConverter,
) -> WasmResult<Vec<ArrayRef>> {
    // For each unique key, we need to find one representative row
    // Build a mapping from key bytes to original row index
    let num_rows = key_arrays.first().map(|a| a.len()).unwrap_or(0);
    let rows = row_converter.convert_columns(key_arrays)
        .map_err(|e| WasmError::execution_error(e.to_string()))?;

    let mut key_to_row: HashMap<Vec<u8>, usize> = HashMap::new();
    for i in 0..num_rows {
        let key_bytes = rows.row(i).as_ref().to_vec();
        key_to_row.entry(key_bytes).or_insert(i);
    }

    // Now build output arrays by taking from original arrays
    let mut result_arrays = Vec::with_capacity(key_arrays.len());

    for array in key_arrays {
        let output = take_by_group_order(array, unique_keys, &key_to_row)?;
        result_arrays.push(output);
    }

    Ok(result_arrays)
}

/// Take values from array in the order of unique_keys
fn take_by_group_order(
    array: &ArrayRef,
    unique_keys: &[Vec<u8>],
    key_to_row: &HashMap<Vec<u8>, usize>,
) -> WasmResult<ArrayRef> {
    match array.data_type() {
        DataType::Int32 => {
            let arr = array.as_any().downcast_ref::<Int32Array>().unwrap();
            let values: Vec<Option<i32>> = unique_keys.iter()
                .map(|k| key_to_row.get(k).map(|&i| {
                    if arr.is_null(i) { None } else { Some(arr.value(i)) }
                }).flatten())
                .collect();
            Ok(Arc::new(Int32Array::from(values)) as ArrayRef)
        }
        DataType::Int64 => {
            let arr = array.as_any().downcast_ref::<Int64Array>().unwrap();
            let values: Vec<Option<i64>> = unique_keys.iter()
                .map(|k| key_to_row.get(k).map(|&i| {
                    if arr.is_null(i) { None } else { Some(arr.value(i)) }
                }).flatten())
                .collect();
            Ok(Arc::new(Int64Array::from(values)) as ArrayRef)
        }
        DataType::Float32 => {
            let arr = array.as_any().downcast_ref::<Float32Array>().unwrap();
            let values: Vec<Option<f32>> = unique_keys.iter()
                .map(|k| key_to_row.get(k).map(|&i| {
                    if arr.is_null(i) { None } else { Some(arr.value(i)) }
                }).flatten())
                .collect();
            Ok(Arc::new(Float32Array::from(values)) as ArrayRef)
        }
        DataType::Float64 => {
            let arr = array.as_any().downcast_ref::<Float64Array>().unwrap();
            let values: Vec<Option<f64>> = unique_keys.iter()
                .map(|k| key_to_row.get(k).map(|&i| {
                    if arr.is_null(i) { None } else { Some(arr.value(i)) }
                }).flatten())
                .collect();
            Ok(Arc::new(Float64Array::from(values)) as ArrayRef)
        }
        DataType::Utf8 => {
            let arr = array.as_any().downcast_ref::<StringArray>().unwrap();
            let values: Vec<Option<String>> = unique_keys.iter()
                .map(|k| key_to_row.get(k).map(|&i| {
                    if arr.is_null(i) { None } else { Some(arr.value(i).to_string()) }
                }).flatten())
                .collect();
            Ok(Arc::new(StringArray::from(values)) as ArrayRef)
        }
        dt => Err(WasmError::unsupported(format!("Unsupported key type: {:?}", dt))),
    }
}

/// Generate output column name for aggregation
fn agg_column_name(schema: &Schema, agg: &Aggregation) -> String {
    if let Some(ref alias) = agg.alias {
        alias.clone()
    } else {
        let col_name = schema.field(agg.column).name();
        let op_name = match agg.op {
            AggregationOp::Sum => "sum",
            AggregationOp::Mean => "mean",
            AggregationOp::Min => "min",
            AggregationOp::Max => "max",
            AggregationOp::Count => "count",
        };
        format!("{}_{}", col_name, op_name)
    }
}

/// Perform hash-based groupby operation
pub fn hash_groupby(table: &Table, config: &GroupByConfig) -> WasmResult<Table> {
    if config.keys.is_empty() {
        return Err(WasmError::invalid("Must specify at least one key column"));
    }

    let batch = table.batch();
    let num_rows = batch.num_rows();

    // Extract key columns
    let key_arrays: Vec<ArrayRef> = config.keys.iter()
        .map(|&i| batch.column(i).clone())
        .collect();

    // Create row converter for keys
    let key_fields: Vec<SortField> = key_arrays.iter()
        .map(|arr| SortField::new(arr.data_type().clone()))
        .collect();
    let mut row_converter = RowConverter::new(key_fields)
        .map_err(|e| WasmError::execution_error(e.to_string()))?;

    let rows = row_converter.convert_columns(&key_arrays)
        .map_err(|e| WasmError::execution_error(e.to_string()))?;

    // Group rows by key
    // Map: key_bytes -> (group_index, Vec<row_indices>)
    let mut group_indices: HashMap<Vec<u8>, usize> = HashMap::new();
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut unique_keys: Vec<Vec<u8>> = Vec::new();

    for i in 0..num_rows {
        let key_bytes = rows.row(i).as_ref().to_vec();
        let group_idx = *group_indices.entry(key_bytes.clone()).or_insert_with(|| {
            let idx = groups.len();
            groups.push(Vec::new());
            unique_keys.push(key_bytes);
            idx
        });
        groups[group_idx].push(i);
    }

    let num_groups = groups.len();

    // Compute aggregations for each group
    let mut agg_results: Vec<Vec<f64>> = vec![vec![0.0; num_groups]; config.aggregations.len()];

    for (agg_idx, agg) in config.aggregations.iter().enumerate() {
        let value_array = batch.column(agg.column);

        for (group_idx, row_indices) in groups.iter().enumerate() {
            let mut acc = GroupAccumulator::new();

            for &row_idx in row_indices {
                if let Some(value) = get_numeric_value(value_array, row_idx) {
                    acc.add(value);
                }
            }

            agg_results[agg_idx][group_idx] = acc.result(agg.op);
        }
    }

    // Build output schema and arrays
    let mut fields = Vec::new();
    let mut arrays: Vec<ArrayRef> = Vec::new();
    let schema = batch.schema();

    // Add key columns
    let key_output = build_key_columns(&unique_keys, &key_arrays, &mut row_converter)?;
    for (i, &key_idx) in config.keys.iter().enumerate() {
        let field = schema.field(key_idx);
        fields.push(Field::new(field.name(), field.data_type().clone(), true));
        arrays.push(key_output[i].clone());
    }

    // Add aggregation result columns
    for (agg_idx, agg) in config.aggregations.iter().enumerate() {
        let col_name = agg_column_name(&schema, agg);

        // Count outputs Int64, others output Float64
        if agg.op == AggregationOp::Count {
            fields.push(Field::new(&col_name, DataType::Int64, false));
            let values: Vec<i64> = agg_results[agg_idx].iter().map(|&v| v as i64).collect();
            arrays.push(Arc::new(Int64Array::from(values)) as ArrayRef);
        } else {
            fields.push(Field::new(&col_name, DataType::Float64, true));
            arrays.push(Arc::new(Float64Array::from(agg_results[agg_idx].clone())) as ArrayRef);
        }
    }

    let schema = Arc::new(Schema::new(fields));
    let result_batch = RecordBatch::try_new(schema, arrays)
        .map_err(|e| WasmError::arrow_error(e.to_string()))?;

    Ok(Table::new(result_batch))
}

/// Compute single aggregation over entire column (no grouping)
/// Uses SIMD optimization where applicable
pub fn aggregate_column(table: &Table, column: usize, op: AggregationOp) -> WasmResult<f64> {
    let batch = table.batch();

    if column >= batch.num_columns() {
        return Err(WasmError::index_error(format!(
            "Column {} out of bounds (table has {} columns)",
            column, batch.num_columns()
        )));
    }

    let array = batch.column(column);

    match (array.data_type(), op) {
        // Use SIMD-optimized paths for f32
        (DataType::Float32, AggregationOp::Sum) => {
            let arr = array.as_any().downcast_ref::<Float32Array>().unwrap();
            let values: Vec<f32> = (0..arr.len())
                .filter(|&i| !arr.is_null(i))
                .map(|i| arr.value(i))
                .collect();
            Ok(simd_sum_f32(&values) as f64)
        }
        (DataType::Float32, AggregationOp::Min) => {
            let arr = array.as_any().downcast_ref::<Float32Array>().unwrap();
            let values: Vec<f32> = (0..arr.len())
                .filter(|&i| !arr.is_null(i))
                .map(|i| arr.value(i))
                .collect();
            Ok(simd_min_f32(&values).unwrap_or(f32::NAN) as f64)
        }
        (DataType::Float32, AggregationOp::Max) => {
            let arr = array.as_any().downcast_ref::<Float32Array>().unwrap();
            let values: Vec<f32> = (0..arr.len())
                .filter(|&i| !arr.is_null(i))
                .map(|i| arr.value(i))
                .collect();
            Ok(simd_max_f32(&values).unwrap_or(f32::NAN) as f64)
        }

        // Use SIMD-optimized path for f64 sum
        (DataType::Float64, AggregationOp::Sum) => {
            let arr = array.as_any().downcast_ref::<Float64Array>().unwrap();
            let values: Vec<f64> = (0..arr.len())
                .filter(|&i| !arr.is_null(i))
                .map(|i| arr.value(i))
                .collect();
            Ok(simd_sum_f64(&values))
        }

        // Generic path for other types/operations
        _ => {
            let mut acc = GroupAccumulator::new();
            for i in 0..array.len() {
                if let Some(value) = get_numeric_value(array, i) {
                    acc.add(value);
                }
            }
            Ok(acc.result(op))
        }
    }
}
