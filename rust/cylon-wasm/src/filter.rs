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

//! Filter operations for WASM
//!
//! Row selection based on predicates - equivalent to SQL WHERE clause.
//! Supports multiple predicates combined with AND/OR logic.

use std::sync::Arc;
use arrow::array::{ArrayRef, Int32Array, Int64Array, Float32Array, Float64Array,
                   StringArray, BooleanArray, Array,
                   Int32Builder, Int64Builder, Float32Builder, Float64Builder,
                   StringBuilder, BooleanBuilder};
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;
use serde::{Deserialize, Serialize};

use crate::table::Table;
use crate::error::{WasmError, WasmResult};

/// Comparison operators
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompareOp {
    Eq,  // ==
    Ne,  // !=
    Lt,  // <
    Le,  // <=
    Gt,  // >
    Ge,  // >=
}

/// Filter predicate value (supports multiple types)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FilterValue {
    Int(i64),
    Float(f64),
    String(String),
    Bool(bool),
}

/// Single filter predicate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Predicate {
    pub column: usize,
    pub op: CompareOp,
    pub value: FilterValue,
}

/// Logical combination of predicates
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LogicOp {
    And,
    Or,
}

/// Filter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterConfig {
    pub predicates: Vec<Predicate>,
    #[serde(default = "default_logic")]
    pub logic: LogicOp,
}

fn default_logic() -> LogicOp { LogicOp::And }

impl FilterConfig {
    pub fn new(predicates: Vec<Predicate>) -> Self {
        Self { predicates, logic: LogicOp::And }
    }

    pub fn with_logic(mut self, logic: LogicOp) -> Self {
        self.logic = logic;
        self
    }
}

// =============================================================================
// Predicate Evaluation
// =============================================================================

fn eval_i64(value: i64, op: CompareOp, target: i64) -> bool {
    match op {
        CompareOp::Eq => value == target,
        CompareOp::Ne => value != target,
        CompareOp::Lt => value < target,
        CompareOp::Le => value <= target,
        CompareOp::Gt => value > target,
        CompareOp::Ge => value >= target,
    }
}

fn eval_f64(value: f64, op: CompareOp, target: f64) -> bool {
    match op {
        CompareOp::Eq => (value - target).abs() < f64::EPSILON,
        CompareOp::Ne => (value - target).abs() >= f64::EPSILON,
        CompareOp::Lt => value < target,
        CompareOp::Le => value <= target,
        CompareOp::Gt => value > target,
        CompareOp::Ge => value >= target,
    }
}

fn eval_str(value: &str, op: CompareOp, target: &str) -> bool {
    match op {
        CompareOp::Eq => value == target,
        CompareOp::Ne => value != target,
        CompareOp::Lt => value < target,
        CompareOp::Le => value <= target,
        CompareOp::Gt => value > target,
        CompareOp::Ge => value >= target,
    }
}

fn eval_bool(value: bool, op: CompareOp, target: bool) -> bool {
    match op {
        CompareOp::Eq => value == target,
        CompareOp::Ne => value != target,
        _ => false,
    }
}

/// Evaluate predicate on array at index
fn evaluate_predicate(array: &ArrayRef, idx: usize, pred: &Predicate) -> WasmResult<bool> {
    if array.is_null(idx) {
        return Ok(false); // NULL doesn't match any predicate
    }

    match (array.data_type(), &pred.value) {
        (DataType::Int32, FilterValue::Int(target)) => {
            let arr = array.as_any().downcast_ref::<Int32Array>().unwrap();
            Ok(eval_i64(arr.value(idx) as i64, pred.op, *target))
        }
        (DataType::Int64, FilterValue::Int(target)) => {
            let arr = array.as_any().downcast_ref::<Int64Array>().unwrap();
            Ok(eval_i64(arr.value(idx), pred.op, *target))
        }
        (DataType::Float32, FilterValue::Float(target)) => {
            let arr = array.as_any().downcast_ref::<Float32Array>().unwrap();
            Ok(eval_f64(arr.value(idx) as f64, pred.op, *target))
        }
        (DataType::Float64, FilterValue::Float(target)) => {
            let arr = array.as_any().downcast_ref::<Float64Array>().unwrap();
            Ok(eval_f64(arr.value(idx), pred.op, *target))
        }
        (DataType::Int32, FilterValue::Float(target)) => {
            let arr = array.as_any().downcast_ref::<Int32Array>().unwrap();
            Ok(eval_f64(arr.value(idx) as f64, pred.op, *target))
        }
        (DataType::Int64, FilterValue::Float(target)) => {
            let arr = array.as_any().downcast_ref::<Int64Array>().unwrap();
            Ok(eval_f64(arr.value(idx) as f64, pred.op, *target))
        }
        (DataType::Utf8, FilterValue::String(target)) => {
            let arr = array.as_any().downcast_ref::<StringArray>().unwrap();
            Ok(eval_str(arr.value(idx), pred.op, target))
        }
        (DataType::Boolean, FilterValue::Bool(target)) => {
            let arr = array.as_any().downcast_ref::<BooleanArray>().unwrap();
            Ok(eval_bool(arr.value(idx), pred.op, *target))
        }
        (dt, val) => Err(WasmError::type_error(format!(
            "Type mismatch: {:?} vs {:?}", dt, val
        ))),
    }
}

/// Compute selection mask for all rows
fn compute_mask(batch: &RecordBatch, config: &FilterConfig) -> WasmResult<Vec<bool>> {
    let num_rows = batch.num_rows();
    // Initialize based on logic: AND starts true, OR starts false
    let mut mask = vec![config.logic == LogicOp::And; num_rows];

    for pred in &config.predicates {
        if pred.column >= batch.num_columns() {
            return Err(WasmError::index_error(format!(
                "Column {} out of bounds", pred.column
            )));
        }

        let array = batch.column(pred.column);

        for i in 0..num_rows {
            let result = evaluate_predicate(array, i, pred)?;
            match config.logic {
                LogicOp::And => mask[i] = mask[i] && result,
                LogicOp::Or => mask[i] = mask[i] || result,
            }
        }
    }

    Ok(mask)
}

// =============================================================================
// Array Filtering
// =============================================================================

/// Filter array by boolean mask
fn filter_array(array: &ArrayRef, mask: &[bool]) -> WasmResult<ArrayRef> {
    let count = mask.iter().filter(|&&m| m).count();

    match array.data_type() {
        DataType::Int32 => {
            let arr = array.as_any().downcast_ref::<Int32Array>().unwrap();
            let mut builder = Int32Builder::with_capacity(count);
            for (i, &selected) in mask.iter().enumerate() {
                if selected {
                    if arr.is_null(i) { builder.append_null(); }
                    else { builder.append_value(arr.value(i)); }
                }
            }
            Ok(Arc::new(builder.finish()) as ArrayRef)
        }
        DataType::Int64 => {
            let arr = array.as_any().downcast_ref::<Int64Array>().unwrap();
            let mut builder = Int64Builder::with_capacity(count);
            for (i, &selected) in mask.iter().enumerate() {
                if selected {
                    if arr.is_null(i) { builder.append_null(); }
                    else { builder.append_value(arr.value(i)); }
                }
            }
            Ok(Arc::new(builder.finish()) as ArrayRef)
        }
        DataType::Float32 => {
            let arr = array.as_any().downcast_ref::<Float32Array>().unwrap();
            let mut builder = Float32Builder::with_capacity(count);
            for (i, &selected) in mask.iter().enumerate() {
                if selected {
                    if arr.is_null(i) { builder.append_null(); }
                    else { builder.append_value(arr.value(i)); }
                }
            }
            Ok(Arc::new(builder.finish()) as ArrayRef)
        }
        DataType::Float64 => {
            let arr = array.as_any().downcast_ref::<Float64Array>().unwrap();
            let mut builder = Float64Builder::with_capacity(count);
            for (i, &selected) in mask.iter().enumerate() {
                if selected {
                    if arr.is_null(i) { builder.append_null(); }
                    else { builder.append_value(arr.value(i)); }
                }
            }
            Ok(Arc::new(builder.finish()) as ArrayRef)
        }
        DataType::Utf8 => {
            let arr = array.as_any().downcast_ref::<StringArray>().unwrap();
            let mut builder = StringBuilder::with_capacity(count, count * 32);
            for (i, &selected) in mask.iter().enumerate() {
                if selected {
                    if arr.is_null(i) { builder.append_null(); }
                    else { builder.append_value(arr.value(i)); }
                }
            }
            Ok(Arc::new(builder.finish()) as ArrayRef)
        }
        DataType::Boolean => {
            let arr = array.as_any().downcast_ref::<BooleanArray>().unwrap();
            let mut builder = BooleanBuilder::with_capacity(count);
            for (i, &selected) in mask.iter().enumerate() {
                if selected {
                    if arr.is_null(i) { builder.append_null(); }
                    else { builder.append_value(arr.value(i)); }
                }
            }
            Ok(Arc::new(builder.finish()) as ArrayRef)
        }
        dt => Err(WasmError::unsupported(format!("Unsupported filter type: {:?}", dt))),
    }
}

// =============================================================================
// Public API
// =============================================================================

/// Filter table rows based on predicates
pub fn filter(table: &Table, config: &FilterConfig) -> WasmResult<Table> {
    if config.predicates.is_empty() {
        return Ok(table.clone());
    }

    let batch = table.batch();
    let mask = compute_mask(batch, config)?;

    let arrays: WasmResult<Vec<ArrayRef>> = batch.columns()
        .iter()
        .map(|col| filter_array(col, &mask))
        .collect();

    let result = RecordBatch::try_new(batch.schema(), arrays?)
        .map_err(|e| WasmError::arrow_error(e.to_string()))?;

    Ok(Table::new(result))
}

/// Convenience: single predicate filter
pub fn filter_single(
    table: &Table,
    column: usize,
    op: CompareOp,
    value: FilterValue,
) -> WasmResult<Table> {
    filter(table, &FilterConfig::new(vec![Predicate { column, op, value }]))
}
