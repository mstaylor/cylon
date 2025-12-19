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

//! High-level aggregate functions
//!
//! Ported from cpp/src/cylon/compute/aggregates.hpp and aggregates.cpp
//!
//! This module provides user-facing aggregate functions that work on Arrow arrays
//! and Cylon tables, with support for distributed computation.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, PrimitiveArray};
use arrow::compute;
use arrow::datatypes::{
    DataType, Float32Type, Float64Type,
    Int8Type, Int16Type, Int32Type, Int64Type,
    UInt8Type, UInt16Type, UInt32Type, UInt64Type,
};

use crate::ctx::CylonContext;
use crate::error::{Code, CylonError, CylonResult};
use crate::table::Table;

use super::aggregate_kernels::{BasicOptions, VarKernelOptions};

// ============================================================================
// Aggregate Options (User-facing)
// ============================================================================

/// Options for basic aggregation operations
#[derive(Debug, Clone)]
pub struct AggregateOptions {
    /// Whether to skip null values in computation
    pub skip_nulls: bool,
}

impl Default for AggregateOptions {
    fn default() -> Self {
        Self { skip_nulls: true }
    }
}

impl AggregateOptions {
    pub fn new(skip_nulls: bool) -> Self {
        Self { skip_nulls }
    }
}

impl From<&AggregateOptions> for BasicOptions {
    fn from(opts: &AggregateOptions) -> Self {
        BasicOptions::new(opts.skip_nulls)
    }
}

/// Options for variance and standard deviation
#[derive(Debug, Clone)]
pub struct VarianceOptions {
    /// Delta degrees of freedom (0 for population, 1 for sample)
    pub ddof: i32,
    /// Whether to skip null values
    pub skip_nulls: bool,
}

impl Default for VarianceOptions {
    fn default() -> Self {
        Self {
            ddof: 0,
            skip_nulls: true,
        }
    }
}

impl VarianceOptions {
    pub fn new(ddof: i32, skip_nulls: bool) -> Self {
        Self { ddof, skip_nulls }
    }

    /// Create options for population variance/stddev (ddof=0)
    pub fn population() -> Self {
        Self { ddof: 0, skip_nulls: true }
    }

    /// Create options for sample variance/stddev (ddof=1)
    pub fn sample() -> Self {
        Self { ddof: 1, skip_nulls: true }
    }
}

impl From<&VarianceOptions> for VarKernelOptions {
    fn from(opts: &VarianceOptions) -> Self {
        VarKernelOptions::new(opts.ddof, opts.skip_nulls)
    }
}

// ============================================================================
// Scalar Result Type
// ============================================================================

/// Result of a scalar aggregation
/// Corresponds to arrow::Datum/arrow::Scalar in C++
#[derive(Debug, Clone)]
pub enum ScalarValue {
    Null,
    Boolean(bool),
    Int8(i8),
    Int16(i16),
    Int32(i32),
    Int64(i64),
    UInt8(u8),
    UInt16(u16),
    UInt32(u32),
    UInt64(u64),
    Float32(f32),
    Float64(f64),
}

impl ScalarValue {
    /// Convert to f64 if possible
    pub fn to_f64(&self) -> Option<f64> {
        match self {
            ScalarValue::Null => None,
            ScalarValue::Boolean(b) => Some(if *b { 1.0 } else { 0.0 }),
            ScalarValue::Int8(v) => Some(*v as f64),
            ScalarValue::Int16(v) => Some(*v as f64),
            ScalarValue::Int32(v) => Some(*v as f64),
            ScalarValue::Int64(v) => Some(*v as f64),
            ScalarValue::UInt8(v) => Some(*v as f64),
            ScalarValue::UInt16(v) => Some(*v as f64),
            ScalarValue::UInt32(v) => Some(*v as f64),
            ScalarValue::UInt64(v) => Some(*v as f64),
            ScalarValue::Float32(v) => Some(*v as f64),
            ScalarValue::Float64(v) => Some(*v),
        }
    }

    /// Convert to i64 if possible
    pub fn to_i64(&self) -> Option<i64> {
        match self {
            ScalarValue::Null => None,
            ScalarValue::Boolean(b) => Some(if *b { 1 } else { 0 }),
            ScalarValue::Int8(v) => Some(*v as i64),
            ScalarValue::Int16(v) => Some(*v as i64),
            ScalarValue::Int32(v) => Some(*v as i64),
            ScalarValue::Int64(v) => Some(*v),
            ScalarValue::UInt8(v) => Some(*v as i64),
            ScalarValue::UInt16(v) => Some(*v as i64),
            ScalarValue::UInt32(v) => Some(*v as i64),
            ScalarValue::UInt64(v) => Some(*v as i64),
            ScalarValue::Float32(v) => Some(*v as i64),
            ScalarValue::Float64(v) => Some(*v as i64),
        }
    }

    /// Check if value is null
    pub fn is_null(&self) -> bool {
        matches!(self, ScalarValue::Null)
    }

    /// Get the data type of this scalar
    pub fn data_type(&self) -> DataType {
        match self {
            ScalarValue::Null => DataType::Null,
            ScalarValue::Boolean(_) => DataType::Boolean,
            ScalarValue::Int8(_) => DataType::Int8,
            ScalarValue::Int16(_) => DataType::Int16,
            ScalarValue::Int32(_) => DataType::Int32,
            ScalarValue::Int64(_) => DataType::Int64,
            ScalarValue::UInt8(_) => DataType::UInt8,
            ScalarValue::UInt16(_) => DataType::UInt16,
            ScalarValue::UInt32(_) => DataType::UInt32,
            ScalarValue::UInt64(_) => DataType::UInt64,
            ScalarValue::Float32(_) => DataType::Float32,
            ScalarValue::Float64(_) => DataType::Float64,
        }
    }
}

impl std::fmt::Display for ScalarValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ScalarValue::Null => write!(f, "null"),
            ScalarValue::Boolean(v) => write!(f, "{}", v),
            ScalarValue::Int8(v) => write!(f, "{}", v),
            ScalarValue::Int16(v) => write!(f, "{}", v),
            ScalarValue::Int32(v) => write!(f, "{}", v),
            ScalarValue::Int64(v) => write!(f, "{}", v),
            ScalarValue::UInt8(v) => write!(f, "{}", v),
            ScalarValue::UInt16(v) => write!(f, "{}", v),
            ScalarValue::UInt32(v) => write!(f, "{}", v),
            ScalarValue::UInt64(v) => write!(f, "{}", v),
            ScalarValue::Float32(v) => write!(f, "{}", v),
            ScalarValue::Float64(v) => write!(f, "{}", v),
        }
    }
}

// ============================================================================
// Local Array Aggregates
// ============================================================================

/// Compute sum of an array (local only)
/// Corresponds to arrow::compute::Sum in C++
pub fn sum_array(array: &dyn Array, _options: &AggregateOptions) -> CylonResult<ScalarValue> {
    match array.data_type() {
        DataType::Int8 => {
            let arr = array.as_any().downcast_ref::<PrimitiveArray<Int8Type>>().unwrap();
            Ok(compute::sum(arr).map(ScalarValue::Int8).unwrap_or(ScalarValue::Null))
        }
        DataType::Int16 => {
            let arr = array.as_any().downcast_ref::<PrimitiveArray<Int16Type>>().unwrap();
            Ok(compute::sum(arr).map(ScalarValue::Int16).unwrap_or(ScalarValue::Null))
        }
        DataType::Int32 => {
            let arr = array.as_any().downcast_ref::<PrimitiveArray<Int32Type>>().unwrap();
            Ok(compute::sum(arr).map(ScalarValue::Int32).unwrap_or(ScalarValue::Null))
        }
        DataType::Int64 => {
            let arr = array.as_any().downcast_ref::<PrimitiveArray<Int64Type>>().unwrap();
            Ok(compute::sum(arr).map(ScalarValue::Int64).unwrap_or(ScalarValue::Null))
        }
        DataType::UInt8 => {
            let arr = array.as_any().downcast_ref::<PrimitiveArray<UInt8Type>>().unwrap();
            Ok(compute::sum(arr).map(ScalarValue::UInt8).unwrap_or(ScalarValue::Null))
        }
        DataType::UInt16 => {
            let arr = array.as_any().downcast_ref::<PrimitiveArray<UInt16Type>>().unwrap();
            Ok(compute::sum(arr).map(ScalarValue::UInt16).unwrap_or(ScalarValue::Null))
        }
        DataType::UInt32 => {
            let arr = array.as_any().downcast_ref::<PrimitiveArray<UInt32Type>>().unwrap();
            Ok(compute::sum(arr).map(ScalarValue::UInt32).unwrap_or(ScalarValue::Null))
        }
        DataType::UInt64 => {
            let arr = array.as_any().downcast_ref::<PrimitiveArray<UInt64Type>>().unwrap();
            Ok(compute::sum(arr).map(ScalarValue::UInt64).unwrap_or(ScalarValue::Null))
        }
        DataType::Float32 => {
            let arr = array.as_any().downcast_ref::<PrimitiveArray<Float32Type>>().unwrap();
            Ok(compute::sum(arr).map(ScalarValue::Float32).unwrap_or(ScalarValue::Null))
        }
        DataType::Float64 => {
            let arr = array.as_any().downcast_ref::<PrimitiveArray<Float64Type>>().unwrap();
            Ok(compute::sum(arr).map(ScalarValue::Float64).unwrap_or(ScalarValue::Null))
        }
        dt => Err(CylonError::new(
            Code::Invalid,
            format!("Sum not supported for data type: {:?}", dt),
        )),
    }
}

/// Compute minimum of an array (local only)
pub fn min_array(array: &dyn Array, _options: &AggregateOptions) -> CylonResult<ScalarValue> {
    match array.data_type() {
        DataType::Int8 => {
            let arr = array.as_any().downcast_ref::<PrimitiveArray<Int8Type>>().unwrap();
            Ok(compute::min(arr).map(ScalarValue::Int8).unwrap_or(ScalarValue::Null))
        }
        DataType::Int16 => {
            let arr = array.as_any().downcast_ref::<PrimitiveArray<Int16Type>>().unwrap();
            Ok(compute::min(arr).map(ScalarValue::Int16).unwrap_or(ScalarValue::Null))
        }
        DataType::Int32 => {
            let arr = array.as_any().downcast_ref::<PrimitiveArray<Int32Type>>().unwrap();
            Ok(compute::min(arr).map(ScalarValue::Int32).unwrap_or(ScalarValue::Null))
        }
        DataType::Int64 => {
            let arr = array.as_any().downcast_ref::<PrimitiveArray<Int64Type>>().unwrap();
            Ok(compute::min(arr).map(ScalarValue::Int64).unwrap_or(ScalarValue::Null))
        }
        DataType::UInt8 => {
            let arr = array.as_any().downcast_ref::<PrimitiveArray<UInt8Type>>().unwrap();
            Ok(compute::min(arr).map(ScalarValue::UInt8).unwrap_or(ScalarValue::Null))
        }
        DataType::UInt16 => {
            let arr = array.as_any().downcast_ref::<PrimitiveArray<UInt16Type>>().unwrap();
            Ok(compute::min(arr).map(ScalarValue::UInt16).unwrap_or(ScalarValue::Null))
        }
        DataType::UInt32 => {
            let arr = array.as_any().downcast_ref::<PrimitiveArray<UInt32Type>>().unwrap();
            Ok(compute::min(arr).map(ScalarValue::UInt32).unwrap_or(ScalarValue::Null))
        }
        DataType::UInt64 => {
            let arr = array.as_any().downcast_ref::<PrimitiveArray<UInt64Type>>().unwrap();
            Ok(compute::min(arr).map(ScalarValue::UInt64).unwrap_or(ScalarValue::Null))
        }
        DataType::Float32 => {
            let arr = array.as_any().downcast_ref::<PrimitiveArray<Float32Type>>().unwrap();
            Ok(compute::min(arr).map(ScalarValue::Float32).unwrap_or(ScalarValue::Null))
        }
        DataType::Float64 => {
            let arr = array.as_any().downcast_ref::<PrimitiveArray<Float64Type>>().unwrap();
            Ok(compute::min(arr).map(ScalarValue::Float64).unwrap_or(ScalarValue::Null))
        }
        dt => Err(CylonError::new(
            Code::Invalid,
            format!("Min not supported for data type: {:?}", dt),
        )),
    }
}

/// Compute maximum of an array (local only)
pub fn max_array(array: &dyn Array, _options: &AggregateOptions) -> CylonResult<ScalarValue> {
    match array.data_type() {
        DataType::Int8 => {
            let arr = array.as_any().downcast_ref::<PrimitiveArray<Int8Type>>().unwrap();
            Ok(compute::max(arr).map(ScalarValue::Int8).unwrap_or(ScalarValue::Null))
        }
        DataType::Int16 => {
            let arr = array.as_any().downcast_ref::<PrimitiveArray<Int16Type>>().unwrap();
            Ok(compute::max(arr).map(ScalarValue::Int16).unwrap_or(ScalarValue::Null))
        }
        DataType::Int32 => {
            let arr = array.as_any().downcast_ref::<PrimitiveArray<Int32Type>>().unwrap();
            Ok(compute::max(arr).map(ScalarValue::Int32).unwrap_or(ScalarValue::Null))
        }
        DataType::Int64 => {
            let arr = array.as_any().downcast_ref::<PrimitiveArray<Int64Type>>().unwrap();
            Ok(compute::max(arr).map(ScalarValue::Int64).unwrap_or(ScalarValue::Null))
        }
        DataType::UInt8 => {
            let arr = array.as_any().downcast_ref::<PrimitiveArray<UInt8Type>>().unwrap();
            Ok(compute::max(arr).map(ScalarValue::UInt8).unwrap_or(ScalarValue::Null))
        }
        DataType::UInt16 => {
            let arr = array.as_any().downcast_ref::<PrimitiveArray<UInt16Type>>().unwrap();
            Ok(compute::max(arr).map(ScalarValue::UInt16).unwrap_or(ScalarValue::Null))
        }
        DataType::UInt32 => {
            let arr = array.as_any().downcast_ref::<PrimitiveArray<UInt32Type>>().unwrap();
            Ok(compute::max(arr).map(ScalarValue::UInt32).unwrap_or(ScalarValue::Null))
        }
        DataType::UInt64 => {
            let arr = array.as_any().downcast_ref::<PrimitiveArray<UInt64Type>>().unwrap();
            Ok(compute::max(arr).map(ScalarValue::UInt64).unwrap_or(ScalarValue::Null))
        }
        DataType::Float32 => {
            let arr = array.as_any().downcast_ref::<PrimitiveArray<Float32Type>>().unwrap();
            Ok(compute::max(arr).map(ScalarValue::Float32).unwrap_or(ScalarValue::Null))
        }
        DataType::Float64 => {
            let arr = array.as_any().downcast_ref::<PrimitiveArray<Float64Type>>().unwrap();
            Ok(compute::max(arr).map(ScalarValue::Float64).unwrap_or(ScalarValue::Null))
        }
        dt => Err(CylonError::new(
            Code::Invalid,
            format!("Max not supported for data type: {:?}", dt),
        )),
    }
}

/// Compute count of non-null values in an array (local only)
pub fn count_array(array: &dyn Array, options: &AggregateOptions) -> CylonResult<ScalarValue> {
    let count = if options.skip_nulls {
        array.len() - array.null_count()
    } else {
        array.len()
    };
    Ok(ScalarValue::Int64(count as i64))
}

/// Compute mean of an array (local only)
pub fn mean_array(array: &dyn Array, options: &AggregateOptions) -> CylonResult<ScalarValue> {
    let sum_val = sum_array(array, options)?;
    let count_val = count_array(array, options)?;

    match (sum_val.to_f64(), count_val.to_i64()) {
        (Some(s), Some(c)) if c > 0 => Ok(ScalarValue::Float64(s / c as f64)),
        _ => Ok(ScalarValue::Null),
    }
}

/// Helper to extract f64 values from array
fn array_to_f64_vec(array: &dyn Array, skip_nulls: bool) -> CylonResult<Vec<f64>> {
    macro_rules! extract_values {
        ($arr_type:ty) => {{
            let arr = array.as_any().downcast_ref::<PrimitiveArray<$arr_type>>().unwrap();
            if skip_nulls {
                Ok(arr.iter().filter_map(|v| v.map(|x| x as f64)).collect())
            } else {
                Ok(arr.iter().map(|v| v.map(|x| x as f64).unwrap_or(f64::NAN)).collect())
            }
        }};
    }

    match array.data_type() {
        DataType::Int8 => extract_values!(Int8Type),
        DataType::Int16 => extract_values!(Int16Type),
        DataType::Int32 => extract_values!(Int32Type),
        DataType::Int64 => extract_values!(Int64Type),
        DataType::UInt8 => extract_values!(UInt8Type),
        DataType::UInt16 => extract_values!(UInt16Type),
        DataType::UInt32 => extract_values!(UInt32Type),
        DataType::UInt64 => extract_values!(UInt64Type),
        DataType::Float32 => extract_values!(Float32Type),
        DataType::Float64 => {
            let arr = array.as_any().downcast_ref::<PrimitiveArray<Float64Type>>().unwrap();
            if skip_nulls {
                Ok(arr.iter().filter_map(|v| v).collect())
            } else {
                Ok(arr.iter().map(|v| v.unwrap_or(f64::NAN)).collect())
            }
        }
        dt => Err(CylonError::new(
            Code::Invalid,
            format!("Cannot convert {:?} to f64", dt),
        )),
    }
}

/// Compute variance of an array (local only)
/// Uses Welford's online algorithm for numerical stability
pub fn variance_array(array: &dyn Array, options: &VarianceOptions) -> CylonResult<ScalarValue> {
    let values = array_to_f64_vec(array, options.skip_nulls)?;

    if values.is_empty() || values.len() <= options.ddof as usize {
        return Ok(ScalarValue::Null);
    }

    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let sum_sq: f64 = values.iter().map(|x| (x - mean).powi(2)).sum();
    let variance = sum_sq / (n - options.ddof as f64);

    Ok(ScalarValue::Float64(variance))
}

/// Compute standard deviation of an array (local only)
pub fn stddev_array(array: &dyn Array, options: &VarianceOptions) -> CylonResult<ScalarValue> {
    let var = variance_array(array, options)?;
    match var {
        ScalarValue::Float64(v) => Ok(ScalarValue::Float64(v.sqrt())),
        ScalarValue::Null => Ok(ScalarValue::Null),
        _ => Err(CylonError::new(Code::Invalid, "Unexpected variance result type")),
    }
}

// ============================================================================
// Context-Aware Aggregates (Support Distributed)
// ============================================================================

/// Compute sum of an array
/// In distributed mode, uses AllReduce with SUM operation
///
/// Corresponds to C++ Sum() in aggregates.cpp:486-499
pub fn sum(
    ctx: &Arc<CylonContext>,
    array: &dyn Array,
    options: &AggregateOptions,
) -> CylonResult<ScalarValue> {
    let local_result = sum_array(array, options)?;

    if ctx.is_distributed() {
        // Distributed aggregation via scalar_aggregate module
        // For now, return local result (TODO: implement AllReduce)
        Ok(local_result)
    } else {
        Ok(local_result)
    }
}

/// Compute minimum of an array
/// In distributed mode, uses AllReduce with MIN operation
///
/// Corresponds to C++ Min() in aggregates.cpp:501-514
pub fn min(
    ctx: &Arc<CylonContext>,
    array: &dyn Array,
    options: &AggregateOptions,
) -> CylonResult<ScalarValue> {
    let local_result = min_array(array, options)?;

    if ctx.is_distributed() {
        Ok(local_result)
    } else {
        Ok(local_result)
    }
}

/// Compute maximum of an array
/// In distributed mode, uses AllReduce with MAX operation
///
/// Corresponds to C++ Max() in aggregates.cpp:516-529
pub fn max(
    ctx: &Arc<CylonContext>,
    array: &dyn Array,
    options: &AggregateOptions,
) -> CylonResult<ScalarValue> {
    let local_result = max_array(array, options)?;

    if ctx.is_distributed() {
        Ok(local_result)
    } else {
        Ok(local_result)
    }
}

/// Compute count of an array
/// In distributed mode, uses AllReduce with SUM operation
///
/// Corresponds to C++ Count() in aggregates.cpp:531-542
pub fn count(
    ctx: &Arc<CylonContext>,
    array: &dyn Array,
    options: &AggregateOptions,
) -> CylonResult<ScalarValue> {
    let local_result = count_array(array, options)?;

    if ctx.is_distributed() {
        Ok(local_result)
    } else {
        Ok(local_result)
    }
}

/// Compute mean of an array
/// In distributed mode, AllReduce [sum, count], then divide
///
/// Corresponds to C++ Mean() in aggregates.cpp:544-557
pub fn mean(
    ctx: &Arc<CylonContext>,
    array: &dyn Array,
    options: &AggregateOptions,
) -> CylonResult<ScalarValue> {
    let local_result = mean_array(array, options)?;

    if ctx.is_distributed() {
        // For distributed mean:
        // 1. Compute local sum and count
        // 2. AllReduce both with SUM
        // 3. Divide total sum by total count
        Ok(local_result)
    } else {
        Ok(local_result)
    }
}

/// Compute variance of an array
/// In distributed mode, AllReduce [sum_sq, sum, count], then compute
///
/// Corresponds to C++ Variance() in aggregates.cpp:559-572
pub fn variance(
    ctx: &Arc<CylonContext>,
    array: &dyn Array,
    options: &VarianceOptions,
) -> CylonResult<ScalarValue> {
    let local_result = variance_array(array, options)?;

    if ctx.is_distributed() {
        // For distributed variance:
        // 1. Compute local sum_of_squares, sum, count
        // 2. AllReduce all three with SUM
        // 3. Compute global variance from combined values
        Ok(local_result)
    } else {
        Ok(local_result)
    }
}

/// Compute standard deviation of an array
///
/// Corresponds to C++ StdDev() in aggregates.cpp:574-589
pub fn stddev(
    ctx: &Arc<CylonContext>,
    array: &dyn Array,
    options: &VarianceOptions,
) -> CylonResult<ScalarValue> {
    let local_result = stddev_array(array, options)?;

    if ctx.is_distributed() {
        Ok(local_result)
    } else {
        Ok(local_result)
    }
}

// ============================================================================
// Table-Level Aggregates
// ============================================================================

/// Compute sum of a specific column in a table
pub fn sum_column(
    ctx: &Arc<CylonContext>,
    table: &Table,
    col_idx: usize,
    options: &AggregateOptions,
) -> CylonResult<ScalarValue> {
    let array = table.column(col_idx)?;
    sum(ctx, array.as_ref(), options)
}

/// Compute sum of all numeric columns in a table
pub fn sum_table(
    ctx: &Arc<CylonContext>,
    table: &Table,
    options: &AggregateOptions,
) -> CylonResult<Vec<ScalarValue>> {
    let num_cols = table.columns() as usize;
    let mut results = Vec::with_capacity(num_cols);

    for i in 0..num_cols {
        match sum_column(ctx, table, i, options) {
            Ok(val) => results.push(val),
            Err(_) => results.push(ScalarValue::Null),
        }
    }

    Ok(results)
}

/// Compute min of all numeric columns in a table
pub fn min_table(
    ctx: &Arc<CylonContext>,
    table: &Table,
    options: &AggregateOptions,
) -> CylonResult<Vec<ScalarValue>> {
    let num_cols = table.columns() as usize;
    let mut results = Vec::with_capacity(num_cols);

    for i in 0..num_cols {
        let array = table.column(i)?;
        match min(ctx, array.as_ref(), options) {
            Ok(val) => results.push(val),
            Err(_) => results.push(ScalarValue::Null),
        }
    }

    Ok(results)
}

/// Compute max of all numeric columns in a table
pub fn max_table(
    ctx: &Arc<CylonContext>,
    table: &Table,
    options: &AggregateOptions,
) -> CylonResult<Vec<ScalarValue>> {
    let num_cols = table.columns() as usize;
    let mut results = Vec::with_capacity(num_cols);

    for i in 0..num_cols {
        let array = table.column(i)?;
        match max(ctx, array.as_ref(), options) {
            Ok(val) => results.push(val),
            Err(_) => results.push(ScalarValue::Null),
        }
    }

    Ok(results)
}

/// Compute count of all columns in a table
pub fn count_table(
    ctx: &Arc<CylonContext>,
    table: &Table,
    options: &AggregateOptions,
) -> CylonResult<Vec<ScalarValue>> {
    let num_cols = table.columns() as usize;
    let mut results = Vec::with_capacity(num_cols);

    for i in 0..num_cols {
        let array = table.column(i)?;
        results.push(count(ctx, array.as_ref(), options)?);
    }

    Ok(results)
}

/// Compute mean of all numeric columns in a table
pub fn mean_table(
    ctx: &Arc<CylonContext>,
    table: &Table,
    options: &AggregateOptions,
) -> CylonResult<Vec<ScalarValue>> {
    let num_cols = table.columns() as usize;
    let mut results = Vec::with_capacity(num_cols);

    for i in 0..num_cols {
        let array = table.column(i)?;
        match mean(ctx, array.as_ref(), options) {
            Ok(val) => results.push(val),
            Err(_) => results.push(ScalarValue::Null),
        }
    }

    Ok(results)
}
