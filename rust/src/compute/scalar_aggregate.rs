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

//! Distributed scalar aggregate implementation
//!
//! Ported from cpp/src/cylon/compute/scalar_aggregate.cpp
//!
//! This module implements the distributed aggregation pattern:
//! 1. CombineLocally - Reduce local array to intermediate result(s)
//! 2. AllReduce - Combine intermediate results across all workers
//! 3. Finalize - Convert combined intermediate result to final scalar
//!
//! # Intermediate Result Sizes
//!
//! Different aggregations produce different numbers of intermediate values:
//! - Sum, Min, Max, Count: 1 value
//! - Mean: 2 values [sum, count]
//! - Variance, StdDev: 3 values [sum_of_squares, sum, count]

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Float64Array, PrimitiveArray};
use arrow::datatypes::{DataType, Float64Type};

use crate::ctx::CylonContext;
use crate::error::{Code, CylonError, CylonResult};
use crate::net::comm_operations::ReduceOp;

use super::aggregate_kernels::{BasicOptions, KernelOptions, VarKernelOptions};
use super::aggregates::ScalarValue;

// ============================================================================
// Scalar Aggregate Kernel Trait
// ============================================================================

/// Trait for scalar aggregate kernels that support distributed computation
///
/// Corresponds to C++ ScalarAggregateKernel in aggregates.hpp:135-168
///
/// The kernel operates in three phases:
/// 1. `combine_locally` - Reduce local array to intermediate array
/// 2. AllReduce intermediate arrays using `reduce_op()`
/// 3. `finalize` - Convert combined intermediate to final scalar
pub trait ScalarAggregateKernel: Send + Sync {
    /// Initialize the kernel with options
    fn init(&mut self, options: Option<&dyn KernelOptions>);

    /// Combine local values into intermediate result array
    ///
    /// The returned array contains `num_intermediate_results()` elements
    fn combine_locally(&self, values: &dyn Array) -> CylonResult<ArrayRef>;

    /// Finalize combined intermediate results to scalar
    fn finalize(&self, combined_results: &dyn Array) -> CylonResult<ScalarValue>;

    /// The reduce operation to use for AllReduce
    fn reduce_op(&self) -> ReduceOp;

    /// Number of intermediate results produced by combine_locally
    fn num_intermediate_results(&self) -> usize;

    /// Get output data type for the result
    fn output_type(&self, input_type: &DataType) -> DataType {
        input_type.clone()
    }
}

// ============================================================================
// Sum Kernel Implementation
// ============================================================================

/// Sum kernel - combines locally with sum, AllReduce with SUM
///
/// Corresponds to C++ SumKernelImpl in scalar_aggregate.cpp:77-79
pub struct SumKernelImpl {
    skip_nulls: bool,
}

impl Default for SumKernelImpl {
    fn default() -> Self {
        Self { skip_nulls: true }
    }
}

impl ScalarAggregateKernel for SumKernelImpl {
    fn init(&mut self, options: Option<&dyn KernelOptions>) {
        if let Some(opts) = options {
            if let Some(basic) = opts.as_any().downcast_ref::<BasicOptions>() {
                self.skip_nulls = basic.skip_nulls;
            }
        }
    }

    fn combine_locally(&self, values: &dyn Array) -> CylonResult<ArrayRef> {
        use arrow::compute;

        let sum_result = match values.data_type() {
            DataType::Float64 => {
                let arr = values.as_any().downcast_ref::<PrimitiveArray<Float64Type>>().unwrap();
                compute::sum(arr).unwrap_or(0.0)
            }
            _ => {
                // Cast to f64 for uniform intermediate representation
                let cast_arr = arrow::compute::cast(values, &DataType::Float64)?;
                let arr = cast_arr.as_any().downcast_ref::<Float64Array>().unwrap();
                compute::sum(arr).unwrap_or(0.0)
            }
        };

        Ok(Arc::new(Float64Array::from(vec![sum_result])))
    }

    fn finalize(&self, combined_results: &dyn Array) -> CylonResult<ScalarValue> {
        let arr = combined_results.as_any().downcast_ref::<Float64Array>()
            .ok_or_else(|| CylonError::new(Code::Invalid, "Expected Float64Array"))?;

        if arr.is_empty() || arr.is_null(0) {
            Ok(ScalarValue::Null)
        } else {
            Ok(ScalarValue::Float64(arr.value(0)))
        }
    }

    fn reduce_op(&self) -> ReduceOp {
        ReduceOp::Sum
    }

    fn num_intermediate_results(&self) -> usize {
        1
    }

    fn output_type(&self, _input_type: &DataType) -> DataType {
        DataType::Float64
    }
}

// ============================================================================
// Min Kernel Implementation
// ============================================================================

/// Min kernel - combines locally with min, AllReduce with MIN
///
/// Corresponds to C++ MinKernelImpl in scalar_aggregate.cpp:94-96
pub struct MinKernelImpl {
    skip_nulls: bool,
}

impl Default for MinKernelImpl {
    fn default() -> Self {
        Self { skip_nulls: true }
    }
}

impl ScalarAggregateKernel for MinKernelImpl {
    fn init(&mut self, options: Option<&dyn KernelOptions>) {
        if let Some(opts) = options {
            if let Some(basic) = opts.as_any().downcast_ref::<BasicOptions>() {
                self.skip_nulls = basic.skip_nulls;
            }
        }
    }

    fn combine_locally(&self, values: &dyn Array) -> CylonResult<ArrayRef> {
        use arrow::compute;

        let min_result = match values.data_type() {
            DataType::Float64 => {
                let arr = values.as_any().downcast_ref::<PrimitiveArray<Float64Type>>().unwrap();
                compute::min(arr)
            }
            _ => {
                let cast_arr = arrow::compute::cast(values, &DataType::Float64)?;
                let arr = cast_arr.as_any().downcast_ref::<Float64Array>().unwrap();
                compute::min(arr)
            }
        };

        match min_result {
            Some(v) => Ok(Arc::new(Float64Array::from(vec![v]))),
            None => Ok(Arc::new(Float64Array::from(vec![f64::MAX]))), // Sentinel for AllReduce
        }
    }

    fn finalize(&self, combined_results: &dyn Array) -> CylonResult<ScalarValue> {
        let arr = combined_results.as_any().downcast_ref::<Float64Array>()
            .ok_or_else(|| CylonError::new(Code::Invalid, "Expected Float64Array"))?;

        if arr.is_empty() || arr.is_null(0) {
            Ok(ScalarValue::Null)
        } else {
            let val = arr.value(0);
            if val == f64::MAX {
                Ok(ScalarValue::Null) // Was sentinel
            } else {
                Ok(ScalarValue::Float64(val))
            }
        }
    }

    fn reduce_op(&self) -> ReduceOp {
        ReduceOp::Min
    }

    fn num_intermediate_results(&self) -> usize {
        1
    }
}

// ============================================================================
// Max Kernel Implementation
// ============================================================================

/// Max kernel - combines locally with max, AllReduce with MAX
///
/// Corresponds to C++ MaxKernelImpl in scalar_aggregate.cpp:97-99
pub struct MaxKernelImpl {
    skip_nulls: bool,
}

impl Default for MaxKernelImpl {
    fn default() -> Self {
        Self { skip_nulls: true }
    }
}

impl ScalarAggregateKernel for MaxKernelImpl {
    fn init(&mut self, options: Option<&dyn KernelOptions>) {
        if let Some(opts) = options {
            if let Some(basic) = opts.as_any().downcast_ref::<BasicOptions>() {
                self.skip_nulls = basic.skip_nulls;
            }
        }
    }

    fn combine_locally(&self, values: &dyn Array) -> CylonResult<ArrayRef> {
        use arrow::compute;

        let max_result = match values.data_type() {
            DataType::Float64 => {
                let arr = values.as_any().downcast_ref::<PrimitiveArray<Float64Type>>().unwrap();
                compute::max(arr)
            }
            _ => {
                let cast_arr = arrow::compute::cast(values, &DataType::Float64)?;
                let arr = cast_arr.as_any().downcast_ref::<Float64Array>().unwrap();
                compute::max(arr)
            }
        };

        match max_result {
            Some(v) => Ok(Arc::new(Float64Array::from(vec![v]))),
            None => Ok(Arc::new(Float64Array::from(vec![f64::MIN]))), // Sentinel for AllReduce
        }
    }

    fn finalize(&self, combined_results: &dyn Array) -> CylonResult<ScalarValue> {
        let arr = combined_results.as_any().downcast_ref::<Float64Array>()
            .ok_or_else(|| CylonError::new(Code::Invalid, "Expected Float64Array"))?;

        if arr.is_empty() || arr.is_null(0) {
            Ok(ScalarValue::Null)
        } else {
            let val = arr.value(0);
            if val == f64::MIN {
                Ok(ScalarValue::Null)
            } else {
                Ok(ScalarValue::Float64(val))
            }
        }
    }

    fn reduce_op(&self) -> ReduceOp {
        ReduceOp::Max
    }

    fn num_intermediate_results(&self) -> usize {
        1
    }
}

// ============================================================================
// Count Kernel Implementation
// ============================================================================

/// Count kernel - combines locally with count, AllReduce with SUM
///
/// Corresponds to C++ CountKernelImpl in scalar_aggregate.cpp:110-118
pub struct CountKernelImpl;

impl Default for CountKernelImpl {
    fn default() -> Self {
        Self
    }
}

impl ScalarAggregateKernel for CountKernelImpl {
    fn init(&mut self, _options: Option<&dyn KernelOptions>) {}

    fn combine_locally(&self, values: &dyn Array) -> CylonResult<ArrayRef> {
        let count = values.len() as f64;
        Ok(Arc::new(Float64Array::from(vec![count])))
    }

    fn finalize(&self, combined_results: &dyn Array) -> CylonResult<ScalarValue> {
        let arr = combined_results.as_any().downcast_ref::<Float64Array>()
            .ok_or_else(|| CylonError::new(Code::Invalid, "Expected Float64Array"))?;

        if arr.is_empty() {
            Ok(ScalarValue::Int64(0))
        } else {
            Ok(ScalarValue::Int64(arr.value(0) as i64))
        }
    }

    fn reduce_op(&self) -> ReduceOp {
        ReduceOp::Sum
    }

    fn num_intermediate_results(&self) -> usize {
        1
    }

    fn output_type(&self, _input_type: &DataType) -> DataType {
        DataType::Int64
    }
}

// ============================================================================
// Mean Kernel Implementation
// ============================================================================

/// Mean kernel - intermediate: [sum, count], AllReduce with SUM
///
/// Corresponds to C++ MeanKernelImpl in scalar_aggregate.cpp:120-179
pub struct MeanKernelImpl {
    skip_nulls: bool,
}

impl Default for MeanKernelImpl {
    fn default() -> Self {
        Self { skip_nulls: true }
    }
}

impl ScalarAggregateKernel for MeanKernelImpl {
    fn init(&mut self, options: Option<&dyn KernelOptions>) {
        if let Some(opts) = options {
            if let Some(basic) = opts.as_any().downcast_ref::<BasicOptions>() {
                self.skip_nulls = basic.skip_nulls;
            }
        }
    }

    fn combine_locally(&self, values: &dyn Array) -> CylonResult<ArrayRef> {
        use arrow::compute;

        // Cast to f64 for computation
        let cast_arr = arrow::compute::cast(values, &DataType::Float64)?;
        let arr = cast_arr.as_any().downcast_ref::<Float64Array>().unwrap();

        let sum_val = compute::sum(arr).unwrap_or(0.0);
        let count_val = arr.len() as f64;

        // Return [sum, count]
        Ok(Arc::new(Float64Array::from(vec![sum_val, count_val])))
    }

    fn finalize(&self, combined_results: &dyn Array) -> CylonResult<ScalarValue> {
        let arr = combined_results.as_any().downcast_ref::<Float64Array>()
            .ok_or_else(|| CylonError::new(Code::Invalid, "Expected Float64Array"))?;

        if arr.len() < 2 {
            return Ok(ScalarValue::Null);
        }

        let sum_val = arr.value(0);
        let count_val = arr.value(1);

        if count_val == 0.0 {
            Ok(ScalarValue::Null)
        } else {
            Ok(ScalarValue::Float64(sum_val / count_val))
        }
    }

    fn reduce_op(&self) -> ReduceOp {
        ReduceOp::Sum
    }

    fn num_intermediate_results(&self) -> usize {
        2
    }

    fn output_type(&self, _input_type: &DataType) -> DataType {
        DataType::Float64
    }
}

// ============================================================================
// Variance Kernel Implementation
// ============================================================================

/// Variance kernel - intermediate: [sum_sq, sum, count], AllReduce with SUM
///
/// Corresponds to C++ VarianceKernelImpl in scalar_aggregate.cpp:181-268
pub struct VarianceKernelImpl {
    ddof: i32,
    do_std: bool,
    skip_nulls: bool,
}

impl VarianceKernelImpl {
    pub fn variance(ddof: i32) -> Self {
        Self { ddof, do_std: false, skip_nulls: true }
    }

    pub fn stddev(ddof: i32) -> Self {
        Self { ddof, do_std: true, skip_nulls: true }
    }
}

impl Default for VarianceKernelImpl {
    fn default() -> Self {
        Self::variance(0)
    }
}

impl ScalarAggregateKernel for VarianceKernelImpl {
    fn init(&mut self, options: Option<&dyn KernelOptions>) {
        if let Some(opts) = options {
            if let Some(var_opts) = opts.as_any().downcast_ref::<VarKernelOptions>() {
                self.ddof = var_opts.ddof;
                self.skip_nulls = var_opts.skip_nulls;
            }
        }
    }

    fn combine_locally(&self, values: &dyn Array) -> CylonResult<ArrayRef> {
        // Cast to f64 for computation
        let cast_arr = arrow::compute::cast(values, &DataType::Float64)?;
        let arr = cast_arr.as_any().downcast_ref::<Float64Array>().unwrap();

        let mut sum_sq = 0.0;
        let mut sum_val = 0.0;
        let mut count = 0.0;

        for i in 0..arr.len() {
            if !arr.is_null(i) || !self.skip_nulls {
                let v = arr.value(i);
                sum_sq += v * v;
                sum_val += v;
                count += 1.0;
            }
        }

        // Return [sum_of_squares, sum, count]
        Ok(Arc::new(Float64Array::from(vec![sum_sq, sum_val, count])))
    }

    fn finalize(&self, combined_results: &dyn Array) -> CylonResult<ScalarValue> {
        let arr = combined_results.as_any().downcast_ref::<Float64Array>()
            .ok_or_else(|| CylonError::new(Code::Invalid, "Expected Float64Array"))?;

        if arr.len() < 3 {
            return Ok(ScalarValue::Null);
        }

        let sum_sq = arr.value(0);
        let sum_val = arr.value(1);
        let count = arr.value(2);

        if count <= self.ddof as f64 {
            return Ok(ScalarValue::Null);
        }

        let mean = sum_val / count;
        let mean_sq = sum_sq / count;
        let var = (mean_sq - mean * mean) * count / (count - self.ddof as f64);

        let result = if self.do_std { var.sqrt() } else { var };
        Ok(ScalarValue::Float64(result))
    }

    fn reduce_op(&self) -> ReduceOp {
        ReduceOp::Sum
    }

    fn num_intermediate_results(&self) -> usize {
        3
    }

    fn output_type(&self, _input_type: &DataType) -> DataType {
        DataType::Float64
    }
}

// ============================================================================
// Distributed Scalar Aggregate Function
// ============================================================================

/// Execute a distributed scalar aggregation
///
/// Corresponds to C++ ScalarAggregate() in scalar_aggregate.cpp:280-314
///
/// This function:
/// 1. Calls kernel.combine_locally() to get intermediate results
/// 2. If distributed, AllReduces the intermediate results
/// 3. Calls kernel.finalize() to produce the final scalar
pub fn scalar_aggregate(
    ctx: &Arc<CylonContext>,
    kernel: &dyn ScalarAggregateKernel,
    values: &dyn Array,
) -> CylonResult<ScalarValue> {
    // Step 1: Combine locally
    let combined = kernel.combine_locally(values)?;

    // Step 2: AllReduce if distributed
    let final_combined = if ctx.is_distributed() && ctx.get_world_size() > 1 {
        // TODO: Implement AllReduce for arrays when available
        // For now, return local combined result
        // let reduced = ctx.get_communicator()?.allreduce_array(&combined, kernel.reduce_op())?;
        combined
    } else {
        combined
    };

    // Step 3: Finalize
    kernel.finalize(final_combined.as_ref())
}

// ============================================================================
// Factory Functions
// ============================================================================

/// Create a scalar aggregate kernel for the given operation
pub fn create_scalar_aggregate_kernel(
    op: super::aggregate_kernels::AggregationOpId,
) -> Box<dyn ScalarAggregateKernel> {
    use super::aggregate_kernels::AggregationOpId;

    match op {
        AggregationOpId::Sum => Box::new(SumKernelImpl::default()),
        AggregationOpId::Min => Box::new(MinKernelImpl::default()),
        AggregationOpId::Max => Box::new(MaxKernelImpl::default()),
        AggregationOpId::Count => Box::new(CountKernelImpl::default()),
        AggregationOpId::Mean => Box::new(MeanKernelImpl::default()),
        AggregationOpId::Var => Box::new(VarianceKernelImpl::variance(0)),
        AggregationOpId::StdDev => Box::new(VarianceKernelImpl::stddev(0)),
        _ => Box::new(SumKernelImpl::default()), // Fallback
    }
}
