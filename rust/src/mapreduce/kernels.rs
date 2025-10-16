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

//! Aggregation kernel implementations
//!
//! Ported from cpp/src/cylon/mapreduce/mapreduce.cpp

use arrow::array::{
    Array, ArrayRef, Float64Array, Int64Array, PrimitiveArray, ArrowPrimitiveType,
};
use arrow::datatypes::DataType;
use std::sync::Arc;

use crate::error::{CylonResult, CylonError, Code};
use super::MapReduceKernel;

/// Helper trait for numeric operations
trait NumericOp<T: Copy> {
    fn call(x: T, y: T) -> T;
}

/// Sum operation
struct SumOp<T>(std::marker::PhantomData<T>);
impl<T: Copy + std::ops::Add<Output = T>> NumericOp<T> for SumOp<T> {
    fn call(x: T, y: T) -> T {
        x + y
    }
}

/// Min operation
struct MinOp<T>(std::marker::PhantomData<T>);
impl<T: Copy + PartialOrd> NumericOp<T> for MinOp<T> {
    fn call(x: T, y: T) -> T {
        if x < y { x } else { y }
    }
}

/// Max operation
struct MaxOp<T>(std::marker::PhantomData<T>);
impl<T: Copy + PartialOrd> NumericOp<T> for MaxOp<T> {
    fn call(x: T, y: T) -> T {
        if x > y { x } else { y }
    }
}

/// Base implementation for 1D MapReduce kernels
/// Corresponds to C++ MapReduceKernelImpl1D
struct MapReduceKernel1D<T, Op>
where
    T: ArrowPrimitiveType,
    Op: NumericOp<T::Native>,
{
    name: String,
    output_type: DataType,
    _phantom: std::marker::PhantomData<(T, Op)>,
}

impl<T, Op> MapReduceKernel1D<T, Op>
where
    T: ArrowPrimitiveType,
    Op: NumericOp<T::Native>,
{
    fn new(name: String, output_type: DataType) -> Self {
        Self {
            name,
            output_type,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T, Op> MapReduceKernel for MapReduceKernel1D<T, Op>
where
    T: ArrowPrimitiveType + Send + Sync + 'static,
    T::Native: Send + Sync + Default,
    Op: NumericOp<T::Native> + Send + Sync + 'static,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn output_type(&self) -> &DataType {
        &self.output_type
    }

    fn intermediate_types(&self) -> &[DataType] {
        std::slice::from_ref(&self.output_type)
    }

    fn combine_locally(
        &self,
        value_col: &ArrayRef,
        local_group_ids: &ArrayRef,
        local_num_groups: i64,
        combined_results: &mut Vec<ArrayRef>,
    ) -> CylonResult<()> {
        let value_arr = value_col
            .as_any()
            .downcast_ref::<PrimitiveArray<T>>()
            .ok_or_else(|| CylonError::new(Code::Invalid, "Invalid array type".to_string()))?;

        let group_ids = local_group_ids
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| CylonError::new(Code::Invalid, "Invalid group ids type".to_string()))?;

        // Allocate result array with default values
        let mut result_data = vec![T::Native::default(); local_num_groups as usize];
        // Track which groups have been initialized
        let mut initialized = vec![false; local_num_groups as usize];

        // Combine values by group
        for i in 0..value_arr.len() {
            if !value_arr.is_null(i) {
                let gid = group_ids.value(i) as usize;
                let val = value_arr.value(i);

                if !initialized[gid] {
                    // First value for this group - initialize with it
                    result_data[gid] = val;
                    initialized[gid] = true;
                } else {
                    // Subsequent values - combine
                    result_data[gid] = Op::call(result_data[gid], val);
                }
            }
        }

        // Create result array
        let result_arr = PrimitiveArray::<T>::from_iter_values(result_data.into_iter());
        combined_results.clear();
        combined_results.push(Arc::new(result_arr));

        Ok(())
    }

    fn reduce_shuffled_results(
        &self,
        combined_results: &[ArrayRef],
        local_group_ids: &ArrayRef,
        _local_group_indices: &ArrayRef,
        local_num_groups: i64,
        reduced_results: &mut Vec<ArrayRef>,
    ) -> CylonResult<()> {
        // For 1D kernels, reduce is the same as combine
        self.combine_locally(&combined_results[0], local_group_ids, local_num_groups, reduced_results)
    }

    fn finalize(
        &self,
        combined_results: &[ArrayRef],
        output: &mut Option<ArrayRef>,
    ) -> CylonResult<()> {
        *output = Some(combined_results[0].clone());
        Ok(())
    }
}

/// Sum kernel
/// Corresponds to C++ SumKernelImpl
pub struct SumKernel<T>
where
    T: ArrowPrimitiveType,
    T::Native: std::ops::Add<Output = T::Native> + Copy,
{
    inner: MapReduceKernel1D<T, SumOp<T::Native>>,
}

impl<T> SumKernel<T>
where
    T: ArrowPrimitiveType,
    T::Native: std::ops::Add<Output = T::Native> + Copy,
{
    pub fn new(data_type: DataType) -> Self {
        Self {
            inner: MapReduceKernel1D::new("sum".to_string(), data_type),
        }
    }
}

impl<T> MapReduceKernel for SumKernel<T>
where
    T: ArrowPrimitiveType + Send + Sync + 'static,
    T::Native: std::ops::Add<Output = T::Native> + Send + Sync + Copy,
{
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn output_type(&self) -> &DataType {
        self.inner.output_type()
    }

    fn intermediate_types(&self) -> &[DataType] {
        self.inner.intermediate_types()
    }

    fn combine_locally(
        &self,
        value_col: &ArrayRef,
        local_group_ids: &ArrayRef,
        local_num_groups: i64,
        combined_results: &mut Vec<ArrayRef>,
    ) -> CylonResult<()> {
        self.inner.combine_locally(value_col, local_group_ids, local_num_groups, combined_results)
    }

    fn reduce_shuffled_results(
        &self,
        combined_results: &[ArrayRef],
        local_group_ids: &ArrayRef,
        local_group_indices: &ArrayRef,
        local_num_groups: i64,
        reduced_results: &mut Vec<ArrayRef>,
    ) -> CylonResult<()> {
        self.inner.reduce_shuffled_results(
            combined_results,
            local_group_ids,
            local_group_indices,
            local_num_groups,
            reduced_results,
        )
    }

    fn finalize(
        &self,
        combined_results: &[ArrayRef],
        output: &mut Option<ArrayRef>,
    ) -> CylonResult<()> {
        self.inner.finalize(combined_results, output)
    }
}

/// Min kernel
/// Corresponds to C++ MinKernelImpl
pub struct MinKernel<T>
where
    T: ArrowPrimitiveType,
    T::Native: PartialOrd + Copy,
{
    inner: MapReduceKernel1D<T, MinOp<T::Native>>,
}

impl<T> MinKernel<T>
where
    T: ArrowPrimitiveType,
    T::Native: PartialOrd + Copy,
{
    pub fn new(data_type: DataType) -> Self {
        Self {
            inner: MapReduceKernel1D::new("min".to_string(), data_type),
        }
    }
}

impl<T> MapReduceKernel for MinKernel<T>
where
    T: ArrowPrimitiveType + Send + Sync + 'static,
    T::Native: PartialOrd + Send + Sync + Copy,
{
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn output_type(&self) -> &DataType {
        self.inner.output_type()
    }

    fn intermediate_types(&self) -> &[DataType] {
        self.inner.intermediate_types()
    }

    fn combine_locally(
        &self,
        value_col: &ArrayRef,
        local_group_ids: &ArrayRef,
        local_num_groups: i64,
        combined_results: &mut Vec<ArrayRef>,
    ) -> CylonResult<()> {
        self.inner.combine_locally(value_col, local_group_ids, local_num_groups, combined_results)
    }

    fn reduce_shuffled_results(
        &self,
        combined_results: &[ArrayRef],
        local_group_ids: &ArrayRef,
        local_group_indices: &ArrayRef,
        local_num_groups: i64,
        reduced_results: &mut Vec<ArrayRef>,
    ) -> CylonResult<()> {
        self.inner.reduce_shuffled_results(
            combined_results,
            local_group_ids,
            local_group_indices,
            local_num_groups,
            reduced_results,
        )
    }

    fn finalize(
        &self,
        combined_results: &[ArrayRef],
        output: &mut Option<ArrayRef>,
    ) -> CylonResult<()> {
        self.inner.finalize(combined_results, output)
    }
}

/// Max kernel
/// Corresponds to C++ MaxKernelImpl
pub struct MaxKernel<T>
where
    T: ArrowPrimitiveType,
    T::Native: PartialOrd + Copy,
{
    inner: MapReduceKernel1D<T, MaxOp<T::Native>>,
}

impl<T> MaxKernel<T>
where
    T: ArrowPrimitiveType,
    T::Native: PartialOrd + Copy,
{
    pub fn new(data_type: DataType) -> Self {
        Self {
            inner: MapReduceKernel1D::new("max".to_string(), data_type),
        }
    }
}

impl<T> MapReduceKernel for MaxKernel<T>
where
    T: ArrowPrimitiveType + Send + Sync + 'static,
    T::Native: PartialOrd + Send + Sync + Copy,
{
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn output_type(&self) -> &DataType {
        self.inner.output_type()
    }

    fn intermediate_types(&self) -> &[DataType] {
        self.inner.intermediate_types()
    }

    fn combine_locally(
        &self,
        value_col: &ArrayRef,
        local_group_ids: &ArrayRef,
        local_num_groups: i64,
        combined_results: &mut Vec<ArrayRef>,
    ) -> CylonResult<()> {
        self.inner.combine_locally(value_col, local_group_ids, local_num_groups, combined_results)
    }

    fn reduce_shuffled_results(
        &self,
        combined_results: &[ArrayRef],
        local_group_ids: &ArrayRef,
        local_group_indices: &ArrayRef,
        local_num_groups: i64,
        reduced_results: &mut Vec<ArrayRef>,
    ) -> CylonResult<()> {
        self.inner.reduce_shuffled_results(
            combined_results,
            local_group_ids,
            local_group_indices,
            local_num_groups,
            reduced_results,
        )
    }

    fn finalize(
        &self,
        combined_results: &[ArrayRef],
        output: &mut Option<ArrayRef>,
    ) -> CylonResult<()> {
        self.inner.finalize(combined_results, output)
    }
}

/// Count kernel
/// Corresponds to C++ CountKernelImpl
pub struct CountKernel {
    output_type: DataType,
}

impl CountKernel {
    pub fn new() -> Self {
        Self {
            output_type: DataType::Int64,
        }
    }
}

impl Default for CountKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl MapReduceKernel for CountKernel {
    fn name(&self) -> &str {
        "count"
    }

    fn output_type(&self) -> &DataType {
        &self.output_type
    }

    fn intermediate_types(&self) -> &[DataType] {
        std::slice::from_ref(&self.output_type)
    }

    fn combine_locally(
        &self,
        value_col: &ArrayRef,
        local_group_ids: &ArrayRef,
        local_num_groups: i64,
        combined_results: &mut Vec<ArrayRef>,
    ) -> CylonResult<()> {
        let group_ids = local_group_ids
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| CylonError::new(Code::Invalid, "Invalid group ids type".to_string()))?;

        // Count occurrences per group
        let mut counts = vec![0i64; local_num_groups as usize];
        for i in 0..value_col.len() {
            let gid = group_ids.value(i) as usize;
            counts[gid] += 1;
        }

        let result_arr = Int64Array::from_iter_values(counts.into_iter());
        combined_results.clear();
        combined_results.push(Arc::new(result_arr));

        Ok(())
    }

    fn reduce_shuffled_results(
        &self,
        combined_results: &[ArrayRef],
        local_group_ids: &ArrayRef,
        _local_group_indices: &ArrayRef,
        local_num_groups: i64,
        reduced_results: &mut Vec<ArrayRef>,
    ) -> CylonResult<()> {
        let combined = combined_results[0]
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| CylonError::new(Code::Invalid, "Invalid array type".to_string()))?;

        let group_ids = local_group_ids
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| CylonError::new(Code::Invalid, "Invalid group ids type".to_string()))?;

        // Sum counts per group
        let mut result = vec![0i64; local_num_groups as usize];
        for i in 0..combined.len() {
            let gid = group_ids.value(i) as usize;
            let val = combined.value(i);
            result[gid] += val;
        }

        let result_arr = Int64Array::from_iter_values(result.into_iter());
        reduced_results.clear();
        reduced_results.push(Arc::new(result_arr));

        Ok(())
    }

    fn finalize(
        &self,
        combined_results: &[ArrayRef],
        output: &mut Option<ArrayRef>,
    ) -> CylonResult<()> {
        *output = Some(combined_results[0].clone());
        Ok(())
    }
}

/// Mean kernel
/// Corresponds to C++ MeanKernelImpl
/// Uses two intermediate arrays: sum and count
pub struct MeanKernel<T: ArrowPrimitiveType> {
    input_type: DataType,
    intermediate_types: Vec<DataType>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: ArrowPrimitiveType> MeanKernel<T> {
    pub fn new(input_type: DataType) -> Self {
        Self {
            input_type: input_type.clone(),
            intermediate_types: vec![input_type, DataType::Int64],
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T> MapReduceKernel for MeanKernel<T>
where
    T: ArrowPrimitiveType + Send + Sync + 'static,
    T::Native: std::ops::Add<Output = T::Native> + std::ops::Div<Output = T::Native> + num_traits::NumCast + Default + Send + Sync + Copy,
{
    fn name(&self) -> &str {
        "mean"
    }

    fn output_type(&self) -> &DataType {
        &self.intermediate_types[0]
    }

    fn intermediate_types(&self) -> &[DataType] {
        &self.intermediate_types
    }

    fn combine_locally(
        &self,
        value_col: &ArrayRef,
        local_group_ids: &ArrayRef,
        local_num_groups: i64,
        combined_results: &mut Vec<ArrayRef>,
    ) -> CylonResult<()> {
        let value_arr = value_col
            .as_any()
            .downcast_ref::<PrimitiveArray<T>>()
            .ok_or_else(|| CylonError::new(Code::Invalid, "Invalid array type".to_string()))?;

        let group_ids = local_group_ids
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| CylonError::new(Code::Invalid, "Invalid group ids type".to_string()))?;

        // Allocate sum and count arrays
        let mut sums = vec![T::Native::default(); local_num_groups as usize];
        let mut counts = vec![0i64; local_num_groups as usize];

        // Combine values by group
        for i in 0..value_arr.len() {
            if !value_arr.is_null(i) {
                let gid = group_ids.value(i) as usize;
                let val = value_arr.value(i);
                sums[gid] = sums[gid] + val;
                counts[gid] += 1;
            }
        }

        // Create result arrays
        let sum_arr = PrimitiveArray::<T>::from_iter_values(sums.into_iter());
        let count_arr = Int64Array::from_iter_values(counts.into_iter());

        combined_results.clear();
        combined_results.push(Arc::new(sum_arr));
        combined_results.push(Arc::new(count_arr));

        Ok(())
    }

    fn reduce_shuffled_results(
        &self,
        combined_results: &[ArrayRef],
        local_group_ids: &ArrayRef,
        _local_group_indices: &ArrayRef,
        local_num_groups: i64,
        reduced_results: &mut Vec<ArrayRef>,
    ) -> CylonResult<()> {
        let sum_arr = combined_results[0]
            .as_any()
            .downcast_ref::<PrimitiveArray<T>>()
            .ok_or_else(|| CylonError::new(Code::Invalid, "Invalid sum array type".to_string()))?;

        let count_arr = combined_results[1]
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| CylonError::new(Code::Invalid, "Invalid count array type".to_string()))?;

        let group_ids = local_group_ids
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| CylonError::new(Code::Invalid, "Invalid group ids type".to_string()))?;

        // Allocate result arrays
        let mut result_sums = vec![T::Native::default(); local_num_groups as usize];
        let mut result_counts = vec![0i64; local_num_groups as usize];

        // Reduce by group
        for i in 0..sum_arr.len() {
            let gid = group_ids.value(i) as usize;
            result_sums[gid] = result_sums[gid] + sum_arr.value(i);
            result_counts[gid] += count_arr.value(i);
        }

        let result_sum_arr = PrimitiveArray::<T>::from_iter_values(result_sums.into_iter());
        let result_count_arr = Int64Array::from_iter_values(result_counts.into_iter());

        reduced_results.clear();
        reduced_results.push(Arc::new(result_sum_arr));
        reduced_results.push(Arc::new(result_count_arr));

        Ok(())
    }

    fn finalize(
        &self,
        combined_results: &[ArrayRef],
        output: &mut Option<ArrayRef>,
    ) -> CylonResult<()> {
        let sum_arr = combined_results[0]
            .as_any()
            .downcast_ref::<PrimitiveArray<T>>()
            .ok_or_else(|| CylonError::new(Code::Invalid, "Invalid sum array type".to_string()))?;

        let count_arr = combined_results[1]
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| CylonError::new(Code::Invalid, "Invalid count array type".to_string()))?;

        // Calculate mean = sum / count (modify in place to match C++)
        // For integers this does integer division (truncation), for floats it's proper division
        // This matches C++: sums[i] = static_cast<T>(sums[i] / counts[i])

        // We need to get mutable access to the sum array data
        // Since we can't modify the Arrow array directly, we'll create new values
        let num_groups = sum_arr.len();
        let mut means_data = Vec::with_capacity(num_groups);

        for i in 0..num_groups {
            let sum = sum_arr.value(i);
            let count = count_arr.value(i);

            // Cast count to the same type as sum, then divide
            // For integer types, this truncates. For float types, it's proper division.
            // We use num_traits to handle the conversion generically
            if let Some(count_as_t) = num_traits::NumCast::from(count) {
                means_data.push(sum / count_as_t);
            } else {
                // Fallback: shouldn't happen for standard numeric types
                means_data.push(T::Native::default());
            }
        }

        let means = means_data;

        let mean_arr = PrimitiveArray::<T>::from_iter_values(means.into_iter());
        *output = Some(Arc::new(mean_arr));

        Ok(())
    }
}

/// Variance kernel
/// Corresponds to C++ VarKernelImpl<ArrowT, false>
/// Uses three intermediate arrays: sum_of_squares, sum, and count
pub struct VarKernel<T: ArrowPrimitiveType> {
    ddof: i32, // delta degrees of freedom
    intermediate_types: Vec<DataType>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: ArrowPrimitiveType> VarKernel<T> {
    pub fn new(ddof: i32) -> Self {
        Self {
            ddof,
            intermediate_types: vec![DataType::Float64, DataType::Float64, DataType::Int64],
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T> MapReduceKernel for VarKernel<T>
where
    T: ArrowPrimitiveType + Send + Sync + 'static,
    T::Native: Into<f64> + Copy + Send + Sync,
{
    fn name(&self) -> &str {
        "var"
    }

    fn output_type(&self) -> &DataType {
        &self.intermediate_types[0]
    }

    fn intermediate_types(&self) -> &[DataType] {
        &self.intermediate_types
    }

    fn combine_locally(
        &self,
        value_col: &ArrayRef,
        local_group_ids: &ArrayRef,
        local_num_groups: i64,
        combined_results: &mut Vec<ArrayRef>,
    ) -> CylonResult<()> {
        let value_arr = value_col
            .as_any()
            .downcast_ref::<PrimitiveArray<T>>()
            .ok_or_else(|| CylonError::new(Code::Invalid, "Invalid array type".to_string()))?;

        let group_ids = local_group_ids
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| CylonError::new(Code::Invalid, "Invalid group ids type".to_string()))?;

        // Allocate arrays: sum_of_squares, sum, count
        let mut sq_sums = vec![0.0f64; local_num_groups as usize];
        let mut sums = vec![0.0f64; local_num_groups as usize];
        let mut counts = vec![0i64; local_num_groups as usize];

        // Combine values by group
        for i in 0..value_arr.len() {
            if !value_arr.is_null(i) {
                let gid = group_ids.value(i) as usize;
                let val: f64 = value_arr.value(i).into();
                sq_sums[gid] += val * val;
                sums[gid] += val;
                counts[gid] += 1;
            }
        }

        let sq_sum_arr = Float64Array::from_iter_values(sq_sums.into_iter());
        let sum_arr = Float64Array::from_iter_values(sums.into_iter());
        let count_arr = Int64Array::from_iter_values(counts.into_iter());

        combined_results.clear();
        combined_results.push(Arc::new(sq_sum_arr));
        combined_results.push(Arc::new(sum_arr));
        combined_results.push(Arc::new(count_arr));

        Ok(())
    }

    fn reduce_shuffled_results(
        &self,
        combined_results: &[ArrayRef],
        local_group_ids: &ArrayRef,
        _local_group_indices: &ArrayRef,
        local_num_groups: i64,
        reduced_results: &mut Vec<ArrayRef>,
    ) -> CylonResult<()> {
        use arrow::array::Float64Array;

        let sq_sum_arr = combined_results[0]
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| CylonError::new(Code::Invalid, "Invalid sq_sum array type".to_string()))?;

        let sum_arr = combined_results[1]
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| CylonError::new(Code::Invalid, "Invalid sum array type".to_string()))?;

        let count_arr = combined_results[2]
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| CylonError::new(Code::Invalid, "Invalid count array type".to_string()))?;

        let group_ids = local_group_ids
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| CylonError::new(Code::Invalid, "Invalid group ids type".to_string()))?;

        // Allocate result arrays
        let mut result_sq_sums = vec![0.0f64; local_num_groups as usize];
        let mut result_sums = vec![0.0f64; local_num_groups as usize];
        let mut result_counts = vec![0i64; local_num_groups as usize];

        // Reduce by group
        for i in 0..sq_sum_arr.len() {
            let gid = group_ids.value(i) as usize;
            result_sq_sums[gid] += sq_sum_arr.value(i);
            result_sums[gid] += sum_arr.value(i);
            result_counts[gid] += count_arr.value(i);
        }

        let result_sq_sum_arr = Float64Array::from_iter_values(result_sq_sums.into_iter());
        let result_sum_arr = Float64Array::from_iter_values(result_sums.into_iter());
        let result_count_arr = Int64Array::from_iter_values(result_counts.into_iter());

        reduced_results.clear();
        reduced_results.push(Arc::new(result_sq_sum_arr));
        reduced_results.push(Arc::new(result_sum_arr));
        reduced_results.push(Arc::new(result_count_arr));

        Ok(())
    }

    fn finalize(
        &self,
        combined_results: &[ArrayRef],
        output: &mut Option<ArrayRef>,
    ) -> CylonResult<()> {
        use arrow::array::Float64Array;

        let sq_sum_arr = combined_results[0]
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| CylonError::new(Code::Invalid, "Invalid sq_sum array type".to_string()))?;

        let sum_arr = combined_results[1]
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| CylonError::new(Code::Invalid, "Invalid sum array type".to_string()))?;

        let count_arr = combined_results[2]
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| CylonError::new(Code::Invalid, "Invalid count array type".to_string()))?;

        // Calculate variance: var = (sum_sq / count - (sum / count)^2) * count / (count - ddof)
        let variances: Vec<f64> = (0..sq_sum_arr.len())
            .map(|i| {
                let count = count_arr.value(i) as f64;
                let sq_sum = sq_sum_arr.value(i);
                let sum = sum_arr.value(i);

                if count > 0.0 {
                    let mean = sum / count;
                    let mean_sum_sq = sq_sum / count;
                    sq_sum * (mean_sum_sq - mean * mean) / (count - self.ddof as f64)
                } else {
                    0.0
                }
            })
            .collect();

        let var_arr = Float64Array::from_iter_values(variances.into_iter());
        *output = Some(Arc::new(var_arr));

        Ok(())
    }
}

/// Standard deviation kernel
/// Corresponds to C++ VarKernelImpl<ArrowT, true>
pub struct StdDevKernel<T: ArrowPrimitiveType> {
    inner: VarKernel<T>,
}

impl<T: ArrowPrimitiveType> StdDevKernel<T> {
    pub fn new(ddof: i32) -> Self {
        Self {
            inner: VarKernel::new(ddof),
        }
    }
}

impl<T> MapReduceKernel for StdDevKernel<T>
where
    T: ArrowPrimitiveType + Send + Sync + 'static,
    T::Native: Into<f64> + Copy + Send + Sync,
{
    fn name(&self) -> &str {
        "std"
    }

    fn output_type(&self) -> &DataType {
        self.inner.output_type()
    }

    fn intermediate_types(&self) -> &[DataType] {
        self.inner.intermediate_types()
    }

    fn combine_locally(
        &self,
        value_col: &ArrayRef,
        local_group_ids: &ArrayRef,
        local_num_groups: i64,
        combined_results: &mut Vec<ArrayRef>,
    ) -> CylonResult<()> {
        self.inner.combine_locally(value_col, local_group_ids, local_num_groups, combined_results)
    }

    fn reduce_shuffled_results(
        &self,
        combined_results: &[ArrayRef],
        local_group_ids: &ArrayRef,
        local_group_indices: &ArrayRef,
        local_num_groups: i64,
        reduced_results: &mut Vec<ArrayRef>,
    ) -> CylonResult<()> {
        self.inner.reduce_shuffled_results(
            combined_results,
            local_group_ids,
            local_group_indices,
            local_num_groups,
            reduced_results,
        )
    }

    fn finalize(
        &self,
        combined_results: &[ArrayRef],
        output: &mut Option<ArrayRef>,
    ) -> CylonResult<()> {
        // Compute variance first
        self.inner.finalize(combined_results, output)?;

        // Take square root of variance to get standard deviation
        if let Some(var_arr) = output {
            use arrow::array::Float64Array;
            let var_arr = var_arr
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| CylonError::new(Code::Invalid, "Invalid variance array".to_string()))?;

            let std_devs: Vec<f64> = (0..var_arr.len())
                .map(|i| var_arr.value(i).sqrt())
                .collect();

            *output = Some(Arc::new(Float64Array::from_iter_values(std_devs.into_iter())));
        }

        Ok(())
    }
}
