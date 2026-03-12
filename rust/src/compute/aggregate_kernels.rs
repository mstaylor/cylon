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

//! Aggregate kernel definitions
//!
//! Ported from cpp/src/cylon/compute/aggregate_kernels.hpp
//!
//! This module defines:
//! - `AggregationOpId` - Enum of aggregation operation types
//! - `KernelOptions` - Base trait for kernel options
//! - `AggregationOp` - Trait for aggregation operations
//! - Concrete kernel implementations for each operation

use std::collections::HashSet;
use std::hash::Hash;

/// Aggregation operation identifiers
/// Corresponds to C++ AggregationOpId enum in aggregate_kernels.hpp:44-54
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AggregationOpId {
    Sum,
    Min,
    Max,
    Count,
    Mean,
    Var,
    NUnique,
    Quantile,
    StdDev,
}

impl AggregationOpId {
    /// Get the column name prefix for this aggregation
    pub fn name_prefix(&self) -> &'static str {
        match self {
            AggregationOpId::Sum => "sum_",
            AggregationOpId::Min => "min_",
            AggregationOpId::Max => "max_",
            AggregationOpId::Count => "count_",
            AggregationOpId::Mean => "mean_",
            AggregationOpId::Var => "var_",
            AggregationOpId::NUnique => "nunique_",
            AggregationOpId::Quantile => "quantile_",
            AggregationOpId::StdDev => "std_",
        }
    }
}

impl std::fmt::Display for AggregationOpId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AggregationOpId::Sum => write!(f, "SUM"),
            AggregationOpId::Min => write!(f, "MIN"),
            AggregationOpId::Max => write!(f, "MAX"),
            AggregationOpId::Count => write!(f, "COUNT"),
            AggregationOpId::Mean => write!(f, "MEAN"),
            AggregationOpId::Var => write!(f, "VAR"),
            AggregationOpId::NUnique => write!(f, "NUNIQUE"),
            AggregationOpId::Quantile => write!(f, "QUANTILE"),
            AggregationOpId::StdDev => write!(f, "STDDEV"),
        }
    }
}

// ============================================================================
// Kernel Options
// ============================================================================

/// Base trait for kernel options
/// Corresponds to C++ KernelOptions struct in aggregate_kernels.hpp:59-61
pub trait KernelOptions: Send + Sync {
    fn as_any(&self) -> &dyn std::any::Any;
}

/// Basic options for aggregation operations
/// Corresponds to C++ BasicOptions in aggregate_kernels.hpp:63-66
#[derive(Debug, Clone)]
pub struct BasicOptions {
    pub skip_nulls: bool,
}

impl Default for BasicOptions {
    fn default() -> Self {
        Self { skip_nulls: true }
    }
}

impl BasicOptions {
    pub fn new(skip_nulls: bool) -> Self {
        Self { skip_nulls }
    }
}

impl KernelOptions for BasicOptions {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Options for variance and standard deviation
/// Corresponds to C++ VarKernelOptions in aggregate_kernels.hpp:71-79
#[derive(Debug, Clone)]
pub struct VarKernelOptions {
    /// Delta degrees of freedom (0 for population, 1 for sample)
    pub ddof: i32,
    /// Whether to skip null values
    pub skip_nulls: bool,
}

impl Default for VarKernelOptions {
    fn default() -> Self {
        Self {
            ddof: 0,
            skip_nulls: true,
        }
    }
}

impl VarKernelOptions {
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

impl KernelOptions for VarKernelOptions {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Options for quantile computation
/// Corresponds to C++ QuantileKernelOptions in aggregate_kernels.hpp:81-89
#[derive(Debug, Clone)]
pub struct QuantileKernelOptions {
    /// Quantile value (0.0 to 1.0)
    pub quantile: f64,
}

impl Default for QuantileKernelOptions {
    fn default() -> Self {
        Self { quantile: 0.5 } // Median by default
    }
}

impl QuantileKernelOptions {
    pub fn new(quantile: f64) -> Self {
        Self { quantile }
    }

    pub fn median() -> Self {
        Self { quantile: 0.5 }
    }
}

impl KernelOptions for QuantileKernelOptions {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// ============================================================================
// Aggregation Operations
// ============================================================================

/// Trait for aggregation operations
/// Corresponds to C++ AggregationOp struct in aggregate_kernels.hpp:116-120
pub trait AggregationOp: Send + Sync {
    fn id(&self) -> AggregationOpId;
    fn options(&self) -> Option<&dyn KernelOptions> {
        None
    }
}

/// Sum operation
#[derive(Debug, Clone, Default)]
pub struct SumOp;

impl AggregationOp for SumOp {
    fn id(&self) -> AggregationOpId {
        AggregationOpId::Sum
    }
}

/// Min operation
#[derive(Debug, Clone, Default)]
pub struct MinOp;

impl AggregationOp for MinOp {
    fn id(&self) -> AggregationOpId {
        AggregationOpId::Min
    }
}

/// Max operation
#[derive(Debug, Clone, Default)]
pub struct MaxOp;

impl AggregationOp for MaxOp {
    fn id(&self) -> AggregationOpId {
        AggregationOpId::Max
    }
}

/// Count operation
#[derive(Debug, Clone, Default)]
pub struct CountOp;

impl AggregationOp for CountOp {
    fn id(&self) -> AggregationOpId {
        AggregationOpId::Count
    }
}

/// Mean operation
#[derive(Debug, Clone, Default)]
pub struct MeanOp;

impl AggregationOp for MeanOp {
    fn id(&self) -> AggregationOpId {
        AggregationOpId::Mean
    }
}

/// NUnique operation
#[derive(Debug, Clone, Default)]
pub struct NUniqueOp;

impl AggregationOp for NUniqueOp {
    fn id(&self) -> AggregationOpId {
        AggregationOpId::NUnique
    }
}

/// Variance operation
#[derive(Debug, Clone)]
pub struct VarOp {
    pub options: VarKernelOptions,
}

impl Default for VarOp {
    fn default() -> Self {
        Self {
            options: VarKernelOptions::default(),
        }
    }
}

impl VarOp {
    pub fn new(ddof: i32) -> Self {
        Self {
            options: VarKernelOptions::new(ddof, true),
        }
    }
}

impl AggregationOp for VarOp {
    fn id(&self) -> AggregationOpId {
        AggregationOpId::Var
    }

    fn options(&self) -> Option<&dyn KernelOptions> {
        Some(&self.options)
    }
}

/// Standard deviation operation
#[derive(Debug, Clone)]
pub struct StdDevOp {
    pub options: VarKernelOptions,
}

impl Default for StdDevOp {
    fn default() -> Self {
        Self {
            options: VarKernelOptions::default(),
        }
    }
}

impl StdDevOp {
    pub fn new(ddof: i32) -> Self {
        Self {
            options: VarKernelOptions::new(ddof, true),
        }
    }
}

impl AggregationOp for StdDevOp {
    fn id(&self) -> AggregationOpId {
        AggregationOpId::StdDev
    }

    fn options(&self) -> Option<&dyn KernelOptions> {
        Some(&self.options)
    }
}

/// Quantile operation
#[derive(Debug, Clone)]
pub struct QuantileOp {
    pub options: QuantileKernelOptions,
}

impl Default for QuantileOp {
    fn default() -> Self {
        Self {
            options: QuantileKernelOptions::default(),
        }
    }
}

impl QuantileOp {
    pub fn new(quantile: f64) -> Self {
        Self {
            options: QuantileKernelOptions::new(quantile),
        }
    }
}

impl AggregationOp for QuantileOp {
    fn id(&self) -> AggregationOpId {
        AggregationOpId::Quantile
    }

    fn options(&self) -> Option<&dyn KernelOptions> {
        Some(&self.options)
    }
}

/// Create an aggregation operation from an ID
/// Corresponds to C++ MakeAggregationOpFromID in aggregate_kernels.hpp:198
pub fn make_aggregation_op(id: AggregationOpId) -> Box<dyn AggregationOp> {
    match id {
        AggregationOpId::Sum => Box::new(SumOp),
        AggregationOpId::Min => Box::new(MinOp),
        AggregationOpId::Max => Box::new(MaxOp),
        AggregationOpId::Count => Box::new(CountOp),
        AggregationOpId::Mean => Box::new(MeanOp),
        AggregationOpId::Var => Box::new(VarOp::default()),
        AggregationOpId::StdDev => Box::new(StdDevOp::default()),
        AggregationOpId::NUnique => Box::new(NUniqueOp),
        AggregationOpId::Quantile => Box::new(QuantileOp::default()),
    }
}

// ============================================================================
// Aggregation Kernels
// ============================================================================

/// Trait for aggregation kernels that can be executed on typed data
/// Corresponds to C++ AggregationKernel in aggregate_kernels.hpp:298-327
///
/// The kernel operates in three phases:
/// 1. Setup - Configure with options
/// 2. InitializeState + Update (repeated) - Build running state
/// 3. Finalize - Convert state to result
pub trait AggregationKernel<T, State, Result>: Send + Sync {
    /// Setup the kernel with options
    fn setup(&mut self, options: Option<&dyn KernelOptions>);

    /// Initialize a new state
    fn initialize_state(&self) -> State;

    /// Update state with a value
    fn update(&self, value: &T, state: &mut State);

    /// Finalize state to result
    fn finalize(&self, state: &State) -> Result;
}

/// Sum kernel
/// Corresponds to C++ SumKernel in aggregate_kernels.hpp:443-457
pub struct SumKernel<T> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T> Default for SumKernel<T> {
    fn default() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T> AggregationKernel<T, T, T> for SumKernel<T>
where
    T: num_traits::Zero + std::ops::AddAssign + Copy + Send + Sync,
{
    fn setup(&mut self, _options: Option<&dyn KernelOptions>) {}

    fn initialize_state(&self) -> T {
        T::zero()
    }

    fn update(&self, value: &T, state: &mut T) {
        *state += *value;
    }

    fn finalize(&self, state: &T) -> T {
        *state
    }
}

/// Count kernel
/// Corresponds to C++ CountKernel in aggregate_kernels.hpp:462-477
pub struct CountKernel<T> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T> Default for CountKernel<T> {
    fn default() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Send + Sync> AggregationKernel<T, i64, i64> for CountKernel<T> {
    fn setup(&mut self, _options: Option<&dyn KernelOptions>) {}

    fn initialize_state(&self) -> i64 {
        0
    }

    fn update(&self, _value: &T, state: &mut i64) {
        *state += 1;
    }

    fn finalize(&self, state: &i64) -> i64 {
        *state
    }
}

/// Min kernel
/// Corresponds to C++ MinKernel in aggregate_kernels.hpp:482-496
pub struct MinKernel<T> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T> Default for MinKernel<T> {
    fn default() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T> AggregationKernel<T, Option<T>, Option<T>> for MinKernel<T>
where
    T: PartialOrd + Copy + Send + Sync,
{
    fn setup(&mut self, _options: Option<&dyn KernelOptions>) {}

    fn initialize_state(&self) -> Option<T> {
        None
    }

    fn update(&self, value: &T, state: &mut Option<T>) {
        *state = Some(match *state {
            Some(current) if current < *value => current,
            _ => *value,
        });
    }

    fn finalize(&self, state: &Option<T>) -> Option<T> {
        *state
    }
}

/// Max kernel
/// Corresponds to C++ MaxKernel in aggregate_kernels.hpp:501-515
pub struct MaxKernel<T> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T> Default for MaxKernel<T> {
    fn default() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T> AggregationKernel<T, Option<T>, Option<T>> for MaxKernel<T>
where
    T: PartialOrd + Copy + Send + Sync,
{
    fn setup(&mut self, _options: Option<&dyn KernelOptions>) {}

    fn initialize_state(&self) -> Option<T> {
        None
    }

    fn update(&self, value: &T, state: &mut Option<T>) {
        *state = Some(match *state {
            Some(current) if current > *value => current,
            _ => *value,
        });
    }

    fn finalize(&self, state: &Option<T>) -> Option<T> {
        *state
    }
}

/// Mean kernel state: (sum, count)
/// Corresponds to C++ MeanKernel in aggregate_kernels.hpp:377-397
pub struct MeanKernel<T> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T> Default for MeanKernel<T> {
    fn default() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T> AggregationKernel<T, (f64, i64), f64> for MeanKernel<T>
where
    T: num_traits::ToPrimitive + Send + Sync,
{
    fn setup(&mut self, _options: Option<&dyn KernelOptions>) {}

    fn initialize_state(&self) -> (f64, i64) {
        (0.0, 0)
    }

    fn update(&self, value: &T, state: &mut (f64, i64)) {
        if let Some(v) = value.to_f64() {
            state.0 += v;
            state.1 += 1;
        }
    }

    fn finalize(&self, state: &(f64, i64)) -> f64 {
        if state.1 > 0 {
            state.0 / state.1 as f64
        } else {
            f64::NAN
        }
    }
}

/// Variance kernel state: (sum_of_squares, sum, count)
/// Corresponds to C++ VarianceKernel in aggregate_kernels.hpp:402-438
pub struct VarianceKernel<T> {
    ddof: i32,
    do_std: bool,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> VarianceKernel<T> {
    pub fn new(ddof: i32, do_std: bool) -> Self {
        Self {
            ddof,
            do_std,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn variance(ddof: i32) -> Self {
        Self::new(ddof, false)
    }

    pub fn stddev(ddof: i32) -> Self {
        Self::new(ddof, true)
    }
}

impl<T> Default for VarianceKernel<T> {
    fn default() -> Self {
        Self::new(0, false)
    }
}

impl<T> AggregationKernel<T, (f64, f64, i64), f64> for VarianceKernel<T>
where
    T: num_traits::ToPrimitive + Send + Sync,
{
    fn setup(&mut self, options: Option<&dyn KernelOptions>) {
        if let Some(opts) = options {
            if let Some(var_opts) = opts.as_any().downcast_ref::<VarKernelOptions>() {
                self.ddof = var_opts.ddof;
            }
        }
    }

    fn initialize_state(&self) -> (f64, f64, i64) {
        (0.0, 0.0, 0)
    }

    fn update(&self, value: &T, state: &mut (f64, f64, i64)) {
        if let Some(v) = value.to_f64() {
            state.0 += v * v; // sum of squares
            state.1 += v;     // sum
            state.2 += 1;     // count
        }
    }

    fn finalize(&self, state: &(f64, f64, i64)) -> f64 {
        let (sum_sq, sum, count) = *state;
        if count <= self.ddof as i64 {
            return f64::NAN;
        }

        let n = count as f64;
        let mean = sum / n;
        let mean_sq = sum_sq / n;
        let var = (mean_sq - mean * mean) * n / (n - self.ddof as f64);

        if self.do_std {
            var.sqrt()
        } else {
            var
        }
    }
}

/// NUnique kernel - counts unique values
/// Corresponds to C++ NUniqueKernel in aggregate_kernels.hpp:517-530
pub struct NUniqueKernel<T> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T> Default for NUniqueKernel<T> {
    fn default() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T> AggregationKernel<T, HashSet<T>, i64> for NUniqueKernel<T>
where
    T: Eq + Hash + Clone + Send + Sync,
{
    fn setup(&mut self, _options: Option<&dyn KernelOptions>) {}

    fn initialize_state(&self) -> HashSet<T> {
        HashSet::new()
    }

    fn update(&self, value: &T, state: &mut HashSet<T>) {
        state.insert(value.clone());
    }

    fn finalize(&self, state: &HashSet<T>) -> i64 {
        state.len() as i64
    }
}
