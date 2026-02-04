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

//! Compute operations and aggregations
//!
//! Ported from cpp/src/cylon/compute/

pub mod aggregate_kernels;
pub mod aggregates;
pub mod scalar_aggregate;

// Re-export commonly used types
pub use aggregate_kernels::{
    AggregationOpId,
    AggregationOp,
    KernelOptions,
    BasicOptions,
    VarKernelOptions,
    QuantileKernelOptions,
    SumOp, MinOp, MaxOp, CountOp, MeanOp, VarOp, StdDevOp,
    make_aggregation_op,
};

pub use aggregates::{
    ScalarValue,
    AggregateOptions,
    VarianceOptions,
    // Local array aggregates
    sum_array,
    min_array,
    max_array,
    count_array,
    mean_array,
    variance_array,
    stddev_array,
    // Context-aware aggregates
    sum,
    min,
    max,
    count,
    mean,
    variance,
    stddev,
    // Table-level aggregates
    sum_column,
    sum_table,
    min_table,
    max_table,
    count_table,
    mean_table,
};

pub use scalar_aggregate::{
    ScalarAggregateKernel,
    SumKernelImpl,
    MinKernelImpl,
    MaxKernelImpl,
    CountKernelImpl,
    MeanKernelImpl,
    VarianceKernelImpl,
    scalar_aggregate,
    create_scalar_aggregate_kernel,
};
