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

//! Table operations for WASM
//!
//! Wraps cylon operations for use in WASM environment.

use std::sync::Arc;
use serde::{Deserialize, Serialize};

use cylon::CylonContext;
use cylon::table::{
    Table as CylonTable,
    union as cylon_union,
    subtract as cylon_subtract,
    intersect as cylon_intersect,
    unique as cylon_unique,
};
use cylon::groupby::hash_groupby as cylon_hash_groupby;
use cylon::mapreduce::AggregationOpId;
use cylon::compute::{
    sum_array, min_array, max_array, count_array, mean_array,
    variance_array, stddev_array,
    AggregateOptions, VarianceOptions,
};
use cylon::ops::partition::hash_partition_table as cylon_hash_partition;

use crate::table::Table;
use crate::error::{WasmError, WasmResult};

// =============================================================================
// Table Conversion
// =============================================================================

fn ctx() -> Arc<CylonContext> {
    CylonContext::init()
}

fn to_cylon(table: &Table) -> WasmResult<CylonTable> {
    CylonTable::from_record_batch(ctx(), table.batch().clone())
        .map_err(|e| WasmError::execution_error(e.to_string()))
}

fn from_cylon(table: CylonTable) -> WasmResult<Table> {
    let num_batches = table.num_batches();
    if num_batches == 0 {
        return Err(WasmError::execution_error("Cylon table has no batches"));
    }

    if num_batches == 1 {
        let batch = table.batch(0).unwrap().clone();
        return Ok(Table::new(batch));
    }

    // Multiple batches - concatenate them
    let schema = table.schema()
        .ok_or_else(|| WasmError::execution_error("Cylon table has no schema"))?;

    let batches: Vec<_> = (0..num_batches)
        .filter_map(|i| table.batch(i).cloned())
        .collect();

    let concatenated = arrow::compute::concat_batches(&schema, &batches)
        .map_err(|e| WasmError::execution_error(format!("Failed to concat batches: {}", e)))?;

    Ok(Table::new(concatenated))
}

// =============================================================================
// Column Selection
// =============================================================================

pub fn project(table: &Table, column_indices: &[usize]) -> WasmResult<Table> {
    let cylon_table = to_cylon(table)?;
    let result = cylon_table.project(column_indices)
        .map_err(|e| WasmError::execution_error(e.to_string()))?;
    from_cylon(result)
}

pub fn project_by_names(table: &Table, column_names: &[&str]) -> WasmResult<Table> {
    let cylon_table = to_cylon(table)?;
    let result = cylon_table.project_by_names(column_names)
        .map_err(|e| WasmError::execution_error(e.to_string()))?;
    from_cylon(result)
}

// =============================================================================
// Row Selection
// =============================================================================

pub fn slice(table: &Table, offset: usize, length: usize) -> WasmResult<Table> {
    let cylon_table = to_cylon(table)?;
    let result = cylon_table.slice(offset, length)
        .map_err(|e| WasmError::execution_error(e.to_string()))?;
    from_cylon(result)
}

pub fn head(table: &Table, n: usize) -> WasmResult<Table> {
    let cylon_table = to_cylon(table)?;
    let result = cylon_table.head(n)
        .map_err(|e| WasmError::execution_error(e.to_string()))?;
    from_cylon(result)
}

pub fn tail(table: &Table, n: usize) -> WasmResult<Table> {
    let cylon_table = to_cylon(table)?;
    let result = cylon_table.tail(n)
        .map_err(|e| WasmError::execution_error(e.to_string()))?;
    from_cylon(result)
}

// =============================================================================
// Sort
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SortDirection {
    Ascending,
    Descending,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SortSpec {
    pub column: usize,
    #[serde(default)]
    pub ascending: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SortConfig {
    pub columns: Vec<SortSpec>,
}

pub fn sort(table: &Table, column: usize, ascending: bool) -> WasmResult<Table> {
    let cylon_table = to_cylon(table)?;
    let result = cylon_table.sort(column, ascending)
        .map_err(|e| WasmError::execution_error(e.to_string()))?;
    from_cylon(result)
}

pub fn sort_multi(table: &Table, config: &SortConfig) -> WasmResult<Table> {
    let columns: Vec<usize> = config.columns.iter().map(|s| s.column).collect();
    let directions: Vec<bool> = config.columns.iter().map(|s| s.ascending).collect();

    let cylon_table = to_cylon(table)?;
    let result = cylon_table.sort_multi(&columns, &directions)
        .map_err(|e| WasmError::execution_error(e.to_string()))?;
    from_cylon(result)
}

// =============================================================================
// Merge
// =============================================================================

pub fn merge(tables: &[&Table]) -> WasmResult<Table> {
    if tables.is_empty() {
        return Err(WasmError::invalid("Must provide at least one table"));
    }
    if tables.len() == 1 {
        return Ok(tables[0].clone());
    }

    let first = to_cylon(tables[0])?;
    let others: WasmResult<Vec<CylonTable>> = tables[1..].iter().map(|t| to_cylon(t)).collect();
    let others = others?;
    let other_refs: Vec<&CylonTable> = others.iter().collect();

    let result = first.merge(&other_refs)
        .map_err(|e| WasmError::execution_error(e.to_string()))?;
    from_cylon(result)
}

// =============================================================================
// Set Operations
// =============================================================================

pub fn union(left: &Table, right: &Table) -> WasmResult<Table> {
    let left_cylon = to_cylon(left)?;
    let right_cylon = to_cylon(right)?;
    let result = cylon_union(&left_cylon, &right_cylon)
        .map_err(|e| WasmError::execution_error(e.to_string()))?;
    from_cylon(result)
}

pub fn subtract(left: &Table, right: &Table) -> WasmResult<Table> {
    let left_cylon = to_cylon(left)?;
    let right_cylon = to_cylon(right)?;
    let result = cylon_subtract(&left_cylon, &right_cylon)
        .map_err(|e| WasmError::execution_error(e.to_string()))?;
    from_cylon(result)
}

pub fn intersect(left: &Table, right: &Table) -> WasmResult<Table> {
    let left_cylon = to_cylon(left)?;
    let right_cylon = to_cylon(right)?;
    let result = cylon_intersect(&left_cylon, &right_cylon)
        .map_err(|e| WasmError::execution_error(e.to_string()))?;
    from_cylon(result)
}

pub fn unique(table: &Table, column_indices: &[usize], keep_first: bool) -> WasmResult<Table> {
    let cylon_table = to_cylon(table)?;
    let result = cylon_unique(&cylon_table, column_indices, keep_first)
        .map_err(|e| WasmError::execution_error(e.to_string()))?;
    from_cylon(result)
}

// =============================================================================
// GroupBy
// =============================================================================

/// Aggregation operation for groupby
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AggOp {
    Sum,
    Min,
    Max,
    Count,
    Mean,
    Var,
    Stddev,
    Nunique,
}

impl AggOp {
    fn to_cylon(self) -> AggregationOpId {
        match self {
            AggOp::Sum => AggregationOpId::Sum,
            AggOp::Min => AggregationOpId::Min,
            AggOp::Max => AggregationOpId::Max,
            AggOp::Count => AggregationOpId::Count,
            AggOp::Mean => AggregationOpId::Mean,
            AggOp::Var => AggregationOpId::Var,
            AggOp::Stddev => AggregationOpId::Stddev,
            AggOp::Nunique => AggregationOpId::Nunique,
        }
    }
}

/// Aggregation specification for groupby
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggSpec {
    pub column: usize,
    pub op: AggOp,
}

/// GroupBy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupByConfig {
    pub keys: Vec<usize>,
    pub aggregations: Vec<AggSpec>,
}

/// Hash-based groupby using cylon
pub fn groupby(table: &Table, config: &GroupByConfig) -> WasmResult<Table> {
    if config.keys.is_empty() {
        return Err(WasmError::invalid("Must specify at least one key column"));
    }
    if config.aggregations.is_empty() {
        return Err(WasmError::invalid("Must specify at least one aggregation"));
    }

    let cylon_table = to_cylon(table)?;

    let agg_cols: Vec<usize> = config.aggregations.iter().map(|a| a.column).collect();
    let agg_ops: Vec<AggregationOpId> = config.aggregations.iter().map(|a| a.op.to_cylon()).collect();

    let result = cylon_hash_groupby(&cylon_table, &config.keys, &agg_cols, &agg_ops)
        .map_err(|e| WasmError::execution_error(e.to_string()))?;
    from_cylon(result)
}

// =============================================================================
// Compute Aggregates
// =============================================================================

/// Compute sum of a column
pub fn compute_sum(table: &Table, column: usize) -> WasmResult<f64> {
    let batch = table.batch();
    if column >= batch.num_columns() {
        return Err(WasmError::index_error(format!("Column {} out of bounds", column)));
    }
    let array = batch.column(column);
    let opts = AggregateOptions::default();
    let result = sum_array(array.as_ref(), &opts)
        .map_err(|e| WasmError::execution_error(e.to_string()))?;
    result.to_f64().ok_or_else(|| WasmError::execution_error("Sum returned null"))
}

/// Compute min of a column
pub fn compute_min(table: &Table, column: usize) -> WasmResult<f64> {
    let batch = table.batch();
    if column >= batch.num_columns() {
        return Err(WasmError::index_error(format!("Column {} out of bounds", column)));
    }
    let array = batch.column(column);
    let opts = AggregateOptions::default();
    let result = min_array(array.as_ref(), &opts)
        .map_err(|e| WasmError::execution_error(e.to_string()))?;
    result.to_f64().ok_or_else(|| WasmError::execution_error("Min returned null"))
}

/// Compute max of a column
pub fn compute_max(table: &Table, column: usize) -> WasmResult<f64> {
    let batch = table.batch();
    if column >= batch.num_columns() {
        return Err(WasmError::index_error(format!("Column {} out of bounds", column)));
    }
    let array = batch.column(column);
    let opts = AggregateOptions::default();
    let result = max_array(array.as_ref(), &opts)
        .map_err(|e| WasmError::execution_error(e.to_string()))?;
    result.to_f64().ok_or_else(|| WasmError::execution_error("Max returned null"))
}

/// Compute count of a column (non-null values)
pub fn compute_count(table: &Table, column: usize) -> WasmResult<i64> {
    let batch = table.batch();
    if column >= batch.num_columns() {
        return Err(WasmError::index_error(format!("Column {} out of bounds", column)));
    }
    let array = batch.column(column);
    let opts = AggregateOptions::default();
    let result = count_array(array.as_ref(), &opts)
        .map_err(|e| WasmError::execution_error(e.to_string()))?;
    result.to_i64().ok_or_else(|| WasmError::execution_error("Count returned null"))
}

/// Compute mean of a column
pub fn compute_mean(table: &Table, column: usize) -> WasmResult<f64> {
    let batch = table.batch();
    if column >= batch.num_columns() {
        return Err(WasmError::index_error(format!("Column {} out of bounds", column)));
    }
    let array = batch.column(column);
    let opts = AggregateOptions::default();
    let result = mean_array(array.as_ref(), &opts)
        .map_err(|e| WasmError::execution_error(e.to_string()))?;
    result.to_f64().ok_or_else(|| WasmError::execution_error("Mean returned null"))
}

/// Compute variance of a column
pub fn compute_variance(table: &Table, column: usize, ddof: i32) -> WasmResult<f64> {
    let batch = table.batch();
    if column >= batch.num_columns() {
        return Err(WasmError::index_error(format!("Column {} out of bounds", column)));
    }
    let array = batch.column(column);
    let opts = VarianceOptions::new(ddof, true);
    let result = variance_array(array.as_ref(), &opts)
        .map_err(|e| WasmError::execution_error(e.to_string()))?;
    result.to_f64().ok_or_else(|| WasmError::execution_error("Variance returned null"))
}

/// Compute standard deviation of a column
pub fn compute_stddev(table: &Table, column: usize, ddof: i32) -> WasmResult<f64> {
    let batch = table.batch();
    if column >= batch.num_columns() {
        return Err(WasmError::index_error(format!("Column {} out of bounds", column)));
    }
    let array = batch.column(column);
    let opts = VarianceOptions::new(ddof, true);
    let result = stddev_array(array.as_ref(), &opts)
        .map_err(|e| WasmError::execution_error(e.to_string()))?;
    result.to_f64().ok_or_else(|| WasmError::execution_error("StdDev returned null"))
}

// =============================================================================
// Partitioning (for host-orchestrated distributed operations)
// =============================================================================

/// Hash partition table into multiple partitions
/// This is the key primitive for distributed operations - the host orchestrates:
/// 1. Call hash_partition() to split data by hash key
/// 2. Use native communication (FMI/MPI/UCX) for all-to-all shuffle
/// 3. Call join_tables()/union_tables()/etc. for local compute
pub fn hash_partition(table: &Table, hash_columns: &[usize], num_partitions: usize) -> WasmResult<Vec<Table>> {
    let cylon_table = to_cylon(table)?;
    let partitions = cylon_hash_partition(&cylon_table, hash_columns, num_partitions)
        .map_err(|e| WasmError::execution_error(e.to_string()))?;

    partitions.into_iter()
        .map(|batch| Ok(Table::new(batch)))
        .collect()
}
