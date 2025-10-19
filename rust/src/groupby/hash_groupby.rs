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

//! Hash-based GroupBy implementation
//!
//! Ported from cpp/src/cylon/groupby/hash_groupby.hpp and hash_groupby.cpp

use crate::error::{CylonResult, CylonError, Code};
use crate::table::Table;
use crate::mapreduce::{AggregationOpId, mapred_hash_groupby};

/// Hash-based GroupBy operation (local, non-distributed)
/// Corresponds to C++ HashGroupBy (hash_groupby.cpp:414-431)
///
/// Groups rows by key columns and performs aggregations on value columns.
///
/// # Arguments
/// * `table` - Input table to group
/// * `idx_cols` - Column indices to group by
/// * `aggregate_cols` - Column indices to aggregate
/// * `aggregate_ops` - Aggregation operations (one per aggregate column)
///
/// # Returns
/// A new table with key columns followed by aggregation result columns
///
/// # Example
/// ```ignore
/// use cylon::groupby::hash_groupby;
/// use cylon::mapreduce::AggregationOpId;
///
/// // Group by column 0, compute sum of column 1 and mean of column 2
/// let result = hash_groupby(&table, &[0], &[1, 2], &[AggregationOpId::Sum, AggregationOpId::Mean])?;
/// ```
pub fn hash_groupby(
    table: &Table,
    idx_cols: &[usize],
    aggregate_cols: &[usize],
    aggregate_ops: &[AggregationOpId],
) -> CylonResult<Table> {
    if aggregate_cols.len() != aggregate_ops.len() {
        return Err(CylonError::new(
            Code::Invalid,
            format!("aggregate_cols size {} != aggregate_ops size {}",
                    aggregate_cols.len(), aggregate_ops.len())
        ));
    }

    // Convert to (col, op) pairs as expected by mapred_hash_groupby
    let aggs: Vec<(usize, AggregationOpId)> = aggregate_cols.iter()
        .zip(aggregate_ops.iter())
        .map(|(&col, &op)| (col, op))
        .collect();

    let mut output = None;
    mapred_hash_groupby(table, idx_cols, &aggs, &mut output)?;
    output.ok_or_else(|| CylonError::new(
        Code::ExecutionError,
        "GroupBy operation returned None".to_string()
    ))
}

/// Hash-based GroupBy operation with single index column
/// Corresponds to C++ HashGroupBy (hash_groupby.cpp:433-439)
///
/// Convenience function for grouping by a single column
///
/// # Arguments
/// * `table` - Input table to group
/// * `idx_col` - Column index to group by
/// * `aggregate_cols` - Column indices to aggregate
/// * `aggregate_ops` - Aggregation operations (one per aggregate column)
///
/// # Returns
/// A new table with key column followed by aggregation result columns
pub fn hash_groupby_single(
    table: &Table,
    idx_col: usize,
    aggregate_cols: &[usize],
    aggregate_ops: &[AggregationOpId],
) -> CylonResult<Table> {
    hash_groupby(table, &[idx_col], aggregate_cols, aggregate_ops)
}
