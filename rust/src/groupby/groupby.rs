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

//! Distributed GroupBy operations
//!
//! Ported from cpp/src/cylon/groupby/groupby.hpp and groupby.cpp
//!
//! This module provides distributed groupby functionality that operates
//! across multiple processes using shuffle operations.

use crate::error::{CylonResult, CylonError, Code};
use crate::table::{Table, shuffle};
use crate::mapreduce::AggregationOpId;
use super::hash_groupby::hash_groupby;

/// Associative operations that can be applied locally before and after shuffle
/// Corresponds to C++ ASSOCIATIVE_OPS (groupby.cpp:24-25)
const ASSOCIATIVE_OPS: &[AggregationOpId] = &[
    AggregationOpId::Sum,
    AggregationOpId::Min,
    AggregationOpId::Max,
];

/// Check if all aggregation operations are associative
/// Corresponds to C++ is_associative (groupby.cpp:27-31)
fn is_associative(aggregate_ops: &[AggregationOpId]) -> bool {
    aggregate_ops.iter().all(|op| ASSOCIATIVE_OPS.contains(op))
}

/// Distributed hash-based GroupBy operation
///
/// Performs groupby across multiple processes by:
/// 1. Projecting to only index + aggregation columns
/// 2. Optionally doing local groupby (if world_size == 1 or ops are associative)
/// 3. Shuffling data so same keys end up on same process
/// 4. Doing final local groupby on shuffled data
///
/// Corresponds to C++ DistributedHashGroupBy (groupby.cpp:33-83)
///
/// # Arguments
/// * `table` - Input table to group
/// * `index_cols` - Column indices to group by
/// * `aggregate_cols` - Column indices to aggregate
/// * `aggregate_ops` - Aggregation operations (one per aggregate column)
///
/// # Returns
/// A new table with key columns followed by aggregation result columns
///
/// # Example
/// ```ignore
/// use cylon::groupby::distributed_hash_groupby;
/// use cylon::mapreduce::AggregationOpId;
///
/// // Group by column 0, compute sum of column 1
/// let result = distributed_hash_groupby(&table, &[0], &[1], &[AggregationOpId::Sum])?;
/// ```
pub fn distributed_hash_groupby(
    table: &Table,
    index_cols: &[usize],
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

    // First filter index + aggregation cols
    // Corresponds to C++ (groupby.cpp:43-46)
    let mut project_cols: Vec<usize> = index_cols.to_vec();
    project_cols.extend(aggregate_cols.iter().cloned());
    let projected_table = table.project(&project_cols)?;

    // Adjust local column indices for aggregations after projection
    // Corresponds to C++ (groupby.cpp:48-55)
    // After projection: columns 0..index_cols.len() are key columns
    // columns index_cols.len().. are aggregate columns
    let indices_after_project: Vec<usize> = (0..index_cols.len()).collect();
    let agg_cols_after_project: Vec<usize> = (0..aggregate_cols.len())
        .map(|i| index_cols.len() + i)
        .collect();

    let ctx = table.get_context();
    let world_size = ctx.get_world_size();

    // Do local group by if world_sz is 1 or if all agg ops are associative
    // Corresponds to C++ (groupby.cpp:57-67)
    let local_table = if world_size == 1 || is_associative(aggregate_ops) {
        hash_groupby(
            &projected_table,
            &indices_after_project,
            &agg_cols_after_project,
            aggregate_ops,
        )?
    } else {
        projected_table
    };

    // If distributed, shuffle and do final groupby
    // Corresponds to C++ (groupby.cpp:69-81)
    if world_size > 1 {
        // Shuffle on key columns
        let shuffled_table = shuffle(&local_table, &indices_after_project)?;

        // Do local groupby again on shuffled data
        hash_groupby(
            &shuffled_table,
            &indices_after_project,
            &agg_cols_after_project,
            aggregate_ops,
        )
    } else {
        Ok(local_table)
    }
}

/// Distributed hash-based GroupBy with single index column
///
/// Convenience function for grouping by a single column.
/// Corresponds to C++ DistributedHashGroupBy (groupby.cpp:85-91)
///
/// # Arguments
/// * `table` - Input table to group
/// * `index_col` - Column index to group by
/// * `aggregate_cols` - Column indices to aggregate
/// * `aggregate_ops` - Aggregation operations (one per aggregate column)
///
/// # Returns
/// A new table with key column followed by aggregation result columns
pub fn distributed_hash_groupby_single(
    table: &Table,
    index_col: usize,
    aggregate_cols: &[usize],
    aggregate_ops: &[AggregationOpId],
) -> CylonResult<Table> {
    distributed_hash_groupby(table, &[index_col], aggregate_cols, aggregate_ops)
}
