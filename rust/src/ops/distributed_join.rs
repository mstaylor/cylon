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

//! Distributed join operation
//!
//! Ported from cpp/src/cylon/table.cpp (DistributedJoin)

use crate::error::CylonResult;
use crate::table::Table;
use crate::join::JoinConfig;

/// Perform a distributed join operation
///
/// This function performs a distributed join by:
/// 1. Shuffling both left and right tables by their respective join columns
/// 2. Performing a local join on the shuffled data at each rank
///
/// After the shuffle phase, rows with matching join keys are co-located on the same
/// rank, allowing a local join to complete the operation.
///
/// Corresponds to C++ DistributedJoin() in cpp/src/cylon/table.cpp:861-890
///
/// # Arguments
/// * `left` - The left table
/// * `right` - The right table
/// * `join_config` - Join configuration (type, columns, algorithm)
///
/// # Returns
/// The joined table containing results from all ranks
///
/// # Example
/// ```ignore
/// use cylon::join::{JoinConfig, JoinType};
/// use cylon::ops::distributed_join::distributed_join;
///
/// let ctx = Arc::new(CylonContext::new(true));
/// ctx.set_communicator(MPICommunicator::make()?);
///
/// let left_table = // ... create left table
/// let right_table = // ... create right table
///
/// let config = JoinConfig {
///     join_type: JoinType::Inner,
///     left_on: vec![0],
///     right_on: vec![0],
///     ..Default::default()
/// };
///
/// let result = distributed_join(&left_table, &right_table, &config)?;
/// ```
pub fn distributed_join(
    left: &Table,
    right: &Table,
    join_config: &JoinConfig,
) -> CylonResult<Table> {
    let ctx = left.get_context();

    // If not distributed or world_size == 1, use local join
    // Corresponds to C++ table.cpp:864-866
    if !ctx.is_distributed() || ctx.get_world_size() == 1 {
        return crate::table::join(left, right, join_config);
    }

    // Phase 1: Shuffle both tables by join columns
    // After shuffle, rows with matching keys will be on the same rank
    // Corresponds to C++ shuffle_two_tables_by_hashing (table.cpp:872-880)
    let left_shuffled = crate::table::shuffle(left, join_config.left_column_indices())?;
    let right_shuffled = crate::table::shuffle(right, join_config.right_column_indices())?;

    // Phase 2: Perform local join on shuffled data
    // Each rank joins its local partition independently
    // Corresponds to C++ join::JoinTables (table.cpp:883-887)
    crate::table::join(&left_shuffled, &right_shuffled, join_config)
}
