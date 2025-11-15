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

//! Distributed set operations
//!
//! Ported from cpp/src/cylon/table.cpp (DistributedUnion, DistributedIntersect, DistributedSubtract)

use crate::error::CylonResult;
use crate::table::Table;

/// Type alias for local set operation functions
type LocalSetOperation = fn(&Table, &Table) -> CylonResult<Table>;

/// Common implementation for distributed set operations
/// Corresponds to C++ do_dist_set_op (table.cpp:1118-1150)
///
/// # Algorithm
/// 1. Check if world_size == 1 -> use local operation
/// 2. Shuffle both tables by ALL columns to co-locate identical rows
/// 3. Perform local set operation on co-located data
///
/// # Arguments
/// * `local_operation` - The local set operation to perform after shuffle
/// * `left` - Left table
/// * `right` - Right table
///
/// # Returns
/// Result table distributed across all processes
fn do_dist_set_op(
    local_operation: LocalSetOperation,
    left: &Table,
    right: &Table,
) -> CylonResult<Table> {
    let ctx = left.get_context();

    // Verify schemas match
    // Corresponds to C++ VerifyTableSchema (table.cpp:1125)
    let left_schema = left.schema()
        .ok_or_else(|| crate::error::CylonError::new(
            crate::error::Code::Invalid,
            "Left table has no schema".to_string()
        ))?;
    let right_schema = right.schema()
        .ok_or_else(|| crate::error::CylonError::new(
            crate::error::Code::Invalid,
            "Right table has no schema".to_string()
        ))?;

    if left_schema != right_schema {
        return Err(crate::error::CylonError::new(
            crate::error::Code::Invalid,
            "Table schemas do not match".to_string()
        ));
    }

    // If not distributed or world_size == 1, use local operation
    // Corresponds to C++ table.cpp:1127-1129
    if !ctx.is_distributed() || ctx.get_world_size() == 1 {
        return local_operation(left, right);
    }

    // Shuffle both tables by ALL columns
    // This co-locates identical rows on the same process
    // Corresponds to C++ shuffle_two_tables_by_hashing (table.cpp:1131-1143)
    let num_columns = left_schema.fields().len();
    let all_columns: Vec<usize> = (0..num_columns).collect();

    let left_shuffled = crate::table::shuffle(left, &all_columns)?;
    let right_shuffled = crate::table::shuffle(right, &all_columns)?;

    // Perform local set operation on shuffled data
    // Corresponds to C++ local_operation call (table.cpp:1148)
    local_operation(&left_shuffled, &right_shuffled)
}

/// Distributed union operation
/// Corresponds to C++ DistributedUnion (table.cpp:1152-1155)
///
/// Computes the union of two distributed tables by:
/// 1. Shuffling both tables by all columns to co-locate identical rows
/// 2. Performing local union on co-located data
/// 3. Result is deduplicated and distributed across all processes
///
/// # Arguments
/// * `left` - Left table
/// * `right` - Right table
///
/// # Returns
/// Union of both tables, distributed across all processes
///
/// # Example
/// ```ignore
/// let result = distributed_union(&table1, &table2)?;
/// ```
pub fn distributed_union(left: &Table, right: &Table) -> CylonResult<Table> {
    do_dist_set_op(crate::table::union, left, right)
}

/// Distributed intersect operation
/// Corresponds to C++ DistributedIntersect (table.cpp:1162-1165)
///
/// Computes the intersection of two distributed tables by:
/// 1. Shuffling both tables by all columns to co-locate identical rows
/// 2. Performing local intersect on co-located data
/// 3. Result contains only rows present in both tables
///
/// # Arguments
/// * `left` - Left table
/// * `right` - Right table
///
/// # Returns
/// Intersection of both tables, distributed across all processes
///
/// # Example
/// ```ignore
/// let result = distributed_intersect(&table1, &table2)?;
/// ```
pub fn distributed_intersect(left: &Table, right: &Table) -> CylonResult<Table> {
    do_dist_set_op(crate::table::intersect, left, right)
}

/// Distributed subtract operation
/// Corresponds to C++ DistributedSubtract (table.cpp:1157-1160)
///
/// Computes the difference (left - right) of two distributed tables by:
/// 1. Shuffling both tables by all columns to co-locate identical rows
/// 2. Performing local subtract on co-located data
/// 3. Result contains rows in left but not in right
///
/// # Arguments
/// * `left` - Left table
/// * `right` - Right table
///
/// # Returns
/// Difference (left - right), distributed across all processes
///
/// # Example
/// ```ignore
/// let result = distributed_subtract(&table1, &table2)?;
/// ```
pub fn distributed_subtract(left: &Table, right: &Table) -> CylonResult<Table> {
    do_dist_set_op(crate::table::subtract, left, right)
}
