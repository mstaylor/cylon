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

//! Distributed Operations using Host Imports
//!
//! This module implements distributed DataFrame operations by:
//! 1. Using local compute primitives (join, union, etc.)
//! 2. Calling host imports for communication (all_to_all, barrier, etc.)
//!
//! The orchestration logic lives here in WASM, so it's written once
//! and works across all host environments (Python, Node.js, browser).

use wasm_bindgen::prelude::*;
use arrow::record_batch::RecordBatch;

use crate::table::Table;
use crate::join::{hash_join, JoinConfig};
use crate::ops;
use crate::imports::{get_rank, get_world_size, all_to_all, barrier};

// =============================================================================
// Internal Helpers
// =============================================================================

fn parse_table(data: &[u8]) -> Result<Table, JsValue> {
    Table::from_arrow_ipc(data).map_err(|e| JsValue::from_str(&e.to_string()))
}

fn table_to_ipc(table: &Table) -> Result<Vec<u8>, JsValue> {
    table.to_arrow_ipc().map_err(|e| JsValue::from_str(&e.to_string()))
}

fn merge_tables(tables: Vec<Table>) -> Result<Table, JsValue> {
    if tables.is_empty() {
        return Err(JsValue::from_str("Cannot merge empty table list"));
    }
    let refs: Vec<&Table> = tables.iter().collect();
    ops::merge(&refs).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Create empty table with same schema (using Arrow's standard pattern)
fn empty_table_like(other: &Table) -> Result<Table, JsValue> {
    let schema = other.schema();
    let empty_batch = RecordBatch::new_empty(schema);
    Ok(Table::new(empty_batch))
}

// =============================================================================
// Communication Primitives (exposed for host testing)
// =============================================================================

/// Get current worker's rank (0 to world_size - 1)
#[wasm_bindgen]
pub fn dist_get_rank() -> i32 {
    get_rank()
}

/// Get total number of workers
#[wasm_bindgen]
pub fn dist_get_world_size() -> i32 {
    get_world_size()
}

/// Barrier synchronization - all workers wait
#[wasm_bindgen]
pub fn dist_barrier() {
    barrier()
}

// =============================================================================
// Distributed Join
// =============================================================================

/// Distributed join operation
///
/// Orchestrates:
/// 1. Hash partition both tables by join keys
/// 2. All-to-all shuffle partitions to co-locate matching keys
/// 3. Local join on each worker
///
/// config_json: {
///   join_type: "inner" | "left" | "right" | "full_outer",
///   left_on: [0],   // column indices
///   right_on: [0]
/// }
#[wasm_bindgen]
pub fn distributed_join(
    left: &[u8],
    right: &[u8],
    config_json: &str,
) -> Result<Vec<u8>, JsValue> {
    let config: JoinConfig = serde_json::from_str(config_json)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let left_table = parse_table(left)?;
    let right_table = parse_table(right)?;

    let world_size = get_world_size() as usize;

    // Single worker case - just do local join
    if world_size <= 1 {
        let result = hash_join(&left_table, &right_table, &config)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        return table_to_ipc(&result);
    }

    // Step 1: Hash partition both tables
    let left_partitions = ops::hash_partition(&left_table, &config.left_on, world_size)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let right_partitions = ops::hash_partition(&right_table, &config.right_on, world_size)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Convert to IPC bytes for communication
    let left_ipc: Vec<Vec<u8>> = left_partitions.iter()
        .map(|t| table_to_ipc(t))
        .collect::<Result<Vec<_>, _>>()?;
    let right_ipc: Vec<Vec<u8>> = right_partitions.iter()
        .map(|t| table_to_ipc(t))
        .collect::<Result<Vec<_>, _>>()?;

    // Step 2: All-to-all shuffle
    let left_received = all_to_all(left_ipc)
        .map_err(|e| JsValue::from_str(&e))?;
    let right_received = all_to_all(right_ipc)
        .map_err(|e| JsValue::from_str(&e))?;

    // Step 3: Parse received tables
    let left_tables: Vec<Table> = left_received.iter()
        .filter(|d| !d.is_empty())
        .map(|d| parse_table(d))
        .collect::<Result<Vec<_>, _>>()?;
    let right_tables: Vec<Table> = right_received.iter()
        .filter(|d| !d.is_empty())
        .map(|d| parse_table(d))
        .collect::<Result<Vec<_>, _>>()?;

    // Step 4: Merge partitions locally
    let left_local = if left_tables.is_empty() {
        // Empty table with same schema
        empty_table_like(&left_table)?
    } else {
        merge_tables(left_tables)?
    };

    let right_local = if right_tables.is_empty() {
        empty_table_like(&right_table)?
    } else {
        merge_tables(right_tables)?
    };

    // Step 5: Local join
    let result = hash_join(&left_local, &right_local, &config)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    table_to_ipc(&result)
}

// =============================================================================
// Distributed Union
// =============================================================================

/// Distributed union operation (with deduplication)
///
/// Orchestrates:
/// 1. Hash partition both tables by all columns
/// 2. Combine partitions for same destination
/// 3. All-to-all shuffle
/// 4. Local unique to deduplicate
#[wasm_bindgen]
pub fn distributed_union(left: &[u8], right: &[u8]) -> Result<Vec<u8>, JsValue> {
    let left_table = parse_table(left)?;
    let right_table = parse_table(right)?;

    let world_size = get_world_size() as usize;

    // Single worker - local union
    if world_size <= 1 {
        let result = ops::union(&left_table, &right_table)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        return table_to_ipc(&result);
    }

    // All columns for hash key
    let all_columns: Vec<usize> = (0..left_table.batch().num_columns()).collect();

    // Hash partition both tables
    let left_partitions = ops::hash_partition(&left_table, &all_columns, world_size)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let right_partitions = ops::hash_partition(&right_table, &all_columns, world_size)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Combine partitions for same destination
    let mut combined_ipc: Vec<Vec<u8>> = Vec::with_capacity(world_size);
    for i in 0..world_size {
        let combined = ops::merge(&[&left_partitions[i], &right_partitions[i]])
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        combined_ipc.push(table_to_ipc(&combined)?);
    }

    // All-to-all shuffle
    let received = all_to_all(combined_ipc)
        .map_err(|e| JsValue::from_str(&e))?;

    // Parse and merge received tables
    let tables: Vec<Table> = received.iter()
        .filter(|d| !d.is_empty())
        .map(|d| parse_table(d))
        .collect::<Result<Vec<_>, _>>()?;

    let local = if tables.is_empty() {
        empty_table_like(&left_table)?
    } else {
        merge_tables(tables)?
    };

    // Local unique to deduplicate
    let result = ops::unique(&local, &all_columns, true)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    table_to_ipc(&result)
}

// =============================================================================
// Distributed Intersect
// =============================================================================

/// Distributed intersect operation
///
/// Orchestrates:
/// 1. Hash partition both tables
/// 2. All-to-all shuffle
/// 3. Local intersect
#[wasm_bindgen]
pub fn distributed_intersect(left: &[u8], right: &[u8]) -> Result<Vec<u8>, JsValue> {
    let left_table = parse_table(left)?;
    let right_table = parse_table(right)?;

    let world_size = get_world_size() as usize;

    // Single worker - local intersect
    if world_size <= 1 {
        let result = ops::intersect(&left_table, &right_table)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        return table_to_ipc(&result);
    }

    // All columns for hash key
    let all_columns: Vec<usize> = (0..left_table.batch().num_columns()).collect();

    // Hash partition both tables
    let left_partitions = ops::hash_partition(&left_table, &all_columns, world_size)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let right_partitions = ops::hash_partition(&right_table, &all_columns, world_size)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Convert to IPC
    let left_ipc: Vec<Vec<u8>> = left_partitions.iter()
        .map(|t| table_to_ipc(t))
        .collect::<Result<Vec<_>, _>>()?;
    let right_ipc: Vec<Vec<u8>> = right_partitions.iter()
        .map(|t| table_to_ipc(t))
        .collect::<Result<Vec<_>, _>>()?;

    // All-to-all shuffle
    let left_received = all_to_all(left_ipc)
        .map_err(|e| JsValue::from_str(&e))?;
    let right_received = all_to_all(right_ipc)
        .map_err(|e| JsValue::from_str(&e))?;

    // Parse and merge
    let left_tables: Vec<Table> = left_received.iter()
        .filter(|d| !d.is_empty())
        .map(|d| parse_table(d))
        .collect::<Result<Vec<_>, _>>()?;
    let right_tables: Vec<Table> = right_received.iter()
        .filter(|d| !d.is_empty())
        .map(|d| parse_table(d))
        .collect::<Result<Vec<_>, _>>()?;

    let left_local = if left_tables.is_empty() {
        empty_table_like(&left_table)?
    } else {
        merge_tables(left_tables)?
    };

    let right_local = if right_tables.is_empty() {
        empty_table_like(&right_table)?
    } else {
        merge_tables(right_tables)?
    };

    // Local intersect
    let result = ops::intersect(&left_local, &right_local)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    table_to_ipc(&result)
}

// =============================================================================
// Distributed Subtract
// =============================================================================

/// Distributed subtract operation
///
/// Orchestrates:
/// 1. Hash partition both tables
/// 2. All-to-all shuffle
/// 3. Local subtract
#[wasm_bindgen]
pub fn distributed_subtract(left: &[u8], right: &[u8]) -> Result<Vec<u8>, JsValue> {
    let left_table = parse_table(left)?;
    let right_table = parse_table(right)?;

    let world_size = get_world_size() as usize;

    // Single worker - local subtract
    if world_size <= 1 {
        let result = ops::subtract(&left_table, &right_table)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        return table_to_ipc(&result);
    }

    // All columns for hash key
    let all_columns: Vec<usize> = (0..left_table.batch().num_columns()).collect();

    // Hash partition both tables
    let left_partitions = ops::hash_partition(&left_table, &all_columns, world_size)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let right_partitions = ops::hash_partition(&right_table, &all_columns, world_size)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Convert to IPC
    let left_ipc: Vec<Vec<u8>> = left_partitions.iter()
        .map(|t| table_to_ipc(t))
        .collect::<Result<Vec<_>, _>>()?;
    let right_ipc: Vec<Vec<u8>> = right_partitions.iter()
        .map(|t| table_to_ipc(t))
        .collect::<Result<Vec<_>, _>>()?;

    // All-to-all shuffle
    let left_received = all_to_all(left_ipc)
        .map_err(|e| JsValue::from_str(&e))?;
    let right_received = all_to_all(right_ipc)
        .map_err(|e| JsValue::from_str(&e))?;

    // Parse and merge
    let left_tables: Vec<Table> = left_received.iter()
        .filter(|d| !d.is_empty())
        .map(|d| parse_table(d))
        .collect::<Result<Vec<_>, _>>()?;
    let right_tables: Vec<Table> = right_received.iter()
        .filter(|d| !d.is_empty())
        .map(|d| parse_table(d))
        .collect::<Result<Vec<_>, _>>()?;

    let left_local = if left_tables.is_empty() {
        empty_table_like(&left_table)?
    } else {
        merge_tables(left_tables)?
    };

    let right_local = if right_tables.is_empty() {
        empty_table_like(&right_table)?
    } else {
        merge_tables(right_tables)?
    };

    // Local subtract
    let result = ops::subtract(&left_local, &right_local)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    table_to_ipc(&result)
}

// =============================================================================
// Distributed GroupBy
// =============================================================================

/// Distributed groupby aggregation
///
/// Orchestrates:
/// 1. Local partial aggregation
/// 2. Hash partition by group keys
/// 3. All-to-all shuffle
/// 4. Final aggregation
///
/// config_json: { keys: [0], aggregations: [{ column: 1, op: "sum", alias: "total" }] }
#[wasm_bindgen]
pub fn distributed_groupby(data: &[u8], config_json: &str) -> Result<Vec<u8>, JsValue> {
    let config: crate::groupby::GroupByConfig = serde_json::from_str(config_json)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let table = parse_table(data)?;
    let world_size = get_world_size() as usize;

    // Single worker - local groupby
    if world_size <= 1 {
        let result = crate::groupby::hash_groupby(&table, &config)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        return table_to_ipc(&result);
    }

    // Step 1: Local partial aggregation
    let partial = crate::groupby::hash_groupby(&table, &config)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Step 2: Hash partition by group keys
    let partitions = ops::hash_partition(&partial, &config.keys, world_size)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let ipc_data: Vec<Vec<u8>> = partitions.iter()
        .map(|t| table_to_ipc(t))
        .collect::<Result<Vec<_>, _>>()?;

    // Step 3: All-to-all shuffle
    let received = all_to_all(ipc_data)
        .map_err(|e| JsValue::from_str(&e))?;

    // Step 4: Parse and merge
    let tables: Vec<Table> = received.iter()
        .filter(|d| !d.is_empty())
        .map(|d| parse_table(d))
        .collect::<Result<Vec<_>, _>>()?;

    let local = if tables.is_empty() {
        empty_table_like(&partial)?
    } else {
        merge_tables(tables)?
    };

    // Step 5: Final aggregation (re-aggregate the partial results)
    let result = crate::groupby::hash_groupby(&local, &config)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    table_to_ipc(&result)
}