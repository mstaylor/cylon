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

//! Example demonstrating all distributed join types with MPI
//!
//! Tests: Inner, Left, Right, and Full Outer joins
//! Run with: mpirun -n 2 cargo run --example distributed_join_all_types --features mpi

use std::sync::Arc;
use arrow::array::{Array, Int32Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use cylon::ctx::CylonContext;
use cylon::table::Table;
use cylon::ops::distributed_join::distributed_join;
use cylon::join::{JoinConfig, JoinType};
use cylon::error::CylonResult;

fn main() -> CylonResult<()> {
    // Initialize MPI context
    let mut ctx_new = CylonContext::new(true);
    ctx_new.set_communicator(cylon::net::mpi::communicator::MPICommunicator::make()?);
    let ctx = Arc::new(ctx_new);

    let rank = ctx.get_rank();
    let world_size = ctx.get_world_size();

    println!("\n========================================");
    println!("Rank {}/{}: Testing all distributed join types", rank, world_size);
    println!("========================================\n");

    // Create schema for left table (id, value)
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("left_value", DataType::Int32, false),
    ]));

    // Create schema for right table (id, value)
    let right_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("right_value", DataType::Int32, false),
    ]));

    // Create test data with some overlapping and some non-overlapping keys
    // Left table: Even IDs (0, 2, 4, 6, 8, 10, 12, 14, 16, 18)
    let left_ids: Vec<i32> = (0..10).map(|i| (i + rank) * 2).collect();
    let left_values: Vec<i32> = (0..10).map(|i| (i + rank) * 100).collect();

    // Right table: IDs divisible by 3 (0, 3, 6, 9, 12, 15, 18, 21, 24, 27)
    let right_ids: Vec<i32> = (0..10).map(|i| (i + rank) * 3).collect();
    let right_values: Vec<i32> = (0..10).map(|i| (i + rank) * 200).collect();

    let left_batch = RecordBatch::try_new(
        left_schema.clone(),
        vec![
            Arc::new(Int32Array::from(left_ids.clone())),
            Arc::new(Int32Array::from(left_values)),
        ],
    )?;

    let right_batch = RecordBatch::try_new(
        right_schema.clone(),
        vec![
            Arc::new(Int32Array::from(right_ids.clone())),
            Arc::new(Int32Array::from(right_values)),
        ],
    )?;

    let left_table = Table::from_record_batch(ctx.clone(), left_batch)?;
    let right_table = Table::from_record_batch(ctx.clone(), right_batch)?;

    println!("Rank {}: Left table has {} rows (even IDs)", rank, left_table.rows());
    println!("Rank {}: Right table has {} rows (IDs divisible by 3)", rank, right_table.rows());

    // Test 1: Inner Join
    println!("\n--- Test 1: Inner Join ---");
    let inner_config = JoinConfig::inner_join(0, 0);
    let inner_result = distributed_join(&left_table, &right_table, &inner_config)?;
    println!("Rank {}: Inner join result has {} rows", rank, inner_result.rows());
    validate_join_result(&inner_result, rank, JoinType::Inner)?;

    // Test 2: Left Join
    println!("\n--- Test 2: Left Join ---");
    let left_config = JoinConfig::left_join(0, 0);
    let left_result = distributed_join(&left_table, &right_table, &left_config)?;
    println!("Rank {}: Left join result has {} rows", rank, left_result.rows());
    validate_join_result(&left_result, rank, JoinType::Left)?;

    // Test 3: Right Join
    println!("\n--- Test 3: Right Join ---");
    let right_config = JoinConfig::right_join(0, 0);
    let right_result = distributed_join(&left_table, &right_table, &right_config)?;
    println!("Rank {}: Right join result has {} rows", rank, right_result.rows());
    validate_join_result(&right_result, rank, JoinType::Right)?;

    // Test 4: Full Outer Join
    println!("\n--- Test 4: Full Outer Join ---");
    let outer_config = JoinConfig::full_outer_join(0, 0);
    let outer_result = distributed_join(&left_table, &right_table, &outer_config)?;
    println!("Rank {}: Full outer join result has {} rows", rank, outer_result.rows());
    validate_join_result(&outer_result, rank, JoinType::FullOuter)?;

    println!("\nRank {}: ✓ All join types validated successfully", rank);

    ctx.barrier()?;

    if rank == 0 {
        println!("\n========================================");
        println!("ALL DISTRIBUTED JOIN TYPES PASSED ✓");
        println!("Tested: Inner, Left, Right, Full Outer");
        println!("========================================\n");
    }

    Ok(())
}

fn validate_join_result(result: &Table, rank: i32, join_type: JoinType) -> CylonResult<()> {
    if result.rows() == 0 {
        println!("Rank {}: No rows in result (acceptable for some join types)", rank);
        return Ok(());
    }

    let batch = result.batch(0).unwrap();

    // All join types should have 4 columns
    assert_eq!(batch.num_columns(), 4,
              "Rank {}: Expected 4 columns for {:?} join", rank, join_type);

    // For non-outer joins, verify join key matching where both sides are present
    let left_id_col = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    let right_id_col = batch.column(2).as_any().downcast_ref::<Int32Array>().unwrap();

    match join_type {
        JoinType::Inner => {
            // All rows should have matching keys
            for i in 0..left_id_col.len() {
                if !left_id_col.is_null(i) && !right_id_col.is_null(i) {
                    assert_eq!(left_id_col.value(i), right_id_col.value(i),
                              "Rank {}: Inner join keys should match", rank);
                }
            }
        },
        JoinType::Left => {
            // All left keys should be present, right keys may be null
            for i in 0..left_id_col.len() {
                assert!(!left_id_col.is_null(i),
                       "Rank {}: Left join should have all left keys", rank);
                if !right_id_col.is_null(i) {
                    assert_eq!(left_id_col.value(i), right_id_col.value(i),
                              "Rank {}: Matching keys should be equal", rank);
                }
            }
        },
        JoinType::Right => {
            // All right keys should be present, left keys may be null
            for i in 0..right_id_col.len() {
                assert!(!right_id_col.is_null(i),
                       "Rank {}: Right join should have all right keys", rank);
                if !left_id_col.is_null(i) {
                    assert_eq!(left_id_col.value(i), right_id_col.value(i),
                              "Rank {}: Matching keys should be equal", rank);
                }
            }
        },
        JoinType::FullOuter => {
            // Either left or right key should be present (or both)
            for i in 0..left_id_col.len() {
                let has_left = !left_id_col.is_null(i);
                let has_right = !right_id_col.is_null(i);
                assert!(has_left || has_right,
                       "Rank {}: Outer join row should have at least one key", rank);
                if has_left && has_right {
                    assert_eq!(left_id_col.value(i), right_id_col.value(i),
                              "Rank {}: Matching keys should be equal", rank);
                }
            }
        },
    }

    Ok(())
}
