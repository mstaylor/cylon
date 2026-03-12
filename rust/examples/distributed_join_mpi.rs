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

//! Example demonstrating distributed join operation with MPI
//!
//! Run with: mpirun -n 2 cargo run --example distributed_join_mpi --features mpi

use std::sync::Arc;
use arrow::array::Int32Array;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use cylon::ctx::CylonContext;
use cylon::table::Table;
use cylon::ops::distributed_join::distributed_join;
use cylon::join::JoinConfig;
use cylon::error::CylonResult;

fn main() -> CylonResult<()> {
    // Initialize MPI context
    let mut ctx_new = CylonContext::new(true);
    ctx_new.set_communicator(cylon::net::mpi::communicator::MPICommunicator::make()?);
    let ctx = Arc::new(ctx_new);

    let rank = ctx.get_rank();
    let world_size = ctx.get_world_size();

    println!("\n========================================");
    println!("Rank {}/{}: Testing distributed join", rank, world_size);
    println!("========================================\n");

    // Create schema for left table (id, value_left)
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("value_left", DataType::Int32, false),
    ]));

    // Create schema for right table (id, value_right)
    let right_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("value_right", DataType::Int32, false),
    ]));

    // Each process creates tables with different data
    // Left table: id values based on rank
    let left_ids: Vec<i32> = (0..10).map(|i| i * world_size + rank).collect();
    let left_values: Vec<i32> = (0..10).map(|i| (i * world_size + rank) * 100).collect();

    // Right table: id values based on rank (some overlap with left)
    let right_ids: Vec<i32> = (0..10).map(|i| i * world_size + rank).collect();
    let right_values: Vec<i32> = (0..10).map(|i| (i * world_size + rank) * 200).collect();

    let left_batch = RecordBatch::try_new(
        left_schema.clone(),
        vec![
            Arc::new(Int32Array::from(left_ids)),
            Arc::new(Int32Array::from(left_values)),
        ],
    )?;

    let right_batch = RecordBatch::try_new(
        right_schema.clone(),
        vec![
            Arc::new(Int32Array::from(right_ids)),
            Arc::new(Int32Array::from(right_values)),
        ],
    )?;

    let left_table = Table::from_record_batch(ctx.clone(), left_batch)?;
    let right_table = Table::from_record_batch(ctx.clone(), right_batch)?;

    println!("Rank {}: Left table has {} rows", rank, left_table.rows());
    println!("Rank {}: Right table has {} rows", rank, right_table.rows());

    // Perform distributed inner join on column 0 (id)
    let join_config = JoinConfig::inner_join(0, 0);
    let result = distributed_join(&left_table, &right_table, &join_config)?;

    println!("Rank {}: Join result has {} rows", rank, result.rows());

    // Verify the result
    if result.rows() > 0 {
        let result_batch = result.batch(0).unwrap();

        // Result should have 4 columns: id (left), value_left, id (right), value_right
        assert_eq!(result_batch.num_columns(), 4, "Expected 4 columns in result");

        let left_id_col = result_batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        let left_val_col = result_batch.column(1).as_any().downcast_ref::<Int32Array>().unwrap();
        let right_id_col = result_batch.column(2).as_any().downcast_ref::<Int32Array>().unwrap();
        let right_val_col = result_batch.column(3).as_any().downcast_ref::<Int32Array>().unwrap();

        // Verify join semantics: for each row, id from left matches id from right
        for i in 0..left_id_col.len() {
            let left_id = left_id_col.value(i);
            let right_id = right_id_col.value(i);
            let left_val = left_val_col.value(i);
            let right_val = right_val_col.value(i);

            // Verify that join keys match
            assert_eq!(left_id, right_id,
                      "Rank {}: Row {}: Join keys don't match: left_id={} != right_id={}",
                      rank, i, left_id, right_id);

            // Verify relationship: left_val = id * 100, right_val = id * 200
            assert_eq!(left_val, left_id * 100,
                      "Rank {}: Row {}: left_val={} != id*100={}", rank, i, left_val, left_id * 100);
            assert_eq!(right_val, right_id * 200,
                      "Rank {}: Row {}: right_val={} != id*200={}", rank, i, right_val, right_id * 200);
        }
    }

    println!("Rank {}: ✓ Join validation successful", rank);

    ctx.barrier()?;

    if rank == 0 {
        println!("\n========================================");
        println!("DISTRIBUTED JOIN TEST PASSED ✓");
        println!("Inner join working correctly!");
        println!("========================================\n");
    }

    Ok(())
}
