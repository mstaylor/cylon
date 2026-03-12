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

//! UCC Operators Example
//!
//! This example demonstrates UCC collective operations (allgather, allreduce, bcast, etc.)
//! using Redis for out-of-band coordination.
//!
//! Corresponds to cpp/src/examples/ucc_operators_example.cpp
//!
//! Usage:
//!   # Set session ID to isolate Redis keys
//!   export CYLON_SESSION_ID=$(uuidgen)
//!   export CYLON_UCX_OOB_REDIS_ADDR=127.0.0.1:6379
//!   export CYLON_UCX_OOB_WORLD_SIZE=4
//!   cargo run --example ucc_operators_example --features "ucx ucc redis"

use std::env;
use std::sync::Arc;

use arrow::array::{ArrayRef, Int32Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use cylon::ctx::CylonContext;
use cylon::error::CylonResult;
use cylon::net::comm_operations::ReduceOp;
use cylon::net::ucx::communicator::{UCXCommunicator, UCXUCCCommunicator};
use cylon::net::ucx::redis_oob::UCXRedisOOBContext;
use cylon::scalar::Scalar;
use cylon::table::{Column, Table};

fn create_test_table(ctx: Arc<CylonContext>, rank: i32) -> CylonResult<Table> {
    // Create a simple table with rank-dependent data
    let schema = Arc::new(Schema::new(vec![
        Field::new("col0", DataType::Int32, false),
        Field::new("col1", DataType::Int32, false),
    ]));

    let col0: Vec<i32> = (0..10).map(|i| i + rank * 10).collect();
    let col1: Vec<i32> = (0..10).map(|i| (i + rank * 10) * 2).collect();

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(col0)) as ArrayRef,
            Arc::new(Int32Array::from(col1)) as ArrayRef,
        ],
    )?;

    Table::from_record_batch(ctx, batch)
}

fn test_column_allgather(ctx: &Arc<CylonContext>) -> CylonResult<()> {
    let rank = ctx.get_rank();
    let world_size = ctx.get_world_size();

    // Create input column: values i + rank * 10 + 1 for i in 0..10
    let values: Vec<i32> = (0..10).map(|i| i + rank * 10 + 1).collect();
    let array = Arc::new(Int32Array::from(values)) as ArrayRef;
    let input = Column::new(array);

    // Perform allgather
    let output = ctx.get_communicator().allgather_column(&input)?;

    // Verify results
    for (i, col) in output.iter().enumerate() {
        let arr = col.data().as_any().downcast_ref::<Int32Array>().unwrap();
        for j in 0..10 {
            let expected = j as i32 + (i as i32) * 10 + 1;
            let actual = arr.value(j);
            if actual != expected {
                println!(
                    "Column allgather test FAILED at rank {}: expected {}, got {}",
                    rank, expected, actual
                );
                return Ok(());
            }
        }
    }

    println!("Column allgather test passed at rank {}", rank);
    Ok(())
}

fn test_scalar_allgather(ctx: &Arc<CylonContext>) -> CylonResult<()> {
    let rank = ctx.get_rank();
    let world_size = ctx.get_world_size();

    // Create scalar with rank value
    let array = Arc::new(Int32Array::from(vec![rank])) as ArrayRef;
    let scalar = Scalar::new(arrow::array::make_array(array.to_data()));

    // Perform allgather
    let output = ctx.get_communicator().allgather_scalar(&scalar)?;

    // Verify results
    let arr = output.data().as_any().downcast_ref::<Int32Array>().unwrap();
    for i in 0..world_size {
        let actual = arr.value(i as usize);
        if actual != i {
            println!(
                "Scalar allgather test FAILED at rank {}: expected {}, got {}",
                rank, i, actual
            );
            return Ok(());
        }
    }

    println!("Scalar allgather test passed at rank {}", rank);
    Ok(())
}

fn test_column_allreduce(ctx: &Arc<CylonContext>) -> CylonResult<()> {
    let rank = ctx.get_rank();
    let world_size = ctx.get_world_size();

    // Create input column: values i + rank * 10 for i in 0..10
    let values: Vec<i32> = (0..10).map(|i| i + rank * 10).collect();
    let array = Arc::new(Int32Array::from(values)) as ArrayRef;
    let input = Column::new(array);

    // Perform allreduce with SUM
    let output = ctx.get_communicator().all_reduce_column(&input, ReduceOp::Sum)?;

    // Verify results
    // Expected: world_size * i + (world_size - 1) * world_size / 2 * 10
    let arr = output.data().as_any().downcast_ref::<Int32Array>().unwrap();
    for i in 0..10 {
        let expected = world_size * i + (world_size - 1) * world_size / 2 * 10;
        let actual = arr.value(i as usize);
        if actual != expected {
            println!(
                "Column allreduce test FAILED at rank {}: expected {}, got {}",
                rank, expected, actual
            );
            return Ok(());
        }
    }

    println!("Column allreduce test passed at rank {}", rank);
    Ok(())
}

fn test_scalar_allreduce(ctx: &Arc<CylonContext>) -> CylonResult<()> {
    let rank = ctx.get_rank();
    let world_size = ctx.get_world_size();

    // Create scalar with rank + 1 value
    let array = Arc::new(Int32Array::from(vec![rank + 1])) as ArrayRef;
    let scalar = Scalar::new(arrow::array::make_array(array.to_data()));

    // Perform allreduce with SUM
    let output = ctx.get_communicator().all_reduce_scalar(&scalar, ReduceOp::Sum)?;

    // Verify results: sum of 1..world_size = (world_size + 1) * world_size / 2
    let expected = (world_size + 1) * world_size / 2;
    let arr = output.data().as_any().downcast_ref::<Int32Array>().unwrap();
    let actual = arr.value(0);

    if actual != expected {
        println!(
            "Scalar allreduce test FAILED at rank {}: expected {}, got {}",
            rank, expected, actual
        );
        return Ok(());
    }

    println!("Scalar allreduce test passed at rank {}", rank);
    Ok(())
}

fn main() -> CylonResult<()> {
    // Get Redis address from environment
    let redis_addr = env::var("CYLON_UCX_OOB_REDIS_ADDR")
        .unwrap_or_else(|_| "127.0.0.1:6379".to_string());
    let redis_addr = format!("redis://{}", redis_addr);

    // Get world size from environment
    let world_size: i32 = env::var("CYLON_UCX_OOB_WORLD_SIZE")
        .unwrap_or_else(|_| "4".to_string())
        .parse()
        .expect("Invalid world size");

    println!("Initializing UCX/UCC with {} processes", world_size);

    // Create UCX OOB context using Redis
    let oob_ctx = UCXRedisOOBContext::make(world_size, &redis_addr)?;

    // Create UCX communicator
    let ucx_comm = UCXCommunicator::make_oob(oob_ctx)?;

    // Wrap with UCC for collective operations
    let ucx_ucc_comm = UCXUCCCommunicator::new(ucx_comm)?;

    // Initialize context with UCX+UCC communicator
    let mut ctx_new = CylonContext::new(true);
    ctx_new.set_communicator(Box::new(ucx_ucc_comm));
    let ctx = Arc::new(ctx_new);

    let rank = ctx.get_rank();
    println!("Rank {} initialized", rank);

    // Run tests
    test_column_allgather(&ctx)?;
    test_scalar_allgather(&ctx)?;
    test_column_allreduce(&ctx)?;
    test_scalar_allreduce(&ctx)?;

    // Test table allgather if world_size <= 4
    if world_size <= 4 {
        let table = create_test_table(ctx.clone(), rank)?;
        let output = ctx.get_communicator().all_gather(&table, ctx.clone())?;
        println!(
            "Table allgather test: input {} rows, output {} tables",
            table.rows(),
            output.len()
        );
    }

    ctx.barrier()?;

    if rank == 0 {
        println!("\nAll UCC operator tests completed!");
    }

    Ok(())
}
