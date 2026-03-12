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

//! Redis UCC/UCX Example
//!
//! This example demonstrates a distributed join using UCX+UCC with Redis
//! for out-of-band coordination. This allows running distributed operations
//! without MPI.
//!
//! Corresponds to cpp/src/examples/redis_ucc_ucx_example.cpp
//!
//! Usage:
//!   # Set session ID to isolate Redis keys
//!   export CYLON_SESSION_ID=$(uuidgen)
//!   export CYLON_UCX_OOB_REDIS_ADDR=127.0.0.1:6379
//!   export CYLON_UCX_OOB_WORLD_SIZE=2
//!   cargo run --example redis_ucc_ucx_example --features "ucx ucc redis" -- /path/to/data/

use std::env;
use std::sync::Arc;

use cylon::ctx::CylonContext;
use cylon::error::CylonResult;
use cylon::join::{JoinAlgorithm, JoinConfig, JoinType};
use cylon::net::ucx::communicator::{UCXCommunicator, UCXUCCCommunicator};
use cylon::net::ucx::redis_oob::UCXRedisOOBContext;
use cylon::ops::distributed_join::distributed_join;
use cylon::table::Table;

fn main() -> CylonResult<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: ./redis_ucc_ucx_example <directory>");
        eprintln!("There should be an argument to the directory for data files");
        eprintln!("");
        eprintln!("Environment variables required:");
        eprintln!("  CYLON_SESSION_ID - Unique session identifier (e.g., $(uuidgen))");
        eprintln!("  CYLON_UCX_OOB_REDIS_ADDR - Redis address (e.g., 127.0.0.1:6379)");
        eprintln!("  CYLON_UCX_OOB_WORLD_SIZE - Number of processes");
        std::process::exit(1);
    }

    // Get Redis address from environment
    let redis_addr = env::var("CYLON_UCX_OOB_REDIS_ADDR")
        .unwrap_or_else(|_| "127.0.0.1:6379".to_string());
    let redis_addr = format!("redis://{}", redis_addr);

    // Get world size from environment
    let world_size: i32 = env::var("CYLON_UCX_OOB_WORLD_SIZE")
        .unwrap_or_else(|_| "2".to_string())
        .parse()
        .expect("Invalid world size");

    println!("Connecting to Redis at {}", redis_addr);

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

    // Use 1-indexed rank for file names (like C++ example)
    let modified_rank = ctx.get_rank() + 1;
    let directory = &args[1];

    // Construct file paths (same as C++ example)
    let csv1 = format!("{}user_device_tm_{}.csv", directory, modified_rank);
    let csv2 = format!("{}user_usage_tm_{}.csv", directory, modified_rank);

    println!("Reading CSV files:");
    println!("  File 1: {}", csv1);
    println!("  File 2: {}", csv2);

    // Read tables
    let first_table = Table::from_csv_default(ctx.clone(), &csv1)?;
    let second_table = Table::from_csv_default(ctx.clone(), &csv2)?;

    // Configure inner join (column 0 on left, column 3 on right)
    let join_config = JoinConfig::new(
        JoinType::Inner,
        vec![0],
        vec![3],
        JoinAlgorithm::Sort,
        "l_".to_string(),
        "r_".to_string(),
    )?;

    // Perform distributed join
    let joined_table = distributed_join(&first_table, &second_table, &join_config)?;

    println!(
        "First table had: {} and Second table had: {}, Joined has: {} rows",
        first_table.rows(),
        second_table.rows(),
        joined_table.rows()
    );

    ctx.barrier()?;

    if ctx.get_rank() == 0 {
        println!("\nRedis UCC/UCX example completed successfully!");
    }

    Ok(())
}
