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

//! UCX Join Example
//!
//! This example demonstrates distributed join operations using UCX/UCC
//! with Redis for out-of-band coordination.
//!
//! Corresponds to cpp/src/examples/ucx_join_example.cpp
//!
//! Usage:
//!   # Set session ID to isolate Redis keys
//!   export CYLON_SESSION_ID=$(uuidgen)
//!   cargo run --example ucx_join_example --features "ucx ucc redis" -- file1.csv file2.csv

use std::env;
use std::sync::Arc;
use std::time::Instant;

use cylon::ctx::CylonContext;
use cylon::error::CylonResult;
use cylon::join::{JoinAlgorithm, JoinConfig, JoinType};
use cylon::net::ucx::communicator::UCXCommunicator;
use cylon::net::ucx::redis_oob::UCXRedisOOBContext;
use cylon::ops::distributed_join::distributed_join;
use cylon::table::Table;

fn main() -> CylonResult<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage: ./ucx_join_example <csv_file1> <csv_file2>");
        eprintln!("There should be two arguments with paths to csv files");
        eprintln!("");
        eprintln!("Environment variables required:");
        eprintln!("  CYLON_SESSION_ID - Unique session identifier (e.g., $(uuidgen))");
        eprintln!("  CYLON_UCX_OOB_REDIS_ADDR - Redis address (e.g., 127.0.0.1:6379)");
        eprintln!("  CYLON_UCX_OOB_WORLD_SIZE - Number of processes");
        std::process::exit(1);
    }

    let start_time = Instant::now();

    // Get Redis address from environment
    let redis_addr = env::var("CYLON_UCX_OOB_REDIS_ADDR")
        .unwrap_or_else(|_| "127.0.0.1:6379".to_string());
    let redis_addr = format!("redis://{}", redis_addr);

    // Get world size from environment
    let world_size: i32 = env::var("CYLON_UCX_OOB_WORLD_SIZE")
        .unwrap_or_else(|_| "2".to_string())
        .parse()
        .expect("Invalid world size");

    // Create UCX OOB context using Redis
    let oob_ctx = UCXRedisOOBContext::make(world_size, &redis_addr)?;

    // Create UCX communicator
    let ucx_comm = UCXCommunicator::make_oob(oob_ctx)?;

    // Initialize context with UCX communicator
    let mut ctx_new = CylonContext::new(true);
    ctx_new.set_communicator(Box::new(ucx_comm));
    let ctx = Arc::new(ctx_new);

    let rank = ctx.get_rank();

    // Read CSV files
    let first_table = Table::from_csv_default(ctx.clone(), &args[1])?;
    let second_table = Table::from_csv_default(ctx.clone(), &args[2])?;

    let read_end_time = Instant::now();
    println!(
        "Rank {}: Read tables in {}ms",
        rank,
        read_end_time.duration_since(start_time).as_millis()
    );

    // Configure multi-column inner join (columns 0 and 1)
    let join_config = JoinConfig::new(
        JoinType::Inner,
        vec![0, 1],
        vec![0, 1],
        JoinAlgorithm::Sort,
        "l_".to_string(),
        "r_".to_string(),
    )?;

    // Perform distributed join
    let joined = distributed_join(&first_table, &second_table, &join_config)?;
    let join_end_time = Instant::now();

    println!(
        "Rank {}: First table had: {} and Second table had: {}, Joined has: {} rows",
        rank,
        first_table.rows(),
        second_table.rows(),
        joined.rows()
    );

    if rank == 0 {
        println!(
            "Join done in {}ms",
            join_end_time.duration_since(read_end_time).as_millis()
        );
    }

    println!("UCX join example completed successfully!");

    Ok(())
}
