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

//! FMI Example
//!
//! This example demonstrates distributed operations using the FMI
//! (Flexible Message Interface) communication layer with direct TCP
//! connections and Redis for coordination.
//!
//! Corresponds to cpp/src/examples/fmi_example.cpp
//!
//! Usage:
//!   cargo run --example fmi_example --features "fmi redis" -- \
//!     <directory> <rank> <world_size> <comm_name> <host> <port> \
//!     <max_timeout> <nonblocking> <redis_host> <redis_port> <redis_namespace>

use std::env;
use std::sync::Arc;

use arrow::array::{ArrayRef, Int64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use cylon::ctx::CylonContext;
use cylon::error::CylonResult;
use cylon::join::{JoinAlgorithm, JoinConfig, JoinType};
use cylon::net::fmi::cylon_communicator::{FMIConfig, FMICommunicator};
use cylon::ops::distributed_join::distributed_join;
use cylon::table::Table;

static K_COUNT: i64 = 5000000;
static K_DUP: f64 = 0.9;

/// Create an in-memory table with int64 data
fn create_in_memory_table(
    ctx: Arc<CylonContext>,
    count: i64,
    dup_ratio: f64,
) -> CylonResult<Table> {
    let rank = ctx.get_rank();
    let world_size = ctx.get_world_size() as i64;

    let schema = Arc::new(Schema::new(vec![
        Field::new("col1", DataType::Int64, false),
        Field::new("col2", DataType::Int64, false),
    ]));

    let unique_count = ((count as f64) * (1.0 - dup_ratio)) as i64;
    let dup_count = count - unique_count;

    let mut col1: Vec<i64> = Vec::with_capacity(count as usize);
    let mut col2: Vec<i64> = Vec::with_capacity(count as usize);

    // Generate unique values
    for i in 0..unique_count {
        let val = i * world_size + rank as i64;
        col1.push(val);
        col2.push(val * 10);
    }

    // Generate duplicate values
    for i in 0..dup_count {
        let val = (i % unique_count.max(1)) * world_size + rank as i64;
        col1.push(val);
        col2.push(val * 10);
    }

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(col1)) as ArrayRef,
            Arc::new(Int64Array::from(col2)) as ArrayRef,
        ],
    )?;

    Table::from_record_batch(ctx, batch)
}

fn print_usage() {
    eprintln!("Usage:");
    eprintln!("  ./fmi_example <directory> <rank> <world_size> <comm_name> <host> <port> \\");
    eprintln!("    <max_timeout> <nonblocking> <redis_host> <redis_port> <redis_namespace>");
    eprintln!("");
    eprintln!("Arguments:");
    eprintln!("  directory      - Directory for data files (not used in in-memory mode)");
    eprintln!("  rank           - This process's rank (0-indexed)");
    eprintln!("  world_size     - Total number of processes");
    eprintln!("  comm_name      - Communication name/identifier");
    eprintln!("  host           - Rendezvous host");
    eprintln!("  port           - Rendezvous port");
    eprintln!("  max_timeout    - Maximum timeout for connections (ms)");
    eprintln!("  nonblocking    - 1 for non-blocking, 0 for blocking");
    eprintln!("  redis_host     - Redis server host");
    eprintln!("  redis_port     - Redis server port");
    eprintln!("  redis_namespace- Redis namespace for this run");
}

fn main() -> CylonResult<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 12 {
        print_usage();
        std::process::exit(1);
    }

    let _directory = &args[1];
    let rank: i32 = args[2].parse().expect("Invalid rank");
    let world_size: i32 = args[3].parse().expect("Invalid world_size");
    let comm_name = &args[4];
    let host = &args[5];
    let port: i32 = args[6].parse().expect("Invalid port");
    let max_timeout: i32 = args[7].parse().expect("Invalid max_timeout");
    let nonblocking: i32 = args[8].parse().expect("Invalid nonblocking");
    let redis_host = &args[9];
    let redis_port: i32 = args[10].parse().expect("Invalid redis_port");
    let redis_namespace = &args[11];

    // Create FMI config
    let config = FMIConfig::builder()
        .rank(rank)
        .world_size(world_size)
        .host(host)
        .port(port)
        .max_timeout(max_timeout)
        .resolve_ip(false)
        .comm_name(comm_name)
        .nonblocking(nonblocking != 0)
        .enable_ping(true)
        .redis_host(redis_host)
        .redis_port(redis_port)
        .redis_namespace(redis_namespace)
        .build()?;

    // Create FMI communicator
    let fmi_comm = FMICommunicator::new(config)?;

    // Initialize context with FMI communicator
    let mut ctx_new = CylonContext::new(true);
    ctx_new.set_communicator(Box::new(fmi_comm));
    let ctx = Arc::new(ctx_new);

    println!(
        "FMI initialized: rank={} world_size={}",
        ctx.get_rank(),
        ctx.get_world_size()
    );

    ctx.barrier()?;

    // Create in-memory tables for the join test
    let first_table = create_in_memory_table(ctx.clone(), K_COUNT, K_DUP)?;
    let second_table = create_in_memory_table(ctx.clone(), K_COUNT, K_DUP)?;

    // Configure inner join
    let join_config = JoinConfig::new(
        JoinType::Inner,
        vec![0],
        vec![0],
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
        println!("\nFMI example completed successfully!");
    }

    Ok(())
}
