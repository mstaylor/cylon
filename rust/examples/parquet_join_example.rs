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

//! Parquet Join Example
//!
//! This example reads two parquet files and performs a distributed join on them.
//!
//! Corresponds to cpp/src/examples/parquet_join_example.cpp
//!
//! Usage:
//!   mpirun -n 2 cargo run --example parquet_join_example --features "mpi parquet" -- file1.parquet file2.parquet

use std::env;
use std::sync::Arc;
use std::time::Instant;

use cylon::ctx::CylonContext;
use cylon::error::CylonResult;
use cylon::join::{JoinAlgorithm, JoinConfig, JoinType};
use cylon::ops::distributed_join::distributed_join;
use cylon::table::Table;

fn main() -> CylonResult<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage: ./parquet_join_example <parquet_file1> <parquet_file2>");
        eprintln!("There should be two arguments with paths to parquet files");
        std::process::exit(1);
    }

    let start_time = Instant::now();

    // Initialize MPI context
    let mut ctx_new = CylonContext::new(true);
    ctx_new.set_communicator(cylon::net::mpi::communicator::MPICommunicator::make()?);
    let ctx = Arc::new(ctx_new);

    let rank = ctx.get_rank();

    // Read first parquet file
    let first_table = Table::from_parquet(ctx.clone(), &args[1])?;

    // Read second parquet file
    let second_table = Table::from_parquet(ctx.clone(), &args[2])?;

    let read_end_time = Instant::now();
    if rank == 0 {
        println!(
            "Read tables in {}ms",
            read_end_time.duration_since(start_time).as_millis()
        );
    }

    // Configure inner join on column 0 for both tables
    let join_config = JoinConfig::new(
        JoinType::Inner,
        vec![0],
        vec![0],
        JoinAlgorithm::Sort,
        "l_".to_string(),
        "r_".to_string(),
    )?;

    // Perform distributed join
    let joined = distributed_join(&first_table, &second_table, &join_config)?;
    let join_end_time = Instant::now();

    println!(
        "First table had: {} and Second table had: {}, Joined has: {} rows",
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

    ctx.barrier()?;

    if rank == 0 {
        println!("\nParquet join example completed successfully!");
    }

    Ok(())
}
