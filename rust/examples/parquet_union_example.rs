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

//! Parquet Union Example
//!
//! This example reads two parquet files and performs a distributed union on them.
//!
//! Corresponds to cpp/src/examples/parquet_union_example.cpp
//!
//! Usage:
//!   mpirun -n 2 cargo run --example parquet_union_example --features "mpi parquet" -- file1.parquet file2.parquet

use std::env;
use std::sync::Arc;
use std::time::Instant;

use cylon::ctx::CylonContext;
use cylon::error::CylonResult;
use cylon::ops::distributed_set_ops::distributed_union;
use cylon::table::Table;

fn main() -> CylonResult<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage: ./parquet_union_example <parquet_file1> <parquet_file2>");
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
            "Read all in {}ms",
            read_end_time.duration_since(start_time).as_millis()
        );
    }

    // Perform distributed union
    let union_start_time = Instant::now();
    let unioned_table = distributed_union(&first_table, &second_table)?;
    let union_end_time = Instant::now();

    println!(
        "First table had: {} and Second table had: {}, Union has: {} rows",
        first_table.rows(),
        second_table.rows(),
        unioned_table.rows()
    );

    if rank == 0 {
        println!(
            "Union done in {}ms",
            union_end_time.duration_since(union_start_time).as_millis()
        );
    }

    ctx.barrier()?;

    let end_time = Instant::now();
    if rank == 0 {
        println!(
            "Operation took: {}ms",
            end_time.duration_since(start_time).as_millis()
        );
    }

    Ok(())
}
