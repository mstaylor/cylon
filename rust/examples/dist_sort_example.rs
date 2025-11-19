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

//! Distributed sort example
//!
//! Ported from cpp/src/examples/dist_sort_example.cpp
//!
//! Run with: mpirun -n 4 cargo run --example dist_sort_example --features mpi --release -- 10000 3
//!
//! Arguments:
//!   num_rows_per_worker - Number of rows each worker will generate
//!   num_iterations - Number of times to run the sort (for benchmarking)

use std::sync::Arc;
use std::time::Instant;
use std::env;
use arrow::array::{Int64Array, Float64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use cylon::ctx::CylonContext;
use cylon::table::Table;
use cylon::ops::distributed_sort::distributed_sort;
use cylon::error::CylonResult;

/// Create in-memory table with random data
/// Corresponds to C++ cylon::examples::create_in_memory_tables
fn create_in_memory_table(
    count: usize,
    ctx: Arc<CylonContext>,
) -> CylonResult<Table> {
    let rank = ctx.get_rank();

    // Create columns with pseudo-random data based on rank
    let mut col1_data = Vec::with_capacity(count);
    let mut col2_data = Vec::with_capacity(count);

    for i in 0..count {
        // Generate values that will be distributed across ranks after sort
        let val1 = ((i as i64 * 31 + rank as i64 * 17) % 1000000) as i64;
        let val2 = ((i as f64 * 3.14159 + rank as f64 * 2.71828) % 1000.0) as f64;
        col1_data.push(val1);
        col2_data.push(val2);
    }

    let schema = Arc::new(Schema::new(vec![
        Field::new("col1", DataType::Int64, false),
        Field::new("col2", DataType::Float64, false),
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(col1_data)),
            Arc::new(Float64Array::from(col2_data)),
        ],
    )?;

    Table::from_record_batch(ctx, batch)
}

/// Run distributed sort and return elapsed time in milliseconds
fn run_example(table: &Table) -> CylonResult<u128> {
    let start = Instant::now();

    // Sort by first column (col1), ascending
    let _output = distributed_sort(table, 0, true)?;

    let elapsed = start.elapsed().as_millis();
    Ok(elapsed)
}

fn main() -> CylonResult<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage: {} <num_rows_per_worker> <num_iterations>", args[0]);
        eprintln!("Example: mpirun -n 4 cargo run --example dist_sort_example --features mpi --release -- 10000 3");
        std::process::exit(1);
    }

    let count: usize = args[1].parse().expect("Invalid num_rows_per_worker");
    let iters: i32 = args[2].parse().expect("Invalid num_iterations");

    // Initialize MPI context
    let mut ctx_new = CylonContext::new(true);
    ctx_new.set_communicator(cylon::net::mpi::communicator::MPICommunicator::make()?);
    let ctx = Arc::new(ctx_new);

    let rank = ctx.get_rank();
    let world_size = ctx.get_world_size();

    if rank == 0 {
        println!("\n========================================");
        println!("Distributed Sort Example");
        println!("========================================");
        println!("World size: {}", world_size);
        println!("Rows per worker: {}", count);
        println!("Iterations: {}", iters);
        println!("Total rows: {}", count * world_size as usize);
        println!("========================================\n");
    }

    // Create in-memory table
    let table = create_in_memory_table(count, ctx.clone())?;

    if rank == 0 {
        println!("Tables created, starting sort...\n");
    }

    // Barrier before timing
    ctx.barrier()?;

    let mut total_time: u128 = 0;

    for i in 0..iters {
        let time = run_example(&table)?;
        total_time += time;

        if rank == 0 {
            println!("Iteration {}: {} ms", i + 1, time);
        }
    }

    if rank == 0 {
        println!("\n========================================");
        println!("Total time: {} ms", total_time);
        println!("Average time: {} ms", total_time / iters as u128);
        println!("========================================\n");
    }

    ctx.barrier()?;

    Ok(())
}
