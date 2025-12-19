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

//! Distributed Unique Example
//!
//! This example demonstrates distributed unique operations using MPI.
//! It reads a table and performs a distributed unique operation.
//!
//! Corresponds to cpp/src/examples/unique_example.cpp
//!
//! Usage:
//!   mpirun -n 4 cargo run --example unique_example --features mpi -- m 1000 0.5
//!   mpirun -n 4 cargo run --example unique_example --features mpi -- f csv_file

use std::env;
use std::sync::Arc;
use std::time::Instant;

use arrow::array::Int64Array;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use cylon::ctx::CylonContext;
use cylon::error::CylonResult;
use cylon::ops::distributed_set_ops::distributed_unique;
use cylon::table::Table;

/// Create an in-memory table with int64 data including duplicates
fn create_in_memory_table(
    ctx: Arc<CylonContext>,
    count: i64,
    dup_ratio: f64,
) -> CylonResult<Table> {
    let rank = ctx.get_rank();
    let world_size = ctx.get_world_size() as i64;

    // Create schema: col1 (Int64), col2 (Int64)
    let schema = Arc::new(Schema::new(vec![
        Field::new("col1", DataType::Int64, false),
        Field::new("col2", DataType::Int64, false),
    ]));

    let unique_count = ((count as f64) * (1.0 - dup_ratio)) as i64;
    let dup_count = count - unique_count;

    let mut col1: Vec<i64> = Vec::with_capacity(count as usize);
    let mut col2: Vec<i64> = Vec::with_capacity(count as usize);

    // Generate unique values (distributed across ranks)
    for i in 0..unique_count {
        let val = i * world_size + rank as i64;
        col1.push(val);
        col2.push(val * 10);
    }

    // Generate duplicate values (same row multiple times)
    for i in 0..dup_count {
        let val = (i % unique_count.max(1)) * world_size + rank as i64;
        col1.push(val);
        col2.push(val * 10);
    }

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(col1)),
            Arc::new(Int64Array::from(col2)),
        ],
    )?;

    Table::from_record_batch(ctx, batch)
}

fn print_usage() {
    eprintln!("Usage:");
    eprintln!("  ./unique_example m num_tuples_per_worker 0.0-1.0");
    eprintln!("  ./unique_example f csv_file");
}

fn main() -> CylonResult<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        print_usage();
        std::process::exit(1);
    }

    let start_time = Instant::now();

    // Initialize MPI context
    let mut ctx_new = CylonContext::new(true);
    ctx_new.set_communicator(cylon::net::mpi::communicator::MPICommunicator::make()?);
    let ctx = Arc::new(ctx_new);

    let rank = ctx.get_rank();

    let mode = &args[1];

    let input_table = if mode == "m" {
        // In-memory mode
        if args.len() < 4 {
            print_usage();
            std::process::exit(1);
        }
        let count: i64 = args[2].parse().expect("Invalid count");
        let dup_ratio: f64 = args[3].parse().expect("Invalid dup ratio");

        if rank == 0 {
            println!("Using in-memory tables");
        }
        create_in_memory_table(ctx.clone(), count, dup_ratio)?
    } else if mode == "f" {
        // CSV file mode
        if rank == 0 {
            println!("Using files");
        }
        let file = format!("{}{}.csv", &args[2], rank);
        Table::from_csv_default(ctx.clone(), &file)?
    } else {
        print_usage();
        std::process::exit(1);
    };

    ctx.barrier()?;
    let read_end_time = Instant::now();
    if rank == 0 {
        println!(
            "Input tables created in {}ms",
            read_end_time.duration_since(start_time).as_millis()
        );
    }

    // Get all columns for unique operation
    let columns: Vec<usize> = (0..input_table.columns() as usize).collect();

    // Perform distributed unique
    let unique_start_time = Instant::now();
    let unique_table = distributed_unique(&input_table, &columns)?;
    let unique_end_time = Instant::now();

    println!(
        "Rank {}: Table had: {}, Output has: {} rows",
        rank,
        input_table.rows(),
        unique_table.rows()
    );

    if rank == 0 {
        println!(
            "Completed in {}ms",
            unique_end_time.duration_since(unique_start_time).as_millis()
        );
    }

    ctx.barrier()?;

    if rank == 0 {
        println!("Unique example completed successfully!");
    }

    Ok(())
}
