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

//! Distributed Intersect Example
//!
//! This example demonstrates distributed intersection operations using MPI.
//! It reads two tables and performs a distributed intersect operation.
//!
//! Corresponds to cpp/src/examples/intersect_example.cpp
//!
//! Usage:
//!   mpirun -n 4 cargo run --example intersect_example --features mpi -- m 1000 0.5
//!   mpirun -n 4 cargo run --example intersect_example --features mpi -- f file1 file2

use std::env;
use std::sync::Arc;
use std::time::Instant;

use arrow::array::Int64Array;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use cylon::ctx::CylonContext;
use cylon::error::CylonResult;
use cylon::ops::distributed_set_ops::distributed_intersect;
use cylon::table::Table;

/// Create two in-memory tables with int64 data
fn create_two_in_memory_tables(
    ctx: Arc<CylonContext>,
    count: u64,
    dup_ratio: f64,
) -> CylonResult<(Table, Table)> {
    let rank = ctx.get_rank();
    let world_size = ctx.get_world_size() as u64;

    // Create schema: col1 (Int64), col2 (Int64)
    let schema = Arc::new(Schema::new(vec![
        Field::new("col1", DataType::Int64, false),
        Field::new("col2", DataType::Int64, false),
    ]));

    // Generate left table data
    let unique_count = ((count as f64) * (1.0 - dup_ratio)) as u64;
    let dup_count = count - unique_count;

    let mut left_col1: Vec<i64> = Vec::with_capacity(count as usize);
    let mut left_col2: Vec<i64> = Vec::with_capacity(count as usize);

    for i in 0..unique_count {
        let val = (i * world_size + rank as u64) as i64;
        left_col1.push(val);
        left_col2.push(val * 10);
    }

    for i in 0..dup_count {
        let val = ((i % unique_count.max(1)) * world_size + rank as u64) as i64;
        left_col1.push(val);
        left_col2.push(val * 10);
    }

    // Generate right table data (same pattern to create overlap for intersection)
    let mut right_col1: Vec<i64> = Vec::with_capacity(count as usize);
    let mut right_col2: Vec<i64> = Vec::with_capacity(count as usize);

    for i in 0..unique_count {
        // Use same values to create intersection
        let val = (i * world_size + rank as u64) as i64;
        right_col1.push(val);
        right_col2.push(val * 10);
    }

    for i in 0..dup_count {
        let val = ((i % unique_count.max(1)) * world_size + rank as u64) as i64;
        right_col1.push(val);
        right_col2.push(val * 10);
    }

    let left_batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(left_col1)),
            Arc::new(Int64Array::from(left_col2)),
        ],
    )?;

    let right_batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(right_col1)),
            Arc::new(Int64Array::from(right_col2)),
        ],
    )?;

    let left_table = Table::from_record_batch(ctx.clone(), left_batch)?;
    let right_table = Table::from_record_batch(ctx, right_batch)?;

    Ok((left_table, right_table))
}

fn print_usage() {
    eprintln!("Usage:");
    eprintln!("  ./intersect_example m num_tuples_per_worker 0.0-1.0");
    eprintln!("  ./intersect_example f csv_file1 csv_file2");
}

fn main() -> CylonResult<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 4 {
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

    let (first_table, second_table) = if mode == "m" {
        // In-memory mode
        let count: u64 = args[2].parse().expect("Invalid count");
        let dup_ratio: f64 = args[3].parse().expect("Invalid dup ratio");

        create_two_in_memory_tables(ctx.clone(), count, dup_ratio)?
    } else if mode == "f" {
        // CSV file mode
        let file1 = format!("{}{}.csv", &args[2], rank);
        let file2 = format!("{}{}.csv", &args[3], rank);

        let first_table = Table::from_csv_default(ctx.clone(), &file1)?;
        let second_table = Table::from_csv_default(ctx.clone(), &file2)?;

        (first_table, second_table)
    } else {
        print_usage();
        std::process::exit(1);
    };

    let read_end_time = Instant::now();
    if rank == 0 {
        println!(
            "Read all in {}ms",
            read_end_time.duration_since(start_time).as_millis()
        );
    }

    ctx.barrier()?;

    // Perform distributed intersect
    let intersect_start_time = Instant::now();
    let intersect_table = distributed_intersect(&first_table, &second_table)?;
    let intersect_end_time = Instant::now();

    println!(
        "Rank {}: First table had: {} and Second table had: {}, Intersect has: {} rows",
        rank,
        first_table.rows(),
        second_table.rows(),
        intersect_table.rows()
    );

    if rank == 0 {
        println!(
            "Intersect done in {}ms",
            intersect_end_time
                .duration_since(intersect_start_time)
                .as_millis()
        );
    }

    ctx.barrier()?;
    let end_time = Instant::now();

    if rank == 0 {
        println!(
            "Operation took: {}ms",
            end_time.duration_since(start_time).as_millis()
        );
        println!("Intersect example completed successfully!");
    }

    Ok(())
}
