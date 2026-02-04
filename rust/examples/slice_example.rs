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

//! Slice Example
//!
//! This example demonstrates slice, head, and tail operations on tables.
//!
//! Corresponds to cpp/src/examples/slice_example.cpp
//!
//! Usage:
//!   mpirun -n 2 cargo run --example slice_example --features mpi -- m 1000 0.5 100 50
//!   mpirun -n 2 cargo run --example slice_example --features mpi -- f csv_file 100 50

use std::env;
use std::sync::Arc;
use std::time::Instant;

use arrow::array::Int64Array;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use cylon::ctx::CylonContext;
use cylon::error::CylonResult;
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

    // Generate duplicate values
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

fn print_table(table: &Table, name: &str, max_rows: usize) {
    if let Some(batch) = table.batch(0) {
        let col1 = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let col2 = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();

        println!("\n{} ({} rows):", name, table.rows());
        println!("  col1 | col2");
        println!("  -----|------");
        let rows_to_print = std::cmp::min(max_rows, col1.len());
        for i in 0..rows_to_print {
            println!("  {:4} | {:5}", col1.value(i), col2.value(i));
        }
        if col1.len() > max_rows {
            println!("  ... ({} more rows)", col1.len() - max_rows);
        }
    }
}

fn print_usage() {
    eprintln!("Usage:");
    eprintln!("  ./slice_example m num_tuples_per_worker 0.0-1.0 offset length");
    eprintln!("  ./slice_example f csv_file offset length");
}

fn main() -> CylonResult<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 5 {
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

    let (in_table, offset, length) = if mode == "m" {
        if args.len() < 6 {
            print_usage();
            std::process::exit(1);
        }
        let count: i64 = args[2].parse().expect("Invalid count");
        let dup_ratio: f64 = args[3].parse().expect("Invalid dup ratio");
        let offset: usize = args[4].parse().expect("Invalid offset");
        let length: usize = args[5].parse().expect("Invalid length");

        if rank == 0 {
            println!("Using in-memory tables with {} rows", count);
        }
        (create_in_memory_table(ctx.clone(), count, dup_ratio)?, offset, length)
    } else if mode == "f" {
        let file = &args[2];
        let offset: usize = args[3].parse().expect("Invalid offset");
        let length: usize = args[4].parse().expect("Invalid length");

        if rank == 0 {
            println!("Loading from CSV file: {}", file);
        }
        (Table::from_csv_default(ctx.clone(), file)?, offset, length)
    } else {
        print_usage();
        std::process::exit(1);
    };

    ctx.barrier()?;
    let read_end_time = Instant::now();
    if rank == 0 {
        println!(
            "Read table in {}ms",
            read_end_time.duration_since(start_time).as_millis()
        );
    }

    println!("Rank {}: Input table has {} rows", rank, in_table.rows());

    // Slice operation
    let slice_start = Instant::now();
    let sliced = in_table.slice(offset, length)?;
    let slice_end = Instant::now();

    println!(
        "Rank {}: Sliced table (offset={}, length={}) has {} rows",
        rank, offset, length, sliced.rows()
    );
    if rank == 0 {
        println!(
            "Slice done in {}ms",
            slice_end.duration_since(slice_start).as_millis()
        );
    }

    if sliced.rows() <= 20 {
        print_table(&sliced, &format!("Rank {} Sliced", rank), 20);
    }

    // Head operation
    let num_rows: usize = 10;
    let head_start = Instant::now();
    let head_table = in_table.head(num_rows)?;
    let head_end = Instant::now();

    println!("Rank {}: Head table has {} rows", rank, head_table.rows());
    if rank == 0 {
        println!(
            "Head done in {}ms",
            head_end.duration_since(head_start).as_millis()
        );
    }

    if head_table.rows() <= 20 {
        print_table(&head_table, &format!("Rank {} Head", rank), 20);
    }

    // Tail operation
    let tail_start = Instant::now();
    let tail_table = in_table.tail(num_rows)?;
    let tail_end = Instant::now();

    println!("Rank {}: Tail table has {} rows", rank, tail_table.rows());
    if rank == 0 {
        println!(
            "Tail done in {}ms",
            tail_end.duration_since(tail_start).as_millis()
        );
    }

    if tail_table.rows() <= 20 {
        print_table(&tail_table, &format!("Rank {} Tail", rank), 20);
    }

    ctx.barrier()?;

    if rank == 0 {
        println!("\nSlice example completed successfully!");
    }

    Ok(())
}
