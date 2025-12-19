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

//! Multi-column Sorting Example
//!
//! This example demonstrates distributed sorting on multiple columns with
//! different sort directions (ascending/descending).
//!
//! Corresponds to cpp/src/examples/multicolumn_sorting_example.cpp
//!
//! Usage:
//!   mpirun -n 4 cargo run --example multicolumn_sorting_example --features mpi -- m 1000 0.5
//!   mpirun -n 4 cargo run --example multicolumn_sorting_example --features mpi -- f csv_file

use std::env;
use std::sync::Arc;
use std::time::Instant;

use arrow::array::Int64Array;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use cylon::ctx::CylonContext;
use cylon::error::CylonResult;
use cylon::ops::distributed_sort::{distributed_sort_multi, SortOptions};
use cylon::table::Table;

/// Create an in-memory table with int64 data
fn create_in_memory_table(ctx: Arc<CylonContext>, count: u64, dup_ratio: f64) -> CylonResult<Table> {
    let rank = ctx.get_rank();
    let world_size = ctx.get_world_size() as u64;

    // Create schema: col1 (Int64), col2 (Int64)
    let schema = Arc::new(Schema::new(vec![
        Field::new("col1", DataType::Int64, false),
        Field::new("col2", DataType::Int64, false),
    ]));

    let unique_count = ((count as f64) * (1.0 - dup_ratio)) as u64;
    let dup_count = count - unique_count;

    let mut col1: Vec<i64> = Vec::with_capacity(count as usize);
    let mut col2: Vec<i64> = Vec::with_capacity(count as usize);

    // Generate unique values (distributed across ranks)
    for i in 0..unique_count {
        let val = (i * world_size + rank as u64) as i64;
        col1.push(val);
        col2.push((val * 7) % 100); // Different pattern for second column
    }

    // Generate duplicate values
    for i in 0..dup_count {
        let val = ((i % unique_count.max(1)) * world_size + rank as u64) as i64;
        col1.push(val);
        col2.push((val * 7) % 100);
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
    eprintln!("  ./multicolumn_sorting_example m num_tuples_per_worker 0.0-1.0");
    eprintln!("  ./multicolumn_sorting_example f csv_file");
}

fn main() -> CylonResult<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        print_usage();
        std::process::exit(1);
    }

    let _start_time = Instant::now();

    // Initialize MPI context
    let mut ctx_new = CylonContext::new(true);
    ctx_new.set_communicator(cylon::net::mpi::communicator::MPICommunicator::make()?);
    let ctx = Arc::new(ctx_new);

    let rank = ctx.get_rank();

    let mode = &args[1];

    let input_table = if mode == "m" {
        if args.len() < 4 {
            print_usage();
            std::process::exit(1);
        }
        let count: u64 = args[2].parse().expect("Invalid count");
        let dup_ratio: f64 = args[3].parse().expect("Invalid dup ratio");

        create_in_memory_table(ctx.clone(), count, dup_ratio)?
    } else if mode == "f" {
        let file = format!("{}{}.csv", &args[2], rank);
        Table::from_csv_default(ctx.clone(), &file)?
    } else {
        print_usage();
        std::process::exit(1);
    };

    ctx.barrier()?;

    // Sort by columns [0, 1] with directions [ascending, descending]
    let sort_columns = vec![0, 1];
    let sort_directions = vec![true, false]; // true = ascending, false = descending

    let sort_start_time = Instant::now();
    let sorted_table = distributed_sort_multi(&input_table, &sort_columns, &sort_directions, SortOptions::default())?;
    let sort_end_time = Instant::now();

    println!(
        "Rank {}: Input table had: {}, Sorted table has: {} rows",
        rank,
        input_table.rows(),
        sorted_table.rows()
    );

    if rank == 0 {
        println!(
            "Multi-column sort done in {}ms",
            sort_end_time.duration_since(sort_start_time).as_millis()
        );
    }

    // Print table if small enough
    if sorted_table.rows() <= 100 {
        if let Some(batch) = sorted_table.batch(0) {
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

            println!("\nRank {} sorted data (col1 ASC, col2 DESC):", rank);
            println!("  col1 | col2");
            println!("  -----|-----");
            let rows_to_print = std::cmp::min(20, col1.len());
            for i in 0..rows_to_print {
                println!("  {:4} | {:4}", col1.value(i), col2.value(i));
            }
            if col1.len() > 20 {
                println!("  ... ({} more rows)", col1.len() - 20);
            }
        }
    }

    ctx.barrier()?;

    if rank == 0 {
        println!("\nMulti-column sorting example completed successfully!");
    }

    Ok(())
}
