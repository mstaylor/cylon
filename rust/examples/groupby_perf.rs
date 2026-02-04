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

//! GroupBy Performance Benchmark
//!
//! This example benchmarks distributed hash groupby operations with
//! aggregations (sum, mean, stddev).
//!
//! Corresponds to cpp/src/examples/groupby_perf.cpp
//!
//! Usage:
//!   mpirun -n 4 cargo run --example groupby_perf --features mpi -- m 1000000 0.9
//!   mpirun -n 4 cargo run --example groupby_perf --features mpi -- f csv_file

use std::env;
use std::sync::Arc;
use std::time::Instant;

use arrow::array::{ArrayRef, Int64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use cylon::ctx::CylonContext;
use cylon::error::CylonResult;
use cylon::groupby::distributed_hash_groupby_single;
use cylon::mapreduce::AggregationOpId;
use cylon::table::Table;

/// Create an in-memory table with int64 data
fn create_in_memory_table(
    ctx: Arc<CylonContext>,
    count: i64,
    dup_ratio: f64,
) -> CylonResult<Table> {
    let rank = ctx.get_rank();
    let world_size = ctx.get_world_size() as i64;

    // Create schema with two columns
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, false),
        Field::new("value", DataType::Int64, false),
    ]));

    let unique_count = ((count as f64) * (1.0 - dup_ratio)) as i64;
    let dup_count = count - unique_count;

    let mut keys: Vec<i64> = Vec::with_capacity(count as usize);
    let mut values: Vec<i64> = Vec::with_capacity(count as usize);

    // Generate unique values
    for i in 0..unique_count {
        let key = i * world_size + rank as i64;
        keys.push(key);
        values.push(key * 10);
    }

    // Generate duplicate values (same key multiple times)
    for i in 0..dup_count {
        let key = (i % unique_count.max(1)) * world_size + rank as i64;
        keys.push(key);
        values.push(key * 10 + (i % 5)); // Slightly different values for variation
    }

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(keys)) as ArrayRef,
            Arc::new(Int64Array::from(values)) as ArrayRef,
        ],
    )?;

    Table::from_record_batch(ctx, batch)
}

fn print_usage() {
    eprintln!("Usage:");
    eprintln!("  ./groupby_perf m num_tuples_per_worker dup_factor[0.0-1.0]");
    eprintln!("  ./groupby_perf f csv_file");
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

    let table = if mode == "m" && args.len() >= 4 {
        if rank == 0 {
            println!("Using in-memory tables");
        }
        let count: i64 = args[2].parse().expect("Invalid count");
        let dup: f64 = args[3].parse().expect("Invalid dup ratio");
        create_in_memory_table(ctx.clone(), count, dup)?
    } else if mode == "f" {
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

    // Distributed Hash GroupBy with Sum, Mean, StdDev on column 1
    let groupby_start = Instant::now();
    let output = distributed_hash_groupby_single(
        &table,
        0,  // group by column 0 (key)
        &[1, 1, 1],  // aggregate column 1 three times
        &[AggregationOpId::Sum, AggregationOpId::Mean, AggregationOpId::Stddev],
    )?;
    let groupby_end = Instant::now();

    println!(
        "Rank {}: Table had: {} rows, output has: {} rows",
        rank,
        table.rows(),
        output.rows()
    );

    if rank == 0 {
        println!(
            "Distributed Hash GroupBy completed in {}ms",
            groupby_end.duration_since(groupby_start).as_millis()
        );
    }

    // Print column names of output
    let column_names = output.column_names();
    if rank == 0 {
        println!("Output columns: {:?}", column_names);
    }

    // Print a few rows if small enough
    if output.rows() <= 20 {
        if let Some(batch) = output.batch(0) {
            println!("\nRank {} output (first {} rows):", rank, batch.num_rows().min(10));
            let rows_to_print = batch.num_rows().min(10);
            for i in 0..rows_to_print {
                let mut row_str = String::new();
                for col_idx in 0..batch.num_columns() {
                    let col = batch.column(col_idx);
                    if let Some(arr) = col.as_any().downcast_ref::<Int64Array>() {
                        row_str.push_str(&format!("{:10} ", arr.value(i)));
                    } else if let Some(arr) = col.as_any().downcast_ref::<arrow::array::Float64Array>() {
                        row_str.push_str(&format!("{:10.2} ", arr.value(i)));
                    }
                }
                println!("  {}", row_str);
            }
        }
    }

    ctx.barrier()?;

    if rank == 0 {
        println!("\nGroupBy performance test completed!");
    }

    Ok(())
}
