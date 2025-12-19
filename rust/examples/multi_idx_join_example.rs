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

//! Multi-Index Join Example
//!
//! This example demonstrates distributed join operations on multiple columns.
//!
//! Corresponds to cpp/src/examples/multi_idx_join_example.cpp
//!
//! Usage:
//!   mpirun -n 4 cargo run --example multi_idx_join_example --features mpi -- m 1000 0.5 [hash]
//!   mpirun -n 4 cargo run --example multi_idx_join_example --features mpi -- f file1 file2 [hash]

use std::env;
use std::sync::Arc;
use std::time::Instant;

use arrow::array::Int64Array;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use cylon::ctx::CylonContext;
use cylon::error::CylonResult;
use cylon::join::{JoinAlgorithm, JoinConfig, JoinType};
use cylon::ops::distributed_join::distributed_join;
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

    let unique_count = ((count as f64) * (1.0 - dup_ratio)) as u64;
    let dup_count = count - unique_count;

    // Generate left table data
    let mut left_col1: Vec<i64> = Vec::with_capacity(count as usize);
    let mut left_col2: Vec<i64> = Vec::with_capacity(count as usize);

    for i in 0..unique_count {
        let val = (i * world_size + rank as u64) as i64;
        left_col1.push(val);
        left_col2.push(val % 10); // Second join key
    }

    for i in 0..dup_count {
        let val = ((i % unique_count.max(1)) * world_size + rank as u64) as i64;
        left_col1.push(val);
        left_col2.push(val % 10);
    }

    // Generate right table data (same pattern for join)
    let mut right_col1: Vec<i64> = Vec::with_capacity(count as usize);
    let mut right_col2: Vec<i64> = Vec::with_capacity(count as usize);

    for i in 0..unique_count {
        let val = (i * world_size + rank as u64) as i64;
        right_col1.push(val);
        right_col2.push(val % 10);
    }

    for i in 0..dup_count {
        let val = ((i % unique_count.max(1)) * world_size + rank as u64) as i64;
        right_col1.push(val);
        right_col2.push(val % 10);
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
    eprintln!("  ./multi_idx_join_example m num_tuples_per_worker 0.0-1.0 [hash]");
    eprintln!("  ./multi_idx_join_example f csv_file1 csv_file2 [hash]");
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

    // Determine join algorithm
    let algorithm = if args.len() >= 5 && args[4] == "hash" {
        if rank == 0 {
            println!("Using hash join algorithm");
        }
        JoinAlgorithm::Hash
    } else {
        if rank == 0 {
            println!("Using sort join algorithm");
        }
        JoinAlgorithm::Sort
    };

    let (first_table, second_table) = if mode == "m" {
        let count: u64 = args[2].parse().expect("Invalid count");
        let dup_ratio: f64 = args[3].parse().expect("Invalid dup ratio");

        create_two_in_memory_tables(ctx.clone(), count, dup_ratio)?
    } else if mode == "f" {
        let file1 = format!("{}{}.csv", &args[2], rank);
        let file2 = format!("{}{}.csv", &args[3], rank);

        let first_table = Table::from_csv_default(ctx.clone(), &file1)?;
        let second_table = Table::from_csv_default(ctx.clone(), &file2)?;

        (first_table, second_table)
    } else {
        print_usage();
        std::process::exit(1);
    };

    ctx.barrier()?;
    let read_end_time = Instant::now();

    if rank == 0 {
        println!(
            "Read tables in {}ms",
            read_end_time.duration_since(start_time).as_millis()
        );
    }

    // Multi-column join: join on columns [0, 1]
    let join_config = JoinConfig::new(
        JoinType::Inner,
        vec![0, 1], // Left columns
        vec![0, 1], // Right columns
        algorithm,
        "l_".to_string(),
        "r_".to_string(),
    )?;

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

    // Print column names
    let column_names = joined.column_names();
    println!("Rank {}: Result columns: {:?}", rank, column_names);

    // Print a few rows if small enough
    if joined.rows() <= 100 {
        if let Some(batch) = joined.batch(0) {
            println!("\nRank {} joined data (first 10 rows):", rank);
            let rows_to_print = std::cmp::min(10, batch.num_rows());
            for i in 0..rows_to_print {
                let mut row_str = String::new();
                for col_idx in 0..batch.num_columns() {
                    let col = batch
                        .column(col_idx)
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .unwrap();
                    row_str.push_str(&format!("{:6} ", col.value(i)));
                }
                println!("  {}", row_str);
            }
        }
    }

    ctx.barrier()?;

    if rank == 0 {
        println!("\nMulti-index join example completed successfully!");
    }

    Ok(())
}
