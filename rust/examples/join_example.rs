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

//! Distributed Join Example
//!
//! This example demonstrates distributed join operations using MPI.
//! It supports both in-memory generated tables and CSV file input.
//!
//! Corresponds to cpp/src/examples/join_example.cpp
//!
//! Usage:
//!   mpirun -n 4 cargo run --example join_example --features mpi -- m 1000 0.5 [hash]
//!   mpirun -n 4 cargo run --example join_example --features mpi -- f file1 file2 [hash]
//!
//! Arguments:
//!   m num_tuples dup_ratio [hash] - Generate in-memory tables
//!   f csv_file1 csv_file2 [hash]  - Read from CSV files

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
///
/// Corresponds to cylon::examples::create_two_in_memory_tables
fn create_two_in_memory_tables(
    ctx: Arc<CylonContext>,
    count: u64,
    dup_ratio: f64,
) -> CylonResult<(Table, Table)> {
    let rank = ctx.get_rank();
    let world_size = ctx.get_world_size() as u64;

    // Create schema: id (Int64), value (Int64)
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("value", DataType::Int64, false),
    ]));

    // Generate left table data
    // id values distributed across ranks, with some duplicates based on dup_ratio
    let unique_count = ((count as f64) * (1.0 - dup_ratio)) as u64;
    let dup_count = count - unique_count;

    let mut left_ids: Vec<i64> = Vec::with_capacity(count as usize);
    let mut left_values: Vec<i64> = Vec::with_capacity(count as usize);

    // Unique values
    for i in 0..unique_count {
        let id = (i * world_size + rank as u64) as i64;
        left_ids.push(id);
        left_values.push(id * 100);
    }

    // Duplicates (repeat some values)
    for i in 0..dup_count {
        let id = ((i % unique_count.max(1)) * world_size + rank as u64) as i64;
        left_ids.push(id);
        left_values.push(id * 100);
    }

    // Generate right table data (similar pattern but with different value multiplier)
    let mut right_ids: Vec<i64> = Vec::with_capacity(count as usize);
    let mut right_values: Vec<i64> = Vec::with_capacity(count as usize);

    for i in 0..unique_count {
        let id = (i * world_size + rank as u64) as i64;
        right_ids.push(id);
        right_values.push(id * 200);
    }

    for i in 0..dup_count {
        let id = ((i % unique_count.max(1)) * world_size + rank as u64) as i64;
        right_ids.push(id);
        right_values.push(id * 200);
    }

    let left_batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(left_ids)),
            Arc::new(Int64Array::from(left_values)),
        ],
    )?;

    let right_batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(right_ids)),
            Arc::new(Int64Array::from(right_values)),
        ],
    )?;

    let left_table = Table::from_record_batch(ctx.clone(), left_batch)?;
    let right_table = Table::from_record_batch(ctx, right_batch)?;

    Ok((left_table, right_table))
}

fn print_usage() {
    eprintln!("Usage:");
    eprintln!("  ./join_example m num_tuples_per_worker 0.0-1.0 [hash]");
    eprintln!("  ./join_example f csv_file1 csv_file2 [hash]");
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
    let _world_size = ctx.get_world_size();

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

    ctx.barrier()?;
    let read_end_time = Instant::now();
    if rank == 0 {
        println!(
            "Read tables in {}ms",
            read_end_time.duration_since(start_time).as_millis()
        );
    }

    // Configure and perform join
    let join_config = JoinConfig::new(
        JoinType::Inner,
        vec![0],
        vec![0],
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
    if rank == 0 {
        let column_names = joined.column_names();
        println!("Result columns: {:?}", column_names);
    }

    ctx.barrier()?;

    if rank == 0 {
        println!("Join example completed successfully!");
    }

    Ok(())
}
