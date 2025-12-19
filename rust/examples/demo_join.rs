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

//! Demo Join Example
//!
//! This example demonstrates a simple join operation on CSV files.
//!
//! Corresponds to cpp/src/examples/demo_join.cpp
//!
//! Usage:
//!   mpirun -n 2 cargo run --example demo_join --features mpi -- /path/to/data/

use std::env;
use std::sync::Arc;

use cylon::ctx::CylonContext;
use cylon::error::CylonResult;
use cylon::join::{JoinAlgorithm, JoinConfig, JoinType};
use cylon::ops::distributed_join::distributed_join;
use cylon::table::Table;

fn main() -> CylonResult<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: ./demo_join <directory>");
        eprintln!("There should be an argument to the directory for data files");
        std::process::exit(1);
    }

    // Initialize MPI context
    let mut ctx_new = CylonContext::new(true);
    ctx_new.set_communicator(cylon::net::mpi::communicator::MPICommunicator::make()?);
    let ctx = Arc::new(ctx_new);

    let rank = ctx.get_rank() + 1; // 1-indexed for file names
    let directory = &args[1];

    // Construct file paths (similar to C++ example)
    let csv1 = format!("{}user_device_tm_{}.csv", directory, rank);
    let csv2 = format!("{}user_usage_tm_{}.csv", directory, rank);

    println!("Reading CSV files:");
    println!("  File 1: {}", csv1);
    println!("  File 2: {}", csv2);

    // Read first table
    let first_table = Table::from_csv_default(ctx.clone(), &csv1)?;
    println!("First table loaded: {} rows", first_table.rows());

    // Read second table
    let second_table = Table::from_csv_default(ctx.clone(), &csv2)?;
    println!("Second table loaded: {} rows", second_table.rows());

    // Configure inner join on column 0 (left) and column 3 (right)
    let join_config = JoinConfig::new(
        JoinType::Inner,
        vec![0],    // Left join column
        vec![3],    // Right join column
        JoinAlgorithm::Sort,
        "l_".to_string(),
        "r_".to_string(),
    )?;

    // Perform distributed join
    let joined_table = distributed_join(&first_table, &second_table, &join_config)?;

    println!(
        "First table had: {} and Second table had: {}, Joined has: {} rows",
        first_table.rows(),
        second_table.rows(),
        joined_table.rows()
    );

    // Print column names
    let column_names = joined_table.column_names();
    println!("Result columns: {:?}", column_names);

    ctx.barrier()?;

    if ctx.get_rank() == 0 {
        println!("\nDemo join completed successfully!");
    }

    Ok(())
}
