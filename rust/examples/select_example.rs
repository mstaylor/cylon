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

//! Select Example
//!
//! This example demonstrates the select operation to filter rows based on a condition.
//! It reads a table and selects rows where a specific condition is met.
//!
//! Corresponds to cpp/src/examples/select_example.cpp
//!
//! Usage:
//!   cargo run --example select_example -- csv_file.csv
//!   mpirun -n 2 cargo run --example select_example --features mpi -- csv_file.csv

use std::env;
use std::sync::Arc;
use std::time::Instant;

use arrow::array::{Array, BooleanArray, Float64Array, Int64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use cylon::ctx::CylonContext;
use cylon::error::CylonResult;
use cylon::table::Table;

fn create_sample_table(ctx: Arc<CylonContext>) -> CylonResult<Table> {
    // Create sample data with some values to filter
    // Simulates a table with: id, value1, value2 (float)
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("value1", DataType::Int64, false),
        Field::new("value2", DataType::Float64, false),
    ]));

    let ids: Vec<i64> = (0..100).collect();
    let values1: Vec<i64> = (0..100).map(|i| i * 10).collect();
    let values2: Vec<f64> = (0..100).map(|i| (i as f64) * 0.01).collect();

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(ids)),
            Arc::new(Int64Array::from(values1)),
            Arc::new(Float64Array::from(values2)),
        ],
    )?;

    Table::from_record_batch(ctx, batch)
}

fn print_usage() {
    eprintln!("Usage:");
    eprintln!("  ./select_example [csv_file]");
    eprintln!("If no file is provided, uses sample data");
}

fn main() -> CylonResult<()> {
    let args: Vec<String> = env::args().collect();

    let start_time = Instant::now();

    // Initialize context (check if MPI is available)
    #[cfg(feature = "mpi")]
    let ctx = {
        let mut ctx_new = CylonContext::new(true);
        ctx_new.set_communicator(cylon::net::mpi::communicator::MPICommunicator::make()?);
        Arc::new(ctx_new)
    };

    #[cfg(not(feature = "mpi"))]
    let ctx = Arc::new(CylonContext::new(false));

    let rank = ctx.get_rank();

    // Load table
    let table = if args.len() > 1 {
        // Read from CSV file
        Table::from_csv_default(ctx.clone(), &args[1])?
    } else {
        // Use sample data
        if rank == 0 {
            println!("Using sample data (no CSV file provided)");
        }
        create_sample_table(ctx.clone())?
    };

    let read_end_time = Instant::now();
    if rank == 0 {
        println!(
            "Read table in {}ms",
            read_end_time.duration_since(start_time).as_millis()
        );
    }

    // Create a selection mask: select rows where value2 >= 0.3
    // In C++: Select(table, [](cylon::Row row) { return row.GetDouble(8) >= 0.3; }, select);
    // In Rust: we use a boolean mask

    // Get the value2 column (index 2)
    let batch = table.batch(0).unwrap();
    let value2_col = batch
        .column(2)
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("Expected Float64Array for column 2");

    // Create mask: value2 >= 0.3
    let mask_values: Vec<bool> = (0..value2_col.len())
        .map(|i| value2_col.value(i) >= 0.3)
        .collect();
    let mask = BooleanArray::from(mask_values);

    // Perform select
    let select_start_time = Instant::now();
    let selected = table.select(&mask)?;
    let select_end_time = Instant::now();

    println!(
        "Rank {}: Table had: {} rows, Select has: {} rows",
        rank,
        table.rows(),
        selected.rows()
    );

    if rank == 0 {
        println!(
            "Select done in {}ms",
            select_end_time.duration_since(select_start_time).as_millis()
        );
    }

    // Print some results
    if selected.rows() > 0 && selected.rows() <= 20 {
        println!("\nSelected rows (value2 >= 0.3):");
        if let Some(result_batch) = selected.batch(0) {
            let id_col = result_batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            let v1_col = result_batch
                .column(1)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            let v2_col = result_batch
                .column(2)
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap();

            println!("  id | value1 | value2");
            println!("  ---|--------|-------");
            for i in 0..std::cmp::min(10, id_col.len()) {
                println!(
                    "  {:3} | {:6} | {:.2}",
                    id_col.value(i),
                    v1_col.value(i),
                    v2_col.value(i)
                );
            }
            if id_col.len() > 10 {
                println!("  ... ({} more rows)", id_col.len() - 10);
            }
        }
    }

    if rank == 0 {
        let total_time = Instant::now();
        println!(
            "\nOperation took: {}ms",
            total_time.duration_since(start_time).as_millis()
        );
        println!("Select example completed successfully!");
    }

    Ok(())
}
