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

//! Project Example
//!
//! This example demonstrates the project operation to select specific columns.
//! It reads a table and projects (selects) specific columns.
//!
//! Corresponds to cpp/src/examples/project_example.cpp
//!
//! Usage:
//!   cargo run --example project_example -- csv_file.csv
//!   mpirun -n 2 cargo run --example project_example --features mpi -- csv_file.csv

use std::env;
use std::sync::Arc;
use std::time::Instant;

use arrow::array::{Float64Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use cylon::ctx::CylonContext;
use cylon::error::CylonResult;
use cylon::table::Table;

fn create_sample_table(ctx: Arc<CylonContext>) -> CylonResult<Table> {
    // Create sample data with multiple columns
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("value1", DataType::Int64, false),
        Field::new("value2", DataType::Float64, false),
        Field::new("category", DataType::Utf8, false),
    ]));

    let ids: Vec<i64> = (0..20).collect();
    let names: Vec<&str> = vec![
        "Alice", "Bob", "Carol", "David", "Eve",
        "Frank", "Grace", "Henry", "Ivy", "Jack",
        "Kate", "Leo", "Mary", "Nick", "Olivia",
        "Paul", "Quinn", "Rose", "Sam", "Tina",
    ];
    let values1: Vec<i64> = (0..20).map(|i| i * 100).collect();
    let values2: Vec<f64> = (0..20).map(|i| (i as f64) * 1.5).collect();
    let categories: Vec<&str> = (0..20)
        .map(|i| match i % 3 {
            0 => "A",
            1 => "B",
            _ => "C",
        })
        .collect();

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(ids)),
            Arc::new(StringArray::from(names)),
            Arc::new(Int64Array::from(values1)),
            Arc::new(Float64Array::from(values2)),
            Arc::new(StringArray::from(categories)),
        ],
    )?;

    Table::from_record_batch(ctx, batch)
}

fn print_usage() {
    eprintln!("Usage:");
    eprintln!("  ./project_example [csv_file]");
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
        println!(
            "Original table: {} columns, {} rows",
            table.columns(),
            table.rows()
        );
        println!("Columns: {:?}", table.column_names());
    }

    // Project column 0 (id) - like C++ Project(table, {0}, project)
    let project_start_time = Instant::now();
    let projected = table.project(&[0])?;
    let project_end_time = Instant::now();

    if rank == 0 {
        println!(
            "\nProjected to column 0 (id): {} columns, {} rows",
            projected.columns(),
            projected.rows()
        );
        println!(
            "Project done in {}ms",
            project_end_time
                .duration_since(project_start_time)
                .as_millis()
        );
    }

    // Also demonstrate projecting multiple columns
    let projected_multi = table.project(&[0, 1, 3])?; // id, name, value2

    if rank == 0 {
        println!(
            "\nProjected to columns [0, 1, 3]: {} columns",
            projected_multi.columns()
        );
        println!("Columns: {:?}", projected_multi.column_names());

        // Print first few rows
        if let Some(batch) = projected_multi.batch(0) {
            let id_col = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            let name_col = batch
                .column(1)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            let val_col = batch
                .column(2)
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap();

            println!("\n  id | name    | value2");
            println!("  ---|---------|-------");
            for i in 0..std::cmp::min(10, id_col.len()) {
                println!(
                    "  {:3} | {:7} | {:.2}",
                    id_col.value(i),
                    name_col.value(i),
                    val_col.value(i)
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
        println!("Project example completed successfully!");
    }

    Ok(())
}
