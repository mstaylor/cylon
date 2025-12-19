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

//! Table From Vectors Example
//!
//! This example demonstrates creating a table from Rust vectors/arrays.
//!
//! Corresponds to cpp/src/examples/table_from_vectors_example.cpp
//!
//! Usage:
//!   cargo run --example table_from_vectors_example

use std::sync::Arc;

use arrow::array::{Float64Array, Int32Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use cylon::ctx::CylonContext;
use cylon::error::CylonResult;
use cylon::table::Table;

fn main() -> CylonResult<()> {
    println!("=== Table From Vectors Example ===\n");

    let size = 10;

    // Initialize local context
    let ctx = Arc::new(CylonContext::new(false));

    // Create vectors similar to C++ example
    let col0: Vec<i32> = (0..size).collect();
    let col1: Vec<f64> = (0..size).map(|i| i as f64 + 10.0).collect();

    // Create Arrow arrays from vectors
    let col0_array = Int32Array::from(col0);
    let col1_array = Float64Array::from(col1);

    // Create schema
    let schema = Arc::new(Schema::new(vec![
        Field::new("col0", DataType::Int32, false),
        Field::new("col1", DataType::Float64, false),
    ]));

    // Create RecordBatch
    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(col0_array), Arc::new(col1_array)],
    )?;

    // Create Table from RecordBatch
    let table = Table::from_record_batch(ctx.clone(), batch)?;

    println!("Created table. Row Count: {}", table.rows());

    if table.columns() == 2 && table.rows() == size as i64 {
        // Print the table
        println!("\nTable contents:");
        println!("  col0 | col1");
        println!("  -----|------");

        if let Some(batch) = table.batch(0) {
            let c0 = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            let c1 = batch
                .column(1)
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap();

            for i in 0..c0.len() {
                println!("  {:4} | {:5.1}", c0.value(i), c1.value(i));
            }
        }

        // Print just the float column values
        println!("\nFloat column values:");
        if let Some(batch) = table.batch(0) {
            let c1 = batch
                .column(1)
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap();
            let values: Vec<String> = (0..c1.len()).map(|i| format!("{:.1}", c1.value(i))).collect();
            println!("  {}", values.join(" "));
        }
    }

    println!("\n=== Example Complete ===");
    Ok(())
}
