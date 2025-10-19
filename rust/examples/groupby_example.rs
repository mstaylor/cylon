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

//! Example demonstrating MapReduce-based groupby operations
//!
//! This example shows how to use the mapred_hash_groupby function to perform
//! aggregations on grouped data. It creates sample sales data and demonstrates
//! various aggregation operations.

use std::sync::Arc;
use cylon::ctx::CylonContext;
use cylon::table::Table;
use cylon::mapreduce::{mapred_hash_groupby, AggregationOpId};
use arrow::array::{Array, Int64Array, Float64Array, StringArray};
use arrow::datatypes::{Schema, Field, DataType};
use arrow::record_batch::RecordBatch;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Cylon MapReduce GroupBy Example ===\n");

    // Create a context (local mode, no MPI)
    let ctx = Arc::new(CylonContext::new(false));

    // Create sample sales data:
    // Product: [A, A, B, B, C, C, A, B]
    // Quantity: [10, 20, 15, 25, 30, 35, 40, 50]
    // Revenue: [100.0, 200.0, 150.0, 250.0, 300.0, 350.0, 400.0, 500.0]
    println!("Creating sample sales data...");
    let products = StringArray::from(vec!["A", "A", "B", "B", "C", "C", "A", "B"]);
    let quantities = Int64Array::from(vec![10, 20, 15, 25, 30, 35, 40, 50]);
    let revenues = Float64Array::from(vec![100.0, 200.0, 150.0, 250.0, 300.0, 350.0, 400.0, 500.0]);

    let schema = Arc::new(Schema::new(vec![
        Field::new("product", DataType::Utf8, false),
        Field::new("quantity", DataType::Int64, false),
        Field::new("revenue", DataType::Float64, false),
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(products), Arc::new(quantities), Arc::new(revenues)],
    )?;

    let table = Table::from_record_batch(ctx.clone(), batch)?;

    println!("Original data:");
    println!("  Product | Quantity | Revenue");
    println!("  --------|----------|--------");
    let batch = table.batch(0).unwrap();
    let prod_arr = batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
    let qty_arr = batch.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
    let rev_arr = batch.column(2).as_any().downcast_ref::<Float64Array>().unwrap();
    for i in 0..prod_arr.len() {
        println!("  {:7} | {:8} | {:7.2}",
            prod_arr.value(i), qty_arr.value(i), rev_arr.value(i));
    }
    println!();

    // Example 1: Sum of quantities by product
    println!("Example 1: Total quantity by product (SUM)");
    let mut output = None;
    mapred_hash_groupby(&table, &[0], &[(1, AggregationOpId::Sum)], &mut output)?;
    print_results(&output.unwrap(), "Product", "Total Quantity");

    // Example 2: Count of sales by product
    println!("Example 2: Number of sales by product (COUNT)");
    let mut output = None;
    mapred_hash_groupby(&table, &[0], &[(1, AggregationOpId::Count)], &mut output)?;
    print_results(&output.unwrap(), "Product", "Sales Count");

    // Example 3: Average quantity by product
    println!("Example 3: Average quantity by product (MEAN)");
    let mut output = None;
    mapred_hash_groupby(&table, &[0], &[(1, AggregationOpId::Mean)], &mut output)?;
    print_results(&output.unwrap(), "Product", "Avg Quantity");

    // Example 4: Min and Max quantity by product
    println!("Example 4: Min quantity by product (MIN)");
    let mut output = None;
    mapred_hash_groupby(&table, &[0], &[(1, AggregationOpId::Min)], &mut output)?;
    print_results(&output.unwrap(), "Product", "Min Quantity");

    println!("Example 5: Max quantity by product (MAX)");
    let mut output = None;
    mapred_hash_groupby(&table, &[0], &[(1, AggregationOpId::Max)], &mut output)?;
    print_results(&output.unwrap(), "Product", "Max Quantity");

    // Example 6: Sum of revenue (Float64) by product
    println!("Example 6: Total revenue by product (SUM on Float64)");
    let mut output = None;
    mapred_hash_groupby(&table, &[0], &[(2, AggregationOpId::Sum)], &mut output)?;
    print_results_f64(&output.unwrap(), "Product", "Total Revenue");

    // Example 7: Mean revenue by product
    println!("Example 7: Average revenue by product (MEAN on Float64)");
    let mut output = None;
    mapred_hash_groupby(&table, &[0], &[(2, AggregationOpId::Mean)], &mut output)?;
    print_results_f64(&output.unwrap(), "Product", "Avg Revenue");

    // Example 8: Variance of revenue by product
    println!("Example 8: Variance of revenue by product (VAR)");
    let mut output = None;
    mapred_hash_groupby(&table, &[0], &[(2, AggregationOpId::Var)], &mut output)?;
    print_results_f64(&output.unwrap(), "Product", "Revenue Variance");

    // Example 9: Standard deviation of revenue by product
    println!("Example 9: Standard deviation of revenue by product (STDDEV)");
    let mut output = None;
    mapred_hash_groupby(&table, &[0], &[(2, AggregationOpId::Stddev)], &mut output)?;
    print_results_f64(&output.unwrap(), "Product", "Revenue StdDev");

    println!("\n=== Example Complete ===");
    Ok(())
}

/// Helper to print results with Int64 aggregation column
fn print_results(table: &Table, col1_name: &str, col2_name: &str) {
    let batch = table.batch(0).unwrap();
    let col1 = batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
    let col2 = batch.column(1).as_any().downcast_ref::<Int64Array>().unwrap();

    println!("  {} | {}", col1_name, col2_name);
    println!("  ---------|---------------");
    for i in 0..col1.len() {
        println!("  {:7} | {}", col1.value(i), col2.value(i));
    }
    println!();
}

/// Helper to print results with Float64 aggregation column
fn print_results_f64(table: &Table, col1_name: &str, col2_name: &str) {
    let batch = table.batch(0).unwrap();
    let col1 = batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
    let col2 = batch.column(1).as_any().downcast_ref::<Float64Array>().unwrap();

    println!("  {} | {}", col1_name, col2_name);
    println!("  ---------|------------------");
    for i in 0..col1.len() {
        println!("  {:7} | {:.2}", col1.value(i), col2.value(i));
    }
    println!();
}
