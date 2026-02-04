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

//! Compute Example
//!
//! This example demonstrates compute/aggregation operations on tables.
//!
//! Corresponds to cpp/src/examples/compute_example.cpp
//!
//! Usage:
//!   mpirun -n 2 cargo run --example compute_example --features mpi

use std::sync::Arc;

use arrow::array::{Array, Float64Array, Int32Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use cylon::ctx::CylonContext;
use cylon::error::CylonResult;
use cylon::table::Table;
use cylon::compute::{sum_array, count_array, min_array, max_array, mean_array, AggregateOptions, ScalarValue};

fn create_table(ctx: Arc<CylonContext>, rows: usize) -> CylonResult<Table> {
    let rank = ctx.get_rank() + 1;

    let col0: Vec<i32> = (0..rows).map(|i| (i as i32) * rank).collect();
    let col1: Vec<f64> = (0..rows).map(|i| (i as f64) * (rank as f64) + 10.0).collect();

    let schema = Arc::new(Schema::new(vec![
        Field::new("col0", DataType::Int32, false),
        Field::new("col1", DataType::Float64, false),
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(col0)),
            Arc::new(Float64Array::from(col1)),
        ],
    )?;

    Table::from_record_batch(ctx, batch)
}

fn main() -> CylonResult<()> {
    println!("=== Compute Example ===\n");

    let rows = 4;
    let agg_index = 1; // Aggregate on column 1 (Float64)

    // Initialize MPI context
    #[cfg(feature = "mpi")]
    let ctx = {
        let mut ctx_new = CylonContext::new(true);
        ctx_new.set_communicator(cylon::net::mpi::communicator::MPICommunicator::make()?);
        Arc::new(ctx_new)
    };

    #[cfg(not(feature = "mpi"))]
    let ctx = Arc::new(CylonContext::new(false));

    let rank = ctx.get_rank();

    let table = create_table(ctx.clone(), rows)?;

    if table.columns() == 2 && table.rows() == rows as i64 {
        println!("Rank {}: Table created successfully", rank);

        // Print table
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

            println!("\nRank {} table:", rank);
            println!("  col0 | col1");
            println!("  -----|------");
            for i in 0..c0.len() {
                println!("  {:4} | {:6.1}", c0.value(i), c1.value(i));
            }
        }
    } else {
        println!("Rank {}: Table creation failed!", rank);
        return Ok(());
    }

    // Get the column to aggregate
    if let Some(batch) = table.batch(0) {
        let col = batch.column(agg_index);
        let opts = AggregateOptions::default();

        // Sum
        let sum_result = sum_array(col.as_ref(), &opts)?;
        println!("\nRank {}: sum = {:?}", rank, sum_result);

        // Count
        let count_result = count_array(col.as_ref(), &opts)?;
        println!("Rank {}: count = {:?}", rank, count_result);

        // Min
        let min_result = min_array(col.as_ref(), &opts)?;
        println!("Rank {}: min = {:?}", rank, min_result);

        // Max
        let max_result = max_array(col.as_ref(), &opts)?;
        println!("Rank {}: max = {:?}", rank, max_result);

        // Mean
        let mean_result = mean_array(col.as_ref(), &opts)?;
        println!("Rank {}: mean = {:?}", rank, mean_result);
    }

    // Demonstrate table-level aggregation using compute functions
    println!("\n--- Table-level aggregations ---");

    if let Some(batch) = table.batch(0) {
        let col = batch.column(agg_index);
        let opts = AggregateOptions::default();

        // Sum
        let sum_val = sum_array(col.as_ref(), &opts)?;
        match sum_val {
            ScalarValue::Float64(v) => println!("Rank {}: Table sum = {:.1}", rank, v),
            ScalarValue::Int64(v) => println!("Rank {}: Table sum = {}", rank, v),
            _ => println!("Rank {}: Table sum = {:?}", rank, sum_val),
        }

        // Count
        let count_val = count_array(col.as_ref(), &opts)?;
        match count_val {
            ScalarValue::Int64(v) => println!("Rank {}: Table count = {}", rank, v),
            _ => println!("Rank {}: Table count = {:?}", rank, count_val),
        }

        // Max
        let max_val = max_array(col.as_ref(), &opts)?;
        match max_val {
            ScalarValue::Float64(v) => println!("Rank {}: Table max = {:.1}", rank, v),
            ScalarValue::Int64(v) => println!("Rank {}: Table max = {}", rank, v),
            _ => println!("Rank {}: Table max = {:?}", rank, max_val),
        }

        // Min
        let min_val = min_array(col.as_ref(), &opts)?;
        match min_val {
            ScalarValue::Float64(v) => println!("Rank {}: Table min = {:.1}", rank, v),
            ScalarValue::Int64(v) => println!("Rank {}: Table min = {}", rank, v),
            _ => println!("Rank {}: Table min = {:?}", rank, min_val),
        }
    }

    println!("\n=== Compute Example Complete ===");
    Ok(())
}
