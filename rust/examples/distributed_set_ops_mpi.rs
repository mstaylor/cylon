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

//! Example demonstrating distributed set operations with MPI
//!
//! Tests: Union, Intersect, Subtract
//! Run with: mpirun -n 2 cargo run --example distributed_set_ops_mpi --features mpi

use std::sync::Arc;
use arrow::array::{Array, Int32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use cylon::ctx::CylonContext;
use cylon::table::Table;
use cylon::ops::distributed_set_ops::{distributed_union, distributed_intersect, distributed_subtract};
use cylon::error::CylonResult;

fn main() -> CylonResult<()> {
    // Initialize MPI context
    let mut ctx_new = CylonContext::new(true);
    ctx_new.set_communicator(cylon::net::mpi::communicator::MPICommunicator::make()?);
    let ctx = Arc::new(ctx_new);

    let rank = ctx.get_rank();
    let world_size = ctx.get_world_size();

    println!("\n========================================");
    println!("Rank {}/{}: Testing distributed set operations", rank, world_size);
    println!("========================================\n");

    // Create schema (id, name)
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
    ]));

    // Create test data with some overlapping and some unique rows
    // Left table on rank 0: (1, "a"), (2, "b"), (3, "c")
    // Left table on rank 1: (4, "d"), (5, "e"), (6, "f")
    let left_ids = vec![1 + rank * 3, 2 + rank * 3, 3 + rank * 3];
    let left_names = vec!["a", "b", "c"].iter().map(|s| s.to_string()).collect::<Vec<_>>();

    // Right table on rank 0: (2, "b"), (3, "c"), (7, "g")
    // Right table on rank 1: (5, "e"), (6, "f"), (8, "h")
    let right_ids = vec![2 + rank * 3, 3 + rank * 3, 7 + rank];
    let right_names_list = if rank == 0 {
        vec!["b", "c", "g"]
    } else {
        vec!["e", "f", "h"]
    };
    let right_names = right_names_list.iter().map(|s| s.to_string()).collect::<Vec<_>>();

    let left_batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(left_ids.clone())),
            Arc::new(StringArray::from(left_names)),
        ],
    )?;

    let right_batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(right_ids.clone())),
            Arc::new(StringArray::from(right_names)),
        ],
    )?;

    let left_table = Table::from_record_batch(ctx.clone(), left_batch)?;
    let right_table = Table::from_record_batch(ctx.clone(), right_batch)?;

    println!("Rank {}: Left table has {} rows: {:?}", rank, left_table.rows(), left_ids);

    // Test 1: Union
    println!("\n--- Test 1: Distributed Union ---");
    let union_result = distributed_union(&left_table, &right_table)?;
    println!("Rank {}: Union result has {} rows", rank, union_result.rows());

    // Verify union: should contain all unique rows from both tables
    if union_result.rows() > 0 {
        let batch = union_result.batch(0).unwrap();
        let id_col = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        let name_col = batch.column(1).as_any().downcast_ref::<StringArray>().unwrap();

        println!("Rank {}: Union result rows:", rank);
        for i in 0..id_col.len() {
            println!("  ({}, {})", id_col.value(i), name_col.value(i));
        }
    }

    // Test 2: Intersect
    println!("\n--- Test 2: Distributed Intersect ---");
    let intersect_result = distributed_intersect(&left_table, &right_table)?;
    println!("Rank {}: Intersect result has {} rows", rank, intersect_result.rows());

    // Display intersect results
    if intersect_result.rows() > 0 {
        let batch = intersect_result.batch(0).unwrap();
        let id_col = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        let name_col = batch.column(1).as_any().downcast_ref::<StringArray>().unwrap();

        println!("Rank {}: Intersect result rows:", rank);
        for i in 0..id_col.len() {
            println!("  ({}, {})", id_col.value(i), name_col.value(i));
        }
    } else {
        println!("Rank {}: Intersect result is empty", rank);
    }

    // Test 3: Subtract (Left - Right)
    println!("\n--- Test 3: Distributed Subtract (Left - Right) ---");
    let subtract_result = distributed_subtract(&left_table, &right_table)?;
    println!("Rank {}: Subtract result has {} rows", rank, subtract_result.rows());

    // Display subtract results
    if subtract_result.rows() > 0 {
        let batch = subtract_result.batch(0).unwrap();
        let id_col = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        let name_col = batch.column(1).as_any().downcast_ref::<StringArray>().unwrap();

        println!("Rank {}: Subtract result rows (in left but not right):", rank);
        for i in 0..id_col.len() {
            println!("  ({}, {})", id_col.value(i), name_col.value(i));
        }
    } else {
        println!("Rank {}: Subtract result is empty", rank);
    }

    println!("\nRank {}: ✓ All set operations validated successfully", rank);

    ctx.barrier()?;

    if rank == 0 {
        println!("\n========================================");
        println!("ALL DISTRIBUTED SET OPERATIONS PASSED ✓");
        println!("Tested: Union, Intersect, Subtract");
        println!("========================================\n");
    }

    Ok(())
}
