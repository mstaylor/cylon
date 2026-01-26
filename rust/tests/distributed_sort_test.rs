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

//! Tests for distributed sort operation
//! Corresponds to C++ DistributedSort tests
//!
//! NOTE: All tests combined into one function because MPI can only be
//! initialized once per process.

#[cfg(feature = "mpi")]
mod mpi_tests {
    use std::sync::Arc;
    use arrow::array::{Int32Array, Float64Array, RecordBatch};
    use arrow::datatypes::{DataType, Field, Schema};
    use cylon::ctx::CylonContext;
    use cylon::table::Table;
    use cylon::ops::distributed_sort::{distributed_sort, distributed_sort_multi, SortOptions};
    use cylon::error::CylonResult;

    /// Verify that a table is locally sorted ascending
    fn verify_local_sort_ascending_i32(table: &Table, column_idx: usize) -> bool {
        if table.rows() == 0 {
            return true;
        }
        let batch = match table.batch(0) {
            Some(b) => b,
            None => return true,
        };
        let column = batch.column(column_idx);
        let int32_arr = column.as_any().downcast_ref::<Int32Array>().unwrap();
        for i in 1..int32_arr.len() {
            if int32_arr.value(i) < int32_arr.value(i - 1) {
                return false;
            }
        }
        true
    }

    /// All distributed sort tests combined
    #[test]
    fn test_distributed_sort_all() -> CylonResult<()> {
        let mut ctx_new = CylonContext::new(true);
        ctx_new.set_communicator(cylon::net::mpi::communicator::MPICommunicator::make()?);
        let ctx = Arc::new(ctx_new);

        let rank = ctx.get_rank();
        let world_size = ctx.get_world_size();

        println!("\n========================================");
        println!("Rank {}/{}: Distributed Sort Tests", rank, world_size);
        println!("========================================\n");

        // Test 1: Single column int32 ascending
        test_single_column_int32(&ctx, rank, world_size)?;
        ctx.barrier()?;

        // Test 2: Single column descending
        test_single_column_descending(&ctx, rank, world_size)?;
        ctx.barrier()?;

        // Test 3: Multi-column sort
        test_multi_column(&ctx, rank, world_size)?;
        ctx.barrier()?;

        // Test 4: Float sort
        test_float_sort(&ctx, rank, world_size)?;
        ctx.barrier()?;

        // Test 5: Empty on some ranks
        test_empty_ranks(&ctx, rank, world_size)?;
        ctx.barrier()?;

        if rank == 0 {
            println!("\n========================================");
            println!("ALL DISTRIBUTED SORT TESTS PASSED");
            println!("========================================\n");
        }

        Ok(())
    }

    fn test_single_column_int32(ctx: &Arc<CylonContext>, rank: i32, world_size: i32) -> CylonResult<()> {
        println!("Rank {}: TEST 1 - Single column int32 ascending", rank);

        let mut values: Vec<i32> = Vec::new();
        for i in 0..20 {
            let base = (i % 4) * 25;
            let offset = rank * 5 + (i / 4);
            values.push((base + offset) as i32);
        }

        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Int32, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(values))],
        )?;

        let table = Table::from_record_batch(ctx.clone(), batch)?;
        let sorted = distributed_sort(&table, 0, true)?;

        assert!(verify_local_sort_ascending_i32(&sorted, 0),
                "Rank {}: Local partition not sorted", rank);

        // Verify global correctness
        let gathered = ctx.get_communicator().unwrap()
            .gather(&sorted, 0, true, ctx.clone())?;

        if rank == 0 {
            let mut all_values: Vec<i32> = Vec::new();
            for t in &gathered {
                if t.rows() > 0 {
                    let b = t.batch(0).unwrap();
                    let col = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
                    for i in 0..col.len() {
                        all_values.push(col.value(i));
                    }
                }
            }
            for i in 1..all_values.len() {
                assert!(all_values[i] >= all_values[i - 1], "Global sort failed at {}", i);
            }
            println!("Rank 0: Global verification passed ({} values)", all_values.len());
        }

        println!("Rank {}: Test 1 passed", rank);
        Ok(())
    }

    fn test_single_column_descending(ctx: &Arc<CylonContext>, rank: i32, world_size: i32) -> CylonResult<()> {
        println!("Rank {}: TEST 2 - Single column descending", rank);

        let mut values: Vec<i32> = Vec::new();
        for i in 0..20 {
            let base = (i % 4) * 25;
            let offset = rank * 5 + (i / 4);
            values.push((base + offset) as i32);
        }

        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Int32, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(values))],
        )?;

        let table = Table::from_record_batch(ctx.clone(), batch)?;
        let sorted = distributed_sort(&table, 0, false)?;

        // Verify descending
        if sorted.rows() > 0 {
            let result_batch = sorted.batch(0).unwrap();
            let col = result_batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
            for i in 1..col.len() {
                assert!(col.value(i) <= col.value(i - 1),
                        "Rank {}: Not descending at {}", rank, i);
            }
        }

        println!("Rank {}: Test 2 passed", rank);
        Ok(())
    }

    fn test_multi_column(ctx: &Arc<CylonContext>, rank: i32, _world_size: i32) -> CylonResult<()> {
        println!("Rank {}: TEST 3 - Multi-column sort", rank);

        let mut categories: Vec<i32> = Vec::new();
        let mut values: Vec<i32> = Vec::new();

        for i in 0..20 {
            categories.push((i % 3) as i32);
            values.push(((rank * 20 + i) * 7 % 100) as i32);
        }

        let schema = Arc::new(Schema::new(vec![
            Field::new("category", DataType::Int32, false),
            Field::new("value", DataType::Int32, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(categories)),
                Arc::new(Int32Array::from(values)),
            ],
        )?;

        let table = Table::from_record_batch(ctx.clone(), batch)?;

        let sorted = distributed_sort_multi(
            &table,
            &[0, 1],
            &[true, false],  // category asc, value desc
            SortOptions::default()
        )?;

        // Verify multi-column sort
        if sorted.rows() > 0 {
            let result_batch = sorted.batch(0).unwrap();
            let cat_col = result_batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
            let val_col = result_batch.column(1).as_any().downcast_ref::<Int32Array>().unwrap();

            for i in 1..cat_col.len() {
                let prev_cat = cat_col.value(i - 1);
                let curr_cat = cat_col.value(i);
                let prev_val = val_col.value(i - 1);
                let curr_val = val_col.value(i);

                assert!(curr_cat >= prev_cat, "Rank {}: Category not ascending at {}", rank, i);
                if curr_cat == prev_cat {
                    assert!(curr_val <= prev_val, "Rank {}: Value not descending at {}", rank, i);
                }
            }
        }

        println!("Rank {}: Test 3 passed", rank);
        Ok(())
    }

    fn test_float_sort(ctx: &Arc<CylonContext>, rank: i32, _world_size: i32) -> CylonResult<()> {
        println!("Rank {}: TEST 4 - Float sort", rank);

        let mut values: Vec<f64> = Vec::new();
        for i in 0..20 {
            let base = (i as f64) * 3.14159 + (rank as f64) * 2.71828;
            values.push(base % 100.0);
        }

        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Float64, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Float64Array::from(values))],
        )?;

        let table = Table::from_record_batch(ctx.clone(), batch)?;
        let sorted = distributed_sort(&table, 0, true)?;

        // Verify ascending
        if sorted.rows() > 0 {
            let result_batch = sorted.batch(0).unwrap();
            let col = result_batch.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
            for i in 1..col.len() {
                assert!(col.value(i) >= col.value(i - 1),
                        "Rank {}: Float not sorted at {}", rank, i);
            }
        }

        println!("Rank {}: Test 4 passed", rank);
        Ok(())
    }

    fn test_empty_ranks(ctx: &Arc<CylonContext>, rank: i32, _world_size: i32) -> CylonResult<()> {
        println!("Rank {}: TEST 5 - Empty on some ranks", rank);

        // Only rank 0 has data
        let values = if rank == 0 {
            vec![30, 10, 20, 40, 5]
        } else {
            vec![]
        };

        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Int32, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(values))],
        )?;

        let table = Table::from_record_batch(ctx.clone(), batch)?;
        let sorted = distributed_sort(&table, 0, true)?;

        // Verify local sorting
        if sorted.rows() > 0 {
            assert!(verify_local_sort_ascending_i32(&sorted, 0),
                    "Rank {}: Result not locally sorted", rank);
        }

        println!("Rank {}: Test 5 passed", rank);
        Ok(())
    }
}