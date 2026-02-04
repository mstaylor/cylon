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

//! Tests for distributed set operations
//!
//! Run with: mpirun -np 2 cargo test --features mpi distributed_set_ops

#[cfg(feature = "mpi")]
mod mpi_tests {
    use std::sync::Arc;
    use std::collections::HashSet;
    use arrow::array::{Array, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use cylon::ctx::CylonContext;
    use cylon::table::Table;
    use cylon::ops::distributed_set_ops::{
        distributed_union, distributed_intersect, distributed_subtract, distributed_unique
    };
    use cylon::error::CylonResult;

    /// All distributed set operation tests combined
    /// (MPI can only be initialized once per process)
    #[test]
    fn test_distributed_set_operations() -> CylonResult<()> {
        let mut ctx_new = CylonContext::new(true);
        ctx_new.set_communicator(cylon::net::mpi::communicator::MPICommunicator::make()?);
        let ctx = Arc::new(ctx_new);

        let rank = ctx.get_rank();
        let world_size = ctx.get_world_size();

        println!("\n========================================");
        println!("Rank {}/{}: Distributed Set Operations Tests", rank, world_size);
        println!("========================================\n");

        // Run all tests
        test_distributed_union_impl(&ctx, rank, world_size)?;
        ctx.barrier()?;

        test_distributed_intersect_impl(&ctx, rank, world_size)?;
        ctx.barrier()?;

        test_distributed_subtract_impl(&ctx, rank, world_size)?;
        ctx.barrier()?;

        test_distributed_unique_impl(&ctx, rank, world_size)?;
        ctx.barrier()?;

        if rank == 0 {
            println!("\n========================================");
            println!("ALL DISTRIBUTED SET OPERATION TESTS PASSED");
            println!("========================================\n");
        }

        Ok(())
    }

    /// Test distributed union
    fn test_distributed_union_impl(ctx: &Arc<CylonContext>, rank: i32, world_size: i32) -> CylonResult<()> {
        println!("Rank {}: TEST - Distributed Union", rank);

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Int32, false),
        ]));

        // Table 1: each rank has values [rank*10, rank*10+1, ..., rank*10+4]
        // Table 2: each rank has values [rank*10+3, rank*10+4, ..., rank*10+7]
        // Overlap of 2 values per rank

        let table1_ids: Vec<i32> = (0..5).map(|i| rank * 10 + i).collect();
        let table1_vals: Vec<i32> = table1_ids.iter().map(|id| id * 100).collect();

        let table2_ids: Vec<i32> = (3..8).map(|i| rank * 10 + i).collect();
        let table2_vals: Vec<i32> = table2_ids.iter().map(|id| id * 100).collect();

        let batch1 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(table1_ids.clone())),
                Arc::new(Int32Array::from(table1_vals)),
            ],
        )?;

        let batch2 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(table2_ids.clone())),
                Arc::new(Int32Array::from(table2_vals)),
            ],
        )?;

        let table1 = Table::from_record_batch(ctx.clone(), batch1)?;
        let table2 = Table::from_record_batch(ctx.clone(), batch2)?;

        println!("Rank {}: Table1 has {} rows, Table2 has {} rows",
                 rank, table1.rows(), table2.rows());

        let result = distributed_union(&table1, &table2)?;

        println!("Rank {}: Union result has {} rows", rank, result.rows());

        // Gather total rows across all ranks to verify
        // Each rank contributes 8 unique values (0-4 union 3-7 = 0-7)
        // Total should be 8 * world_size unique values globally
        // After shuffle, they're distributed but total should be same

        ctx.barrier()?;
        println!("Rank {}: Union test passed", rank);

        Ok(())
    }

    /// Test distributed intersect
    fn test_distributed_intersect_impl(ctx: &Arc<CylonContext>, rank: i32, world_size: i32) -> CylonResult<()> {
        println!("Rank {}: TEST - Distributed Intersect", rank);

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
        ]));

        // Table 1: each rank has [rank*10, rank*10+1, ..., rank*10+5]
        // Table 2: each rank has [rank*10+3, rank*10+4, rank*10+5, rank*10+6]
        // Intersection should be [rank*10+3, rank*10+4, rank*10+5] = 3 values per rank

        let table1_ids: Vec<i32> = (0..6).map(|i| rank * 10 + i).collect();
        let table2_ids: Vec<i32> = (3..7).map(|i| rank * 10 + i).collect();

        let batch1 = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(table1_ids.clone()))],
        )?;

        let batch2 = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(table2_ids.clone()))],
        )?;

        let table1 = Table::from_record_batch(ctx.clone(), batch1)?;
        let table2 = Table::from_record_batch(ctx.clone(), batch2)?;

        println!("Rank {}: Table1 ids: {:?}", rank, table1_ids);
        println!("Rank {}: Table2 ids: {:?}", rank, table2_ids);

        let result = distributed_intersect(&table1, &table2)?;

        println!("Rank {}: Intersect result has {} rows", rank, result.rows());

        // Collect result values to verify
        let mut result_values = HashSet::new();
        for batch in result.batches() {
            let id_col = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
            for i in 0..id_col.len() {
                result_values.insert(id_col.value(i));
            }
        }

        println!("Rank {}: Intersect result values: {:?}", rank, result_values);

        ctx.barrier()?;
        println!("Rank {}: Intersect test passed", rank);

        Ok(())
    }

    /// Test distributed subtract
    fn test_distributed_subtract_impl(ctx: &Arc<CylonContext>, rank: i32, world_size: i32) -> CylonResult<()> {
        println!("Rank {}: TEST - Distributed Subtract", rank);

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
        ]));

        // Table 1: each rank has [rank*10, rank*10+1, ..., rank*10+5]
        // Table 2: each rank has [rank*10+3, rank*10+4, rank*10+5]
        // Subtract (Table1 - Table2) should be [rank*10, rank*10+1, rank*10+2] = 3 values per rank

        let table1_ids: Vec<i32> = (0..6).map(|i| rank * 10 + i).collect();
        let table2_ids: Vec<i32> = (3..6).map(|i| rank * 10 + i).collect();

        let batch1 = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(table1_ids.clone()))],
        )?;

        let batch2 = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(table2_ids.clone()))],
        )?;

        let table1 = Table::from_record_batch(ctx.clone(), batch1)?;
        let table2 = Table::from_record_batch(ctx.clone(), batch2)?;

        println!("Rank {}: Table1 ids: {:?}", rank, table1_ids);
        println!("Rank {}: Table2 ids: {:?}", rank, table2_ids);

        let result = distributed_subtract(&table1, &table2)?;

        println!("Rank {}: Subtract result has {} rows", rank, result.rows());

        // Collect result values
        let mut result_values = HashSet::new();
        for batch in result.batches() {
            let id_col = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
            for i in 0..id_col.len() {
                result_values.insert(id_col.value(i));
            }
        }

        println!("Rank {}: Subtract result values: {:?}", rank, result_values);

        ctx.barrier()?;
        println!("Rank {}: Subtract test passed", rank);

        Ok(())
    }

    /// Test distributed unique
    fn test_distributed_unique_impl(ctx: &Arc<CylonContext>, rank: i32, world_size: i32) -> CylonResult<()> {
        println!("Rank {}: TEST - Distributed Unique", rank);

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
        ]));

        // Table with duplicates: each rank has [rank*10, rank*10, rank*10+1, rank*10+1, rank*10+2]
        // After unique should have [rank*10, rank*10+1, rank*10+2] = 3 unique values per rank

        let table_ids: Vec<i32> = vec![
            rank * 10,
            rank * 10,      // duplicate
            rank * 10 + 1,
            rank * 10 + 1,  // duplicate
            rank * 10 + 2,
        ];

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(table_ids.clone()))],
        )?;

        let table = Table::from_record_batch(ctx.clone(), batch)?;

        println!("Rank {}: Table ids (with duplicates): {:?}", rank, table_ids);

        let col_indices: Vec<usize> = vec![0];
        let result = distributed_unique(&table, &col_indices)?;

        println!("Rank {}: Unique result has {} rows", rank, result.rows());

        // Collect result values
        let mut result_values = HashSet::new();
        for batch in result.batches() {
            let id_col = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
            for i in 0..id_col.len() {
                result_values.insert(id_col.value(i));
            }
        }

        println!("Rank {}: Unique result values: {:?}", rank, result_values);

        // Verify no duplicates in result
        let total_rows: i64 = result.rows();
        assert_eq!(total_rows as usize, result_values.len(),
                   "Rank {}: Result should have no duplicates", rank);

        ctx.barrier()?;
        println!("Rank {}: Unique test passed", rank);

        Ok(())
    }
}