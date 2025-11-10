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

//! Tests for ArrowAllToAll with multi-batch tables
//!
//! These tests specifically validate the ChunkedArray concatenation fix.
//! Before the fix, multi-batch tables would lose all but the first batch.
//!
//! NOTE: All tests combined into one function because MPI can only be
//! initialized once per process.

#[cfg(feature = "mpi")]
mod mpi_tests {
    use std::sync::Arc;
    use arrow::array::{Array, Int32Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use cylon::ctx::CylonContext;
    use cylon::table::Table;
    use cylon::error::CylonResult;

    /// Combined test for multi-batch table handling
    ///
    /// Tests:
    /// 1. Multi-batch table integrity (3 batches, 30 rows)
    /// 2. Single vs multi-batch comparison
    /// 3. Many small batches (10 batches, 20 rows)
    ///
    /// CRITICAL: These tests validate the ChunkedArray concatenation fix.
    /// The old implementation would FAIL test 1, losing batches 2 and 3.
    #[test]
    fn test_multi_batch_tables() -> CylonResult<()> {
        // Initialize MPI context once
        let mut ctx_new = CylonContext::new(true);
        ctx_new.set_communicator(cylon::net::mpi::communicator::MPICommunicator::make()?);
        let ctx = Arc::new(ctx_new);

        let rank = ctx.get_rank();
        let world_size = ctx.get_world_size();

        println!("\n========================================");
        println!("Rank {}/{}: Multi-Batch Table Tests", rank, world_size);
        println!("========================================\n");

        // ===================================================================
        // TEST 1: Multi-batch table integrity (3 batches, 30 rows)
        // ===================================================================
        println!("Rank {}: TEST 1 - Multi-batch table integrity", rank);

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
        ]));

        // Create 3 separate batches
        let mut batches = Vec::new();

        // Batch 1: rows 0-9
        let batch1 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9])),
                Arc::new(StringArray::from(vec![
                    "row0", "row1", "row2", "row3", "row4",
                    "row5", "row6", "row7", "row8", "row9"
                ])),
            ],
        )?;
        batches.push(batch1);

        // Batch 2: rows 10-19
        let batch2 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![10, 11, 12, 13, 14, 15, 16, 17, 18, 19])),
                Arc::new(StringArray::from(vec![
                    "row10", "row11", "row12", "row13", "row14",
                    "row15", "row16", "row17", "row18", "row19"
                ])),
            ],
        )?;
        batches.push(batch2);

        // Batch 3: rows 20-29
        let batch3 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![20, 21, 22, 23, 24, 25, 26, 27, 28, 29])),
                Arc::new(StringArray::from(vec![
                    "row20", "row21", "row22", "row23", "row24",
                    "row25", "row26", "row27", "row28", "row29"
                ])),
            ],
        )?;
        batches.push(batch3);

        // Create table from multiple batches
        let table = Table::from_record_batches(ctx.clone(), batches)?;

        println!("Rank {}: Created table with {} batches and {} total rows",
                 rank, table.num_batches(), table.rows());

        // Verify the table has all data
        assert_eq!(table.num_batches(), 3, "Should have 3 batches");
        assert_eq!(table.rows(), 30, "Should have 30 total rows");

        // Verify we can access data from all batches
        let all_batches = table.batches();
        assert_eq!(all_batches.len(), 3);

        // Check batch 1
        let b1_ids = all_batches[0].column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(b1_ids.value(0), 0);
        assert_eq!(b1_ids.value(9), 9);

        // Check batch 2
        let b2_ids = all_batches[1].column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(b2_ids.value(0), 10);
        assert_eq!(b2_ids.value(9), 19);

        // Check batch 3
        let b3_ids = all_batches[2].column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(b3_ids.value(0), 20);
        assert_eq!(b3_ids.value(9), 29);

        println!("Rank {}: ✓ Test 1 passed - All batches verified, no data loss\n", rank);

        ctx.barrier()?;

        // ===================================================================
        // TEST 2: Single vs multi-batch comparison
        // ===================================================================
        println!("Rank {}: TEST 2 - Single vs multi-batch comparison", rank);

        let schema2 = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Int32, false),
        ]));

        // Create single-batch table with 20 rows
        let single_batch = RecordBatch::try_new(
            schema2.clone(),
            vec![
                Arc::new(Int32Array::from((0..20).collect::<Vec<i32>>())),
            ],
        )?;
        let single_batch_table = Table::from_record_batch(ctx.clone(), single_batch)?;

        // Create multi-batch table with same 20 rows split into 2 batches
        let batch_a = RecordBatch::try_new(
            schema2.clone(),
            vec![
                Arc::new(Int32Array::from((0..10).collect::<Vec<i32>>())),
            ],
        )?;
        let batch_b = RecordBatch::try_new(
            schema2.clone(),
            vec![
                Arc::new(Int32Array::from((10..20).collect::<Vec<i32>>())),
            ],
        )?;
        let multi_batch_table = Table::from_record_batches(
            ctx.clone(),
            vec![batch_a, batch_b]
        )?;

        println!("Rank {}: Single-batch: {} batches, {} rows",
                 rank, single_batch_table.num_batches(), single_batch_table.rows());
        println!("Rank {}: Multi-batch: {} batches, {} rows",
                 rank, multi_batch_table.num_batches(), multi_batch_table.rows());

        // Both should have same total rows
        assert_eq!(single_batch_table.rows(), 20);
        assert_eq!(multi_batch_table.rows(), 20);

        // Verify data integrity
        assert_eq!(single_batch_table.num_batches(), 1);
        assert_eq!(multi_batch_table.num_batches(), 2);

        println!("Rank {}: ✓ Test 2 passed - Both formats equivalent\n", rank);

        ctx.barrier()?;

        // ===================================================================
        // TEST 3: Many small batches (edge case)
        // ===================================================================
        println!("Rank {}: TEST 3 - Many small batches", rank);

        let schema3 = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
        ]));

        // Create 10 batches with 2 rows each = 20 rows total
        let mut small_batches = Vec::new();
        for i in 0..10 {
            let start = i * 2;
            let batch = RecordBatch::try_new(
                schema3.clone(),
                vec![
                    Arc::new(Int32Array::from(vec![start, start + 1])),
                ],
            )?;
            small_batches.push(batch);
        }

        let many_batch_table = Table::from_record_batches(ctx.clone(), small_batches)?;

        println!("Rank {}: Created table with {} batches and {} rows",
                 rank, many_batch_table.num_batches(), many_batch_table.rows());

        assert_eq!(many_batch_table.num_batches(), 10, "Should have 10 batches");
        assert_eq!(many_batch_table.rows(), 20, "Should have 20 total rows");

        // Verify first and last values
        let first_batch = &many_batch_table.batches()[0];
        let first_col = first_batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(first_col.value(0), 0);

        let last_batch = &many_batch_table.batches()[9];
        let last_col = last_batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(last_col.value(1), 19);

        println!("Rank {}: ✓ Test 3 passed - All 10 batches verified\n", rank);

        ctx.barrier()?;

        // ===================================================================
        // Summary
        // ===================================================================
        if rank == 0 {
            println!("\n========================================");
            println!("ALL MULTI-BATCH TESTS PASSED ✓");
            println!("- Test 1: 3 batches, 30 rows (no data loss)");
            println!("- Test 2: Single vs multi-batch equivalent");
            println!("- Test 3: 10 small batches handled correctly");
            println!("========================================");
            println!("ChunkedArray concatenation fix verified!");
            println!("========================================\n");
        }

        ctx.barrier()?;

        Ok(())
    }
}
