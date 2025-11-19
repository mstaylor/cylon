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

#[cfg(feature = "mpi")]
mod mpi_tests {
    use std::sync::Arc;
    use arrow::array::{Int32Array, Int64Array, Float64Array, RecordBatch};
    use arrow::datatypes::{DataType, Field, Schema};
    use cylon::ctx::CylonContext;
    use cylon::table::Table;
    use cylon::ops::distributed_sort::{distributed_sort, distributed_sort_multi, SortOptions};
    use cylon::error::CylonResult;

    /// Helper to create CylonContext with MPI
    fn create_mpi_context() -> CylonResult<Arc<CylonContext>> {
        let mut ctx_new = CylonContext::new(true);
        ctx_new.set_communicator(cylon::net::mpi::communicator::MPICommunicator::make()?);
        Ok(Arc::new(ctx_new))
    }

    /// Verify that a table is locally sorted
    fn verify_local_sort_ascending(table: &Table, column_idx: usize) -> bool {
        let batch = match table.batch(0) {
            Some(b) => b,
            None => return true, // Empty table is trivially sorted
        };

        let column = batch.column(column_idx);

        // Check based on data type
        if let Some(int32_arr) = column.as_any().downcast_ref::<Int32Array>() {
            for i in 1..int32_arr.len() {
                if int32_arr.value(i) < int32_arr.value(i - 1) {
                    return false;
                }
            }
            true
        } else if let Some(int64_arr) = column.as_any().downcast_ref::<Int64Array>() {
            for i in 1..int64_arr.len() {
                if int64_arr.value(i) < int64_arr.value(i - 1) {
                    return false;
                }
            }
            true
        } else if let Some(float_arr) = column.as_any().downcast_ref::<Float64Array>() {
            for i in 1..float_arr.len() {
                if float_arr.value(i) < float_arr.value(i - 1) {
                    return false;
                }
            }
            true
        } else {
            panic!("Unsupported type for sort verification");
        }
    }

    #[test]
    fn test_distributed_sort_single_column_int32() -> CylonResult<()> {
        let ctx = create_mpi_context()?;
        let rank = ctx.get_rank();
        let world_size = ctx.get_world_size();

        println!("Rank {}/{} starting distributed sort test (int32)", rank, world_size);

        // Create unsorted data: each rank has 20 rows with random distribution
        // To ensure we have data across different future partitions, use this distribution:
        // Rank 0: values [80, 60, 20, 10, 85, 65, 25, 15, ...]
        // Rank 1: values [90, 70, 30, 5, 95, 75, 35, 12, ...]
        // This ensures overlap and proper shuffle
        let mut values: Vec<i32> = Vec::new();
        for i in 0..20 {
            // Create values that will be distributed across ranks after sort
            let base = (i % 4) * 25;  // 0, 25, 50, 75
            let offset = rank * 5 + (i / 4);
            values.push((base + offset) as i32);
        }

        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Int32, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(values.clone()))],
        )?;

        let table = Table::from_record_batch(ctx.clone(), batch)?;

        println!("Rank {}: Created table with {} rows", rank, table.rows());

        // Perform distributed sort on column 0, ascending
        let sorted = distributed_sort(&table, 0, true)?;

        println!("Rank {}: Sort completed, result has {} rows", rank, sorted.rows());

        // Verify local sorting
        assert!(verify_local_sort_ascending(&sorted, 0),
                "Rank {}: Local partition is not sorted", rank);

        // Verify global correctness by gathering all results to rank 0
        let gathered_tables = ctx.get_communicator().unwrap()
            .gather(&sorted, 0, true, ctx.clone())?;

        if rank == 0 {
            // Rank 0 verifies global sort correctness
            println!("Rank 0: Verifying global sort correctness");

            let mut all_values: Vec<i32> = Vec::new();

            // Collect all values from all ranks
            for table in &gathered_tables {
                if table.rows() > 0 {
                    let batch = table.batch(0).unwrap();
                    let col = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
                    for i in 0..col.len() {
                        all_values.push(col.value(i));
                    }
                }
            }

            // Verify all values are in ascending order
            for i in 1..all_values.len() {
                assert!(all_values[i] >= all_values[i - 1],
                        "Global sort failed: value at position {} ({}) < value at position {} ({})",
                        i, all_values[i], i - 1, all_values[i - 1]);
            }

            // Verify total count
            let total_rows: i64 = gathered_tables.iter().map(|t| t.rows()).sum();
            let expected_total = (world_size * 20) as i64;
            assert_eq!(total_rows, expected_total,
                      "Total row count mismatch: expected {}, got {}", expected_total, total_rows);

            println!("Rank 0: Global verification successful - {} total values sorted correctly",
                     all_values.len());
        }

        println!("Rank {}: Verification successful", rank);
        ctx.barrier()?;
        Ok(())
    }

    #[test]
    fn test_distributed_sort_single_column_descending() -> CylonResult<()> {
        let ctx = create_mpi_context()?;
        let rank = ctx.get_rank();
        let world_size = ctx.get_world_size();

        println!("Rank {}/{} starting distributed sort descending test", rank, world_size);

        // Create unsorted data
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

        // Perform distributed sort descending
        let sorted = distributed_sort(&table, 0, false)?;

        // Verify local descending order
        let result_batch = sorted.batch(0).unwrap();
        let result_col = result_batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();

        for i in 1..result_col.len() {
            assert!(result_col.value(i) <= result_col.value(i - 1),
                    "Rank {}: Local partition not sorted descending at index {}", rank, i);
        }

        // Verify global descending order by gathering to rank 0
        let gathered_tables = ctx.get_communicator().unwrap()
            .gather(&sorted, 0, true, ctx.clone())?;

        if rank == 0 {
            let mut all_values: Vec<i32> = Vec::new();
            for table in &gathered_tables {
                if table.rows() > 0 {
                    let batch = table.batch(0).unwrap();
                    let col = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
                    for i in 0..col.len() {
                        all_values.push(col.value(i));
                    }
                }
            }

            // Verify descending order
            for i in 1..all_values.len() {
                assert!(all_values[i] <= all_values[i - 1],
                        "Global descending sort failed at position {}", i);
            }

            println!("Rank 0: Global descending verification successful");
        }

        println!("Rank {}: Descending sort verification successful", rank);
        ctx.barrier()?;
        Ok(())
    }

    #[test]
    fn test_distributed_sort_multi_column() -> CylonResult<()> {
        let ctx = create_mpi_context()?;
        let rank = ctx.get_rank();
        let world_size = ctx.get_world_size();

        println!("Rank {}/{} starting multi-column distributed sort test", rank, world_size);

        // Create data with two columns: category and value
        // Sort by category (ascending), then by value (descending)
        let mut categories: Vec<i32> = Vec::new();
        let mut values: Vec<i32> = Vec::new();

        for i in 0..20 {
            // Categories: 0, 1, 2, 0, 1, 2, ...
            categories.push((i % 3) as i32);
            // Values: mix to ensure we need secondary sort
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

        // Sort by category (ascending), then value (descending)
        let sorted = distributed_sort_multi(
            &table,
            &[0, 1],           // sort columns
            &[true, false],    // directions: category asc, value desc
            SortOptions::default()
        )?;

        println!("Rank {}: Multi-column sort completed", rank);

        // Verify local multi-column sorting
        let result_batch = sorted.batch(0).unwrap();
        let cat_col = result_batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        let val_col = result_batch.column(1).as_any().downcast_ref::<Int32Array>().unwrap();

        for i in 1..cat_col.len() {
            let prev_cat = cat_col.value(i - 1);
            let curr_cat = cat_col.value(i);
            let prev_val = val_col.value(i - 1);
            let curr_val = val_col.value(i);

            // Category should be ascending
            if curr_cat < prev_cat {
                panic!("Rank {}: Category not ascending at index {}", rank, i);
            }

            // Within same category, value should be descending
            if curr_cat == prev_cat && curr_val > prev_val {
                panic!("Rank {}: Value not descending within category at index {}", rank, i);
            }
        }

        println!("Rank {}: Multi-column verification successful", rank);
        ctx.barrier()?;
        Ok(())
    }

    #[test]
    fn test_distributed_sort_with_float() -> CylonResult<()> {
        let ctx = create_mpi_context()?;
        let rank = ctx.get_rank();
        let world_size = ctx.get_world_size();

        println!("Rank {}/{} starting float distributed sort test", rank, world_size);

        // Create float data
        let mut values: Vec<f64> = Vec::new();
        for i in 0..20 {
            let base = (i as f64) * 3.14159 + (rank as f64) * 2.71828;
            values.push(base % 100.0);  // Keep values in 0-100 range
        }

        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Float64, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Float64Array::from(values))],
        )?;

        let table = Table::from_record_batch(ctx.clone(), batch)?;

        // Sort ascending
        let sorted = distributed_sort(&table, 0, true)?;

        // Verify local sorting
        let result_batch = sorted.batch(0).unwrap();
        let result_col = result_batch.column(0).as_any().downcast_ref::<Float64Array>().unwrap();

        for i in 1..result_col.len() {
            assert!(result_col.value(i) >= result_col.value(i - 1),
                    "Rank {}: Float values not sorted at index {}", rank, i);
        }

        println!("Rank {}: Float sort verification successful", rank);
        ctx.barrier()?;
        Ok(())
    }

    #[test]
    fn test_distributed_sort_empty_on_some_ranks() -> CylonResult<()> {
        let ctx = create_mpi_context()?;
        let rank = ctx.get_rank();
        let world_size = ctx.get_world_size();

        println!("Rank {}/{} starting empty rank test", rank, world_size);

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

        println!("Rank {}: Created table with {} rows", rank, table.rows());

        // Perform distributed sort - should handle empty ranks gracefully
        let sorted = distributed_sort(&table, 0, true)?;

        println!("Rank {}: Sort completed with {} rows", rank, sorted.rows());

        // Verify local sorting (empty tables are trivially sorted)
        if sorted.rows() > 0 {
            assert!(verify_local_sort_ascending(&sorted, 0),
                    "Rank {}: Result not locally sorted", rank);
        }

        println!("Rank {}: Empty rank test successful", rank);
        ctx.barrier()?;
        Ok(())
    }

    #[test]
    fn test_distributed_sort_world_size_one() -> CylonResult<()> {
        let ctx = create_mpi_context()?;
        let world_size = ctx.get_world_size();

        // This test only makes sense with world_size == 1
        // Skip if running with multiple processes
        if world_size > 1 {
            println!("Skipping world_size_one test (world_size = {})", world_size);
            return Ok(());
        }

        println!("Testing distributed sort with world_size = 1");

        let values = vec![30, 10, 20, 40, 5, 25, 15];
        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Int32, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(values))],
        )?;

        let table = Table::from_record_batch(ctx.clone(), batch)?;

        // With world_size = 1, should fall back to local sort
        let sorted = distributed_sort(&table, 0, true)?;

        assert_eq!(sorted.rows(), 7);

        let result_batch = sorted.batch(0).unwrap();
        let result_col = result_batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();

        // Verify sorted order
        let expected = vec![5, 10, 15, 20, 25, 30, 40];
        for i in 0..expected.len() {
            assert_eq!(result_col.value(i), expected[i],
                      "Value at index {} incorrect", i);
        }

        println!("World size 1 test successful");
        Ok(())
    }
}
