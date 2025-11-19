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

//! Tests for gather and broadcast operations
//! Corresponds to C++ gather/bcast tests

#[cfg(feature = "mpi")]
mod mpi_tests {
    use std::sync::Arc;
    use arrow::array::{Int32Array, RecordBatch};
    use arrow::datatypes::{DataType, Field, Schema};
    use cylon::ctx::CylonContext;
    use cylon::table::Table;
    use cylon::error::CylonResult;

    /// Helper to create CylonContext with MPI
    fn create_mpi_context() -> CylonResult<Arc<CylonContext>> {
        let mut ctx_new = CylonContext::new(true);
        ctx_new.set_communicator(cylon::net::mpi::communicator::MPICommunicator::make()?);
        Ok(Arc::new(ctx_new))
    }

    /// Helper to create a test table with values
    fn create_test_table(ctx: Arc<CylonContext>, values: Vec<i32>) -> CylonResult<Table> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Int32, false),
        ]));

        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int32Array::from(values))],
        )?;

        Table::from_record_batch(ctx, batch)
    }

    #[test]
    fn test_gather_to_root() -> CylonResult<()> {
        let ctx = create_mpi_context()?;
        let rank = ctx.get_rank();
        let world_size = ctx.get_world_size();

        println!("Rank {}/{} starting gather test", rank, world_size);

        // Each rank creates a table with its rank number repeated
        let values: Vec<i32> = (0..10).map(|i| rank * 10 + i).collect();
        let table = create_test_table(ctx.clone(), values)?;

        println!("Rank {}: Created table with {} rows", rank, table.rows());

        // Gather all tables to root (rank 0)
        let comm = ctx.get_communicator()
            .ok_or_else(|| cylon::error::CylonError::new(
                cylon::error::Code::Invalid, "No communicator".to_string()))?;

        let gathered = comm.gather(&table, 0, true, ctx.clone())?;

        if rank == 0 {
            // Root should have tables from all ranks
            assert_eq!(gathered.len(), world_size as usize,
                      "Root should have {} tables, got {}", world_size, gathered.len());

            // Verify each table
            for (i, t) in gathered.iter().enumerate() {
                assert_eq!(t.rows(), 10, "Table from rank {} should have 10 rows", i);

                let batch = t.batch(0).unwrap();
                let col = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();

                // Verify first value matches expected pattern
                let expected_first = (i * 10) as i32;
                assert_eq!(col.value(0), expected_first,
                          "Table from rank {} should start with {}", i, expected_first);
            }

            println!("Root: Gathered {} tables successfully", gathered.len());
        } else {
            // Non-root should have empty result
            assert!(gathered.is_empty(),
                   "Non-root rank {} should have empty result", rank);
        }

        ctx.barrier()?;
        println!("Rank {}: Gather test passed", rank);
        Ok(())
    }

    #[test]
    fn test_gather_exclude_root() -> CylonResult<()> {
        let ctx = create_mpi_context()?;
        let rank = ctx.get_rank();
        let world_size = ctx.get_world_size();

        println!("Rank {}/{} starting gather_exclude_root test", rank, world_size);

        let values: Vec<i32> = (0..5).map(|i| rank * 100 + i).collect();
        let table = create_test_table(ctx.clone(), values)?;

        // Gather excluding root's own data
        let comm = ctx.get_communicator()
            .ok_or_else(|| cylon::error::CylonError::new(
                cylon::error::Code::Invalid, "No communicator".to_string()))?;

        let gathered = comm.gather(&table, 0, false, ctx.clone())?;

        if rank == 0 {
            // Root should have tables from all ranks except itself
            let expected_count = (world_size - 1) as usize;
            assert_eq!(gathered.len(), expected_count,
                      "Root should have {} tables (excluding itself), got {}",
                      expected_count, gathered.len());

            println!("Root: Gathered {} tables (excluding itself)", gathered.len());
        }

        ctx.barrier()?;
        Ok(())
    }

    #[test]
    fn test_broadcast_from_root() -> CylonResult<()> {
        let ctx = create_mpi_context()?;
        let rank = ctx.get_rank();
        let world_size = ctx.get_world_size();

        println!("Rank {}/{} starting broadcast test", rank, world_size);

        // Only root has data initially
        let mut table_opt = if rank == 0 {
            let values: Vec<i32> = vec![100, 200, 300, 400, 500];
            Some(create_test_table(ctx.clone(), values)?)
        } else {
            None
        };

        // Broadcast from root
        let comm = ctx.get_communicator()
            .ok_or_else(|| cylon::error::CylonError::new(
                cylon::error::Code::Invalid, "No communicator".to_string()))?;

        comm.bcast(&mut table_opt, 0, ctx.clone())?;

        // All ranks should now have the table
        let table = table_opt.expect("All ranks should have table after broadcast");

        assert_eq!(table.rows(), 5, "Rank {}: Broadcast table should have 5 rows", rank);

        let batch = table.batch(0).unwrap();
        let col = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();

        // Verify values
        assert_eq!(col.value(0), 100);
        assert_eq!(col.value(1), 200);
        assert_eq!(col.value(2), 300);
        assert_eq!(col.value(3), 400);
        assert_eq!(col.value(4), 500);

        println!("Rank {}: Broadcast test passed", rank);
        ctx.barrier()?;
        Ok(())
    }

    #[test]
    fn test_all_gather() -> CylonResult<()> {
        let ctx = create_mpi_context()?;
        let rank = ctx.get_rank();
        let world_size = ctx.get_world_size();

        println!("Rank {}/{} starting all_gather test", rank, world_size);

        // Each rank creates a table with unique values
        let values: Vec<i32> = (0..3).map(|i| rank * 1000 + i).collect();
        let table = create_test_table(ctx.clone(), values)?;

        // All gather
        let comm = ctx.get_communicator()
            .ok_or_else(|| cylon::error::CylonError::new(
                cylon::error::Code::Invalid, "No communicator".to_string()))?;

        let gathered = comm.all_gather(&table, ctx.clone())?;

        // All ranks should have tables from all ranks
        assert_eq!(gathered.len(), world_size as usize,
                  "Rank {}: Should have {} tables, got {}",
                  rank, world_size, gathered.len());

        // Verify each table
        for (i, t) in gathered.iter().enumerate() {
            assert_eq!(t.rows(), 3, "Table from rank {} should have 3 rows", i);

            let batch = t.batch(0).unwrap();
            let col = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();

            // Verify first value matches expected pattern
            let expected_first = (i * 1000) as i32;
            assert_eq!(col.value(0), expected_first,
                      "Rank {}: Table from rank {} should start with {}",
                      rank, i, expected_first);
        }

        println!("Rank {}: All_gather test passed", rank);
        ctx.barrier()?;
        Ok(())
    }

    #[test]
    fn test_gather_empty_table() -> CylonResult<()> {
        let ctx = create_mpi_context()?;
        let rank = ctx.get_rank();
        let world_size = ctx.get_world_size();

        println!("Rank {}/{} starting gather empty table test", rank, world_size);

        // Create empty table
        let table = create_test_table(ctx.clone(), vec![])?;

        let comm = ctx.get_communicator()
            .ok_or_else(|| cylon::error::CylonError::new(
                cylon::error::Code::Invalid, "No communicator".to_string()))?;

        let gathered = comm.gather(&table, 0, true, ctx.clone())?;

        if rank == 0 {
            assert_eq!(gathered.len(), world_size as usize);
            for t in &gathered {
                assert_eq!(t.rows(), 0, "Empty tables should remain empty");
            }
        }

        ctx.barrier()?;
        println!("Rank {}: Empty table test passed", rank);
        Ok(())
    }

    #[test]
    fn test_broadcast_non_root() -> CylonResult<()> {
        let ctx = create_mpi_context()?;
        let rank = ctx.get_rank();
        let world_size = ctx.get_world_size();

        if world_size < 2 {
            println!("Skipping non_root broadcast test (need at least 2 ranks)");
            return Ok(());
        }

        println!("Rank {}/{} starting broadcast from non-root test", rank, world_size);

        // Broadcast from rank 1 instead of root
        let broadcast_root = 1;
        let mut table_opt = if rank == broadcast_root {
            let values: Vec<i32> = vec![11, 22, 33];
            Some(create_test_table(ctx.clone(), values)?)
        } else {
            None
        };

        let comm = ctx.get_communicator()
            .ok_or_else(|| cylon::error::CylonError::new(
                cylon::error::Code::Invalid, "No communicator".to_string()))?;

        comm.bcast(&mut table_opt, broadcast_root, ctx.clone())?;

        // All ranks should have the table
        let table = table_opt.expect("All ranks should have table after broadcast");
        assert_eq!(table.rows(), 3);

        let batch = table.batch(0).unwrap();
        let col = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();

        assert_eq!(col.value(0), 11);
        assert_eq!(col.value(1), 22);
        assert_eq!(col.value(2), 33);

        println!("Rank {}: Broadcast from non-root test passed", rank);
        ctx.barrier()?;
        Ok(())
    }

    #[test]
    fn test_gather_large_table() -> CylonResult<()> {
        let ctx = create_mpi_context()?;
        let rank = ctx.get_rank();
        let world_size = ctx.get_world_size();

        println!("Rank {}/{} starting large table gather test", rank, world_size);

        // Create larger table (1000 rows per rank)
        let values: Vec<i32> = (0..1000).map(|i| rank * 10000 + i).collect();
        let table = create_test_table(ctx.clone(), values)?;

        let comm = ctx.get_communicator()
            .ok_or_else(|| cylon::error::CylonError::new(
                cylon::error::Code::Invalid, "No communicator".to_string()))?;

        let gathered = comm.gather(&table, 0, true, ctx.clone())?;

        if rank == 0 {
            assert_eq!(gathered.len(), world_size as usize);

            let total_rows: i64 = gathered.iter().map(|t| t.rows()).sum();
            assert_eq!(total_rows, (world_size * 1000) as i64,
                      "Total gathered rows should be {}", world_size * 1000);

            println!("Root: Gathered {} total rows from {} ranks",
                     total_rows, world_size);
        }

        ctx.barrier()?;
        Ok(())
    }
}
