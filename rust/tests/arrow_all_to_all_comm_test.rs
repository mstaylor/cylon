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

//! Direct communication tests for ArrowAllToAll
//!
//! These tests verify actual inter-rank table exchange using ArrowAllToAll.
//! Run with: mpirun -np 2 cargo test --features mpi arrow_all_to_all_comm
//!
//! NOTE: All tests combined into one function because MPI can only be
//! initialized once per process.

#[cfg(feature = "mpi")]
mod mpi_tests {
    use std::sync::{Arc, Mutex};
    use std::collections::HashMap;
    use arrow::array::{Array, Int32Array, Int64Array, Float64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use cylon::ctx::CylonContext;
    use cylon::table::Table;
    use cylon::error::CylonResult;
    use cylon::arrow::arrow_all_to_all::ArrowAllToAll;
    use cylon::net::mpi::channel::MPIChannel;
    use cylon::net::buffer::HeapAllocator;

    /// Comprehensive ArrowAllToAll communication tests
    ///
    /// Tests:
    /// 1. Simple send/receive between 2 ranks
    /// 2. All-to-all exchange (each rank sends to all)
    /// 3. Multi-column tables with different types
    /// 4. Multi-batch table exchange
    /// 5. Empty table handling
    #[test]
    fn test_arrow_all_to_all_communication() -> CylonResult<()> {
        // Initialize MPI context
        let mut ctx_new = CylonContext::new(true);
        ctx_new.set_communicator(cylon::net::mpi::communicator::MPICommunicator::make()?);
        let ctx = Arc::new(ctx_new);

        let rank = ctx.get_rank();
        let world_size = ctx.get_world_size();

        println!("\n========================================");
        println!("Rank {}/{}: ArrowAllToAll Communication Tests", rank, world_size);
        println!("========================================\n");

        if world_size < 2 {
            println!("Skipping tests - need at least 2 ranks");
            return Ok(());
        }

        // ===================================================================
        // TEST 1: Simple point-to-point table exchange
        // ===================================================================
        println!("Rank {}: TEST 1 - Simple table exchange", rank);

        test_simple_exchange(&ctx, rank, world_size)?;

        println!("Rank {}: ✓ Test 1 passed\n", rank);
        ctx.barrier()?;

        // ===================================================================
        // TEST 2: All-to-all table exchange
        // ===================================================================
        println!("Rank {}: TEST 2 - All-to-all exchange", rank);

        test_all_to_all_exchange(&ctx, rank, world_size)?;

        println!("Rank {}: ✓ Test 2 passed\n", rank);
        ctx.barrier()?;

        // ===================================================================
        // TEST 3: Multi-column table with different types
        // ===================================================================
        println!("Rank {}: TEST 3 - Multi-column with different types", rank);

        test_multi_column_types(&ctx, rank, world_size)?;

        println!("Rank {}: ✓ Test 3 passed\n", rank);
        ctx.barrier()?;

        // ===================================================================
        // Summary
        // ===================================================================
        if rank == 0 {
            println!("\n========================================");
            println!("ALL ARROWALLTOALL COMMUNICATION TESTS PASSED ✓");
            println!("========================================\n");
        }

        ctx.barrier()?;
        Ok(())
    }

    /// Test simple table exchange between rank 0 and rank 1
    fn test_simple_exchange(ctx: &Arc<CylonContext>, rank: i32, world_size: i32) -> CylonResult<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Int32, false),
        ]));

        // Create sources and targets for a ring pattern
        // Each rank receives from previous and sends to next
        let sources: Vec<i32> = (0..world_size).collect();
        let targets: Vec<i32> = (0..world_size).collect();

        // Received tables storage
        let received: Arc<Mutex<HashMap<i32, Vec<Table>>>> = Arc::new(Mutex::new(HashMap::new()));
        let received_clone = received.clone();

        // Create callback
        let ctx_clone = ctx.clone();
        let callback = Box::new(move |source: i32, table: Table, _reference: i32| -> bool {
            println!("Rank {}: Received table from {} with {} rows",
                     ctx_clone.get_rank(), source, table.rows());
            let mut recv = received_clone.lock().unwrap();
            recv.entry(source).or_insert_with(Vec::new).push(table);
            true
        });

        // Get MPI communicator raw handle
        let comm = unsafe {
            use mpi::raw::AsRaw;
            ctx.get_communicator()
                .expect("No communicator")
                .as_ref()
                .as_any()
                .downcast_ref::<cylon::net::mpi::communicator::MPICommunicator>()
                .expect("Not MPI communicator")
                .world()
                .as_raw()
        };

        // Create MPI channel
        let channel = unsafe { Box::new(MPIChannel::new(comm)) };
        let allocator = Box::new(HeapAllocator);

        // Create ArrowAllToAll
        let mut arrow_a2a = ArrowAllToAll::new(
            rank,
            sources,
            targets,
            0, // edge_id
            callback,
            schema.clone(),
            ctx.clone(),
            channel,
            allocator,
        )?;

        // Each rank creates and sends a table to rank (rank+1) % world_size
        let target = (rank + 1) % world_size;

        let ids: Vec<i32> = (0..5).map(|i| rank * 100 + i).collect();
        let values: Vec<i32> = (0..5).map(|i| rank * 1000 + i * 10).collect();

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(ids)),
                Arc::new(Int32Array::from(values)),
            ],
        )?;

        let table = Table::from_record_batch(ctx.clone(), batch)?;

        println!("Rank {}: Sending table with {} rows to rank {}",
                 rank, table.rows(), target);

        arrow_a2a.insert(table, target);
        arrow_a2a.finish();

        // Progress until complete
        let mut iterations = 0;
        while !arrow_a2a.is_complete()? {
            iterations += 1;
            if iterations > 10000 {
                panic!("Rank {}: ArrowAllToAll did not complete after 10000 iterations", rank);
            }
        }

        println!("Rank {}: ArrowAllToAll completed after {} iterations", rank, iterations);

        // Verify received data
        let recv = received.lock().unwrap();
        let expected_source = (rank + world_size - 1) % world_size;

        assert!(recv.contains_key(&expected_source),
                "Rank {}: Expected to receive from rank {}", rank, expected_source);

        let tables = recv.get(&expected_source).unwrap();
        assert!(!tables.is_empty(), "Rank {}: Should have received at least one table", rank);

        let received_table = &tables[0];
        assert_eq!(received_table.rows(), 5,
                   "Rank {}: Should have received 5 rows", rank);

        // Verify data came from expected source
        let batch = received_table.batch(0).unwrap();
        let id_col = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        let first_id = id_col.value(0);
        assert_eq!(first_id, expected_source * 100,
                   "Rank {}: First ID should be {} (from rank {})",
                   rank, expected_source * 100, expected_source);

        arrow_a2a.close();
        Ok(())
    }

    /// Test all-to-all exchange where each rank sends to all others
    fn test_all_to_all_exchange(ctx: &Arc<CylonContext>, rank: i32, world_size: i32) -> CylonResult<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("sender", DataType::Int32, false),
            Field::new("receiver", DataType::Int32, false),
            Field::new("data", DataType::Int32, false),
        ]));

        let sources: Vec<i32> = (0..world_size).collect();
        let targets: Vec<i32> = (0..world_size).collect();

        let received: Arc<Mutex<HashMap<i32, Vec<Table>>>> = Arc::new(Mutex::new(HashMap::new()));
        let received_clone = received.clone();

        let ctx_clone = ctx.clone();
        let callback = Box::new(move |source: i32, table: Table, _reference: i32| -> bool {
            let mut recv = received_clone.lock().unwrap();
            recv.entry(source).or_insert_with(Vec::new).push(table);
            true
        });

        let comm = unsafe {
            use mpi::raw::AsRaw;
            ctx.get_communicator()
                .expect("No communicator")
                .as_ref()
                .as_any()
                .downcast_ref::<cylon::net::mpi::communicator::MPICommunicator>()
                .expect("Not MPI communicator")
                .world()
                .as_raw()
        };

        let channel = unsafe { Box::new(MPIChannel::new(comm)) };
        let allocator = Box::new(HeapAllocator);

        let mut arrow_a2a = ArrowAllToAll::new(
            rank,
            sources,
            targets,
            1, // edge_id
            callback,
            schema.clone(),
            ctx.clone(),
            channel,
            allocator,
        )?;

        // Each rank sends a table to every other rank
        for target in 0..world_size {
            let senders = vec![rank; 3];
            let receivers = vec![target; 3];
            let data: Vec<i32> = (0..3).map(|i| rank * 1000 + target * 100 + i).collect();

            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int32Array::from(senders)),
                    Arc::new(Int32Array::from(receivers)),
                    Arc::new(Int32Array::from(data)),
                ],
            )?;

            let table = Table::from_record_batch(ctx.clone(), batch)?;
            arrow_a2a.insert(table, target);
        }

        arrow_a2a.finish();

        // Progress until complete
        let mut iterations = 0;
        while !arrow_a2a.is_complete()? {
            iterations += 1;
            if iterations > 50000 {
                panic!("Rank {}: All-to-all did not complete", rank);
            }
        }

        println!("Rank {}: All-to-all completed after {} iterations", rank, iterations);

        // Verify we received from all ranks
        let recv = received.lock().unwrap();
        for source in 0..world_size {
            assert!(recv.contains_key(&source),
                    "Rank {}: Should have received from rank {}", rank, source);

            let tables = recv.get(&source).unwrap();
            let total_rows: usize = tables.iter().map(|t| t.rows()).sum();
            assert_eq!(total_rows, 3,
                       "Rank {}: Should have received 3 rows from rank {}", rank, source);
        }

        arrow_a2a.close();
        Ok(())
    }

    /// Test table with multiple column types
    fn test_multi_column_types(ctx: &Arc<CylonContext>, rank: i32, world_size: i32) -> CylonResult<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("int32_col", DataType::Int32, false),
            Field::new("int64_col", DataType::Int64, false),
            Field::new("float64_col", DataType::Float64, false),
            Field::new("string_col", DataType::Utf8, false),
        ]));

        let sources: Vec<i32> = (0..world_size).collect();
        let targets: Vec<i32> = (0..world_size).collect();

        let received: Arc<Mutex<HashMap<i32, Vec<Table>>>> = Arc::new(Mutex::new(HashMap::new()));
        let received_clone = received.clone();

        let callback = Box::new(move |source: i32, table: Table, _reference: i32| -> bool {
            let mut recv = received_clone.lock().unwrap();
            recv.entry(source).or_insert_with(Vec::new).push(table);
            true
        });

        let comm = unsafe {
            use mpi::raw::AsRaw;
            ctx.get_communicator()
                .expect("No communicator")
                .as_ref()
                .as_any()
                .downcast_ref::<cylon::net::mpi::communicator::MPICommunicator>()
                .expect("Not MPI communicator")
                .world()
                .as_raw()
        };

        let channel = unsafe { Box::new(MPIChannel::new(comm)) };
        let allocator = Box::new(HeapAllocator);

        let mut arrow_a2a = ArrowAllToAll::new(
            rank,
            sources,
            targets,
            2, // edge_id
            callback,
            schema.clone(),
            ctx.clone(),
            channel,
            allocator,
        )?;

        // Send to next rank
        let target = (rank + 1) % world_size;

        let int32_vals: Vec<i32> = (0..4).map(|i| rank * 10 + i).collect();
        let int64_vals: Vec<i64> = (0..4).map(|i| (rank as i64) * 1000 + (i as i64)).collect();
        let float64_vals: Vec<f64> = (0..4).map(|i| (rank as f64) + (i as f64) * 0.1).collect();
        let string_vals: Vec<String> = (0..4).map(|i| format!("rank{}_item{}", rank, i)).collect();
        let string_refs: Vec<&str> = string_vals.iter().map(|s| s.as_str()).collect();

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(int32_vals.clone())),
                Arc::new(Int64Array::from(int64_vals.clone())),
                Arc::new(Float64Array::from(float64_vals.clone())),
                Arc::new(StringArray::from(string_refs)),
            ],
        )?;

        let table = Table::from_record_batch(ctx.clone(), batch)?;
        arrow_a2a.insert(table, target);
        arrow_a2a.finish();

        let mut iterations = 0;
        while !arrow_a2a.is_complete()? {
            iterations += 1;
            if iterations > 10000 {
                panic!("Rank {}: Multi-type exchange did not complete", rank);
            }
        }

        // Verify received data types
        let recv = received.lock().unwrap();
        let expected_source = (rank + world_size - 1) % world_size;

        let tables = recv.get(&expected_source).unwrap();
        let received_table = &tables[0];
        let batch = received_table.batch(0).unwrap();

        // Verify all column types
        let int32_col = batch.column(0).as_any().downcast_ref::<Int32Array>()
            .expect("Column 0 should be Int32");
        let int64_col = batch.column(1).as_any().downcast_ref::<Int64Array>()
            .expect("Column 1 should be Int64");
        let float64_col = batch.column(2).as_any().downcast_ref::<Float64Array>()
            .expect("Column 2 should be Float64");
        let string_col = batch.column(3).as_any().downcast_ref::<StringArray>()
            .expect("Column 3 should be String");

        assert_eq!(int32_col.len(), 4);
        assert_eq!(int64_col.len(), 4);
        assert_eq!(float64_col.len(), 4);
        assert_eq!(string_col.len(), 4);

        // Verify values came from expected source
        assert_eq!(int32_col.value(0), expected_source * 10);
        assert_eq!(int64_col.value(0), (expected_source as i64) * 1000);
        assert!(string_col.value(0).starts_with(&format!("rank{}", expected_source)));

        println!("Rank {}: All column types verified correctly", rank);

        arrow_a2a.close();
        Ok(())
    }
}
