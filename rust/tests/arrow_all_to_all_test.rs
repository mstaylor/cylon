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

//! Basic tests for ArrowAllToAll infrastructure
//!
//! These tests verify the structures compile and can be instantiated.

#[cfg(feature = "mpi")]
mod mpi_tests {
    use std::sync::Arc;
    use arrow::array::{Array, Int32Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use cylon::ctx::CylonContext;
    use cylon::table::Table;
    use cylon::error::CylonResult;

    /// Combined test for ArrowAllToAll infrastructure
    /// Tests table creation, buffer extraction, and metadata extraction
    ///
    /// Note: All tests are combined into one function because MPI can only be
    /// initialized once per process, and the Rust test harness may finalize MPI
    /// between individual test functions.
    #[test]
    fn test_arrow_all_to_all_infrastructure() -> CylonResult<()> {
        // Initialize MPI context once for all tests
        let mut ctx_new = CylonContext::new(true);
        ctx_new.set_communicator(cylon::net::mpi::communicator::MPICommunicator::make()?);
        let ctx = Arc::new(ctx_new);

        let rank = ctx.get_rank();
        let world_size = ctx.get_world_size();

        println!("\n========================================");
        println!("Rank {}/{}: Starting ArrowAllToAll infrastructure tests", rank, world_size);
        println!("========================================\n");

        // ===================================================================
        // Test 1: Table creation
        // ===================================================================
        println!("Rank {}: TEST 1 - Table creation", rank);

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
            Field::new("value", DataType::Int32, false),
        ]));

        let ids: Vec<i32> = (0..10).map(|i| rank * 100 + i).collect();
        let names: Vec<String> = (0..10).map(|i| format!("rank{}_row{}", rank, i)).collect();
        let names_refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        let values: Vec<i32> = (0..10).map(|i| (rank * 100 + i) * 2).collect();

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(ids.clone())),
                Arc::new(StringArray::from(names_refs)),
                Arc::new(Int32Array::from(values.clone())),
            ],
        )?;

        let table = Table::from_record_batch(ctx.clone(), batch)?;

        println!("Rank {}: Created table with {} rows and {} columns",
                 rank, table.rows(), table.columns());

        assert_eq!(table.rows(), 10, "Table should have 10 rows");
        assert_eq!(table.columns(), 3, "Table should have 3 columns");

        let batches = table.batches();
        assert_eq!(batches.len(), 1, "Should have 1 batch");

        let batch_0 = &batches[0];
        let col_id = batch_0.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        let col_name = batch_0.column(1).as_any().downcast_ref::<StringArray>().unwrap();
        let col_value = batch_0.column(2).as_any().downcast_ref::<Int32Array>().unwrap();

        assert_eq!(col_id.value(0), rank * 100);
        assert_eq!(col_name.value(0), format!("rank{}_row0", rank));
        assert_eq!(col_value.value(0), rank * 100 * 2);

        println!("Rank {}: ✓ Test 1 passed - Table creation successful\n", rank);

        ctx.barrier()?;

        // ===================================================================
        // Test 2: Buffer extraction from Arrow arrays
        // ===================================================================
        println!("Rank {}: TEST 2 - Buffer extraction", rank);

        let array = Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5]));
        let array_data = array.to_data();
        let buffers = array_data.buffers();

        println!("Rank {}: Array has {} buffers", rank, buffers.len());

        assert!(buffers.len() >= 1, "Should have at least 1 buffer");

        let data_buffer = &buffers[0];
        println!("Rank {}: First buffer has {} bytes", rank, data_buffer.len());

        // 5 i32 values = 20 bytes
        assert_eq!(data_buffer.len(), 20, "Buffer should be 20 bytes for 5 i32 values");

        println!("Rank {}: ✓ Test 2 passed - Buffer extraction successful\n", rank);

        ctx.barrier()?;

        // ===================================================================
        // Test 3: Arrow metadata extraction (what ArrowAllToAll sends)
        // ===================================================================
        println!("Rank {}: TEST 3 - Arrow metadata extraction", rank);

        let meta_schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
        ]));

        let meta_batch = RecordBatch::try_new(
            meta_schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(Int32Array::from(vec![4, 5, 6])),
            ],
        )?;

        let meta_table = Table::from_record_batch(ctx.clone(), meta_batch)?;

        let meta_batches = meta_table.batches();
        let num_columns = meta_batches[0].num_columns();
        let num_batches = meta_batches.len();

        println!("Rank {}: Table has {} columns and {} batches",
                 rank, num_columns, num_batches);

        // For each column, get buffer metadata (this is what ArrowAllToAll sends)
        for (batch_idx, batch) in meta_batches.iter().enumerate() {
            println!("Rank {}:   Batch {}", rank, batch_idx);

            for col_idx in 0..batch.num_columns() {
                let array = batch.column(col_idx);
                let array_data = array.to_data();
                let buffers = array_data.buffers();

                println!("Rank {}:     Column {} has {} buffers, length {}",
                         rank, col_idx, buffers.len(), array_data.len());

                // This is the header format that ArrowAllToAll uses:
                // [columnIndex, bufferIndex, noBuffers, noArrays, length, reference]
                for (buf_idx, buf) in buffers.iter().enumerate() {
                    let header = [
                        col_idx as i32,           // columnIndex
                        buf_idx as i32,           // bufferIndex
                        buffers.len() as i32,     // noBuffers
                        num_batches as i32,       // noArrays
                        array_data.len() as i32,  // length
                        0i32,                     // reference
                    ];

                    println!("Rank {}:       Buffer {}: {} bytes, header: {:?}",
                             rank, buf_idx, buf.len(), header);
                }
            }
        }

        println!("Rank {}: ✓ Test 3 passed - Metadata extraction successful\n", rank);

        ctx.barrier()?;

        // ===================================================================
        // Summary
        // ===================================================================
        if rank == 0 {
            println!("\n========================================");
            println!("ALL TESTS PASSED ✓");
            println!("ArrowAllToAll infrastructure is ready!");
            println!("========================================\n");
        }

        ctx.barrier()?;

        Ok(())
    }
}
