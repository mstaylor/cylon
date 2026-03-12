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

//! Tests for shuffle using ArrowAllToAll
//!
//! This tests the buffer-by-buffer shuffle implementation

#[cfg(feature = "mpi")]
mod mpi_tests {
    use std::sync::Arc;
    use arrow::array::Int32Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use cylon::ctx::CylonContext;
    use cylon::table::{Table, shuffle};
    use cylon::error::CylonResult;

    /// Test shuffle with simple data
    #[test]
    fn test_shuffle_basic() -> CylonResult<()> {
        // Initialize MPI context
        let mut ctx_new = CylonContext::new(true);
        ctx_new.set_communicator(cylon::net::mpi::communicator::MPICommunicator::make()?);
        let ctx = Arc::new(ctx_new);

        let rank = ctx.get_rank();
        let world_size = ctx.get_world_size();

        println!("\n========================================");
        println!("Rank {}/{}: Testing shuffle with ArrowAllToAll", rank, world_size);
        println!("========================================\n");

        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
        ]));

        // Each process creates a table with values based on its rank
        let values_a: Vec<i32> = (0..10).map(|i| i * world_size + rank).collect();
        let values_b: Vec<i32> = (0..10).map(|i| (i * world_size + rank) * 2).collect();

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(values_a)),
                Arc::new(Int32Array::from(values_b)),
            ],
        )?;

        let table = Table::from_record_batch(ctx.clone(), batch)?;

        println!("Rank {}: Created table with {} rows", rank, table.rows());

        // Shuffle on column 'a' (index 0)
        let shuffled_table = shuffle(&table, &[0])?;

        println!("Rank {}: Shuffled table has {} rows", rank, shuffled_table.rows());

        // After shuffle, all rows with the same value in column 'a' should be on the same process
        // The partition for a value is calculated as `hash(value) % world_size`
        if shuffled_table.rows() > 0 {
            let shuffled_batch = shuffled_table.batch(0).unwrap();
            let col_a = shuffled_batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();

            for i in 0..col_a.len() {
                let val_a = col_a.value(i);
                // Simple hash: value % world_size
                let expected_rank = (val_a.abs() as i32) % world_size;
                assert_eq!(expected_rank, rank,
                          "Rank {}: Value {} should be on rank {}, but is on rank {}",
                          rank, val_a, expected_rank, rank);
            }
        }

        println!("Rank {}: ✓ Shuffle validation successful", rank);

        ctx.barrier()?;

        if rank == 0 {
            println!("\n========================================");
            println!("SHUFFLE TEST PASSED ✓");
            println!("ArrowAllToAll-based shuffle working!");
            println!("========================================\n");
        }

        Ok(())
    }
}
