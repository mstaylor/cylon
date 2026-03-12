
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

//! Tests for shuffle operation

#[cfg(feature = "mpi")]
mod mpi_tests {
    use std::sync::Arc;
    use arrow::array::{Int32Array, RecordBatch};
    use arrow::datatypes::{DataType, Field, Schema};
    use cylon::ctx::CylonContext;
    use cylon::table::Table;
    use cylon::ops::shuffle::shuffle;
    use cylon::error::CylonResult;

    #[test]
    fn test_shuffle() -> CylonResult<()> {
        let mut ctx_new = CylonContext::new(true);
        ctx_new.set_communicator(cylon::net::mpi::communicator::MPICommunicator::make()?);
        let ctx = Arc::new(ctx_new);

        let rank = ctx.get_rank();
        let world_size = ctx.get_world_size();

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

        let shuffled_table = shuffle(&ctx, &table, &[0])?;

        // After shuffle, all rows with the same value in column 'a' should be on the same process.
        // The partition for a value is calculated as `hash(value) % world_size`.
        // In our hash implementation, the hash of an integer is the integer itself.
        // So, a value `v` should be on process `v % world_size`.
        let shuffled_batch = shuffled_table.batch(0).unwrap();
        let col_a = shuffled_batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();

        for i in 0..col_a.len() {
            let val_a = col_a.value(i);
            assert_eq!((val_a as i32 % world_size), rank, "Value {} with hash {} is on the wrong rank {}", val_a, val_a % world_size, rank);
        }

        Ok(())
    }
}
