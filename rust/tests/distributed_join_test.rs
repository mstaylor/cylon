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

//! Tests for distributed join operation

#[cfg(feature = "mpi")]
mod mpi_tests {
    use std::sync::Arc;
    use arrow::array::{Int32Array, RecordBatch};
    use arrow::datatypes::{DataType, Field, Schema};
    use cylon::ctx::CylonContext;
    use cylon::table::Table;
    use cylon::join::JoinConfig;
    use cylon::ops::distributed_join::distributed_join;
    use cylon::error::CylonResult;

    #[test]
    fn test_distributed_inner_join() -> CylonResult<()> {
        let mut ctx_new = CylonContext::new(true);
        ctx_new.set_communicator(cylon::net::mpi::communicator::MPICommunicator::make()?);
        let ctx = Arc::new(ctx_new);

        let rank = ctx.get_rank();
        let world_size = ctx.get_world_size();

        println!("Rank {}/{} starting distributed join test", rank, world_size);

        // Create left table: each rank has different keys
        // Rank 0: keys [0, 4, 8, ...]
        // Rank 1: keys [1, 5, 9, ...]
        // Rank 2: keys [2, 6, 10, ...]
        // Rank 3: keys [3, 7, 11, ...]
        let left_schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::Int32, false),
            Field::new("left_val", DataType::Int32, false),
        ]));

        let left_keys: Vec<i32> = (0..10).map(|i| i * world_size + rank).collect();
        let left_vals: Vec<i32> = (0..10).map(|i| (i * world_size + rank) * 10).collect();

        let left_batch = RecordBatch::try_new(
            left_schema.clone(),
            vec![
                Arc::new(Int32Array::from(left_keys.clone())),
                Arc::new(Int32Array::from(left_vals)),
            ],
        )?;

        let left_table = Table::from_record_batch(ctx.clone(), left_batch)?;

        // Create right table: each rank has different keys
        // Similar distribution but different values
        let right_schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::Int32, false),
            Field::new("right_val", DataType::Int32, false),
        ]));

        let right_keys: Vec<i32> = (0..10).map(|i| i * world_size + rank).collect();
        let right_vals: Vec<i32> = (0..10).map(|i| (i * world_size + rank) * 100).collect();

        let right_batch = RecordBatch::try_new(
            right_schema.clone(),
            vec![
                Arc::new(Int32Array::from(right_keys.clone())),
                Arc::new(Int32Array::from(right_vals)),
            ],
        )?;

        let right_table = Table::from_record_batch(ctx.clone(), right_batch)?;

        println!("Rank {}: Created left table with {} rows, right table with {} rows",
                 rank, left_table.rows(), right_table.rows());

        // Perform distributed inner join on 'key' column
        let join_config = JoinConfig::inner_join(0, 0);
        let result = distributed_join(&left_table, &right_table, &join_config)?;

        println!("Rank {}: Join result has {} rows", rank, result.rows());

        // After join, each rank should have rows where keys match after shuffle
        // Result has 4 columns: left_key, left_val, right_key, right_val
        assert_eq!(result.columns(), 4, "Result should have 4 columns (left_key, left_val, right_key, right_val)");

        // Verify that all keys in the result belong to this rank after shuffle
        let result_batch = result.batch(0).unwrap();
        let key_col = result_batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();

        for i in 0..key_col.len() {
            let key = key_col.value(i);
            let expected_rank = (key % world_size) as i32;
            assert_eq!(expected_rank, rank,
                      "Key {} should be on rank {} but is on rank {}",
                      key, expected_rank, rank);
        }

        println!("Rank {}: Verification successful", rank);

        ctx.barrier()?;
        Ok(())
    }
}
