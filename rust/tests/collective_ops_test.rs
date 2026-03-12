
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

//! Tests for collective communication operations


#[cfg(feature = "mpi")]
use std::sync::Arc;
#[cfg(feature = "mpi")]
use arrow::array::{Int32Array, StringArray};
#[cfg(feature = "mpi")]
use arrow::datatypes::{DataType, Field, Schema};
#[cfg(feature = "mpi")]
use arrow::record_batch::RecordBatch;
#[cfg(feature = "mpi")]
use cylon::ctx::CylonContext;
#[cfg(feature = "mpi")]
use cylon::table::Table;
#[cfg(feature = "mpi")]
use cylon::error::CylonResult;

#[test]
#[cfg(feature = "mpi")]
fn test_broadcast() -> CylonResult<()> {
    // Create a distributed context
    let mut ctx_new = CylonContext::new(true);
    ctx_new.set_communicator(cylon::net::mpi::communicator::MPICommunicator::make()?);
    let ctx = Arc::new(ctx_new);
    let rank = ctx.get_rank();
    let world_size = ctx.get_world_size();

    println!("Rank {}/{} starting broadcast test", rank, world_size);

    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int32, false),
        Field::new("b", DataType::Utf8, false),
    ]));

    let mut table_to_bcast = if rank == 0 {
        // Create a table only on rank 0
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3, 4])),
                Arc::new(StringArray::from(vec!["a", "b", "c", "d"])),
            ],
        )?;
        let table = Table::from_record_batch(ctx.clone(), batch)?;
        println!("Rank 0: Created table with {} rows", table.rows());
        Some(table)
    } else {
        // Other ranks start with None
        None
    };

    // Broadcast the table from rank 0
    if let Some(comm) = ctx.get_communicator() {
        comm.bcast(&mut table_to_bcast, 0, ctx.clone())?;
    }

    println!("Rank {}: Broadcast finished", rank);

    // All ranks should now have the table
    assert!(table_to_bcast.is_some(), "Rank {} has no table after broadcast", rank);

    let received_table = table_to_bcast.unwrap();
    println!("Rank {}: Received table with {} rows", rank, received_table.rows());

    // Verify the contents of the received table
    assert_eq!(received_table.rows(), 4);
    assert_eq!(received_table.columns(), 2);

    let batches = received_table.batches();
    assert_eq!(batches.len(), 1);
    let batch = &batches[0];

    let a = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    let b = batch.column(1).as_any().downcast_ref::<StringArray>().unwrap();

    assert_eq!(a.values(), &[1, 2, 3, 4]);
    assert_eq!(b.value(0), "a");
    assert_eq!(b.value(1), "b");
    assert_eq!(b.value(2), "c");
    assert_eq!(b.value(3), "d");

    println!("Rank {}: Verification successful", rank);

    ctx.barrier()?;
    Ok(())
}
