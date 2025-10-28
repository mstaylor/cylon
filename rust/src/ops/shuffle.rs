
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

//! Shuffle operation for distributed tables

use std::sync::Arc;
use arrow::record_batch::RecordBatch;
use crate::ctx::CylonContext;
use crate::error::{CylonResult, CylonError, Code};
use crate::table::Table;
use crate::ops::partition::hash_partition_table;
use crate::net::serialize::{serialize_record_batch, deserialize_record_batch};

/// Shuffle table across all processes using all-to-all communication
/// C++ reference: table.cpp:194-215 (shuffle_table_by_hashing)
pub fn shuffle(
    ctx: &Arc<CylonContext>,
    table: &Table,
    hash_columns: &[usize],
) -> CylonResult<Table> {
    if !ctx.is_distributed() {
        return Ok(table.clone()); // Not a distributed context, return original table
    }

    let world_size = ctx.get_world_size() as usize;

    // 1. Hash partition local table into world_size partitions
    let partitions = hash_partition_table(table, hash_columns, world_size)?;

    // 2. Serialize each partition
    let serialized_partitions: CylonResult<Vec<Vec<u8>>> = partitions
        .iter()
        .map(|batch| serialize_record_batch(batch))
        .collect();
    let serialized_partitions = serialized_partitions?;

    // 3. All-to-all exchange: send partition i to process i
    let received_serialized = ctx.get_communicator()
        .ok_or_else(|| CylonError::new(Code::Invalid, "Communicator not set"))?
        .all_to_all(serialized_partitions)?;

    // 4. Deserialize received data
    let mut received_batches = Vec::with_capacity(received_serialized.len());
    for data in received_serialized {
        if !data.is_empty() {
            let batch = deserialize_record_batch(&data)?;
            received_batches.push(batch);
        }
    }

    // 5. Combine into new table
    Table::from_record_batches(ctx.clone(), received_batches)
}
