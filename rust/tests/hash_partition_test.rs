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

//! Tests for hash partitioning

use std::sync::Arc;
use arrow::array::{Int32Array, StringArray, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use cylon::table::Table;
use cylon::ctx::CylonContext;
use cylon::ops::partition::hash_partition_table;
use cylon::error::CylonResult;

#[test]
fn test_hash_partition() -> CylonResult<()> {
    let ctx = CylonContext::init();
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int32, false),
        Field::new("b", DataType::Utf8, false),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 1, 2, 1, 2, 1, 2])),
            Arc::new(StringArray::from(vec!["a", "b", "a", "b", "a", "b", "a", "b"])),
        ],
    )?;

    let table = Table::from_record_batch(ctx, batch)?;
    let num_partitions = 4;
    let partitions = hash_partition_table(&table, &[0], num_partitions)?;

    assert_eq!(partitions.len(), num_partitions);

    let total_rows: usize = partitions.iter().map(|p| p.num_rows()).sum();
    assert_eq!(total_rows, table.rows() as usize);

    // Check that all rows with the same hash column value are in the same partition
    let mut value_to_partition = std::collections::HashMap::new();

    for (i, partition) in partitions.iter().enumerate() {
        for row_idx in 0..partition.num_rows() {
            let col_a = partition.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
            let val_a = col_a.value(row_idx);

            if let Some(&existing_partition) = value_to_partition.get(&val_a) {
                assert_eq!(existing_partition, i, "value {} found in multiple partitions", val_a);
            } else {
                value_to_partition.insert(val_a, i);
            }
        }
    }

    Ok(())
}