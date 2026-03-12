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

//! Partitioning operation tests

use std::hash::{Hash, Hasher};
use std::sync::Arc;
use ahash::AHasher;
use arrow::array::{Int32Array, Int64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use cylon::ctx::CylonContext;
use cylon::partition::hash_partition;
use cylon::table::Table;

fn create_test_table(ctx: Arc<CylonContext>) -> Table {
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int32, false),
        Field::new("b", DataType::Int64, false),
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])),
            Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])),
        ],
    )
    .unwrap();

    Table::from_record_batch(ctx, batch).unwrap()
}

/// Hash a value using ahash (similar to MurmurHash3 in C++)
#[inline]
fn hash_value<T: Hash>(value: &T) -> u32 {
    let mut hasher = AHasher::default();
    value.hash(&mut hasher);
    hasher.finish() as u32
}

#[test]
fn test_hash_partition() {
    let ctx = Arc::new(CylonContext::new(false));
    let table = create_test_table(ctx.clone());
    let num_partitions = 4;

    let partitions = hash_partition(&table, &[0], num_partitions).unwrap();

    assert_eq!(
        partitions.len(),
        num_partitions,
        "Expected number of partitions to be {}",
        num_partitions
    );

    let mut total_rows = 0;
    for (part_key, part_table) in &partitions {
        total_rows += part_table.rows();
        if part_table.rows() > 0 {
            let batch = part_table.batch(0).unwrap();
            let col_a = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            for i in 0..col_a.len() {
                let val = col_a.value(i);
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        &val as *const i32 as *const u8,
                        std::mem::size_of::<i32>(),
                    )
                };
                let hash = hash_value(&bytes);
                assert_eq!((hash % num_partitions as u32) as usize, *part_key);
            }
        }
    }

    assert_eq!(
        total_rows,
        table.rows(),
        "Expected total rows in partitions to be equal to original table rows"
    );
}

#[test]
fn test_multi_column_hash_partition() {
    let ctx = Arc::new(CylonContext::new(false));
    let table = create_test_table(ctx.clone());
    let num_partitions = 4;

    let partitions = hash_partition(&table, &[0, 1], num_partitions).unwrap();

    assert_eq!(
        partitions.len(),
        num_partitions,
        "Expected number of partitions to be {}",
        num_partitions
    );

    let mut total_rows = 0;
    for (part_key, part_table) in &partitions {
        total_rows += part_table.rows();
        if part_table.rows() > 0 {
            let batch = part_table.batch(0).unwrap();
            let col_a = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            let col_b = batch
                .column(1)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();

            for i in 0..col_a.len() {
                let val_a = col_a.value(i);
                let val_b = col_b.value(i);

                // Replicate composite hash logic from implementation
                let mut partial_hash = 0u32;

                // Hash column 0
                let bytes_a = unsafe {
                    std::slice::from_raw_parts(
                        &val_a as *const i32 as *const u8,
                        std::mem::size_of::<i32>(),
                    )
                };
                let hash_a = hash_value(&bytes_a);
                partial_hash = hash_a.wrapping_add(31u32.wrapping_mul(partial_hash));

                // Hash column 1
                let bytes_b = unsafe {
                    std::slice::from_raw_parts(
                        &val_b as *const i64 as *const u8,
                        std::mem::size_of::<i64>(),
                    )
                };
                let hash_b = hash_value(&bytes_b);
                partial_hash = hash_b.wrapping_add(31u32.wrapping_mul(partial_hash));

                assert_eq!((partial_hash % num_partitions as u32) as usize, *part_key);
            }
        }
    }

    assert_eq!(
        total_rows,
        table.rows(),
        "Expected total rows in partitions to be equal to original table rows"
    );
}