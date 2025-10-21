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

//! Tests for HashPartition operation
//!
//! Corresponds to C++ HashPartition function in table.cpp

use cylon::ctx::CylonContext;
use cylon::table::Table;
use cylon::partition::hash_partition;
use std::sync::Arc;
use arrow::array::{Array, Int32Array, Int64Array, StringArray};
use arrow::datatypes::{Schema, Field, DataType};
use arrow::record_batch::RecordBatch;
use std::collections::HashSet;

fn create_test_table(ctx: Arc<CylonContext>, ids: Vec<i32>, names: Vec<&str>) -> Table {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(ids)),
            Arc::new(StringArray::from(names)),
        ],
    ).unwrap();

    Table::from_record_batch(ctx, batch).unwrap()
}

#[test]
fn test_hash_partition_single_column() {
    let ctx = Arc::new(CylonContext::new(false));

    // Create table with IDs 0-9
    let ids: Vec<i32> = (0..10).collect();
    let names: Vec<String> = ids.iter().map(|i| format!("row{}", i)).collect();
    let name_refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();

    let table = create_test_table(ctx.clone(), ids.clone(), name_refs);

    // Partition into 3 partitions based on ID column (index 0)
    let partitions = hash_partition(&table, &[0], 3).unwrap();

    // Verify we have 3 partitions
    assert_eq!(partitions.len(), 3, "Should have 3 partitions");

    // Verify all rows are accounted for
    let mut total_rows = 0;
    for i in 0..3 {
        let partition = partitions.get(&i).unwrap();
        total_rows += partition.rows();
        assert_eq!(partition.columns(), 2, "Each partition should have 2 columns");
    }
    assert_eq!(total_rows, 10, "Total rows across partitions should be 10");

    // Verify each ID is in exactly one partition
    let mut found_ids = HashSet::new();
    for i in 0..3 {
        let partition = partitions.get(&i).unwrap();
        if partition.rows() > 0 {
            let batch = partition.batch(0).unwrap();
            let partition_ids = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
            for j in 0..partition_ids.len() {
                let id = partition_ids.value(j);
                assert!(!found_ids.contains(&id), "ID {} found in multiple partitions", id);
                found_ids.insert(id);
            }
        }
    }
    assert_eq!(found_ids.len(), 10, "All 10 IDs should be found across partitions");

    println!("Hash partition single column test passed!");
}

#[test]
fn test_hash_partition_multi_column() {
    let ctx = Arc::new(CylonContext::new(false));

    // Create table with ID and value columns
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("value", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
    ]));

    let ids: Vec<i32> = (0..10).collect();
    let values: Vec<i32> = (100..110).collect();
    let names: Vec<String> = ids.iter().map(|i| format!("row{}", i)).collect();
    let name_refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(ids)),
            Arc::new(Int32Array::from(values)),
            Arc::new(StringArray::from(name_refs)),
        ],
    ).unwrap();

    let table = Table::from_record_batch(ctx.clone(), batch).unwrap();

    // Partition based on both ID and value columns
    let partitions = hash_partition(&table, &[0, 1], 4).unwrap();

    // Verify we have 4 partitions
    assert_eq!(partitions.len(), 4, "Should have 4 partitions");

    // Verify all rows are accounted for
    let mut total_rows = 0;
    for i in 0..4 {
        let partition = partitions.get(&i).unwrap();
        total_rows += partition.rows();
    }
    assert_eq!(total_rows, 10, "Total rows across partitions should be 10");

    println!("Hash partition multi-column test passed!");
}

#[test]
fn test_hash_partition_large_table() {
    let ctx = Arc::new(CylonContext::new(false));

    // Create larger table with 1000 rows
    let ids: Vec<i32> = (0..1000).collect();
    let names: Vec<String> = ids.iter().map(|i| format!("row{}", i)).collect();
    let name_refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(ids)),
            Arc::new(StringArray::from(name_refs)),
        ],
    ).unwrap();

    let table = Table::from_record_batch(ctx.clone(), batch).unwrap();

    // Partition into 8 partitions
    let partitions = hash_partition(&table, &[0], 8).unwrap();

    assert_eq!(partitions.len(), 8);

    // Verify all rows are accounted for
    let mut total_rows = 0;
    for i in 0..8 {
        let partition = partitions.get(&i).unwrap();
        total_rows += partition.rows();
        // Each partition should have some rows (probabilistically)
        // but we don't enforce exact distribution
    }
    assert_eq!(total_rows, 1000, "Total rows should be 1000");

    println!("Hash partition large table test passed!");
}

#[test]
fn test_hash_partition_single_partition() {
    let ctx = Arc::new(CylonContext::new(false));

    let table = create_test_table(ctx.clone(), vec![1, 2, 3], vec!["a", "b", "c"]);

    // Partition into just 1 partition (all rows should go there)
    let partitions = hash_partition(&table, &[0], 1).unwrap();

    assert_eq!(partitions.len(), 1);
    let partition = partitions.get(&0).unwrap();
    assert_eq!(partition.rows(), 3, "Single partition should have all 3 rows");

    println!("Hash partition single partition test passed!");
}

#[test]
fn test_hash_partition_more_partitions_than_rows() {
    let ctx = Arc::new(CylonContext::new(false));

    let table = create_test_table(ctx.clone(), vec![1, 2], vec!["a", "b"]);

    // Partition into 5 partitions with only 2 rows
    let partitions = hash_partition(&table, &[0], 5).unwrap();

    assert_eq!(partitions.len(), 5, "Should have 5 partitions");

    // Count non-empty partitions
    let mut total_rows = 0;
    for i in 0..5 {
        let partition = partitions.get(&i).unwrap();
        total_rows += partition.rows();
    }

    // We don't assert exact number of non-empty partitions since hash distribution
    // may place both rows in the same partition or different partitions
    assert_eq!(total_rows, 2, "Total rows should be 2");

    println!("Hash partition more partitions than rows test passed!");
}

#[test]
fn test_hash_partition_empty_table() {
    let ctx = Arc::new(CylonContext::new(false));

    let empty_table = create_test_table(ctx.clone(), vec![], vec![]);

    let partitions = hash_partition(&empty_table, &[0], 3).unwrap();

    assert_eq!(partitions.len(), 3);

    // All partitions should be empty
    for i in 0..3 {
        let partition = partitions.get(&i).unwrap();
        assert_eq!(partition.rows(), 0, "Partition {} should be empty", i);
        assert_eq!(partition.columns(), 2, "Should preserve schema");
    }

    println!("Hash partition empty table test passed!");
}

#[test]
fn test_hash_partition_invalid_column() {
    let ctx = Arc::new(CylonContext::new(false));

    let table = create_test_table(ctx.clone(), vec![1, 2, 3], vec!["a", "b", "c"]);

    // Try to partition on non-existent column
    let result = hash_partition(&table, &[5], 3);
    assert!(result.is_err(), "Should fail with invalid column index");

    println!("Hash partition invalid column test passed!");
}

#[test]
fn test_hash_partition_empty_columns() {
    let ctx = Arc::new(CylonContext::new(false));

    let table = create_test_table(ctx.clone(), vec![1, 2, 3], vec!["a", "b", "c"]);

    // Try to partition with no hash columns
    let result = hash_partition(&table, &[], 3);
    assert!(result.is_err(), "Should fail with empty hash columns");

    println!("Hash partition empty columns test passed!");
}

#[test]
fn test_hash_partition_zero_partitions() {
    let ctx = Arc::new(CylonContext::new(false));

    let table = create_test_table(ctx.clone(), vec![1, 2, 3], vec!["a", "b", "c"]);

    // Try to partition into 0 partitions
    let result = hash_partition(&table, &[0], 0);
    assert!(result.is_err(), "Should fail with 0 partitions");

    println!("Hash partition zero partitions test passed!");
}

#[test]
fn test_hash_partition_consistent_hashing() {
    let ctx = Arc::new(CylonContext::new(false));

    let table = create_test_table(ctx.clone(), vec![1, 2, 3, 4, 5], vec!["a", "b", "c", "d", "e"]);

    // Partition twice with same parameters
    let partitions1 = hash_partition(&table, &[0], 3).unwrap();
    let partitions2 = hash_partition(&table, &[0], 3).unwrap();

    // Verify same rows end up in same partitions
    for i in 0..3 {
        let part1 = partitions1.get(&i).unwrap();
        let part2 = partitions2.get(&i).unwrap();

        assert_eq!(part1.rows(), part2.rows(),
            "Partition {} should have same number of rows", i);

        if part1.rows() > 0 {
            let batch1 = part1.batch(0).unwrap();
            let batch2 = part2.batch(0).unwrap();

            let ids1 = batch1.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
            let ids2 = batch2.column(0).as_any().downcast_ref::<Int32Array>().unwrap();

            for j in 0..ids1.len() {
                assert_eq!(ids1.value(j), ids2.value(j),
                    "IDs should be in same order in partition {}", i);
            }
        }
    }

    println!("Hash partition consistent hashing test passed!");
}

#[test]
fn test_hash_partition_preserves_schema() {
    let ctx = Arc::new(CylonContext::new(false));

    let table = create_test_table(ctx.clone(), vec![1, 2, 3], vec!["a", "b", "c"]);
    let orig_names = table.column_names();

    let partitions = hash_partition(&table, &[0], 2).unwrap();

    // Verify all partitions have same schema as original
    for i in 0..2 {
        let partition = partitions.get(&i).unwrap();
        let part_names = partition.column_names();
        assert_eq!(orig_names, part_names, "Partition {} should preserve schema", i);
    }

    println!("Hash partition preserves schema test passed!");
}

#[test]
fn test_hash_partition_different_data_types() {
    let ctx = Arc::new(CylonContext::new(false));

    // Create table with multiple data types
    let schema = Arc::new(Schema::new(vec![
        Field::new("int_col", DataType::Int32, false),
        Field::new("long_col", DataType::Int64, false),
        Field::new("string_col", DataType::Utf8, false),
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])),
            Arc::new(Int64Array::from(vec![100, 200, 300, 400, 500])),
            Arc::new(StringArray::from(vec!["a", "b", "c", "d", "e"])),
        ],
    ).unwrap();

    let table = Table::from_record_batch(ctx.clone(), batch).unwrap();

    // Test partitioning on different column types
    let partitions_int = hash_partition(&table, &[0], 3).unwrap();
    let partitions_long = hash_partition(&table, &[1], 3).unwrap();
    let partitions_str = hash_partition(&table, &[2], 3).unwrap();

    // All should succeed and preserve row count
    for partitions in &[partitions_int, partitions_long, partitions_str] {
        let mut total = 0;
        for i in 0..3 {
            total += partitions.get(&i).unwrap().rows();
        }
        assert_eq!(total, 5, "Should preserve all rows");
    }

    println!("Hash partition different data types test passed!");
}

#[test]
fn test_hash_partition_distribution() {
    let ctx = Arc::new(CylonContext::new(false));

    // Create table with sequential IDs
    let ids: Vec<i32> = (0..100).collect();
    let names: Vec<String> = ids.iter().map(|i| format!("row{}", i)).collect();
    let name_refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(ids)),
            Arc::new(StringArray::from(name_refs)),
        ],
    ).unwrap();

    let table = Table::from_record_batch(ctx.clone(), batch).unwrap();

    // Partition into 4 partitions
    let partitions = hash_partition(&table, &[0], 4).unwrap();

    // Check that distribution is reasonably balanced
    // (Not perfectly balanced, but no partition should be empty or have > 50% of rows)
    for i in 0..4 {
        let partition = partitions.get(&i).unwrap();
        let rows = partition.rows();
        assert!(rows > 0, "Partition {} should not be empty", i);
        assert!(rows < 50, "Partition {} should not have > 50% of rows (has {})", i, rows);
    }

    println!("Hash partition distribution test passed!");
}
