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

//! Tests for Intersect operation
//!
//! Corresponds to C++ Intersect function in table.cpp

use cylon::ctx::CylonContext;
use cylon::table::{Table, intersect};
use std::sync::Arc;
use arrow::array::{Array, Int32Array, StringArray};
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
fn test_intersect_no_overlap() {
    let ctx = Arc::new(CylonContext::new(false));

    // Table 1: {(1, "a"), (2, "b")}
    let table1 = create_test_table(ctx.clone(), vec![1, 2], vec!["a", "b"]);

    // Table 2: {(3, "c"), (4, "d")}
    let table2 = create_test_table(ctx.clone(), vec![3, 4], vec!["c", "d"]);

    let result = intersect(&table1, &table2).unwrap();

    // Should have 0 rows since there's no overlap
    assert_eq!(result.rows(), 0, "Intersect with no overlap should have 0 rows");
    assert_eq!(result.columns(), 2);

    println!("Intersect no overlap test passed!");
}

#[test]
fn test_intersect_complete_overlap() {
    let ctx = Arc::new(CylonContext::new(false));

    // Both tables have the same data: {(1, "a"), (2, "b")}
    let table1 = create_test_table(ctx.clone(), vec![1, 2], vec!["a", "b"]);
    let table2 = create_test_table(ctx.clone(), vec![1, 2], vec!["a", "b"]);

    let result = intersect(&table1, &table2).unwrap();

    // Should have 2 rows since all rows match
    assert_eq!(result.rows(), 2, "Intersect with complete overlap should have 2 rows");

    let batch = result.batch(0).unwrap();
    let ids = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    let id_set: HashSet<i32> = (0..ids.len()).map(|i| ids.value(i)).collect();

    assert!(id_set.contains(&1));
    assert!(id_set.contains(&2));

    println!("Intersect complete overlap test passed!");
}

#[test]
fn test_intersect_partial_overlap() {
    let ctx = Arc::new(CylonContext::new(false));

    // Table 1: {(1, "a"), (2, "b"), (3, "c")}
    let table1 = create_test_table(ctx.clone(), vec![1, 2, 3], vec!["a", "b", "c"]);

    // Table 2: {(2, "b"), (3, "c"), (4, "d")}
    let table2 = create_test_table(ctx.clone(), vec![2, 3, 4], vec!["b", "c", "d"]);

    let result = intersect(&table1, &table2).unwrap();

    // Should have 2 rows: (2, "b") and (3, "c")
    assert_eq!(result.rows(), 2, "Intersect with partial overlap should have 2 rows");

    let batch = result.batch(0).unwrap();
    let ids = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    let id_set: HashSet<i32> = (0..ids.len()).map(|i| ids.value(i)).collect();

    assert_eq!(id_set.len(), 2);
    assert!(id_set.contains(&2));
    assert!(id_set.contains(&3));
    assert!(!id_set.contains(&1));
    assert!(!id_set.contains(&4));

    println!("Intersect partial overlap test passed!");
}

#[test]
fn test_intersect_empty_first_table() {
    let ctx = Arc::new(CylonContext::new(false));

    let empty_table = create_test_table(ctx.clone(), vec![], vec![]);
    let table2 = create_test_table(ctx.clone(), vec![1, 2], vec!["a", "b"]);

    let result = intersect(&empty_table, &table2).unwrap();

    // Should have 0 rows
    assert_eq!(result.rows(), 0);

    println!("Intersect empty first table test passed!");
}

#[test]
fn test_intersect_empty_second_table() {
    let ctx = Arc::new(CylonContext::new(false));

    let table1 = create_test_table(ctx.clone(), vec![1, 2], vec!["a", "b"]);
    let empty_table = create_test_table(ctx.clone(), vec![], vec![]);

    let result = intersect(&table1, &empty_table).unwrap();

    // Should have 0 rows
    assert_eq!(result.rows(), 0);

    println!("Intersect empty second table test passed!");
}

#[test]
fn test_intersect_both_empty() {
    let ctx = Arc::new(CylonContext::new(false));

    let empty1 = create_test_table(ctx.clone(), vec![], vec![]);
    let empty2 = create_test_table(ctx.clone(), vec![], vec![]);

    let result = intersect(&empty1, &empty2).unwrap();

    assert_eq!(result.rows(), 0);
    assert_eq!(result.columns(), 2);

    println!("Intersect both empty test passed!");
}

#[test]
fn test_intersect_incompatible_schemas() {
    let ctx = Arc::new(CylonContext::new(false));

    // Table 1 with id and name
    let schema1 = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
    ]));

    let batch1 = RecordBatch::try_new(
        schema1,
        vec![
            Arc::new(Int32Array::from(vec![1, 2])),
            Arc::new(StringArray::from(vec!["a", "b"])),
        ],
    ).unwrap();

    let table1 = Table::from_record_batch(ctx.clone(), batch1).unwrap();

    // Table 2 with different schema
    let schema2 = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("value", DataType::Utf8, false),  // Different column name
    ]));

    let batch2 = RecordBatch::try_new(
        schema2,
        vec![
            Arc::new(Int32Array::from(vec![1, 2])),
            Arc::new(StringArray::from(vec!["a", "b"])),
        ],
    ).unwrap();

    let table2 = Table::from_record_batch(ctx.clone(), batch2).unwrap();

    // Intersect should fail due to schema mismatch
    let result = intersect(&table1, &table2);
    assert!(result.is_err(), "Should fail with incompatible schemas");

    println!("Intersect incompatible schemas test passed!");
}

#[test]
fn test_intersect_duplicate_rows() {
    let ctx = Arc::new(CylonContext::new(false));

    // Table 1 has duplicates: {(1, "a"), (1, "a"), (2, "b")}
    let table1 = create_test_table(ctx.clone(), vec![1, 1, 2], vec!["a", "a", "b"]);

    // Table 2: {(1, "a"), (3, "c")}
    let table2 = create_test_table(ctx.clone(), vec![1, 3], vec!["a", "c"]);

    let result = intersect(&table1, &table2).unwrap();

    // Should have only 1 row: (1, "a") - duplicates are removed
    assert_eq!(result.rows(), 1, "Intersect should remove duplicates");

    let batch = result.batch(0).unwrap();
    let ids = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    assert_eq!(ids.value(0), 1);

    println!("Intersect duplicate rows test passed!");
}

#[test]
fn test_intersect_single_row() {
    let ctx = Arc::new(CylonContext::new(false));

    let table1 = create_test_table(ctx.clone(), vec![1], vec!["a"]);
    let table2 = create_test_table(ctx.clone(), vec![1], vec!["a"]);

    let result = intersect(&table1, &table2).unwrap();

    assert_eq!(result.rows(), 1);

    let batch = result.batch(0).unwrap();
    let ids = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    assert_eq!(ids.value(0), 1);

    println!("Intersect single row test passed!");
}

#[test]
fn test_intersect_preserves_schema() {
    let ctx = Arc::new(CylonContext::new(false));

    let table1 = create_test_table(ctx.clone(), vec![1, 2], vec!["a", "b"]);
    let table2 = create_test_table(ctx.clone(), vec![2, 3], vec!["b", "c"]);

    let result = intersect(&table1, &table2).unwrap();

    // Verify schema is preserved
    let orig_names = table1.column_names();
    let result_names = result.column_names();
    assert_eq!(orig_names, result_names);

    println!("Intersect preserves schema test passed!");
}

#[test]
fn test_intersect_large_tables() {
    let ctx = Arc::new(CylonContext::new(false));

    // Table 1: 0-99
    let ids1: Vec<i32> = (0..100).collect();
    let names1: Vec<String> = ids1.iter().map(|i| format!("name{}", i)).collect();
    let name_refs1: Vec<&str> = names1.iter().map(|s| s.as_str()).collect();

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
    ]));

    let batch1 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(ids1)),
            Arc::new(StringArray::from(name_refs1)),
        ],
    ).unwrap();

    // Table 2: 50-149 (overlap on 50-99)
    let ids2: Vec<i32> = (50..150).collect();
    let names2: Vec<String> = ids2.iter().map(|i| format!("name{}", i)).collect();
    let name_refs2: Vec<&str> = names2.iter().map(|s| s.as_str()).collect();

    let batch2 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(ids2)),
            Arc::new(StringArray::from(name_refs2)),
        ],
    ).unwrap();

    let table1 = Table::from_record_batch(ctx.clone(), batch1).unwrap();
    let table2 = Table::from_record_batch(ctx.clone(), batch2).unwrap();

    let result = intersect(&table1, &table2).unwrap();

    // Should have 50 rows in common (50-99)
    assert_eq!(result.rows(), 50);

    // Verify the intersecting IDs are correct
    let batch = result.batch(0).unwrap();
    let ids = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    let id_set: HashSet<i32> = (0..ids.len()).map(|i| ids.value(i)).collect();

    // All IDs should be in the range 50-99
    for id in id_set.iter() {
        assert!(*id >= 50 && *id < 100, "ID {} should be in range 50-99", id);
    }

    println!("Intersect large tables test passed!");
}

#[test]
fn test_intersect_subset() {
    let ctx = Arc::new(CylonContext::new(false));

    // Table 1: {(1, "a"), (2, "b"), (3, "c"), (4, "d"), (5, "e")}
    let table1 = create_test_table(ctx.clone(), vec![1, 2, 3, 4, 5], vec!["a", "b", "c", "d", "e"]);

    // Table 2 is a subset: {(2, "b"), (4, "d")}
    let table2 = create_test_table(ctx.clone(), vec![2, 4], vec!["b", "d"]);

    let result = intersect(&table1, &table2).unwrap();

    // Should have 2 rows: (2, "b") and (4, "d")
    assert_eq!(result.rows(), 2);

    let batch = result.batch(0).unwrap();
    let ids = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    let id_set: HashSet<i32> = (0..ids.len()).map(|i| ids.value(i)).collect();

    assert!(id_set.contains(&2));
    assert!(id_set.contains(&4));

    println!("Intersect subset test passed!");
}
