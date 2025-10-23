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

//! Tests for MergeSortedTable operation
//!
//! Corresponds to C++ MergeSortedTable function in table.cpp

use cylon::ctx::CylonContext;
use cylon::table::{Table, merge_sorted_table};
use std::sync::Arc;
use arrow::array::{Array, Int32Array, StringArray};
use arrow::datatypes::{Schema, Field, DataType};
use arrow::record_batch::RecordBatch;

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
fn test_merge_sorted_table_ascending() {
    let ctx = Arc::new(CylonContext::new(false));

    // Create three sorted tables
    let table1 = create_test_table(ctx.clone(), vec![1, 4, 7], vec!["a", "d", "g"]);
    let table2 = create_test_table(ctx.clone(), vec![2, 5, 8], vec!["b", "e", "h"]);
    let table3 = create_test_table(ctx.clone(), vec![3, 6, 9], vec!["c", "f", "i"]);

    // Merge sorted tables
    let merged = merge_sorted_table(&[&table1, &table2, &table3], &[0], &[true]).unwrap();

    assert_eq!(merged.rows(), 9);

    // Verify the result is sorted
    let batch = merged.batch(0).unwrap();
    let ids = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();

    for i in 0..8 {
        assert!(ids.value(i) <= ids.value(i + 1), "Result should be sorted");
    }

    // Verify all values are present
    for expected in 1..=9 {
        let found = (0..ids.len()).any(|i| ids.value(i) == expected);
        assert!(found, "Should contain value {}", expected);
    }

    println!("Merge sorted table ascending test passed!");
}

#[test]
fn test_merge_sorted_table_descending() {
    let ctx = Arc::new(CylonContext::new(false));

    // Create three sorted tables (descending)
    let table1 = create_test_table(ctx.clone(), vec![9, 6, 3], vec!["i", "f", "c"]);
    let table2 = create_test_table(ctx.clone(), vec![8, 5, 2], vec!["h", "e", "b"]);
    let table3 = create_test_table(ctx.clone(), vec![7, 4, 1], vec!["g", "d", "a"]);

    // Merge sorted tables with descending order
    let merged = merge_sorted_table(&[&table1, &table2, &table3], &[0], &[false]).unwrap();

    assert_eq!(merged.rows(), 9);

    // Verify the result is sorted (descending)
    let batch = merged.batch(0).unwrap();
    let ids = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();

    for i in 0..8 {
        assert!(ids.value(i) >= ids.value(i + 1), "Result should be sorted descending");
    }

    println!("Merge sorted table descending test passed!");
}

#[test]
fn test_merge_sorted_table_two_tables() {
    let ctx = Arc::new(CylonContext::new(false));

    let table1 = create_test_table(ctx.clone(), vec![1, 3, 5], vec!["a", "c", "e"]);
    let table2 = create_test_table(ctx.clone(), vec![2, 4, 6], vec!["b", "d", "f"]);

    let merged = merge_sorted_table(&[&table1, &table2], &[0], &[true]).unwrap();

    assert_eq!(merged.rows(), 6);

    let batch = merged.batch(0).unwrap();
    let ids = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();

    // Verify sorted order
    for i in 0..5 {
        assert!(ids.value(i) < ids.value(i + 1));
    }

    println!("Merge sorted table two tables test passed!");
}

#[test]
fn test_merge_sorted_table_single_table() {
    let ctx = Arc::new(CylonContext::new(false));

    let table = create_test_table(ctx.clone(), vec![1, 2, 3], vec!["a", "b", "c"]);

    let merged = merge_sorted_table(&[&table], &[0], &[true]).unwrap();

    assert_eq!(merged.rows(), 3);

    let batch = merged.batch(0).unwrap();
    let ids = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();

    assert_eq!(ids.value(0), 1);
    assert_eq!(ids.value(1), 2);
    assert_eq!(ids.value(2), 3);

    println!("Merge sorted table single table test passed!");
}

#[test]
fn test_merge_sorted_table_multi_column() {
    let ctx = Arc::new(CylonContext::new(false));

    // Create tables sorted by two columns
    let schema = Arc::new(Schema::new(vec![
        Field::new("group", DataType::Int32, false),
        Field::new("value", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
    ]));

    let batch1 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 1, 2])),
            Arc::new(Int32Array::from(vec![10, 30, 20])),
            Arc::new(StringArray::from(vec!["a", "c", "e"])),
        ],
    ).unwrap();

    let batch2 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 2])),
            Arc::new(Int32Array::from(vec![20, 10, 30])),
            Arc::new(StringArray::from(vec!["b", "d", "f"])),
        ],
    ).unwrap();

    let table1 = Table::from_record_batch(ctx.clone(), batch1).unwrap();
    let table2 = Table::from_record_batch(ctx.clone(), batch2).unwrap();

    // Merge sorted by group and value (both ascending)
    let merged = merge_sorted_table(&[&table1, &table2], &[0, 1], &[true, true]).unwrap();

    assert_eq!(merged.rows(), 6);

    let batch = merged.batch(0).unwrap();
    let groups = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    let values = batch.column(1).as_any().downcast_ref::<Int32Array>().unwrap();

    // Verify sorted order (group then value)
    for i in 0..5 {
        let g1 = groups.value(i);
        let g2 = groups.value(i + 1);
        let v1 = values.value(i);
        let v2 = values.value(i + 1);

        assert!(g1 < g2 || (g1 == g2 && v1 <= v2),
            "Should be sorted by group then value");
    }

    println!("Merge sorted table multi-column test passed!");
}

#[test]
fn test_merge_sorted_table_empty_tables() {
    let ctx = Arc::new(CylonContext::new(false));

    let empty1 = create_test_table(ctx.clone(), vec![], vec![]);
    let empty2 = create_test_table(ctx.clone(), vec![], vec![]);

    let merged = merge_sorted_table(&[&empty1, &empty2], &[0], &[true]).unwrap();

    assert_eq!(merged.rows(), 0);
    assert_eq!(merged.columns(), 2);

    println!("Merge sorted table empty tables test passed!");
}

#[test]
fn test_merge_sorted_table_one_empty() {
    let ctx = Arc::new(CylonContext::new(false));

    let table = create_test_table(ctx.clone(), vec![1, 2, 3], vec!["a", "b", "c"]);
    let empty = create_test_table(ctx.clone(), vec![], vec![]);

    let merged = merge_sorted_table(&[&table, &empty], &[0], &[true]).unwrap();

    assert_eq!(merged.rows(), 3);

    let batch = merged.batch(0).unwrap();
    let ids = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();

    assert_eq!(ids.value(0), 1);
    assert_eq!(ids.value(1), 2);
    assert_eq!(ids.value(2), 3);

    println!("Merge sorted table one empty test passed!");
}

#[test]
fn test_merge_sorted_table_invalid_empty_vector() {
    let result = merge_sorted_table(&[], &[0], &[true]);
    assert!(result.is_err(), "Should fail with empty tables vector");

    println!("Merge sorted table empty vector test passed!");
}

#[test]
fn test_merge_sorted_table_invalid_sort_columns() {
    let ctx = Arc::new(CylonContext::new(false));
    let table = create_test_table(ctx.clone(), vec![1, 2], vec!["a", "b"]);

    let result = merge_sorted_table(&[&table], &[], &[]);
    assert!(result.is_err(), "Should fail with empty sort columns");

    println!("Merge sorted table invalid sort columns test passed!");
}

#[test]
fn test_merge_sorted_table_mismatched_directions() {
    let ctx = Arc::new(CylonContext::new(false));
    let table = create_test_table(ctx.clone(), vec![1, 2], vec!["a", "b"]);

    let result = merge_sorted_table(&[&table], &[0, 1], &[true]);
    assert!(result.is_err(), "Should fail when sort_columns and sort_directions lengths differ");

    println!("Merge sorted table mismatched directions test passed!");
}

#[test]
fn test_merge_sorted_table_preserves_data() {
    let ctx = Arc::new(CylonContext::new(false));

    let table1 = create_test_table(ctx.clone(), vec![1, 5], vec!["one", "five"]);
    let table2 = create_test_table(ctx.clone(), vec![2, 4], vec!["two", "four"]);
    let table3 = create_test_table(ctx.clone(), vec![3], vec!["three"]);

    let merged = merge_sorted_table(&[&table1, &table2, &table3], &[0], &[true]).unwrap();

    assert_eq!(merged.rows(), 5);

    let batch = merged.batch(0).unwrap();
    let ids = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    let names = batch.column(1).as_any().downcast_ref::<StringArray>().unwrap();

    // Verify data is preserved correctly
    assert_eq!(ids.value(0), 1);
    assert_eq!(names.value(0), "one");

    assert_eq!(ids.value(2), 3);
    assert_eq!(names.value(2), "three");

    assert_eq!(ids.value(4), 5);
    assert_eq!(names.value(4), "five");

    println!("Merge sorted table preserves data test passed!");
}

#[test]
fn test_merge_sorted_table_large() {
    let ctx = Arc::new(CylonContext::new(false));

    // Create three tables with 100 rows each, already sorted
    let ids1: Vec<i32> = (0..300).step_by(3).collect();  // 0, 3, 6, ...
    let ids2: Vec<i32> = (1..300).step_by(3).collect();  // 1, 4, 7, ...
    let ids3: Vec<i32> = (2..300).step_by(3).collect();  // 2, 5, 8, ...

    let names1: Vec<String> = ids1.iter().map(|i| format!("row{}", i)).collect();
    let names2: Vec<String> = ids2.iter().map(|i| format!("row{}", i)).collect();
    let names3: Vec<String> = ids3.iter().map(|i| format!("row{}", i)).collect();

    let name_refs1: Vec<&str> = names1.iter().map(|s| s.as_str()).collect();
    let name_refs2: Vec<&str> = names2.iter().map(|s| s.as_str()).collect();
    let name_refs3: Vec<&str> = names3.iter().map(|s| s.as_str()).collect();

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
    ]));

    let batch1 = RecordBatch::try_new(schema.clone(), vec![
        Arc::new(Int32Array::from(ids1)),
        Arc::new(StringArray::from(name_refs1)),
    ]).unwrap();

    let batch2 = RecordBatch::try_new(schema.clone(), vec![
        Arc::new(Int32Array::from(ids2)),
        Arc::new(StringArray::from(name_refs2)),
    ]).unwrap();

    let batch3 = RecordBatch::try_new(schema.clone(), vec![
        Arc::new(Int32Array::from(ids3)),
        Arc::new(StringArray::from(name_refs3)),
    ]).unwrap();

    let table1 = Table::from_record_batch(ctx.clone(), batch1).unwrap();
    let table2 = Table::from_record_batch(ctx.clone(), batch2).unwrap();
    let table3 = Table::from_record_batch(ctx.clone(), batch3).unwrap();

    let merged = merge_sorted_table(&[&table1, &table2, &table3], &[0], &[true]).unwrap();

    assert_eq!(merged.rows(), 300);

    // Verify sorted order
    let batch = merged.batch(0).unwrap();
    let ids = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();

    for i in 0..299 {
        assert!(ids.value(i) < ids.value(i + 1), "Should be sorted");
    }

    println!("Merge sorted table large test passed!");
}

#[test]
fn test_merge_sorted_table_mixed_directions() {
    let ctx = Arc::new(CylonContext::new(false));

    // Create tables sorted by: group (ascending), value (descending)
    let schema = Arc::new(Schema::new(vec![
        Field::new("group", DataType::Int32, false),
        Field::new("value", DataType::Int32, false),
    ]));

    let batch1 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 1, 2])),
            Arc::new(Int32Array::from(vec![30, 10, 20])),  // descending within group
        ],
    ).unwrap();

    let batch2 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 2])),
            Arc::new(Int32Array::from(vec![20, 30, 10])),  // descending within group
        ],
    ).unwrap();

    let table1 = Table::from_record_batch(ctx.clone(), batch1).unwrap();
    let table2 = Table::from_record_batch(ctx.clone(), batch2).unwrap();

    // Merge with group ascending, value descending
    let merged = merge_sorted_table(&[&table1, &table2], &[0, 1], &[true, false]).unwrap();

    assert_eq!(merged.rows(), 6);

    let batch = merged.batch(0).unwrap();
    let groups = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    let values = batch.column(1).as_any().downcast_ref::<Int32Array>().unwrap();

    // Verify: group ascending, value descending within group
    for i in 0..5 {
        let g1 = groups.value(i);
        let g2 = groups.value(i + 1);
        let v1 = values.value(i);
        let v2 = values.value(i + 1);

        assert!(g1 < g2 || (g1 == g2 && v1 >= v2),
            "Should be sorted by group (asc) then value (desc)");
    }

    println!("Merge sorted table mixed directions test passed!");
}
