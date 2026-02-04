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

//! Tests for Slice, Head, and Tail operations
//!
//! Corresponds to C++ Slice, Head, and Tail functions in slice.cpp

use cylon::ctx::CylonContext;
use cylon::table::{Table, slice, head, tail};
use std::sync::Arc;
use arrow::array::{Array, Int32Array, StringArray};
use arrow::datatypes::{Schema, Field, DataType};
use arrow::record_batch::RecordBatch;

fn create_test_table(ctx: Arc<CylonContext>) -> Table {
    // Create table with 10 rows
    // id  | name
    // ----|------
    // 0   | row0
    // 1   | row1
    // ...
    // 9   | row9

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
    ]));

    let ids: Vec<i32> = (0..10).collect();
    let names: Vec<String> = (0..10).map(|i| format!("row{}", i)).collect();
    let name_refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(ids)),
            Arc::new(StringArray::from(name_refs)),
        ],
    ).unwrap();

    Table::from_record_batch(ctx, batch).unwrap()
}

#[test]
fn test_slice_middle() {
    let ctx = Arc::new(CylonContext::new(false));
    let table = create_test_table(ctx.clone());

    // Slice rows 3-6 (offset=3, length=4)
    let sliced = slice(&table, 3, 4).unwrap();

    assert_eq!(sliced.rows(), 4, "Should have 4 rows");
    assert_eq!(sliced.columns(), 2, "Should preserve column count");

    // Verify data
    let batch = sliced.batch(0).unwrap();
    let ids = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    let names = batch.column(1).as_any().downcast_ref::<StringArray>().unwrap();

    assert_eq!(ids.value(0), 3);
    assert_eq!(ids.value(1), 4);
    assert_eq!(ids.value(2), 5);
    assert_eq!(ids.value(3), 6);

    assert_eq!(names.value(0), "row3");
    assert_eq!(names.value(1), "row4");
    assert_eq!(names.value(2), "row5");
    assert_eq!(names.value(3), "row6");

    println!("Slice middle test passed!");
}

#[test]
fn test_slice_from_start() {
    let ctx = Arc::new(CylonContext::new(false));
    let table = create_test_table(ctx.clone());

    // Slice from start (offset=0, length=5)
    let sliced = slice(&table, 0, 5).unwrap();

    assert_eq!(sliced.rows(), 5);

    let batch = sliced.batch(0).unwrap();
    let ids = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();

    assert_eq!(ids.value(0), 0);
    assert_eq!(ids.value(4), 4);

    println!("Slice from start test passed!");
}

#[test]
fn test_slice_to_end() {
    let ctx = Arc::new(CylonContext::new(false));
    let table = create_test_table(ctx.clone());

    // Slice to end (offset=7, length=100) - should get rows 7-9
    let sliced = slice(&table, 7, 100).unwrap();

    assert_eq!(sliced.rows(), 3, "Should clamp to table size");

    let batch = sliced.batch(0).unwrap();
    let ids = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();

    assert_eq!(ids.value(0), 7);
    assert_eq!(ids.value(1), 8);
    assert_eq!(ids.value(2), 9);

    println!("Slice to end test passed!");
}

#[test]
fn test_slice_single_row() {
    let ctx = Arc::new(CylonContext::new(false));
    let table = create_test_table(ctx.clone());

    // Slice single row (offset=5, length=1)
    let sliced = slice(&table, 5, 1).unwrap();

    assert_eq!(sliced.rows(), 1);

    let batch = sliced.batch(0).unwrap();
    let ids = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    let names = batch.column(1).as_any().downcast_ref::<StringArray>().unwrap();

    assert_eq!(ids.value(0), 5);
    assert_eq!(names.value(0), "row5");

    println!("Slice single row test passed!");
}

#[test]
fn test_slice_zero_length() {
    let ctx = Arc::new(CylonContext::new(false));
    let table = create_test_table(ctx.clone());

    // Slice with zero length should return empty table
    let sliced = slice(&table, 5, 0).unwrap();

    assert_eq!(sliced.rows(), 0, "Should have 0 rows");
    assert_eq!(sliced.columns(), 2, "Should preserve columns");

    println!("Slice zero length test passed!");
}

#[test]
fn test_slice_invalid_offset() {
    let ctx = Arc::new(CylonContext::new(false));
    let table = create_test_table(ctx.clone());

    // Offset beyond table size should fail
    let result = slice(&table, 100, 5);
    assert!(result.is_err(), "Should fail with offset out of range");

    println!("Slice invalid offset test passed!");
}

#[test]
fn test_head_basic() {
    let ctx = Arc::new(CylonContext::new(false));
    let table = create_test_table(ctx.clone());

    // Get first 3 rows
    let top3 = head(&table, 3).unwrap();

    assert_eq!(top3.rows(), 3);

    let batch = top3.batch(0).unwrap();
    let ids = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    let names = batch.column(1).as_any().downcast_ref::<StringArray>().unwrap();

    assert_eq!(ids.value(0), 0);
    assert_eq!(ids.value(1), 1);
    assert_eq!(ids.value(2), 2);

    assert_eq!(names.value(0), "row0");
    assert_eq!(names.value(1), "row1");
    assert_eq!(names.value(2), "row2");

    println!("Head basic test passed!");
}

#[test]
fn test_head_more_than_table_size() {
    let ctx = Arc::new(CylonContext::new(false));
    let table = create_test_table(ctx.clone());

    // Request more rows than table has (should return all rows)
    let result = head(&table, 100).unwrap();

    assert_eq!(result.rows(), 10, "Should return all 10 rows");

    println!("Head more than table size test passed!");
}

#[test]
fn test_head_zero_rows() {
    let ctx = Arc::new(CylonContext::new(false));
    let table = create_test_table(ctx.clone());

    // Get zero rows
    let result = head(&table, 0).unwrap();

    assert_eq!(result.rows(), 0, "Should have 0 rows");
    assert_eq!(result.columns(), 2, "Should preserve columns");

    println!("Head zero rows test passed!");
}

#[test]
fn test_tail_basic() {
    let ctx = Arc::new(CylonContext::new(false));
    let table = create_test_table(ctx.clone());

    // Get last 4 rows
    let bottom4 = tail(&table, 4).unwrap();

    assert_eq!(bottom4.rows(), 4);

    let batch = bottom4.batch(0).unwrap();
    let ids = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    let names = batch.column(1).as_any().downcast_ref::<StringArray>().unwrap();

    assert_eq!(ids.value(0), 6);
    assert_eq!(ids.value(1), 7);
    assert_eq!(ids.value(2), 8);
    assert_eq!(ids.value(3), 9);

    assert_eq!(names.value(0), "row6");
    assert_eq!(names.value(3), "row9");

    println!("Tail basic test passed!");
}

#[test]
fn test_tail_more_than_table_size() {
    let ctx = Arc::new(CylonContext::new(false));
    let table = create_test_table(ctx.clone());

    // Request more rows than table has (should return all rows)
    let result = tail(&table, 100).unwrap();

    assert_eq!(result.rows(), 10, "Should return all 10 rows");

    // Verify it's the original data
    let batch = result.batch(0).unwrap();
    let ids = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    assert_eq!(ids.value(0), 0);
    assert_eq!(ids.value(9), 9);

    println!("Tail more than table size test passed!");
}

#[test]
fn test_tail_zero_rows() {
    let ctx = Arc::new(CylonContext::new(false));
    let table = create_test_table(ctx.clone());

    // Get zero rows - this will try to slice(10, 0) which should work
    let result = tail(&table, 0).unwrap();

    assert_eq!(result.rows(), 0, "Should have 0 rows");
    assert_eq!(result.columns(), 2, "Should preserve columns");

    println!("Tail zero rows test passed!");
}

#[test]
fn test_head_and_tail_together() {
    let ctx = Arc::new(CylonContext::new(false));
    let table = create_test_table(ctx.clone());

    // Get first 3 rows
    let top = head(&table, 3).unwrap();
    // Then get last 2 of those
    let result = tail(&top, 2).unwrap();

    assert_eq!(result.rows(), 2);

    let batch = result.batch(0).unwrap();
    let ids = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();

    // Should be rows 1 and 2 from original table
    assert_eq!(ids.value(0), 1);
    assert_eq!(ids.value(1), 2);

    println!("Head and tail together test passed!");
}

#[test]
fn test_slice_empty_table() {
    let ctx = Arc::new(CylonContext::new(false));

    // Create empty table
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
    ]));

    let ids = Int32Array::from(Vec::<i32>::new());
    let batch = RecordBatch::try_new(schema, vec![Arc::new(ids)]).unwrap();
    let table = Table::from_record_batch(ctx, batch).unwrap();

    // Slicing empty table should succeed and return empty table (matches C++ behavior)
    let result = slice(&table, 0, 5).unwrap();
    assert_eq!(result.rows(), 0, "Should return empty table");
    assert_eq!(result.columns(), 1, "Should preserve columns");

    println!("Slice empty table test passed!");
}

#[test]
fn test_head_empty_table() {
    let ctx = Arc::new(CylonContext::new(false));

    // Create empty table
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
    ]));

    let ids = Int32Array::from(Vec::<i32>::new());
    let batch = RecordBatch::try_new(schema, vec![Arc::new(ids)]).unwrap();
    let table = Table::from_record_batch(ctx, batch).unwrap();

    // Head on empty table should succeed and return empty table (matches C++ behavior)
    let result = head(&table, 5).unwrap();
    assert_eq!(result.rows(), 0, "Should return empty table");

    let result = head(&table, 0).unwrap();
    assert_eq!(result.rows(), 0, "Should return empty table");

    println!("Head empty table test passed!");
}

#[test]
fn test_tail_single_row_table() {
    let ctx = Arc::new(CylonContext::new(false));

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
    ]));

    let ids = Int32Array::from(vec![42]);
    let batch = RecordBatch::try_new(schema, vec![Arc::new(ids)]).unwrap();
    let table = Table::from_record_batch(ctx, batch).unwrap();

    // Get last 5 rows from single-row table (should return 1 row)
    let result = tail(&table, 5).unwrap();

    assert_eq!(result.rows(), 1);

    let batch = result.batch(0).unwrap();
    let ids = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    assert_eq!(ids.value(0), 42);

    println!("Tail single row table test passed!");
}

#[test]
fn test_slice_preserves_schema() {
    let ctx = Arc::new(CylonContext::new(false));
    let table = create_test_table(ctx.clone());

    let sliced = slice(&table, 2, 5).unwrap();

    // Verify schema is preserved
    let orig_names = table.column_names();
    let sliced_names = sliced.column_names();

    assert_eq!(orig_names, sliced_names);

    println!("Slice preserves schema test passed!");
}

#[test]
fn test_large_table_slice() {
    let ctx = Arc::new(CylonContext::new(false));

    // Create larger table with 1000 rows
    let ids: Vec<i32> = (0..1000).collect();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(Int32Array::from(ids))],
    ).unwrap();

    let table = Table::from_record_batch(ctx, batch).unwrap();

    // Slice middle 100 rows (500-599)
    let sliced = slice(&table, 500, 100).unwrap();

    assert_eq!(sliced.rows(), 100);

    let batch = sliced.batch(0).unwrap();
    let ids = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();

    assert_eq!(ids.value(0), 500);
    assert_eq!(ids.value(99), 599);

    println!("Large table slice test passed!");
}
