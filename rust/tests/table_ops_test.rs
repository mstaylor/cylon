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

//! Tests for simple table operations
//! Corresponds to C++ table operations in table.cpp

use std::sync::Arc;
use cylon::ctx::CylonContext;
use cylon::table::Table;
use arrow::array::{Int64Array, StringArray, Float64Array, Array};

fn create_test_table() -> Table {
    let ctx = Arc::new(CylonContext::new(false));

    let ids = Int64Array::from(vec![1, 2, 3, 4, 5]);
    let names = StringArray::from(vec!["Alice", "Bob", "Charlie", "David", "Eve"]);
    let scores = Float64Array::from(vec![95.5, 87.3, 92.1, 78.9, 88.7]);

    let schema = Arc::new(arrow::datatypes::Schema::new(vec![
        arrow::datatypes::Field::new("id", arrow::datatypes::DataType::Int64, false),
        arrow::datatypes::Field::new("name", arrow::datatypes::DataType::Utf8, false),
        arrow::datatypes::Field::new("score", arrow::datatypes::DataType::Float64, false),
    ]));

    let batch = arrow::record_batch::RecordBatch::try_new(
        schema,
        vec![Arc::new(ids), Arc::new(names), Arc::new(scores)],
    ).unwrap();

    Table::from_record_batch(ctx, batch).unwrap()
}

#[test]
fn test_project_by_indices() {
    let table = create_test_table();

    // Project to get only id and score columns (indices 0 and 2)
    let projected = table.project(&[0, 2]).unwrap();

    assert_eq!(projected.columns(), 2, "Should have 2 columns");
    assert_eq!(projected.rows(), 5, "Should have same number of rows");

    let col_names = projected.column_names();
    assert_eq!(col_names[0], "id");
    assert_eq!(col_names[1], "score");
}

#[test]
fn test_project_by_names() {
    let table = create_test_table();

    // Project to get only name and id columns
    let projected = table.project_by_names(&["name", "id"]).unwrap();

    assert_eq!(projected.columns(), 2);
    assert_eq!(projected.rows(), 5);

    let col_names = projected.column_names();
    assert_eq!(col_names[0], "name");
    assert_eq!(col_names[1], "id");
}

#[test]
fn test_project_single_column() {
    let table = create_test_table();

    let projected = table.project(&[1]).unwrap();

    assert_eq!(projected.columns(), 1);
    assert_eq!(projected.rows(), 5);

    let col_names = projected.column_names();
    assert_eq!(col_names[0], "name");
}

#[test]
fn test_project_invalid_index() {
    let table = create_test_table();

    // Try to project with an invalid column index
    let result = table.project(&[0, 10]);
    assert!(result.is_err(), "Should fail with invalid column index");
}

#[test]
fn test_project_invalid_name() {
    let table = create_test_table();

    let result = table.project_by_names(&["id", "nonexistent"]);
    assert!(result.is_err(), "Should fail with nonexistent column name");
}

#[test]
fn test_slice() {
    let table = create_test_table();

    // Get rows 1-3 (indices 1, 2, 3)
    let sliced = table.slice(1, 3).unwrap();

    assert_eq!(sliced.rows(), 3);
    assert_eq!(sliced.columns(), 3);

    // Verify the data
    let batch = sliced.batch(0).unwrap();
    let ids = batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();

    assert_eq!(ids.value(0), 2);
    assert_eq!(ids.value(1), 3);
    assert_eq!(ids.value(2), 4);
}

#[test]
fn test_slice_at_offset_zero() {
    let table = create_test_table();

    let sliced = table.slice(0, 2).unwrap();

    assert_eq!(sliced.rows(), 2);

    let batch = sliced.batch(0).unwrap();
    let ids = batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();

    assert_eq!(ids.value(0), 1);
    assert_eq!(ids.value(1), 2);
}

#[test]
fn test_slice_exceeds_bounds() {
    let table = create_test_table();

    // Request more rows than available - should get truncated
    let sliced = table.slice(3, 10).unwrap();

    assert_eq!(sliced.rows(), 2, "Should get only 2 rows (indices 3 and 4)");
}

#[test]
fn test_slice_invalid_offset() {
    let table = create_test_table();

    let result = table.slice(100, 5);
    assert!(result.is_err(), "Should fail with offset out of range");
}

#[test]
fn test_head() {
    let table = create_test_table();

    let head = table.head(3).unwrap();

    assert_eq!(head.rows(), 3);
    assert_eq!(head.columns(), 3);

    let batch = head.batch(0).unwrap();
    let ids = batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();

    assert_eq!(ids.value(0), 1);
    assert_eq!(ids.value(1), 2);
    assert_eq!(ids.value(2), 3);
}

#[test]
fn test_head_exceeds_table_size() {
    let table = create_test_table();

    // Request more rows than table has
    let head = table.head(10).unwrap();

    assert_eq!(head.rows(), 5, "Should get all 5 rows");
}

#[test]
fn test_tail() {
    let table = create_test_table();

    let tail = table.tail(2).unwrap();

    assert_eq!(tail.rows(), 2);
    assert_eq!(tail.columns(), 3);

    let batch = tail.batch(0).unwrap();
    let ids = batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();

    assert_eq!(ids.value(0), 4);
    assert_eq!(ids.value(1), 5);
}

#[test]
fn test_tail_exceeds_table_size() {
    let table = create_test_table();

    let tail = table.tail(10).unwrap();

    assert_eq!(tail.rows(), 5, "Should get all 5 rows");
}

#[test]
fn test_merge_two_tables() {
    let table1 = create_test_table();

    // Create a second table with same schema
    let ctx = Arc::new(CylonContext::new(false));
    let ids = Int64Array::from(vec![6, 7, 8]);
    let names = StringArray::from(vec!["Frank", "Grace", "Henry"]);
    let scores = Float64Array::from(vec![91.2, 85.5, 89.3]);

    let schema = Arc::new(arrow::datatypes::Schema::new(vec![
        arrow::datatypes::Field::new("id", arrow::datatypes::DataType::Int64, false),
        arrow::datatypes::Field::new("name", arrow::datatypes::DataType::Utf8, false),
        arrow::datatypes::Field::new("score", arrow::datatypes::DataType::Float64, false),
    ]));

    let batch = arrow::record_batch::RecordBatch::try_new(
        schema,
        vec![Arc::new(ids), Arc::new(names), Arc::new(scores)],
    ).unwrap();

    let table2 = Table::from_record_batch(ctx, batch).unwrap();

    // Merge tables
    let merged = table1.merge(&[&table2]).unwrap();

    assert_eq!(merged.rows(), 8, "Should have 5 + 3 = 8 rows");
    assert_eq!(merged.columns(), 3);

    // Verify data from both tables
    let batch1 = merged.batch(0).unwrap();
    let ids1 = batch1.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
    assert_eq!(ids1.value(0), 1);
    assert_eq!(ids1.value(4), 5);

    let batch2 = merged.batch(1).unwrap();
    let ids2 = batch2.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
    assert_eq!(ids2.value(0), 6);
    assert_eq!(ids2.value(2), 8);
}

#[test]
fn test_merge_multiple_tables() {
    let table1 = create_test_table();

    let ctx = Arc::new(CylonContext::new(false));

    // Create table2
    let ids2 = Int64Array::from(vec![6, 7]);
    let names2 = StringArray::from(vec!["Frank", "Grace"]);
    let scores2 = Float64Array::from(vec![91.2, 85.5]);

    let schema = Arc::new(arrow::datatypes::Schema::new(vec![
        arrow::datatypes::Field::new("id", arrow::datatypes::DataType::Int64, false),
        arrow::datatypes::Field::new("name", arrow::datatypes::DataType::Utf8, false),
        arrow::datatypes::Field::new("score", arrow::datatypes::DataType::Float64, false),
    ]));

    let batch2 = arrow::record_batch::RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(ids2), Arc::new(names2), Arc::new(scores2)],
    ).unwrap();

    let table2 = Table::from_record_batch(ctx.clone(), batch2).unwrap();

    // Create table3
    let ids3 = Int64Array::from(vec![8, 9, 10]);
    let names3 = StringArray::from(vec!["Henry", "Iris", "Jack"]);
    let scores3 = Float64Array::from(vec![89.3, 94.7, 82.1]);

    let batch3 = arrow::record_batch::RecordBatch::try_new(
        schema,
        vec![Arc::new(ids3), Arc::new(names3), Arc::new(scores3)],
    ).unwrap();

    let table3 = Table::from_record_batch(ctx, batch3).unwrap();

    // Merge all three tables
    let merged = table1.merge(&[&table2, &table3]).unwrap();

    assert_eq!(merged.rows(), 10, "Should have 5 + 2 + 3 = 10 rows");
    assert_eq!(merged.columns(), 3);
}

#[test]
fn test_merge_incompatible_schemas() {
    let table1 = create_test_table();

    // Create a table with different schema
    let ctx = Arc::new(CylonContext::new(false));
    let values = Int64Array::from(vec![1, 2, 3]);

    let schema = Arc::new(arrow::datatypes::Schema::new(vec![
        arrow::datatypes::Field::new("value", arrow::datatypes::DataType::Int64, false),
    ]));

    let batch = arrow::record_batch::RecordBatch::try_new(
        schema,
        vec![Arc::new(values)],
    ).unwrap();

    let table2 = Table::from_record_batch(ctx, batch).unwrap();

    // Try to merge tables with different schemas
    let result = table1.merge(&[&table2]);
    assert!(result.is_err(), "Should fail with incompatible schemas");
}

#[test]
fn test_combined_operations() {
    let table = create_test_table();

    // Chain operations: slice -> project -> head
    let result = table
        .slice(1, 4).unwrap()  // Get rows 1-4 (Bob through Eve)
        .project_by_names(&["name", "score"]).unwrap()  // Keep only name and score
        .head(2).unwrap();  // Get first 2 rows

    assert_eq!(result.rows(), 2);
    assert_eq!(result.columns(), 2);

    let col_names = result.column_names();
    assert_eq!(col_names[0], "name");
    assert_eq!(col_names[1], "score");

    let batch = result.batch(0).unwrap();
    let names = batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!(names.value(0), "Bob");
    assert_eq!(names.value(1), "Charlie");
}
