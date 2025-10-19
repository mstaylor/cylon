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

//! Sort operation tests
//! Corresponds to C++ sort operations in table.cpp

use std::sync::Arc;
use cylon::ctx::CylonContext;
use cylon::table::Table;
use arrow::array::{Int64Array, StringArray, Float64Array, Array};

fn create_unsorted_table() -> Table {
    let ctx = Arc::new(CylonContext::new(false));

    let ids = Int64Array::from(vec![5, 2, 4, 1, 3]);
    let names = StringArray::from(vec!["Eve", "Bob", "David", "Alice", "Charlie"]);
    let scores = Float64Array::from(vec![88.7, 87.3, 78.9, 95.5, 92.1]);

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
fn test_sort_single_column_ascending() {
    let table = create_unsorted_table();

    // Sort by id column (index 0) ascending
    let sorted = table.sort(0, true).unwrap();

    assert_eq!(sorted.rows(), 5);
    assert_eq!(sorted.columns(), 3);

    let batch = sorted.batch(0).unwrap();
    let ids = batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();

    // Verify ascending order
    assert_eq!(ids.value(0), 1);
    assert_eq!(ids.value(1), 2);
    assert_eq!(ids.value(2), 3);
    assert_eq!(ids.value(3), 4);
    assert_eq!(ids.value(4), 5);

    // Verify that other columns were reordered correctly
    let names = batch.column(1).as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!(names.value(0), "Alice");
    assert_eq!(names.value(1), "Bob");
    assert_eq!(names.value(2), "Charlie");
}

#[test]
fn test_sort_single_column_descending() {
    let table = create_unsorted_table();

    // Sort by id column descending
    let sorted = table.sort(0, false).unwrap();

    assert_eq!(sorted.rows(), 5);

    let batch = sorted.batch(0).unwrap();
    let ids = batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();

    // Verify descending order
    assert_eq!(ids.value(0), 5);
    assert_eq!(ids.value(1), 4);
    assert_eq!(ids.value(2), 3);
    assert_eq!(ids.value(3), 2);
    assert_eq!(ids.value(4), 1);
}

#[test]
fn test_sort_by_string_column() {
    let table = create_unsorted_table();

    // Sort by name column (index 1) ascending
    let sorted = table.sort(1, true).unwrap();

    let batch = sorted.batch(0).unwrap();
    let names = batch.column(1).as_any().downcast_ref::<StringArray>().unwrap();

    // Verify alphabetical order
    assert_eq!(names.value(0), "Alice");
    assert_eq!(names.value(1), "Bob");
    assert_eq!(names.value(2), "Charlie");
    assert_eq!(names.value(3), "David");
    assert_eq!(names.value(4), "Eve");

    // Verify corresponding ids
    let ids = batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
    assert_eq!(ids.value(0), 1); // Alice
    assert_eq!(ids.value(1), 2); // Bob
    assert_eq!(ids.value(2), 3); // Charlie
}

#[test]
fn test_sort_by_float_column() {
    let table = create_unsorted_table();

    // Sort by score column (index 2) ascending
    let sorted = table.sort(2, true).unwrap();

    let batch = sorted.batch(0).unwrap();
    let scores = batch.column(2).as_any().downcast_ref::<Float64Array>().unwrap();

    // Verify ascending order
    assert!((scores.value(0) - 78.9).abs() < 0.001);
    assert!((scores.value(1) - 87.3).abs() < 0.001);
    assert!((scores.value(2) - 88.7).abs() < 0.001);
    assert!((scores.value(3) - 92.1).abs() < 0.001);
    assert!((scores.value(4) - 95.5).abs() < 0.001);
}

#[test]
fn test_sort_empty_table() {
    let ctx = Arc::new(CylonContext::new(false));

    let ids = Int64Array::from(Vec::<i64>::new());
    let schema = Arc::new(arrow::datatypes::Schema::new(vec![
        arrow::datatypes::Field::new("id", arrow::datatypes::DataType::Int64, false),
    ]));

    let batch = arrow::record_batch::RecordBatch::try_new(
        schema,
        vec![Arc::new(ids)],
    ).unwrap();

    let table = Table::from_record_batch(ctx, batch).unwrap();

    // Sorting empty table should succeed
    let sorted = table.sort(0, true).unwrap();
    assert_eq!(sorted.rows(), 0);
}

#[test]
fn test_sort_single_row() {
    let ctx = Arc::new(CylonContext::new(false));

    let ids = Int64Array::from(vec![42]);
    let schema = Arc::new(arrow::datatypes::Schema::new(vec![
        arrow::datatypes::Field::new("id", arrow::datatypes::DataType::Int64, false),
    ]));

    let batch = arrow::record_batch::RecordBatch::try_new(
        schema,
        vec![Arc::new(ids)],
    ).unwrap();

    let table = Table::from_record_batch(ctx, batch).unwrap();

    // Sorting single row should succeed without change
    let sorted = table.sort(0, true).unwrap();
    assert_eq!(sorted.rows(), 1);

    let batch = sorted.batch(0).unwrap();
    let ids = batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
    assert_eq!(ids.value(0), 42);
}

#[test]
fn test_sort_invalid_column() {
    let table = create_unsorted_table();

    // Try to sort by column index 10 (doesn't exist)
    let result = table.sort(10, true);
    assert!(result.is_err(), "Should fail with invalid column index");
}

#[test]
fn test_sort_multi_two_columns_ascending() {
    let ctx = Arc::new(CylonContext::new(false));

    // Create table with duplicate values in first column
    let dept = StringArray::from(vec!["Sales", "Sales", "Eng", "Eng", "Sales"]);
    let salary = Int64Array::from(vec![50000, 60000, 70000, 65000, 55000]);
    let names = StringArray::from(vec!["Alice", "Bob", "Charlie", "David", "Eve"]);

    let schema = Arc::new(arrow::datatypes::Schema::new(vec![
        arrow::datatypes::Field::new("dept", arrow::datatypes::DataType::Utf8, false),
        arrow::datatypes::Field::new("salary", arrow::datatypes::DataType::Int64, false),
        arrow::datatypes::Field::new("name", arrow::datatypes::DataType::Utf8, false),
    ]));

    let batch = arrow::record_batch::RecordBatch::try_new(
        schema,
        vec![Arc::new(dept), Arc::new(salary), Arc::new(names)],
    ).unwrap();

    let table = Table::from_record_batch(ctx, batch).unwrap();

    // Sort by dept (0) then by salary (1), both ascending
    let sorted = table.sort_multi(&[0, 1], &[true, true]).unwrap();

    let batch = sorted.batch(0).unwrap();
    let dept_col = batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
    let salary_col = batch.column(1).as_any().downcast_ref::<Int64Array>().unwrap();

    // Verify: Eng dept comes first (alphabetically), then Sales
    assert_eq!(dept_col.value(0), "Eng");
    assert_eq!(dept_col.value(1), "Eng");
    assert_eq!(dept_col.value(2), "Sales");
    assert_eq!(dept_col.value(3), "Sales");
    assert_eq!(dept_col.value(4), "Sales");

    // Within Eng, verify salary is sorted ascending
    assert_eq!(salary_col.value(0), 65000); // David
    assert_eq!(salary_col.value(1), 70000); // Charlie

    // Within Sales, verify salary is sorted ascending
    assert_eq!(salary_col.value(2), 50000); // Alice
    assert_eq!(salary_col.value(3), 55000); // Eve
    assert_eq!(salary_col.value(4), 60000); // Bob
}

#[test]
fn test_sort_multi_mixed_directions() {
    let ctx = Arc::new(CylonContext::new(false));

    // Create table
    let dept = StringArray::from(vec!["Sales", "Sales", "Eng", "Eng", "Sales"]);
    let salary = Int64Array::from(vec![50000, 60000, 70000, 65000, 55000]);

    let schema = Arc::new(arrow::datatypes::Schema::new(vec![
        arrow::datatypes::Field::new("dept", arrow::datatypes::DataType::Utf8, false),
        arrow::datatypes::Field::new("salary", arrow::datatypes::DataType::Int64, false),
    ]));

    let batch = arrow::record_batch::RecordBatch::try_new(
        schema,
        vec![Arc::new(dept), Arc::new(salary)],
    ).unwrap();

    let table = Table::from_record_batch(ctx, batch).unwrap();

    // Sort by dept ascending, then by salary descending
    let sorted = table.sort_multi(&[0, 1], &[true, false]).unwrap();

    let batch = sorted.batch(0).unwrap();
    let dept_col = batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
    let salary_col = batch.column(1).as_any().downcast_ref::<Int64Array>().unwrap();

    // Verify: Eng dept comes first
    assert_eq!(dept_col.value(0), "Eng");
    assert_eq!(dept_col.value(1), "Eng");

    // Within Eng, verify salary is sorted DESCENDING
    assert_eq!(salary_col.value(0), 70000);
    assert_eq!(salary_col.value(1), 65000);

    // Verify Sales dept
    assert_eq!(dept_col.value(2), "Sales");
    assert_eq!(dept_col.value(3), "Sales");
    assert_eq!(dept_col.value(4), "Sales");

    // Within Sales, verify salary is sorted DESCENDING
    assert_eq!(salary_col.value(2), 60000);
    assert_eq!(salary_col.value(3), 55000);
    assert_eq!(salary_col.value(4), 50000);
}

#[test]
fn test_sort_multi_three_columns() {
    let ctx = Arc::new(CylonContext::new(false));

    let year = Int64Array::from(vec![2023, 2023, 2023, 2024, 2024]);
    let month = Int64Array::from(vec![3, 1, 2, 1, 2]);
    let day = Int64Array::from(vec![15, 20, 10, 5, 25]);

    let schema = Arc::new(arrow::datatypes::Schema::new(vec![
        arrow::datatypes::Field::new("year", arrow::datatypes::DataType::Int64, false),
        arrow::datatypes::Field::new("month", arrow::datatypes::DataType::Int64, false),
        arrow::datatypes::Field::new("day", arrow::datatypes::DataType::Int64, false),
    ]));

    let batch = arrow::record_batch::RecordBatch::try_new(
        schema,
        vec![Arc::new(year), Arc::new(month), Arc::new(day)],
    ).unwrap();

    let table = Table::from_record_batch(ctx, batch).unwrap();

    // Sort by year, then month, then day (all ascending)
    let sorted = table.sort_multi(&[0, 1, 2], &[true, true, true]).unwrap();

    let batch = sorted.batch(0).unwrap();
    let year_col = batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
    let month_col = batch.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
    let day_col = batch.column(2).as_any().downcast_ref::<Int64Array>().unwrap();

    // Verify chronological order
    assert_eq!(year_col.value(0), 2023);
    assert_eq!(month_col.value(0), 1);
    assert_eq!(day_col.value(0), 20);

    assert_eq!(year_col.value(1), 2023);
    assert_eq!(month_col.value(1), 2);
    assert_eq!(day_col.value(1), 10);

    assert_eq!(year_col.value(2), 2023);
    assert_eq!(month_col.value(2), 3);
    assert_eq!(day_col.value(2), 15);

    assert_eq!(year_col.value(3), 2024);
    assert_eq!(month_col.value(3), 1);
    assert_eq!(day_col.value(3), 5);

    assert_eq!(year_col.value(4), 2024);
    assert_eq!(month_col.value(4), 2);
    assert_eq!(day_col.value(4), 25);
}

#[test]
fn test_sort_multi_invalid_empty_columns() {
    let table = create_unsorted_table();

    let result = table.sort_multi(&[], &[]);
    assert!(result.is_err(), "Should fail with empty sort columns");
}

#[test]
fn test_sort_multi_mismatched_lengths() {
    let table = create_unsorted_table();

    let result = table.sort_multi(&[0, 1], &[true]);
    assert!(result.is_err(), "Should fail with mismatched array lengths");
}

#[test]
fn test_sort_multi_single_column_delegates() {
    let table = create_unsorted_table();

    // Single column via sort_multi should work same as sort
    let sorted1 = table.sort(0, true).unwrap();
    let sorted2 = table.sort_multi(&[0], &[true]).unwrap();

    // Both should have same result
    let batch1 = sorted1.batch(0).unwrap();
    let batch2 = sorted2.batch(0).unwrap();

    let ids1 = batch1.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
    let ids2 = batch2.column(0).as_any().downcast_ref::<Int64Array>().unwrap();

    for i in 0..5 {
        assert_eq!(ids1.value(i), ids2.value(i));
    }
}

#[test]
fn test_sort_preserves_schema() {
    let table = create_unsorted_table();

    let sorted = table.sort(0, true).unwrap();

    // Verify schema is preserved
    let orig_names = table.column_names();
    let sorted_names = sorted.column_names();

    assert_eq!(orig_names.len(), sorted_names.len());
    for (orig, sorted) in orig_names.iter().zip(sorted_names.iter()) {
        assert_eq!(orig, sorted);
    }
}

#[test]
fn test_sort_large_table() {
    let ctx = Arc::new(CylonContext::new(false));

    // Create a larger table with 1000 rows
    let ids: Vec<i64> = (0..1000).rev().collect(); // 999, 998, ..., 1, 0
    let values: Vec<i64> = (0..1000).map(|i| i * 2).rev().collect();

    let schema = Arc::new(arrow::datatypes::Schema::new(vec![
        arrow::datatypes::Field::new("id", arrow::datatypes::DataType::Int64, false),
        arrow::datatypes::Field::new("value", arrow::datatypes::DataType::Int64, false),
    ]));

    let batch = arrow::record_batch::RecordBatch::try_new(
        schema,
        vec![Arc::new(Int64Array::from(ids)), Arc::new(Int64Array::from(values))],
    ).unwrap();

    let table = Table::from_record_batch(ctx, batch).unwrap();

    // Sort ascending
    let sorted = table.sort(0, true).unwrap();

    // Verify first and last few values
    let batch = sorted.batch(0).unwrap();
    let ids = batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();

    assert_eq!(ids.value(0), 0);
    assert_eq!(ids.value(1), 1);
    assert_eq!(ids.value(998), 998);
    assert_eq!(ids.value(999), 999);
}
