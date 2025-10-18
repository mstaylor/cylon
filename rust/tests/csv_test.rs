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

//! CSV I/O tests

use std::sync::Arc;
use std::fs;
use cylon::ctx::CylonContext;
use cylon::table::Table;
use cylon::io::{CsvReadOptions, CsvWriteOptions};
use arrow::array::{Int64Array, StringArray, Float64Array, Array};

#[test]
fn test_write_and_read_csv() {
    let ctx = Arc::new(CylonContext::new(false));

    // Create a simple table
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

    let table = Table::from_record_batch(ctx.clone(), batch).unwrap();

    // Write to CSV
    let path = "/tmp/cylon_test_data.csv";
    table.to_csv_default(path).unwrap();

    // Read it back
    let loaded_table = Table::from_csv_default(ctx.clone(), path).unwrap();

    // Verify
    assert_eq!(loaded_table.rows(), 5, "Should have 5 rows");
    assert_eq!(loaded_table.columns(), 3, "Should have 3 columns");

    // Verify column names
    let col_names = loaded_table.column_names();
    assert_eq!(col_names[0], "id");
    assert_eq!(col_names[1], "name");
    assert_eq!(col_names[2], "score");

    // Clean up
    fs::remove_file(path).ok();
}

#[test]
fn test_csv_with_custom_delimiter() {
    let ctx = Arc::new(CylonContext::new(false));

    // Create test data
    let ids = Int64Array::from(vec![10, 20, 30]);
    let values = StringArray::from(vec!["A", "B", "C"]);

    let schema = Arc::new(arrow::datatypes::Schema::new(vec![
        arrow::datatypes::Field::new("id", arrow::datatypes::DataType::Int64, false),
        arrow::datatypes::Field::new("value", arrow::datatypes::DataType::Utf8, false),
    ]));

    let batch = arrow::record_batch::RecordBatch::try_new(
        schema,
        vec![Arc::new(ids), Arc::new(values)],
    ).unwrap();

    let table = Table::from_record_batch(ctx.clone(), batch).unwrap();

    // Write with pipe delimiter
    let path = "/tmp/cylon_test_pipe.csv";
    let write_opts = CsvWriteOptions::new().with_delimiter(b'|');
    table.to_csv(path, &write_opts).unwrap();

    // Read with pipe delimiter
    let read_opts = CsvReadOptions::new().with_delimiter(b'|');
    let loaded_table = Table::from_csv(ctx.clone(), path, &read_opts).unwrap();

    // Verify
    assert_eq!(loaded_table.rows(), 3);
    assert_eq!(loaded_table.columns(), 2);

    // Clean up
    fs::remove_file(path).ok();
}

#[test]
fn test_csv_with_column_selection() {
    let ctx = Arc::new(CylonContext::new(false));

    // First create a CSV with multiple columns
    let ids = Int64Array::from(vec![1, 2, 3]);
    let names = StringArray::from(vec!["Alice", "Bob", "Charlie"]);
    let scores = Float64Array::from(vec![95.5, 87.3, 92.1]);
    let grades = StringArray::from(vec!["A", "B", "A"]);

    let schema = Arc::new(arrow::datatypes::Schema::new(vec![
        arrow::datatypes::Field::new("id", arrow::datatypes::DataType::Int64, false),
        arrow::datatypes::Field::new("name", arrow::datatypes::DataType::Utf8, false),
        arrow::datatypes::Field::new("score", arrow::datatypes::DataType::Float64, false),
        arrow::datatypes::Field::new("grade", arrow::datatypes::DataType::Utf8, false),
    ]));

    let batch = arrow::record_batch::RecordBatch::try_new(
        schema,
        vec![Arc::new(ids), Arc::new(names), Arc::new(scores), Arc::new(grades)],
    ).unwrap();

    let table = Table::from_record_batch(ctx.clone(), batch).unwrap();

    let path = "/tmp/cylon_test_select.csv";
    table.to_csv_default(path).unwrap();

    // Read only specific columns
    let read_opts = CsvReadOptions::new()
        .with_include_columns(vec!["id".to_string(), "grade".to_string()]);
    let loaded_table = Table::from_csv(ctx.clone(), path, &read_opts).unwrap();

    // Should only have 2 columns
    assert_eq!(loaded_table.columns(), 2);
    assert_eq!(loaded_table.rows(), 3);

    let col_names = loaded_table.column_names();
    assert!(col_names.contains(&"id".to_string()));
    assert!(col_names.contains(&"grade".to_string()));

    // Clean up
    fs::remove_file(path).ok();
}

#[test]
fn test_csv_roundtrip_integers() {
    let ctx = Arc::new(CylonContext::new(false));

    let ids = Int64Array::from(vec![100, 200, 300, 400, 500]);

    let schema = Arc::new(arrow::datatypes::Schema::new(vec![
        arrow::datatypes::Field::new("id", arrow::datatypes::DataType::Int64, false),
    ]));

    let batch = arrow::record_batch::RecordBatch::try_new(
        schema,
        vec![Arc::new(ids)],
    ).unwrap();

    let table = Table::from_record_batch(ctx.clone(), batch).unwrap();

    let path = "/tmp/cylon_test_integers.csv";
    table.to_csv_default(path).unwrap();

    let loaded_table = Table::from_csv_default(ctx.clone(), path).unwrap();

    assert_eq!(loaded_table.rows(), 5);

    let loaded_batch = loaded_table.batch(0).unwrap();
    let loaded_ids = loaded_batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();

    // Verify values
    assert_eq!(loaded_ids.value(0), 100);
    assert_eq!(loaded_ids.value(1), 200);
    assert_eq!(loaded_ids.value(2), 300);
    assert_eq!(loaded_ids.value(3), 400);
    assert_eq!(loaded_ids.value(4), 500);

    // Clean up
    fs::remove_file(path).ok();
}

#[test]
fn test_csv_write_no_header() {
    let ctx = Arc::new(CylonContext::new(false));

    let values = Int64Array::from(vec![1, 2, 3]);

    let schema = Arc::new(arrow::datatypes::Schema::new(vec![
        arrow::datatypes::Field::new("value", arrow::datatypes::DataType::Int64, false),
    ]));

    let batch = arrow::record_batch::RecordBatch::try_new(
        schema,
        vec![Arc::new(values)],
    ).unwrap();

    let table = Table::from_record_batch(ctx.clone(), batch).unwrap();

    // Write without header
    let path = "/tmp/cylon_test_no_header.csv";
    let write_opts = CsvWriteOptions::new().with_header(false);
    table.to_csv(path, &write_opts).unwrap();

    // Read the file to verify no header
    let content = fs::read_to_string(path).unwrap();
    let lines: Vec<&str> = content.lines().collect();

    // First line should be data, not header
    assert!(lines[0].starts_with("1") || lines[0] == "1");

    // Clean up
    fs::remove_file(path).ok();
}

#[test]
fn test_csv_empty_table_error() {
    let ctx = Arc::new(CylonContext::new(false));

    // Try to write an empty table
    let schema = Arc::new(arrow::datatypes::Schema::new(vec![
        arrow::datatypes::Field::new("id", arrow::datatypes::DataType::Int64, false),
    ]));

    let ids = Int64Array::from(Vec::<i64>::new());
    let batch = arrow::record_batch::RecordBatch::try_new(
        schema,
        vec![Arc::new(ids)],
    ).unwrap();

    let table = Table::from_record_batch(ctx, batch).unwrap();

    let path = "/tmp/cylon_test_empty.csv";
    let result = table.to_csv_default(path);

    // Should succeed (writing empty table is ok)
    assert!(result.is_ok());

    // Clean up
    fs::remove_file(path).ok();
}
