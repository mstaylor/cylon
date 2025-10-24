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

//! Tests for Table print operations
//!
//! Corresponds to C++ Table::PrintToOStream (table.cpp:1233-1292)

use cylon::ctx::CylonContext;
use cylon::table::Table;
use std::sync::Arc;
use arrow::array::{Array, Int32Array, StringArray, Float64Array, BooleanArray};
use arrow::datatypes::{Schema, Field, DataType};
use arrow::record_batch::RecordBatch;

fn create_test_table(ctx: Arc<CylonContext>) -> Table {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("value", DataType::Float64, false),
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec!["alice", "bob", "charlie"])),
            Arc::new(Float64Array::from(vec![1.5, 2.5, 3.5])),
        ],
    ).unwrap();

    Table::from_record_batch(ctx, batch).unwrap()
}

#[test]
fn test_print_to_string_full_table() {
    let ctx = Arc::new(CylonContext::new(false));
    let table = create_test_table(ctx.clone());

    let output = table.print_to_string().unwrap();

    // Should have header and 3 data rows
    let lines: Vec<&str> = output.lines().collect();
    assert_eq!(lines.len(), 4); // 1 header + 3 data rows

    // Check header
    assert_eq!(lines[0], "id,name,value");

    // Check data rows
    assert_eq!(lines[1], "1,alice,1.5");
    assert_eq!(lines[2], "2,bob,2.5");
    assert_eq!(lines[3], "3,charlie,3.5");

    println!("Print to string full table test passed!");
}

#[test]
fn test_print_to_string_subset_columns() {
    let ctx = Arc::new(CylonContext::new(false));
    let table = create_test_table(ctx.clone());

    // Print only columns 0-2 (id and name)
    let output = table.print_to_string_range(0, 2, 0, 3, ',', None).unwrap();

    let lines: Vec<&str> = output.lines().collect();
    assert_eq!(lines.len(), 4);

    assert_eq!(lines[0], "id,name");
    assert_eq!(lines[1], "1,alice");
    assert_eq!(lines[2], "2,bob");
    assert_eq!(lines[3], "3,charlie");

    println!("Print to string subset columns test passed!");
}

#[test]
fn test_print_to_string_subset_rows() {
    let ctx = Arc::new(CylonContext::new(false));
    let table = create_test_table(ctx.clone());

    // Print only rows 1-2 (bob)
    let output = table.print_to_string_range(0, 3, 1, 2, ',', None).unwrap();

    let lines: Vec<&str> = output.lines().collect();
    assert_eq!(lines.len(), 2); // 1 header + 1 data row

    assert_eq!(lines[0], "id,name,value");
    assert_eq!(lines[1], "2,bob,2.5");

    println!("Print to string subset rows test passed!");
}

#[test]
fn test_print_to_string_custom_delimiter() {
    let ctx = Arc::new(CylonContext::new(false));
    let table = create_test_table(ctx.clone());

    let output = table.print_to_string_range(0, 3, 0, 3, '|', None).unwrap();

    let lines: Vec<&str> = output.lines().collect();

    assert_eq!(lines[0], "id|name|value");
    assert_eq!(lines[1], "1|alice|1.5");

    println!("Print to string custom delimiter test passed!");
}

#[test]
fn test_print_to_string_custom_headers() {
    let ctx = Arc::new(CylonContext::new(false));
    let table = create_test_table(ctx.clone());

    let custom_headers = vec![
        "ID".to_string(),
        "Name".to_string(),
        "Score".to_string(),
    ];

    let output = table.print_to_string_range(0, 3, 0, 3, ',', Some(custom_headers)).unwrap();

    let lines: Vec<&str> = output.lines().collect();
    assert_eq!(lines[0], "ID,Name,Score");
    assert_eq!(lines[1], "1,alice,1.5");

    println!("Print to string custom headers test passed!");
}

#[test]
fn test_print_to_string_invalid_column_range() {
    let ctx = Arc::new(CylonContext::new(false));
    let table = create_test_table(ctx.clone());

    // Invalid: col1 >= col2
    let result = table.print_to_string_range(2, 1, 0, 3, ',', None);
    assert!(result.is_err());

    // Invalid: col2 > num_columns
    let result = table.print_to_string_range(0, 10, 0, 3, ',', None);
    assert!(result.is_err());

    // Invalid: col1 < 0
    let result = table.print_to_string_range(-1, 2, 0, 3, ',', None);
    assert!(result.is_err());

    println!("Print to string invalid column range test passed!");
}

#[test]
fn test_print_to_string_invalid_row_range() {
    let ctx = Arc::new(CylonContext::new(false));
    let table = create_test_table(ctx.clone());

    // Invalid: row1 >= row2
    let result = table.print_to_string_range(0, 3, 2, 1, ',', None);
    assert!(result.is_err());

    // Invalid: row2 > num_rows
    let result = table.print_to_string_range(0, 3, 0, 100, ',', None);
    assert!(result.is_err());

    // Invalid: row1 < 0
    let result = table.print_to_string_range(0, 3, -1, 3, ',', None);
    assert!(result.is_err());

    println!("Print to string invalid row range test passed!");
}

#[test]
fn test_print_to_string_invalid_headers() {
    let ctx = Arc::new(CylonContext::new(false));
    let table = create_test_table(ctx.clone());

    // Wrong number of headers
    let custom_headers = vec!["ID".to_string(), "Name".to_string()]; // Only 2, need 3

    let result = table.print_to_string_range(0, 3, 0, 3, ',', Some(custom_headers));
    assert!(result.is_err());

    println!("Print to string invalid headers test passed!");
}

#[test]
fn test_print_to_string_with_nulls() {
    let ctx = Arc::new(CylonContext::new(false));

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, true),  // nullable
        Field::new("name", DataType::Utf8, true), // nullable
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![Some(1), None, Some(3)])),
            Arc::new(StringArray::from(vec![Some("alice"), Some("bob"), None])),
        ],
    ).unwrap();

    let table = Table::from_record_batch(ctx, batch).unwrap();
    let output = table.print_to_string().unwrap();

    let lines: Vec<&str> = output.lines().collect();
    assert_eq!(lines.len(), 4);

    assert_eq!(lines[0], "id,name");
    assert_eq!(lines[1], "1,alice");
    assert_eq!(lines[2], "null,bob");
    assert_eq!(lines[3], "3,null");

    println!("Print to string with nulls test passed!");
}

#[test]
fn test_print_to_string_different_types() {
    let ctx = Arc::new(CylonContext::new(false));

    let schema = Arc::new(Schema::new(vec![
        Field::new("int", DataType::Int32, false),
        Field::new("float", DataType::Float64, false),
        Field::new("string", DataType::Utf8, false),
        Field::new("bool", DataType::Boolean, false),
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![42, 100])),
            Arc::new(Float64Array::from(vec![3.14, 2.71])),
            Arc::new(StringArray::from(vec!["hello", "world"])),
            Arc::new(BooleanArray::from(vec![true, false])),
        ],
    ).unwrap();

    let table = Table::from_record_batch(ctx, batch).unwrap();
    let output = table.print_to_string().unwrap();

    let lines: Vec<&str> = output.lines().collect();
    assert_eq!(lines.len(), 3);

    assert_eq!(lines[0], "int,float,string,bool");
    assert_eq!(lines[1], "42,3.14,hello,true");
    assert_eq!(lines[2], "100,2.71,world,false");

    println!("Print to string different types test passed!");
}

#[test]
fn test_print_to_string_empty_table() {
    let ctx = Arc::new(CylonContext::new(false));

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(Vec::<i32>::new())),
            Arc::new(StringArray::from(Vec::<&str>::new())),
        ],
    ).unwrap();

    let table = Table::from_record_batch(ctx, batch).unwrap();
    let output = table.print_to_string().unwrap();

    let lines: Vec<&str> = output.lines().collect();
    assert_eq!(lines.len(), 1); // Only header

    assert_eq!(lines[0], "id,name");

    println!("Print to string empty table test passed!");
}

#[test]
fn test_print_to_string_single_column() {
    let ctx = Arc::new(CylonContext::new(false));

    let schema = Arc::new(Schema::new(vec![
        Field::new("value", DataType::Int32, false),
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(Int32Array::from(vec![10, 20, 30]))],
    ).unwrap();

    let table = Table::from_record_batch(ctx, batch).unwrap();
    let output = table.print_to_string().unwrap();

    let lines: Vec<&str> = output.lines().collect();
    assert_eq!(lines.len(), 4);

    assert_eq!(lines[0], "value");
    assert_eq!(lines[1], "10");
    assert_eq!(lines[2], "20");
    assert_eq!(lines[3], "30");

    println!("Print to string single column test passed!");
}

#[test]
fn test_print_to_string_multi_batch() {
    let ctx = Arc::new(CylonContext::new(false));

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("value", DataType::Int32, false),
    ]));

    let batch1 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2])),
            Arc::new(Int32Array::from(vec![10, 20])),
        ],
    ).unwrap();

    let batch2 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![3, 4])),
            Arc::new(Int32Array::from(vec![30, 40])),
        ],
    ).unwrap();

    let table = Table::from_record_batches(ctx, vec![batch1, batch2]).unwrap();
    let output = table.print_to_string().unwrap();

    let lines: Vec<&str> = output.lines().collect();
    assert_eq!(lines.len(), 5); // 1 header + 4 data rows

    assert_eq!(lines[0], "id,value");
    assert_eq!(lines[1], "1,10");
    assert_eq!(lines[2], "2,20");
    assert_eq!(lines[3], "3,30");
    assert_eq!(lines[4], "4,40");

    println!("Print to string multi-batch test passed!");
}
