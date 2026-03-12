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

//! Tests for Equals operation
//!
//! Corresponds to C++ Equals function in table.cpp

use cylon::ctx::CylonContext;
use cylon::table::{Table, equals};
use std::sync::Arc;
use arrow::array::{Array, Int32Array, StringArray, Float64Array};
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
fn test_equals_identical_ordered() {
    let ctx = Arc::new(CylonContext::new(false));

    let table1 = create_test_table(ctx.clone(), vec![1, 2, 3], vec!["a", "b", "c"]);
    let table2 = create_test_table(ctx.clone(), vec![1, 2, 3], vec!["a", "b", "c"]);

    let result = equals(&table1, &table2, true).unwrap();
    assert!(result, "Identical tables should be equal (ordered)");

    println!("Equals identical ordered test passed!");
}

#[test]
fn test_equals_identical_unordered() {
    let ctx = Arc::new(CylonContext::new(false));

    let table1 = create_test_table(ctx.clone(), vec![1, 2, 3], vec!["a", "b", "c"]);
    let table2 = create_test_table(ctx.clone(), vec![1, 2, 3], vec!["a", "b", "c"]);

    let result = equals(&table1, &table2, false).unwrap();
    assert!(result, "Identical tables should be equal (unordered)");

    println!("Equals identical unordered test passed!");
}

#[test]
fn test_equals_different_order_ordered() {
    let ctx = Arc::new(CylonContext::new(false));

    // Same data, different order
    let table1 = create_test_table(ctx.clone(), vec![1, 2, 3], vec!["a", "b", "c"]);
    let table2 = create_test_table(ctx.clone(), vec![3, 1, 2], vec!["c", "a", "b"]);

    let result = equals(&table1, &table2, true).unwrap();
    assert!(!result, "Tables with different order should not be equal (ordered comparison)");

    println!("Equals different order ordered test passed!");
}

#[test]
fn test_equals_different_order_unordered() {
    let ctx = Arc::new(CylonContext::new(false));

    // Same data, different order
    let table1 = create_test_table(ctx.clone(), vec![1, 2, 3], vec!["a", "b", "c"]);
    let table2 = create_test_table(ctx.clone(), vec![3, 1, 2], vec!["c", "a", "b"]);

    let result = equals(&table1, &table2, false).unwrap();
    assert!(result, "Tables with same data but different order should be equal (unordered comparison)");

    println!("Equals different order unordered test passed!");
}

#[test]
fn test_equals_different_data() {
    let ctx = Arc::new(CylonContext::new(false));

    let table1 = create_test_table(ctx.clone(), vec![1, 2, 3], vec!["a", "b", "c"]);
    let table2 = create_test_table(ctx.clone(), vec![1, 2, 4], vec!["a", "b", "d"]);

    let result_ordered = equals(&table1, &table2, true).unwrap();
    assert!(!result_ordered, "Tables with different data should not be equal (ordered)");

    let result_unordered = equals(&table1, &table2, false).unwrap();
    assert!(!result_unordered, "Tables with different data should not be equal (unordered)");

    println!("Equals different data test passed!");
}

#[test]
fn test_equals_different_row_count() {
    let ctx = Arc::new(CylonContext::new(false));

    let table1 = create_test_table(ctx.clone(), vec![1, 2, 3], vec!["a", "b", "c"]);
    let table2 = create_test_table(ctx.clone(), vec![1, 2], vec!["a", "b"]);

    let result = equals(&table1, &table2, true).unwrap();
    assert!(!result, "Tables with different row counts should not be equal");

    println!("Equals different row count test passed!");
}

#[test]
fn test_equals_different_column_count() {
    let ctx = Arc::new(CylonContext::new(false));

    let table1 = create_test_table(ctx.clone(), vec![1, 2], vec!["a", "b"]);

    // Table with 3 columns
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("value", DataType::Int32, false),
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![1, 2])),
            Arc::new(StringArray::from(vec!["a", "b"])),
            Arc::new(Int32Array::from(vec![10, 20])),
        ],
    ).unwrap();

    let table2 = Table::from_record_batch(ctx.clone(), batch).unwrap();

    let result = equals(&table1, &table2, true).unwrap();
    assert!(!result, "Tables with different column counts should not be equal");

    println!("Equals different column count test passed!");
}

#[test]
fn test_equals_different_schema() {
    let ctx = Arc::new(CylonContext::new(false));

    let table1 = create_test_table(ctx.clone(), vec![1, 2], vec!["a", "b"]);

    // Table with different schema (different column names)
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("value", DataType::Utf8, false),  // Different name
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![1, 2])),
            Arc::new(StringArray::from(vec!["a", "b"])),
        ],
    ).unwrap();

    let table2 = Table::from_record_batch(ctx.clone(), batch).unwrap();

    let result = equals(&table1, &table2, true).unwrap();
    assert!(!result, "Tables with different schemas should not be equal");

    println!("Equals different schema test passed!");
}

#[test]
fn test_equals_empty_tables() {
    let ctx = Arc::new(CylonContext::new(false));

    let empty1 = create_test_table(ctx.clone(), vec![], vec![]);
    let empty2 = create_test_table(ctx.clone(), vec![], vec![]);

    let result = equals(&empty1, &empty2, true).unwrap();
    assert!(result, "Empty tables with same schema should be equal");

    println!("Equals empty tables test passed!");
}

#[test]
fn test_equals_one_empty_one_not() {
    let ctx = Arc::new(CylonContext::new(false));

    let empty_table = create_test_table(ctx.clone(), vec![], vec![]);
    let non_empty = create_test_table(ctx.clone(), vec![1], vec!["a"]);

    let result = equals(&empty_table, &non_empty, true).unwrap();
    assert!(!result, "Empty and non-empty tables should not be equal");

    println!("Equals one empty one not test passed!");
}

#[test]
fn test_equals_large_tables_ordered() {
    let ctx = Arc::new(CylonContext::new(false));

    // Create two large identical tables
    let ids: Vec<i32> = (0..1000).collect();
    let names: Vec<String> = ids.iter().map(|i| format!("row{}", i)).collect();
    let name_refs1: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
    let name_refs2: Vec<&str> = names.iter().map(|s| s.as_str()).collect();

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
    ]));

    let batch1 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(ids.clone())),
            Arc::new(StringArray::from(name_refs1)),
        ],
    ).unwrap();

    let batch2 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(ids)),
            Arc::new(StringArray::from(name_refs2)),
        ],
    ).unwrap();

    let table1 = Table::from_record_batch(ctx.clone(), batch1).unwrap();
    let table2 = Table::from_record_batch(ctx.clone(), batch2).unwrap();

    let result = equals(&table1, &table2, true).unwrap();
    assert!(result, "Large identical tables should be equal");

    println!("Equals large tables ordered test passed!");
}

#[test]
fn test_equals_single_row_difference() {
    let ctx = Arc::new(CylonContext::new(false));

    let table1 = create_test_table(ctx.clone(), vec![1, 2, 3, 4, 5], vec!["a", "b", "c", "d", "e"]);
    let table2 = create_test_table(ctx.clone(), vec![1, 2, 3, 4, 6], vec!["a", "b", "c", "d", "f"]);

    let result = equals(&table1, &table2, true).unwrap();
    assert!(!result, "Tables differing by one row should not be equal");

    println!("Equals single row difference test passed!");
}

#[test]
fn test_equals_multi_column_unordered() {
    let ctx = Arc::new(CylonContext::new(false));

    // Create tables with 3 columns
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("value", DataType::Float64, false),
        Field::new("name", DataType::Utf8, false),
    ]));

    let batch1 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(Float64Array::from(vec![1.1, 2.2, 3.3])),
            Arc::new(StringArray::from(vec!["a", "b", "c"])),
        ],
    ).unwrap();

    // Same data, different order
    let batch2 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![3, 1, 2])),
            Arc::new(Float64Array::from(vec![3.3, 1.1, 2.2])),
            Arc::new(StringArray::from(vec!["c", "a", "b"])),
        ],
    ).unwrap();

    let table1 = Table::from_record_batch(ctx.clone(), batch1).unwrap();
    let table2 = Table::from_record_batch(ctx.clone(), batch2).unwrap();

    let result = equals(&table1, &table2, false).unwrap();
    assert!(result, "Multi-column tables with same data should be equal (unordered)");

    println!("Equals multi-column unordered test passed!");
}

#[test]
fn test_equals_duplicate_rows_ordered() {
    let ctx = Arc::new(CylonContext::new(false));

    // Table with duplicate rows
    let table1 = create_test_table(ctx.clone(), vec![1, 2, 2, 3], vec!["a", "b", "b", "c"]);
    let table2 = create_test_table(ctx.clone(), vec![1, 2, 2, 3], vec!["a", "b", "b", "c"]);

    let result = equals(&table1, &table2, true).unwrap();
    assert!(result, "Tables with identical duplicate rows should be equal");

    println!("Equals duplicate rows ordered test passed!");
}

#[test]
fn test_equals_duplicate_rows_different_count() {
    let ctx = Arc::new(CylonContext::new(false));

    // Different number of duplicates
    let table1 = create_test_table(ctx.clone(), vec![1, 2, 2, 3], vec!["a", "b", "b", "c"]);
    let table2 = create_test_table(ctx.clone(), vec![1, 2, 3], vec!["a", "b", "c"]);

    let result = equals(&table1, &table2, true).unwrap();
    assert!(!result, "Tables with different duplicate counts should not be equal");

    println!("Equals duplicate rows different count test passed!");
}

#[test]
fn test_equals_self() {
    let ctx = Arc::new(CylonContext::new(false));

    let table = create_test_table(ctx.clone(), vec![1, 2, 3], vec!["a", "b", "c"]);

    let result = equals(&table, &table, true).unwrap();
    assert!(result, "Table should equal itself");

    println!("Equals self test passed!");
}
