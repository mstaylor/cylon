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

//! Tests for Unique operation
//!
//! Porting tests from C++ table_api_test.cpp

use cylon::ctx::CylonContext;
use cylon::table::Table;
use std::sync::Arc;
use arrow::array::{Array, Int32Array, StringArray};
use arrow::datatypes::{Schema, Field, DataType};
use arrow::record_batch::RecordBatch;

#[test]
fn test_unique_keep_first() {
    let ctx = Arc::new(CylonContext::new(false));

    // Create table with duplicates: {id: [1, 2, 2, 3, 3, 3, 4], name: ["A", "B", "B", "C", "C", "C", "D"]}
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
    ]));

    let id = Int32Array::from(vec![1, 2, 2, 3, 3, 3, 4]);
    let name = StringArray::from(vec!["A", "B", "B", "C", "C", "C", "D"]);

    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(id), Arc::new(name)],
    ).unwrap();

    let table = Table::from_record_batch(ctx, batch).unwrap();

    // Get unique rows based on all columns, keeping first occurrence
    let col_indices = vec![0, 1]; // id and name columns
    let result = cylon::table::unique(&table, &col_indices, true).unwrap();

    // Should have 4 unique rows: (1,A), (2,B), (3,C), (4,D)
    assert_eq!(result.rows(), 4, "Unique should return 4 rows");

    let result_batch = result.batch(0).unwrap();
    let result_ids = result_batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    let result_names = result_batch.column(1).as_any().downcast_ref::<StringArray>().unwrap();

    // Verify the unique values
    let ids: Vec<i32> = (0..result_ids.len()).map(|i| result_ids.value(i)).collect();
    assert_eq!(ids, vec![1, 2, 3, 4]);

    let names: Vec<String> = (0..result_names.len()).map(|i| result_names.value(i).to_string()).collect();
    assert_eq!(names, vec!["A", "B", "C", "D"]);

    println!("Unique keep_first test passed!");
}

#[test]
fn test_unique_single_column() {
    let ctx = Arc::new(CylonContext::new(false));

    // Create table: {id: [1, 2, 2, 3], value: [10, 20, 30, 40]}
    // When checking uniqueness only on 'id' column, should keep rows with unique ids
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("value", DataType::Int32, false),
    ]));

    let id = Int32Array::from(vec![1, 2, 2, 3]);
    let value = Int32Array::from(vec![10, 20, 30, 40]);

    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(id), Arc::new(value)],
    ).unwrap();

    let table = Table::from_record_batch(ctx, batch).unwrap();

    // Get unique rows based only on 'id' column (index 0)
    let col_indices = vec![0];
    let result = cylon::table::unique(&table, &col_indices, true).unwrap();

    // Should have 3 unique rows based on id: 1, 2, 3
    assert_eq!(result.rows(), 3, "Unique should return 3 rows");

    let result_batch = result.batch(0).unwrap();
    let result_ids = result_batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();

    // Verify the unique id values
    let ids: Vec<i32> = (0..result_ids.len()).map(|i| result_ids.value(i)).collect();
    assert_eq!(ids, vec![1, 2, 3]);

    // When keeping first, the values should be [10, 20, 40] (first occurrence of each id)
    let result_values = result_batch.column(1).as_any().downcast_ref::<Int32Array>().unwrap();
    let values: Vec<i32> = (0..result_values.len()).map(|i| result_values.value(i)).collect();
    assert_eq!(values, vec![10, 20, 40]);

    println!("Unique single column test passed!");
}

#[test]
fn test_unique_all_unique() {
    let ctx = Arc::new(CylonContext::new(false));

    // Create table with all unique rows
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
    ]));

    let id = Int32Array::from(vec![1, 2, 3, 4, 5]);
    let batch = RecordBatch::try_new(schema, vec![Arc::new(id)]).unwrap();
    let table = Table::from_record_batch(ctx, batch).unwrap();

    let col_indices = vec![0];
    let result = cylon::table::unique(&table, &col_indices, true).unwrap();

    // Should return all rows since they're all unique
    assert_eq!(result.rows(), 5);

    println!("Unique all unique test passed!");
}

#[test]
fn test_unique_all_duplicates() {
    let ctx = Arc::new(CylonContext::new(false));

    // Create table with all duplicate rows
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
    ]));

    let id = Int32Array::from(vec![1, 1, 1, 1, 1]);
    let batch = RecordBatch::try_new(schema, vec![Arc::new(id)]).unwrap();
    let table = Table::from_record_batch(ctx, batch).unwrap();

    let col_indices = vec![0];
    let result = cylon::table::unique(&table, &col_indices, true).unwrap();

    // Should return only 1 row
    assert_eq!(result.rows(), 1);

    let result_batch = result.batch(0).unwrap();
    let result_ids = result_batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    assert_eq!(result_ids.value(0), 1);

    println!("Unique all duplicates test passed!");
}

#[test]
fn test_unique_empty_table() {
    let ctx = Arc::new(CylonContext::new(false));

    // Create empty table
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
    ]));

    let id = Int32Array::from(Vec::<i32>::new());
    let batch = RecordBatch::try_new(schema, vec![Arc::new(id)]).unwrap();
    let table = Table::from_record_batch(ctx, batch).unwrap();

    let col_indices = vec![0];
    let result = cylon::table::unique(&table, &col_indices, true).unwrap();

    // Empty table should return empty result
    assert_eq!(result.rows(), 0);

    println!("Unique empty table test passed!");
}

#[test]
fn test_unique_multi_column() {
    let ctx = Arc::new(CylonContext::new(false));

    // Create table where some rows are duplicates on column1 but unique when considering both columns
    let schema = Arc::new(Schema::new(vec![
        Field::new("col1", DataType::Int32, false),
        Field::new("col2", DataType::Int32, false),
    ]));

    let col1 = Int32Array::from(vec![1, 1, 2, 2, 3]);
    let col2 = Int32Array::from(vec![10, 20, 10, 10, 30]);

    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(col1), Arc::new(col2)],
    ).unwrap();

    let table = Table::from_record_batch(ctx, batch).unwrap();

    // Get unique based on both columns
    let col_indices = vec![0, 1];
    let result = cylon::table::unique(&table, &col_indices, true).unwrap();

    // Should have 4 unique combinations: (1,10), (1,20), (2,10), (3,30)
    assert_eq!(result.rows(), 4);

    println!("Unique multi-column test passed!");
}
