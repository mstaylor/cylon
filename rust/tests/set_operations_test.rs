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

//! Tests for set operations (Union, Intersect, Subtract)
//!
//! Porting tests from cpp/test/table_api_test.cpp

use cylon::ctx::CylonContext;
use cylon::table::Table;
use std::sync::Arc;
use arrow::array::{Array, Int32Array, StringArray};
use arrow::datatypes::{Schema, Field, DataType};
use arrow::record_batch::RecordBatch;

fn create_test_table_1(ctx: Arc<CylonContext>) -> Table {
    // Create table: {id: [1, 2, 3, 4], name: ["A", "B", "C", "D"]}
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
    ]));

    let id = Int32Array::from(vec![1, 2, 3, 4]);
    let name = StringArray::from(vec!["A", "B", "C", "D"]);

    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(id), Arc::new(name)],
    ).unwrap();

    Table::from_record_batch(ctx, batch).unwrap()
}

fn create_test_table_2(ctx: Arc<CylonContext>) -> Table {
    // Create table: {id: [3, 4, 5, 6], name: ["C", "D", "E", "F"]}
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
    ]));

    let id = Int32Array::from(vec![3, 4, 5, 6]);
    let name = StringArray::from(vec!["C", "D", "E", "F"]);

    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(id), Arc::new(name)],
    ).unwrap();

    Table::from_record_batch(ctx, batch).unwrap()
}

#[test]
fn test_union() {
    let ctx = Arc::new(CylonContext::new(false));

    let table1 = create_test_table_1(ctx.clone());
    let table2 = create_test_table_2(ctx.clone());

    // Union should contain unique rows from both tables
    // Expected: {id: [1, 2, 3, 4, 5, 6], name: ["A", "B", "C", "D", "E", "F"]}
    let result = cylon::table::union(&table1, &table2).unwrap();

    // Check the result has 6 rows (unique rows from both tables)
    assert_eq!(result.rows(), 6, "Union should have 6 unique rows");

    // Verify the schema
    let schema = result.schema().unwrap();
    assert_eq!(schema.fields().len(), 2);
    assert_eq!(schema.field(0).name(), "id");
    assert_eq!(schema.field(1).name(), "name");

    // Extract values to verify
    let batch = result.batch(0).unwrap();
    let id_array = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    let name_array = batch.column(1).as_any().downcast_ref::<StringArray>().unwrap();

    // Collect all ids
    let mut ids: Vec<i32> = (0..id_array.len()).map(|i| id_array.value(i)).collect();
    ids.sort();

    // Should contain all unique ids: 1, 2, 3, 4, 5, 6
    assert_eq!(ids, vec![1, 2, 3, 4, 5, 6]);

    println!("Union test passed!");
}

#[test]
fn test_intersect() {
    let ctx = Arc::new(CylonContext::new(false));

    let table1 = create_test_table_1(ctx.clone());
    let table2 = create_test_table_2(ctx.clone());

    // Intersect should contain only rows that exist in both tables
    // Expected: {id: [3, 4], name: ["C", "D"]}
    let result = cylon::table::intersect(&table1, &table2).unwrap();

    // Check the result has 2 rows (rows common to both tables)
    assert_eq!(result.rows(), 2, "Intersect should have 2 common rows");

    // Verify the schema
    let schema = result.schema().unwrap();
    assert_eq!(schema.fields().len(), 2);

    // Extract values to verify
    let batch = result.batch(0).unwrap();
    let id_array = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    let name_array = batch.column(1).as_any().downcast_ref::<StringArray>().unwrap();

    // Collect all ids
    let mut ids: Vec<i32> = (0..id_array.len()).map(|i| id_array.value(i)).collect();
    ids.sort();

    // Should contain only common ids: 3, 4
    assert_eq!(ids, vec![3, 4]);

    // Verify names
    let mut names: Vec<String> = (0..name_array.len()).map(|i| name_array.value(i).to_string()).collect();
    names.sort();
    assert_eq!(names, vec!["C", "D"]);

    println!("Intersect test passed!");
}

#[test]
fn test_subtract() {
    let ctx = Arc::new(CylonContext::new(false));

    let table1 = create_test_table_1(ctx.clone());
    let table2 = create_test_table_2(ctx.clone());

    // Subtract should contain rows from table1 that are not in table2
    // Expected: {id: [1, 2], name: ["A", "B"]}
    let result = cylon::table::subtract(&table1, &table2).unwrap();

    // Check the result has 2 rows (rows in table1 but not in table2)
    assert_eq!(result.rows(), 2, "Subtract should have 2 rows");

    // Verify the schema
    let schema = result.schema().unwrap();
    assert_eq!(schema.fields().len(), 2);

    // Extract values to verify
    let batch = result.batch(0).unwrap();
    let id_array = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    let name_array = batch.column(1).as_any().downcast_ref::<StringArray>().unwrap();

    // Collect all ids
    let mut ids: Vec<i32> = (0..id_array.len()).map(|i| id_array.value(i)).collect();
    ids.sort();

    // Should contain only ids from table1 not in table2: 1, 2
    assert_eq!(ids, vec![1, 2]);

    // Verify names
    let mut names: Vec<String> = (0..name_array.len()).map(|i| name_array.value(i).to_string()).collect();
    names.sort();
    assert_eq!(names, vec!["A", "B"]);

    println!("Subtract test passed!");
}

#[test]
fn test_union_with_duplicates() {
    let ctx = Arc::new(CylonContext::new(false));

    // Create table with duplicate rows within itself
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
    ]));

    let table1 = {
        let id = Int32Array::from(vec![1, 2, 2, 3]);
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(id)]).unwrap();
        Table::from_record_batch(ctx.clone(), batch).unwrap()
    };

    let table2 = {
        let id = Int32Array::from(vec![2, 3, 3, 4]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(id)]).unwrap();
        Table::from_record_batch(ctx, batch).unwrap()
    };

    let result = cylon::table::union(&table1, &table2).unwrap();

    // Should have unique values: 1, 2, 3, 4
    assert_eq!(result.rows(), 4);

    let batch = result.batch(0).unwrap();
    let id_array = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    let mut ids: Vec<i32> = (0..id_array.len()).map(|i| id_array.value(i)).collect();
    ids.sort();

    assert_eq!(ids, vec![1, 2, 3, 4]);

    println!("Union with duplicates test passed!");
}
