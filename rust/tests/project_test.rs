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

//! Tests for Project operation
//!
//! Corresponds to C++ Project function in table.cpp

use cylon::ctx::CylonContext;
use cylon::table::{Table, project};
use std::sync::Arc;
use arrow::array::{Array, Int32Array, Int64Array, StringArray, Float64Array};
use arrow::datatypes::{Schema, Field, DataType};
use arrow::record_batch::RecordBatch;

fn create_test_table(ctx: Arc<CylonContext>) -> Table {
    // Create table with multiple columns of different types
    // id | name    | age | salary  | dept
    // ---|---------|-----|---------|------
    // 1  | Alice   | 30  | 75000.5 | Eng
    // 2  | Bob     | 25  | 65000.0 | Sales
    // 3  | Charlie | 35  | 85000.5 | Eng
    // 4  | David   | 28  | 70000.0 | HR
    // 5  | Eve     | 32  | 80000.0 | Sales

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("age", DataType::Int32, false),
        Field::new("salary", DataType::Float64, false),
        Field::new("dept", DataType::Utf8, false),
    ]));

    let id = Int32Array::from(vec![1, 2, 3, 4, 5]);
    let name = StringArray::from(vec!["Alice", "Bob", "Charlie", "David", "Eve"]);
    let age = Int32Array::from(vec![30, 25, 35, 28, 32]);
    let salary = Float64Array::from(vec![75000.5, 65000.0, 85000.5, 70000.0, 80000.0]);
    let dept = StringArray::from(vec!["Eng", "Sales", "Eng", "HR", "Sales"]);

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(id),
            Arc::new(name),
            Arc::new(age),
            Arc::new(salary),
            Arc::new(dept),
        ],
    ).unwrap();

    Table::from_record_batch(ctx, batch).unwrap()
}

#[test]
fn test_project_single_column() {
    let ctx = Arc::new(CylonContext::new(false));
    let table = create_test_table(ctx.clone());

    // Project just the name column (index 1)
    let projected = project(&table, &[1]).unwrap();

    assert_eq!(projected.rows(), 5, "Should preserve all rows");
    assert_eq!(projected.columns(), 1, "Should have 1 column");

    // Verify schema
    let col_names = projected.column_names();
    assert_eq!(col_names.len(), 1);
    assert_eq!(col_names[0], "name");

    // Verify data
    let batch = projected.batch(0).unwrap();
    let names = batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!(names.value(0), "Alice");
    assert_eq!(names.value(1), "Bob");
    assert_eq!(names.value(2), "Charlie");

    println!("Project single column test passed!");
}

#[test]
fn test_project_multiple_columns() {
    let ctx = Arc::new(CylonContext::new(false));
    let table = create_test_table(ctx.clone());

    // Project id, name, and dept columns (indices 0, 1, 4)
    let projected = project(&table, &[0, 1, 4]).unwrap();

    assert_eq!(projected.rows(), 5);
    assert_eq!(projected.columns(), 3);

    // Verify schema
    let col_names = projected.column_names();
    assert_eq!(col_names, vec!["id", "name", "dept"]);

    // Verify data
    let batch = projected.batch(0).unwrap();
    let ids = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    let names = batch.column(1).as_any().downcast_ref::<StringArray>().unwrap();
    let depts = batch.column(2).as_any().downcast_ref::<StringArray>().unwrap();

    assert_eq!(ids.value(0), 1);
    assert_eq!(names.value(0), "Alice");
    assert_eq!(depts.value(0), "Eng");

    assert_eq!(ids.value(1), 2);
    assert_eq!(names.value(1), "Bob");
    assert_eq!(depts.value(1), "Sales");

    println!("Project multiple columns test passed!");
}

#[test]
fn test_project_reorder_columns() {
    let ctx = Arc::new(CylonContext::new(false));
    let table = create_test_table(ctx.clone());

    // Project columns in different order: dept, name, id (4, 1, 0)
    let projected = project(&table, &[4, 1, 0]).unwrap();

    assert_eq!(projected.columns(), 3);

    // Verify column order changed
    let col_names = projected.column_names();
    assert_eq!(col_names, vec!["dept", "name", "id"]);

    // Verify data is in the new order
    let batch = projected.batch(0).unwrap();
    let depts = batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
    let names = batch.column(1).as_any().downcast_ref::<StringArray>().unwrap();
    let ids = batch.column(2).as_any().downcast_ref::<Int32Array>().unwrap();

    assert_eq!(depts.value(0), "Eng");
    assert_eq!(names.value(0), "Alice");
    assert_eq!(ids.value(0), 1);

    println!("Project reorder columns test passed!");
}

#[test]
fn test_project_all_columns() {
    let ctx = Arc::new(CylonContext::new(false));
    let table = create_test_table(ctx.clone());

    // Project all columns in original order
    let projected = project(&table, &[0, 1, 2, 3, 4]).unwrap();

    assert_eq!(projected.rows(), table.rows());
    assert_eq!(projected.columns(), table.columns());

    // Verify schema matches
    let orig_names = table.column_names();
    let proj_names = projected.column_names();
    assert_eq!(orig_names, proj_names);

    println!("Project all columns test passed!");
}

#[test]
fn test_project_duplicate_columns() {
    let ctx = Arc::new(CylonContext::new(false));
    let table = create_test_table(ctx.clone());

    // Project same column multiple times (id appears twice)
    let projected = project(&table, &[0, 1, 0]).unwrap();

    assert_eq!(projected.columns(), 3);

    // Both column 0 and column 2 should be 'id'
    let col_names = projected.column_names();
    assert_eq!(col_names[0], "id");
    assert_eq!(col_names[1], "name");
    assert_eq!(col_names[2], "id");

    // Verify data
    let batch = projected.batch(0).unwrap();
    let ids1 = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    let ids2 = batch.column(2).as_any().downcast_ref::<Int32Array>().unwrap();

    // Both should have same values
    for i in 0..5 {
        assert_eq!(ids1.value(i), ids2.value(i));
    }

    println!("Project duplicate columns test passed!");
}

#[test]
fn test_project_empty_column_list() {
    let ctx = Arc::new(CylonContext::new(false));
    let table = create_test_table(ctx.clone());

    // Trying to project with empty column list should fail
    let result = project(&table, &[]);
    assert!(result.is_err(), "Should fail with empty column list");

    println!("Project empty column list test passed!");
}

#[test]
fn test_project_invalid_column_index() {
    let ctx = Arc::new(CylonContext::new(false));
    let table = create_test_table(ctx.clone());

    // Table has 5 columns (indices 0-4), so index 10 is invalid
    let result = project(&table, &[0, 1, 10]);
    assert!(result.is_err(), "Should fail with invalid column index");

    println!("Project invalid column index test passed!");
}

#[test]
fn test_project_empty_table() {
    let ctx = Arc::new(CylonContext::new(false));

    // Create empty table
    let schema = Arc::new(Schema::new(vec![
        Field::new("col1", DataType::Int32, false),
        Field::new("col2", DataType::Utf8, false),
        Field::new("col3", DataType::Int32, false),
    ]));

    let col1 = Int32Array::from(Vec::<i32>::new());
    let col2 = StringArray::from(Vec::<&str>::new());
    let col3 = Int32Array::from(Vec::<i32>::new());

    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(col1), Arc::new(col2), Arc::new(col3)],
    ).unwrap();

    let table = Table::from_record_batch(ctx, batch).unwrap();

    // Project columns 0 and 2
    let projected = project(&table, &[0, 2]).unwrap();

    assert_eq!(projected.rows(), 0, "Empty table should remain empty");
    assert_eq!(projected.columns(), 2, "Should have 2 columns");

    // Verify schema
    let col_names = projected.column_names();
    assert_eq!(col_names, vec!["col1", "col3"]);

    println!("Project empty table test passed!");
}

#[test]
fn test_project_preserves_data_types() {
    let ctx = Arc::new(CylonContext::new(false));
    let table = create_test_table(ctx.clone());

    // Project columns of different types
    let projected = project(&table, &[0, 3]).unwrap(); // id (Int32), salary (Float64)

    // Verify data types are preserved
    let batch = projected.batch(0).unwrap();
    let schema = batch.schema();

    assert_eq!(schema.field(0).data_type(), &DataType::Int32);
    assert_eq!(schema.field(1).data_type(), &DataType::Float64);

    // Verify data values
    let ids = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    let salaries = batch.column(1).as_any().downcast_ref::<Float64Array>().unwrap();

    assert_eq!(ids.value(0), 1);
    assert!((salaries.value(0) - 75000.5).abs() < 0.01);

    println!("Project preserves data types test passed!");
}

#[test]
fn test_project_by_names() {
    let ctx = Arc::new(CylonContext::new(false));
    let table = create_test_table(ctx.clone());

    // Project by column names instead of indices
    let projected = table.project_by_names(&["name", "dept", "age"]).unwrap();

    assert_eq!(projected.rows(), 5);
    assert_eq!(projected.columns(), 3);

    // Verify column names and order
    let col_names = projected.column_names();
    assert_eq!(col_names, vec!["name", "dept", "age"]);

    // Verify data
    let batch = projected.batch(0).unwrap();
    let names = batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
    let depts = batch.column(1).as_any().downcast_ref::<StringArray>().unwrap();
    let ages = batch.column(2).as_any().downcast_ref::<Int32Array>().unwrap();

    assert_eq!(names.value(0), "Alice");
    assert_eq!(depts.value(0), "Eng");
    assert_eq!(ages.value(0), 30);

    println!("Project by names test passed!");
}

#[test]
fn test_project_by_invalid_name() {
    let ctx = Arc::new(CylonContext::new(false));
    let table = create_test_table(ctx.clone());

    // Try to project with a column name that doesn't exist
    let result = table.project_by_names(&["name", "invalid_column"]);
    assert!(result.is_err(), "Should fail with invalid column name");

    println!("Project by invalid name test passed!");
}

#[test]
fn test_project_multi_batch_table() {
    let ctx = Arc::new(CylonContext::new(false));

    // Create schema
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int64, false),
        Field::new("b", DataType::Int64, false),
        Field::new("c", DataType::Int64, false),
    ]));

    // Create first batch
    let batch1 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(vec![1, 2])),
            Arc::new(Int64Array::from(vec![10, 20])),
            Arc::new(Int64Array::from(vec![100, 200])),
        ],
    ).unwrap();

    // Create second batch
    let batch2 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(vec![3, 4])),
            Arc::new(Int64Array::from(vec![30, 40])),
            Arc::new(Int64Array::from(vec![300, 400])),
        ],
    ).unwrap();

    let table = Table::from_record_batches(ctx, vec![batch1, batch2]).unwrap();

    // Project columns a and c (indices 0 and 2)
    let projected = project(&table, &[0, 2]).unwrap();

    assert_eq!(projected.rows(), 4);
    assert_eq!(projected.columns(), 2);
    assert_eq!(projected.num_batches(), 2);

    // Verify column names
    let col_names = projected.column_names();
    assert_eq!(col_names, vec!["a", "c"]);

    // Verify data from both batches
    let batch1 = projected.batch(0).unwrap();
    let a1 = batch1.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
    let c1 = batch1.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
    assert_eq!(a1.value(0), 1);
    assert_eq!(c1.value(0), 100);

    let batch2 = projected.batch(1).unwrap();
    let a2 = batch2.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
    let c2 = batch2.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
    assert_eq!(a2.value(0), 3);
    assert_eq!(c2.value(0), 300);

    println!("Project multi-batch table test passed!");
}
