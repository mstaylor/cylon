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

//! Tests for add_column operation
//!
//! Corresponds to C++ Table::AddColumn in table.cpp:1613-1624

use cylon::ctx::CylonContext;
use cylon::table::Table;
use std::sync::Arc;
use arrow::array::{Array, Int32Array, StringArray, Float64Array};
use arrow::datatypes::{Schema, Field, DataType};
use arrow::record_batch::RecordBatch;

fn create_test_table(ctx: Arc<CylonContext>) -> Table {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])),
            Arc::new(StringArray::from(vec!["alice", "bob", "charlie", "david", "eve"])),
        ],
    ).unwrap();

    Table::from_record_batch(ctx, batch).unwrap()
}

#[test]
fn test_add_column_at_end() {
    let ctx = Arc::new(CylonContext::new(false));
    let table = create_test_table(ctx.clone());

    // Add a new column at the end (position 2)
    let new_col = Arc::new(Float64Array::from(vec![1.1, 2.2, 3.3, 4.4, 5.5]));
    let result = table.add_column(2, "score", new_col).unwrap();

    assert_eq!(result.columns(), 3);
    assert_eq!(result.rows(), 5);

    let batch = result.batch(0).unwrap();
    assert_eq!(batch.schema().field(2).name(), "score");

    let score_col = batch.column(2).as_any().downcast_ref::<Float64Array>().unwrap();
    assert_eq!(score_col.value(0), 1.1);
    assert_eq!(score_col.value(4), 5.5);

    println!("Add column at end test passed!");
}

#[test]
fn test_add_column_at_beginning() {
    let ctx = Arc::new(CylonContext::new(false));
    let table = create_test_table(ctx.clone());

    // Add a new column at the beginning (position 0)
    let new_col = Arc::new(Int32Array::from(vec![10, 20, 30, 40, 50]));
    let result = table.add_column(0, "index", new_col).unwrap();

    assert_eq!(result.columns(), 3);
    assert_eq!(result.rows(), 5);

    let batch = result.batch(0).unwrap();
    assert_eq!(batch.schema().field(0).name(), "index");
    assert_eq!(batch.schema().field(1).name(), "id");
    assert_eq!(batch.schema().field(2).name(), "name");

    let index_col = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    assert_eq!(index_col.value(0), 10);
    assert_eq!(index_col.value(4), 50);

    println!("Add column at beginning test passed!");
}

#[test]
fn test_add_column_in_middle() {
    let ctx = Arc::new(CylonContext::new(false));
    let table = create_test_table(ctx.clone());

    // Add a new column in the middle (position 1)
    let new_col = Arc::new(StringArray::from(vec!["A", "B", "C", "D", "E"]));
    let result = table.add_column(1, "grade", new_col).unwrap();

    assert_eq!(result.columns(), 3);
    assert_eq!(result.rows(), 5);

    let batch = result.batch(0).unwrap();
    assert_eq!(batch.schema().field(0).name(), "id");
    assert_eq!(batch.schema().field(1).name(), "grade");
    assert_eq!(batch.schema().field(2).name(), "name");

    let grade_col = batch.column(1).as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!(grade_col.value(0), "A");
    assert_eq!(grade_col.value(4), "E");

    println!("Add column in middle test passed!");
}

#[test]
fn test_add_column_wrong_length() {
    let ctx = Arc::new(CylonContext::new(false));
    let table = create_test_table(ctx.clone());

    // Try to add column with wrong length
    let new_col = Arc::new(Int32Array::from(vec![1, 2, 3])); // Only 3 rows, table has 5
    let result = table.add_column(2, "wrong", new_col);

    assert!(result.is_err());
    if let Err(e) = result {
        let err_msg = e.to_string();
        assert!(err_msg.contains("must match the number of rows"));
    }

    println!("Add column wrong length test passed!");
}

#[test]
fn test_add_column_invalid_position_negative() {
    let ctx = Arc::new(CylonContext::new(false));
    let table = create_test_table(ctx.clone());

    let new_col = Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5]));
    let result = table.add_column(-1, "invalid", new_col);

    assert!(result.is_err());
    if let Err(e) = result {
        let err_msg = e.to_string();
        assert!(err_msg.contains("out of bounds"));
    }

    println!("Add column invalid position negative test passed!");
}

#[test]
fn test_add_column_invalid_position_too_large() {
    let ctx = Arc::new(CylonContext::new(false));
    let table = create_test_table(ctx.clone());

    let new_col = Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5]));
    let result = table.add_column(10, "invalid", new_col);

    assert!(result.is_err());
    if let Err(e) = result {
        let err_msg = e.to_string();
        assert!(err_msg.contains("out of bounds"));
    }

    println!("Add column invalid position too large test passed!");
}

#[test]
fn test_add_column_empty_table() {
    let ctx = Arc::new(CylonContext::new(false));

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(Int32Array::from(Vec::<i32>::new()))],
    ).unwrap();

    let table = Table::from_record_batch(ctx.clone(), batch).unwrap();

    // Add column to empty table
    let new_col = Arc::new(StringArray::from(Vec::<&str>::new()));
    let result = table.add_column(1, "name", new_col).unwrap();

    assert_eq!(result.columns(), 2);
    assert_eq!(result.rows(), 0);

    println!("Add column empty table test passed!");
}

#[test]
fn test_add_column_multiple_times() {
    let ctx = Arc::new(CylonContext::new(false));
    let table = create_test_table(ctx.clone());

    // Add first column
    let col1 = Arc::new(Float64Array::from(vec![1.1, 2.2, 3.3, 4.4, 5.5]));
    let table2 = table.add_column(2, "score", col1).unwrap();

    // Add second column
    let col2 = Arc::new(Int32Array::from(vec![100, 200, 300, 400, 500]));
    let table3 = table2.add_column(3, "points", col2).unwrap();

    assert_eq!(table3.columns(), 4);
    assert_eq!(table3.rows(), 5);

    let batch = table3.batch(0).unwrap();
    assert_eq!(batch.schema().field(0).name(), "id");
    assert_eq!(batch.schema().field(1).name(), "name");
    assert_eq!(batch.schema().field(2).name(), "score");
    assert_eq!(batch.schema().field(3).name(), "points");

    println!("Add column multiple times test passed!");
}

#[test]
fn test_add_column_different_types() {
    let ctx = Arc::new(CylonContext::new(false));
    let table = create_test_table(ctx.clone());

    // Test adding different data types
    let float_col = Arc::new(Float64Array::from(vec![1.1, 2.2, 3.3, 4.4, 5.5]));
    let result = table.add_column(2, "float_col", float_col).unwrap();
    assert_eq!(result.columns(), 3);

    let int_col = Arc::new(Int32Array::from(vec![10, 20, 30, 40, 50]));
    let result2 = result.add_column(3, "int_col", int_col).unwrap();
    assert_eq!(result2.columns(), 4);

    let str_col = Arc::new(StringArray::from(vec!["a", "b", "c", "d", "e"]));
    let result3 = result2.add_column(4, "str_col", str_col).unwrap();
    assert_eq!(result3.columns(), 5);

    println!("Add column different types test passed!");
}

#[test]
fn test_add_column_preserves_original_data() {
    let ctx = Arc::new(CylonContext::new(false));
    let table = create_test_table(ctx.clone());

    // Add a new column
    let new_col = Arc::new(Float64Array::from(vec![1.1, 2.2, 3.3, 4.4, 5.5]));
    let result = table.add_column(2, "score", new_col).unwrap();

    // Verify original columns are unchanged
    let batch = result.batch(0).unwrap();

    let id_col = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    assert_eq!(id_col.value(0), 1);
    assert_eq!(id_col.value(4), 5);

    let name_col = batch.column(1).as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!(name_col.value(0), "alice");
    assert_eq!(name_col.value(4), "eve");

    println!("Add column preserves original data test passed!");
}

#[test]
fn test_add_column_single_row() {
    let ctx = Arc::new(CylonContext::new(false));

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(Int32Array::from(vec![42]))],
    ).unwrap();

    let table = Table::from_record_batch(ctx.clone(), batch).unwrap();

    let new_col = Arc::new(StringArray::from(vec!["test"]));
    let result = table.add_column(1, "name", new_col).unwrap();

    assert_eq!(result.columns(), 2);
    assert_eq!(result.rows(), 1);

    let batch = result.batch(0).unwrap();
    let name_col = batch.column(1).as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!(name_col.value(0), "test");

    println!("Add column single row test passed!");
}
