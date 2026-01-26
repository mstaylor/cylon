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

//! Tests for arrow utility functions
//!
//! Tests sample_table_uniform, take_rows, and other arrow utilities.

use std::sync::Arc;
use arrow::array::{Int32Array, Float64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use cylon::ctx::CylonContext;
use cylon::table::Table;
use cylon::util::arrow_utils::{sample_table_uniform, take_rows};
use cylon::error::CylonResult;

/// Test sampling from an empty table preserves schema
#[test]
fn test_sample_empty_table_preserves_schema() -> CylonResult<()> {
    let ctx = Arc::new(CylonContext::new(false));

    // Create empty table with schema
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("value", DataType::Float64, false),
    ]));
    let empty_batch = RecordBatch::new_empty(schema.clone());
    let empty_table = Table::from_record_batch(ctx.clone(), empty_batch)?;

    // Sample from empty table
    let sampled = sample_table_uniform(&empty_table, 10, None)?;

    // Verify schema is preserved
    assert!(sampled.schema().is_some(), "Empty sampled table should have schema");
    let result_schema = sampled.schema().unwrap();
    assert_eq!(result_schema.fields().len(), 2);
    assert_eq!(result_schema.field(0).name(), "id");
    assert_eq!(result_schema.field(1).name(), "value");

    // Verify table is empty
    assert_eq!(sampled.rows(), 0);

    println!("Test passed: sample_empty_table_preserves_schema");
    Ok(())
}

/// Test sampling with zero samples preserves schema
#[test]
fn test_sample_zero_samples_preserves_schema() -> CylonResult<()> {
    let ctx = Arc::new(CylonContext::new(false));

    // Create table with data
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5]))],
    )?;
    let table = Table::from_record_batch(ctx.clone(), batch)?;

    // Sample with 0 samples
    let sampled = sample_table_uniform(&table, 0, None)?;

    // Verify schema is preserved
    assert!(sampled.schema().is_some(), "Zero-sampled table should have schema");
    let result_schema = sampled.schema().unwrap();
    assert_eq!(result_schema.fields().len(), 1);
    assert_eq!(result_schema.field(0).name(), "id");

    // Verify table is empty
    assert_eq!(sampled.rows(), 0);

    println!("Test passed: sample_zero_samples_preserves_schema");
    Ok(())
}

/// Test sampling with column projection preserves projected schema
#[test]
fn test_sample_empty_with_column_projection() -> CylonResult<()> {
    let ctx = Arc::new(CylonContext::new(false));

    // Create empty table with 3 columns
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int32, false),
        Field::new("b", DataType::Float64, false),
        Field::new("c", DataType::Utf8, false),
    ]));
    let empty_batch = RecordBatch::new_empty(schema.clone());
    let empty_table = Table::from_record_batch(ctx.clone(), empty_batch)?;

    // Sample with column projection (only columns 0 and 2)
    let sampled = sample_table_uniform(&empty_table, 10, Some(&[0, 2]))?;

    // Verify projected schema
    assert!(sampled.schema().is_some(), "Projected empty table should have schema");
    let result_schema = sampled.schema().unwrap();
    assert_eq!(result_schema.fields().len(), 2);
    assert_eq!(result_schema.field(0).name(), "a");
    assert_eq!(result_schema.field(1).name(), "c");

    println!("Test passed: sample_empty_with_column_projection");
    Ok(())
}

/// Test normal sampling produces correct number of samples
#[test]
fn test_sample_uniform_count() -> CylonResult<()> {
    let ctx = Arc::new(CylonContext::new(false));

    // Create table with 100 rows
    let values: Vec<i32> = (0..100).collect();
    let schema = Arc::new(Schema::new(vec![
        Field::new("value", DataType::Int32, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(values))],
    )?;
    let table = Table::from_record_batch(ctx.clone(), batch)?;

    // Sample 10 rows
    let sampled = sample_table_uniform(&table, 10, None)?;

    assert_eq!(sampled.rows(), 10);
    assert!(sampled.schema().is_some());

    println!("Test passed: sample_uniform_count");
    Ok(())
}

/// Test sampling more than available rows
#[test]
fn test_sample_more_than_available() -> CylonResult<()> {
    let ctx = Arc::new(CylonContext::new(false));

    // Create table with 5 rows
    let values: Vec<i32> = vec![1, 2, 3, 4, 5];
    let schema = Arc::new(Schema::new(vec![
        Field::new("value", DataType::Int32, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(values))],
    )?;
    let table = Table::from_record_batch(ctx.clone(), batch)?;

    // Sample 100 rows from 5-row table
    let sampled = sample_table_uniform(&table, 100, None)?;

    // Should get samples based on step calculation, not more than available
    assert!(sampled.rows() <= 100);
    assert!(sampled.schema().is_some());

    println!("Test passed: sample_more_than_available ({} rows)", sampled.rows());
    Ok(())
}

/// Test take_rows with various indices
#[test]
fn test_take_rows() -> CylonResult<()> {
    let ctx = Arc::new(CylonContext::new(false));

    // Create table
    let ids: Vec<i32> = vec![10, 20, 30, 40, 50];
    let names: Vec<&str> = vec!["a", "b", "c", "d", "e"];
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(ids)),
            Arc::new(StringArray::from(names)),
        ],
    )?;
    let table = Table::from_record_batch(ctx.clone(), batch)?;

    // Take rows 0, 2, 4
    let indices = arrow::array::Int64Array::from(vec![0, 2, 4]);
    let result = take_rows(&table, &indices)?;

    assert_eq!(result.rows(), 3);

    // Verify values
    let result_batch = result.batch(0).unwrap();
    let id_col = result_batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    assert_eq!(id_col.value(0), 10);
    assert_eq!(id_col.value(1), 30);
    assert_eq!(id_col.value(2), 50);

    let name_col = result_batch.column(1).as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!(name_col.value(0), "a");
    assert_eq!(name_col.value(1), "c");
    assert_eq!(name_col.value(2), "e");

    println!("Test passed: take_rows");
    Ok(())
}

/// Test that sampled empty table can be serialized (the original bug)
#[test]
fn test_empty_sample_serializable() -> CylonResult<()> {
    use cylon::net::serialize::serialize_table;

    let ctx = Arc::new(CylonContext::new(false));

    // Create empty table
    let schema = Arc::new(Schema::new(vec![
        Field::new("value", DataType::Int32, false),
    ]));
    let empty_batch = RecordBatch::new_empty(schema.clone());
    let empty_table = Table::from_record_batch(ctx.clone(), empty_batch)?;

    // Sample (will be empty)
    let sampled = sample_table_uniform(&empty_table, 10, None)?;

    // This was the original bug - serializing an empty sampled table would fail
    // because the table had no schema
    let serialized = serialize_table(&sampled)?;
    assert!(!serialized.is_empty(), "Serialized data should not be empty");

    println!("Test passed: empty_sample_serializable ({} bytes)", serialized.len());
    Ok(())
}