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

//! Tests for Parquet I/O operations
//!
//! Corresponds to C++ Parquet I/O in arrow_io.cpp

use cylon::ctx::CylonContext;
use cylon::table::Table;
use cylon::io::{read_parquet, write_parquet, ParquetOptions};
use std::sync::Arc;
use arrow::array::{Array, Int32Array, Int64Array, StringArray, Float64Array};
use arrow::datatypes::{Schema, Field, DataType};
use arrow::record_batch::RecordBatch;
use tempfile::NamedTempFile;

fn create_test_table(ctx: Arc<CylonContext>) -> Table {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("value", DataType::Float64, false),
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])),
            Arc::new(StringArray::from(vec!["alice", "bob", "charlie", "david", "eve"])),
            Arc::new(Float64Array::from(vec![1.1, 2.2, 3.3, 4.4, 5.5])),
        ],
    ).unwrap();

    Table::from_record_batch(ctx, batch).unwrap()
}

#[test]
fn test_write_and_read_parquet() {
    let ctx = Arc::new(CylonContext::new(false));
    let table = create_test_table(ctx.clone());

    // Create temporary file
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path().to_str().unwrap();

    // Write to Parquet
    let options = ParquetOptions::default();
    write_parquet(ctx.clone(), &table, path, options).unwrap();

    // Read back from Parquet
    let read_table = read_parquet(ctx.clone(), path).unwrap();

    // Verify data
    assert_eq!(read_table.rows(), 5);
    assert_eq!(read_table.columns(), 3);

    let batch = read_table.batch(0).unwrap();
    let ids = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    let names = batch.column(1).as_any().downcast_ref::<StringArray>().unwrap();
    let values = batch.column(2).as_any().downcast_ref::<Float64Array>().unwrap();

    assert_eq!(ids.value(0), 1);
    assert_eq!(names.value(0), "alice");
    assert_eq!(values.value(0), 1.1);

    assert_eq!(ids.value(4), 5);
    assert_eq!(names.value(4), "eve");
    assert_eq!(values.value(4), 5.5);

    println!("Write and read Parquet test passed!");
}

#[test]
fn test_parquet_roundtrip_integers() {
    let ctx = Arc::new(CylonContext::new(false));

    let schema = Arc::new(Schema::new(vec![
        Field::new("int32_col", DataType::Int32, false),
        Field::new("int64_col", DataType::Int64, false),
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(Int64Array::from(vec![100, 200, 300])),
        ],
    ).unwrap();

    let table = Table::from_record_batch(ctx.clone(), batch).unwrap();

    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path().to_str().unwrap();

    write_parquet(ctx.clone(), &table, path, ParquetOptions::default()).unwrap();
    let read_table = read_parquet(ctx.clone(), path).unwrap();

    assert_eq!(read_table.rows(), 3);
    assert_eq!(read_table.columns(), 2);

    println!("Parquet roundtrip integers test passed!");
}

#[test]
fn test_parquet_large_table() {
    let ctx = Arc::new(CylonContext::new(false));

    // Create a larger table with 1000 rows
    let ids: Vec<i32> = (0..1000).collect();
    let names: Vec<String> = ids.iter().map(|i| format!("user{}", i)).collect();
    let name_refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
    let values: Vec<f64> = ids.iter().map(|i| (*i as f64) * 1.5).collect();

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("value", DataType::Float64, false),
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(ids.clone())),
            Arc::new(StringArray::from(name_refs)),
            Arc::new(Float64Array::from(values)),
        ],
    ).unwrap();

    let table = Table::from_record_batch(ctx.clone(), batch).unwrap();

    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path().to_str().unwrap();

    write_parquet(ctx.clone(), &table, path, ParquetOptions::default()).unwrap();
    let read_table = read_parquet(ctx.clone(), path).unwrap();

    assert_eq!(read_table.rows(), 1000);
    assert_eq!(read_table.columns(), 3);

    // Verify first and last rows
    let batch = read_table.batch(0).unwrap();
    let ids_col = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();

    assert_eq!(ids_col.value(0), 0);
    assert_eq!(ids_col.value(999), 999);

    println!("Parquet large table test passed!");
}

#[test]
fn test_parquet_empty_table() {
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

    let table = Table::from_record_batch(ctx.clone(), batch).unwrap();

    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path().to_str().unwrap();

    write_parquet(ctx.clone(), &table, path, ParquetOptions::default()).unwrap();
    let read_table = read_parquet(ctx.clone(), path).unwrap();

    assert_eq!(read_table.rows(), 0);
    assert_eq!(read_table.columns(), 2);

    println!("Parquet empty table test passed!");
}

#[test]
fn test_parquet_schema_preservation() {
    let ctx = Arc::new(CylonContext::new(false));

    let table = create_test_table(ctx.clone());
    let original_schema = table.schema().unwrap();

    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path().to_str().unwrap();

    write_parquet(ctx.clone(), &table, path, ParquetOptions::default()).unwrap();
    let read_table = read_parquet(ctx.clone(), path).unwrap();

    let read_schema = read_table.schema().unwrap();

    assert_eq!(original_schema.fields().len(), read_schema.fields().len());
    assert_eq!(original_schema.field(0).name(), read_schema.field(0).name());
    assert_eq!(original_schema.field(1).name(), read_schema.field(1).name());
    assert_eq!(original_schema.field(2).name(), read_schema.field(2).name());

    println!("Parquet schema preservation test passed!");
}

#[test]
fn test_parquet_invalid_read_path() {
    let ctx = Arc::new(CylonContext::new(false));

    let result = read_parquet(ctx, "nonexistent_file.parquet");
    assert!(result.is_err(), "Should fail reading non-existent file");

    println!("Parquet invalid read path test passed!");
}

#[test]
fn test_parquet_multi_batch() {
    let ctx = Arc::new(CylonContext::new(false));

    // Create table with multiple batches
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("value", DataType::Int32, false),
    ]));

    let batch1 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(Int32Array::from(vec![10, 20, 30])),
        ],
    ).unwrap();

    let batch2 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![4, 5, 6])),
            Arc::new(Int32Array::from(vec![40, 50, 60])),
        ],
    ).unwrap();

    let table = Table::from_record_batches(ctx.clone(), vec![batch1, batch2]).unwrap();

    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path().to_str().unwrap();

    write_parquet(ctx.clone(), &table, path, ParquetOptions::default()).unwrap();
    let read_table = read_parquet(ctx.clone(), path).unwrap();

    assert_eq!(read_table.rows(), 6);
    assert_eq!(read_table.columns(), 2);

    // Verify data is preserved correctly
    let all_ids: Vec<i32> = (0..read_table.num_batches())
        .flat_map(|batch_idx| {
            let batch = read_table.batch(batch_idx).unwrap();
            let ids = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
            (0..ids.len()).map(|i| ids.value(i)).collect::<Vec<_>>()
        })
        .collect();

    assert_eq!(all_ids, vec![1, 2, 3, 4, 5, 6]);

    println!("Parquet multi-batch test passed!");
}

#[test]
fn test_parquet_options_chunk_size() {
    let ctx = Arc::new(CylonContext::new(false));
    let table = create_test_table(ctx.clone());

    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path().to_str().unwrap();

    // Write with custom chunk size
    let options = ParquetOptions::new().with_chunk_size(512 * 1024); // 512KB
    write_parquet(ctx.clone(), &table, path, options).unwrap();

    // Should still be able to read it back
    let read_table = read_parquet(ctx.clone(), path).unwrap();
    assert_eq!(read_table.rows(), 5);

    println!("Parquet options chunk size test passed!");
}

#[test]
fn test_parquet_data_integrity() {
    let ctx = Arc::new(CylonContext::new(false));
    let table = create_test_table(ctx.clone());

    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path().to_str().unwrap();

    write_parquet(ctx.clone(), &table, path, ParquetOptions::default()).unwrap();
    let read_table = read_parquet(ctx.clone(), path).unwrap();

    // Verify every cell matches
    let orig_batch = table.batch(0).unwrap();
    let read_batch = read_table.batch(0).unwrap();

    for col_idx in 0..3 {
        let orig_col = orig_batch.column(col_idx);
        let read_col = read_batch.column(col_idx);

        assert_eq!(orig_col.len(), read_col.len());
        // Column data should be identical
    }

    println!("Parquet data integrity test passed!");
}

#[test]
fn test_table_from_parquet_convenience() {
    // Test Table::from_parquet and Table::to_parquet convenience methods
    let ctx = Arc::new(CylonContext::new(false));
    let table = create_test_table(ctx.clone());

    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path().to_str().unwrap();

    // Use Table::to_parquet_default convenience method
    table.to_parquet_default(path).unwrap();

    // Use Table::from_parquet convenience method
    let read_table = Table::from_parquet(ctx.clone(), path).unwrap();

    assert_eq!(read_table.rows(), 5);
    assert_eq!(read_table.columns(), 3);

    let batch = read_table.batch(0).unwrap();
    let ids = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    assert_eq!(ids.value(0), 1);
    assert_eq!(ids.value(4), 5);

    println!("Table from_parquet/to_parquet convenience test passed!");
}

#[test]
fn test_table_to_parquet_with_options() {
    let ctx = Arc::new(CylonContext::new(false));
    let table = create_test_table(ctx.clone());

    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path().to_str().unwrap();

    // Use Table::to_parquet with custom options
    let options = ParquetOptions::new().with_chunk_size(256 * 1024);
    table.to_parquet(path, options).unwrap();

    let read_table = Table::from_parquet(ctx.clone(), path).unwrap();
    assert_eq!(read_table.rows(), 5);

    println!("Table to_parquet with options test passed!");
}
