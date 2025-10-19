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

//! Tests for Merge operation
//!
//! Corresponds to C++ Merge function in table.cpp

use cylon::ctx::CylonContext;
use cylon::table::{Table, merge};
use std::sync::Arc;
use arrow::array::{Array, Int32Array, Int64Array, StringArray, Float64Array};
use arrow::datatypes::{Schema, Field, DataType};
use arrow::record_batch::RecordBatch;

fn create_test_table(ctx: Arc<CylonContext>, id_start: i32, num_rows: i32) -> Table {
    // Create table with id and name columns
    // id  | name
    // ----|------
    // id_start   | row{id_start}
    // id_start+1 | row{id_start+1}
    // ...

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
    ]));

    let ids: Vec<i32> = (id_start..id_start + num_rows).collect();
    let names: Vec<String> = ids.iter().map(|i| format!("row{}", i)).collect();
    let name_refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(ids)),
            Arc::new(StringArray::from(name_refs)),
        ],
    ).unwrap();

    Table::from_record_batch(ctx, batch).unwrap()
}

#[test]
fn test_merge_two_tables() {
    let ctx = Arc::new(CylonContext::new(false));

    // Create two tables with 5 rows each
    let table1 = create_test_table(ctx.clone(), 0, 5);
    let table2 = create_test_table(ctx.clone(), 5, 5);

    // Merge them
    let merged = merge(&[&table1, &table2]).unwrap();

    assert_eq!(merged.rows(), 10, "Should have 10 rows total");
    assert_eq!(merged.columns(), 2, "Should preserve column count");

    // Verify data from first table
    let batch0 = merged.batch(0).unwrap();
    let ids = batch0.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    assert_eq!(ids.value(0), 0);
    assert_eq!(ids.value(4), 4);

    // Verify data from second table (in second batch or same batch depending on implementation)
    let all_ids = (0..merged.num_batches())
        .flat_map(|i| {
            let batch = merged.batch(i).unwrap();
            let ids = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
            (0..ids.len()).map(|j| ids.value(j)).collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    assert_eq!(all_ids.len(), 10);
    assert_eq!(all_ids[0], 0);
    assert_eq!(all_ids[9], 9);

    println!("Merge two tables test passed!");
}

#[test]
fn test_merge_three_tables() {
    let ctx = Arc::new(CylonContext::new(false));

    let table1 = create_test_table(ctx.clone(), 0, 3);
    let table2 = create_test_table(ctx.clone(), 3, 3);
    let table3 = create_test_table(ctx.clone(), 6, 3);

    let merged = merge(&[&table1, &table2, &table3]).unwrap();

    assert_eq!(merged.rows(), 9);
    assert_eq!(merged.columns(), 2);

    // Verify all IDs are present
    let all_ids = (0..merged.num_batches())
        .flat_map(|i| {
            let batch = merged.batch(i).unwrap();
            let ids = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
            (0..ids.len()).map(|j| ids.value(j)).collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    assert_eq!(all_ids.len(), 9);
    for i in 0..9 {
        assert!(all_ids.contains(&(i as i32)), "Should contain id {}", i);
    }

    println!("Merge three tables test passed!");
}

#[test]
fn test_merge_different_sizes() {
    let ctx = Arc::new(CylonContext::new(false));

    let table1 = create_test_table(ctx.clone(), 0, 10);  // 10 rows
    let table2 = create_test_table(ctx.clone(), 10, 2);  // 2 rows
    let table3 = create_test_table(ctx.clone(), 12, 5);  // 5 rows

    let merged = merge(&[&table1, &table2, &table3]).unwrap();

    assert_eq!(merged.rows(), 17);

    println!("Merge different sizes test passed!");
}

#[test]
fn test_merge_with_empty_table() {
    let ctx = Arc::new(CylonContext::new(false));

    // Create empty table
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
    ]));

    let empty_batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(Vec::<i32>::new())),
            Arc::new(StringArray::from(Vec::<&str>::new())),
        ],
    ).unwrap();

    let empty_table = Table::from_record_batch(ctx.clone(), empty_batch).unwrap();
    let table1 = create_test_table(ctx.clone(), 0, 5);
    let table2 = create_test_table(ctx.clone(), 5, 5);

    // Merge with empty table in the middle
    let merged = merge(&[&table1, &empty_table, &table2]).unwrap();

    // Empty tables should be filtered out (matches C++ behavior at table.cpp:349-350)
    assert_eq!(merged.rows(), 10, "Empty table should be filtered out");

    println!("Merge with empty table test passed!");
}

#[test]
fn test_merge_all_empty_tables() {
    let ctx = Arc::new(CylonContext::new(false));

    // Create multiple empty tables
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
    ]));

    let empty_batch1 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(Vec::<i32>::new())),
            Arc::new(StringArray::from(Vec::<&str>::new())),
        ],
    ).unwrap();

    let empty_batch2 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(Vec::<i32>::new())),
            Arc::new(StringArray::from(Vec::<&str>::new())),
        ],
    ).unwrap();

    let empty1 = Table::from_record_batch(ctx.clone(), empty_batch1).unwrap();
    let empty2 = Table::from_record_batch(ctx.clone(), empty_batch2).unwrap();

    // When all tables are empty, should return first table (matches C++ behavior at table.cpp:353-355)
    let merged = merge(&[&empty1, &empty2]).unwrap();

    assert_eq!(merged.rows(), 0, "All empty tables should return empty table");
    assert_eq!(merged.columns(), 2, "Should preserve schema");

    println!("Merge all empty tables test passed!");
}

#[test]
fn test_merge_empty_vector() {
    // Trying to merge empty vector should fail
    let result = merge(&[]);
    assert!(result.is_err(), "Should fail with empty vector");

    if let Err(e) = result {
        assert!(e.to_string().contains("empty"), "Error should mention empty vector");
    }

    println!("Merge empty vector test passed!");
}

#[test]
fn test_merge_incompatible_schemas() {
    let ctx = Arc::new(CylonContext::new(false));

    // Create table 1 with id (Int32) and name (String)
    let schema1 = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
    ]));

    let batch1 = RecordBatch::try_new(
        schema1,
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec!["a", "b", "c"])),
        ],
    ).unwrap();

    let table1 = Table::from_record_batch(ctx.clone(), batch1).unwrap();

    // Create table 2 with id (Int64) and name (String) - different type!
    let schema2 = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
    ]));

    let batch2 = RecordBatch::try_new(
        schema2,
        vec![
            Arc::new(Int64Array::from(vec![4, 5, 6])),
            Arc::new(StringArray::from(vec!["d", "e", "f"])),
        ],
    ).unwrap();

    let table2 = Table::from_record_batch(ctx.clone(), batch2).unwrap();

    // Merging should fail due to incompatible schemas
    let result = merge(&[&table1, &table2]);
    assert!(result.is_err(), "Should fail with incompatible schemas");

    if let Err(e) = result {
        assert!(e.to_string().contains("schema"), "Error should mention schema incompatibility");
    }

    println!("Merge incompatible schemas test passed!");
}

#[test]
fn test_merge_different_column_names() {
    let ctx = Arc::new(CylonContext::new(false));

    // Create table 1 with columns "a" and "b"
    let schema1 = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int32, false),
        Field::new("b", DataType::Int32, false),
    ]));

    let batch1 = RecordBatch::try_new(
        schema1,
        vec![
            Arc::new(Int32Array::from(vec![1, 2])),
            Arc::new(Int32Array::from(vec![10, 20])),
        ],
    ).unwrap();

    let table1 = Table::from_record_batch(ctx.clone(), batch1).unwrap();

    // Create table 2 with columns "x" and "y" (different names but same types)
    let schema2 = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Int32, false),
        Field::new("y", DataType::Int32, false),
    ]));

    let batch2 = RecordBatch::try_new(
        schema2,
        vec![
            Arc::new(Int32Array::from(vec![3, 4])),
            Arc::new(Int32Array::from(vec![30, 40])),
        ],
    ).unwrap();

    let table2 = Table::from_record_batch(ctx.clone(), batch2).unwrap();

    // Merging should fail due to different column names
    let result = merge(&[&table1, &table2]);
    assert!(result.is_err(), "Should fail with different column names");

    println!("Merge different column names test passed!");
}

#[test]
fn test_merge_preserves_schema() {
    let ctx = Arc::new(CylonContext::new(false));

    let table1 = create_test_table(ctx.clone(), 0, 5);
    let table2 = create_test_table(ctx.clone(), 5, 5);

    let merged = merge(&[&table1, &table2]).unwrap();

    // Verify schema is preserved
    let orig_names = table1.column_names();
    let merged_names = merged.column_names();

    assert_eq!(orig_names, merged_names);

    println!("Merge preserves schema test passed!");
}

#[test]
fn test_merge_multi_batch_tables() {
    let ctx = Arc::new(CylonContext::new(false));

    // Create schema
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("value", DataType::Int32, false),
    ]));

    // Create first table with 2 batches
    let batch1_1 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2])),
            Arc::new(Int32Array::from(vec![10, 20])),
        ],
    ).unwrap();

    let batch1_2 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![3, 4])),
            Arc::new(Int32Array::from(vec![30, 40])),
        ],
    ).unwrap();

    let table1 = Table::from_record_batches(ctx.clone(), vec![batch1_1, batch1_2]).unwrap();

    // Create second table with 2 batches
    let batch2_1 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![5, 6])),
            Arc::new(Int32Array::from(vec![50, 60])),
        ],
    ).unwrap();

    let batch2_2 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![7, 8])),
            Arc::new(Int32Array::from(vec![70, 80])),
        ],
    ).unwrap();

    let table2 = Table::from_record_batches(ctx.clone(), vec![batch2_1, batch2_2]).unwrap();

    // Merge the multi-batch tables
    let merged = merge(&[&table1, &table2]).unwrap();

    assert_eq!(merged.rows(), 8, "Should have 8 rows total");
    assert_eq!(merged.columns(), 2);

    // Verify all batches are included (should have 4 batches total)
    assert_eq!(merged.num_batches(), 4, "Should combine all batches from both tables");

    println!("Merge multi-batch tables test passed!");
}

#[test]
fn test_merge_single_table() {
    let ctx = Arc::new(CylonContext::new(false));

    let table = create_test_table(ctx.clone(), 0, 5);

    // Merging a single table should work and return equivalent table
    let merged = merge(&[&table]).unwrap();

    assert_eq!(merged.rows(), table.rows());
    assert_eq!(merged.columns(), table.columns());

    println!("Merge single table test passed!");
}

#[test]
fn test_merge_large_tables() {
    let ctx = Arc::new(CylonContext::new(false));

    // Create larger tables with 500 rows each
    let table1 = create_test_table(ctx.clone(), 0, 500);
    let table2 = create_test_table(ctx.clone(), 500, 500);
    let table3 = create_test_table(ctx.clone(), 1000, 500);

    let merged = merge(&[&table1, &table2, &table3]).unwrap();

    assert_eq!(merged.rows(), 1500);
    assert_eq!(merged.columns(), 2);

    // Verify a few key values
    let all_ids = (0..merged.num_batches())
        .flat_map(|i| {
            let batch = merged.batch(i).unwrap();
            let ids = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
            (0..ids.len()).map(|j| ids.value(j)).collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    assert_eq!(all_ids.len(), 1500);
    assert_eq!(all_ids[0], 0);
    assert_eq!(all_ids[1499], 1499);

    println!("Merge large tables test passed!");
}

#[test]
fn test_merge_with_different_data_types() {
    let ctx = Arc::new(CylonContext::new(false));

    // Create table with multiple data types
    let schema = Arc::new(Schema::new(vec![
        Field::new("int_col", DataType::Int32, false),
        Field::new("float_col", DataType::Float64, false),
        Field::new("string_col", DataType::Utf8, false),
    ]));

    let batch1 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(Float64Array::from(vec![1.1, 2.2, 3.3])),
            Arc::new(StringArray::from(vec!["a", "b", "c"])),
        ],
    ).unwrap();

    let batch2 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![4, 5, 6])),
            Arc::new(Float64Array::from(vec![4.4, 5.5, 6.6])),
            Arc::new(StringArray::from(vec!["d", "e", "f"])),
        ],
    ).unwrap();

    let table1 = Table::from_record_batch(ctx.clone(), batch1).unwrap();
    let table2 = Table::from_record_batch(ctx.clone(), batch2).unwrap();

    let merged = merge(&[&table1, &table2]).unwrap();

    assert_eq!(merged.rows(), 6);
    assert_eq!(merged.columns(), 3);

    // Verify data types are preserved
    let batch = merged.batch(0).unwrap();
    assert_eq!(batch.schema().field(0).data_type(), &DataType::Int32);
    assert_eq!(batch.schema().field(1).data_type(), &DataType::Float64);
    assert_eq!(batch.schema().field(2).data_type(), &DataType::Utf8);

    println!("Merge with different data types test passed!");
}
