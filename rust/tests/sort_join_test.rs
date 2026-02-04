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

//! Sort join operation tests
//!
//! Tests for sort-merge join implementation (sort_join.rs)

use std::sync::Arc;
use cylon::ctx::CylonContext;
use cylon::table::Table;
use cylon::join::{join, JoinConfig, JoinAlgorithm};
use arrow::array::{Int64Array, Int32Array, Float64Array, StringArray, Array};
use arrow::datatypes::{Schema, Field, DataType};
use arrow::record_batch::RecordBatch;

/// Create a test table with ID and value columns
fn create_left_table(ctx: Arc<CylonContext>) -> Table {
    // Left table:
    // id | value
    // ---|------
    // 1  | A
    // 2  | B
    // 3  | C
    // 4  | D
    let ids = Int64Array::from(vec![1, 2, 3, 4]);
    let values = StringArray::from(vec!["A", "B", "C", "D"]);

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("value", DataType::Utf8, false),
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(ids), Arc::new(values)],
    ).unwrap();

    Table::from_record_batch(ctx, batch).unwrap()
}

fn create_right_table(ctx: Arc<CylonContext>) -> Table {
    // Right table:
    // id | score
    // ---|------
    // 2  | 10
    // 3  | 20
    // 4  | 30
    // 5  | 40
    let ids = Int64Array::from(vec![2, 3, 4, 5]);
    let scores = Int64Array::from(vec![10, 20, 30, 40]);

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("score", DataType::Int64, false),
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(ids), Arc::new(scores)],
    ).unwrap();

    Table::from_record_batch(ctx, batch).unwrap()
}

/// Create tables with Int32 keys for testing different types
fn create_int32_tables(ctx: Arc<CylonContext>) -> (Table, Table) {
    let left_ids = Int32Array::from(vec![1, 2, 3, 4, 5]);
    let left_vals = Int32Array::from(vec![10, 20, 30, 40, 50]);

    let left_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("value", DataType::Int32, false),
    ]));

    let left_batch = RecordBatch::try_new(
        left_schema,
        vec![Arc::new(left_ids), Arc::new(left_vals)],
    ).unwrap();

    let left = Table::from_record_batch(ctx.clone(), left_batch).unwrap();

    let right_ids = Int32Array::from(vec![3, 4, 5, 6, 7]);
    let right_vals = Int32Array::from(vec![300, 400, 500, 600, 700]);

    let right_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("score", DataType::Int32, false),
    ]));

    let right_batch = RecordBatch::try_new(
        right_schema,
        vec![Arc::new(right_ids), Arc::new(right_vals)],
    ).unwrap();

    let right = Table::from_record_batch(ctx, right_batch).unwrap();

    (left, right)
}

/// Create tables with Float64 keys for testing float handling
fn create_float64_tables(ctx: Arc<CylonContext>) -> (Table, Table) {
    let left_ids = Float64Array::from(vec![1.0, 2.0, 3.0, 4.0]);
    let left_vals = Int32Array::from(vec![10, 20, 30, 40]);

    let left_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Float64, false),
        Field::new("value", DataType::Int32, false),
    ]));

    let left_batch = RecordBatch::try_new(
        left_schema,
        vec![Arc::new(left_ids), Arc::new(left_vals)],
    ).unwrap();

    let left = Table::from_record_batch(ctx.clone(), left_batch).unwrap();

    let right_ids = Float64Array::from(vec![2.0, 3.0, 4.0, 5.0]);
    let right_vals = Int32Array::from(vec![200, 300, 400, 500]);

    let right_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Float64, false),
        Field::new("score", DataType::Int32, false),
    ]));

    let right_batch = RecordBatch::try_new(
        right_schema,
        vec![Arc::new(right_ids), Arc::new(right_vals)],
    ).unwrap();

    let right = Table::from_record_batch(ctx, right_batch).unwrap();

    (left, right)
}

#[test]
fn test_sort_inner_join() {
    let ctx = Arc::new(CylonContext::new(false));
    let left = create_left_table(ctx.clone());
    let right = create_right_table(ctx.clone());

    let config = JoinConfig::inner_join(0, 0)
        .with_algorithm(JoinAlgorithm::Sort);

    let result = join(&left, &right, &config).unwrap();

    // Inner join should have 3 rows (ids 2, 3, 4)
    assert_eq!(result.rows(), 3, "Sort inner join should have 3 rows");
    assert_eq!(result.columns(), 4, "Result should have 4 columns");

    let batch = result.batch(0).unwrap();
    let result_ids = batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();

    let mut found_ids: Vec<i64> = (0..result_ids.len())
        .map(|i| result_ids.value(i))
        .collect();
    found_ids.sort();

    assert_eq!(found_ids, vec![2, 3, 4], "Sort inner join should contain ids 2, 3, 4");
}

#[test]
fn test_sort_left_join() {
    let ctx = Arc::new(CylonContext::new(false));
    let left = create_left_table(ctx.clone());
    let right = create_right_table(ctx.clone());

    let config = JoinConfig::left_join(0, 0)
        .with_algorithm(JoinAlgorithm::Sort);

    let result = join(&left, &right, &config).unwrap();

    // Left join should have 4 rows (all left table rows)
    assert_eq!(result.rows(), 4, "Sort left join should have 4 rows");
    assert_eq!(result.columns(), 4, "Result should have 4 columns");

    let batch = result.batch(0).unwrap();
    let result_ids = batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();

    let mut found_ids: Vec<i64> = (0..result_ids.len())
        .map(|i| result_ids.value(i))
        .collect();
    found_ids.sort();

    assert_eq!(found_ids, vec![1, 2, 3, 4], "Sort left join should contain all left table ids");
}

#[test]
fn test_sort_right_join() {
    let ctx = Arc::new(CylonContext::new(false));
    let left = create_left_table(ctx.clone());
    let right = create_right_table(ctx.clone());

    let config = JoinConfig::right_join(0, 0)
        .with_algorithm(JoinAlgorithm::Sort);

    let result = join(&left, &right, &config).unwrap();

    // Right join should have 4 rows (all right table rows)
    assert_eq!(result.rows(), 4, "Sort right join should have 4 rows");
    assert_eq!(result.columns(), 4, "Result should have 4 columns");

    let batch = result.batch(0).unwrap();
    let result_right_ids = batch.column(2).as_any().downcast_ref::<Int64Array>().unwrap();

    let mut found_ids: Vec<i64> = (0..result_right_ids.len())
        .map(|i| result_right_ids.value(i))
        .collect();
    found_ids.sort();

    assert_eq!(found_ids, vec![2, 3, 4, 5], "Sort right join should contain all right table ids");
}

#[test]
fn test_sort_full_outer_join() {
    let ctx = Arc::new(CylonContext::new(false));
    let left = create_left_table(ctx.clone());
    let right = create_right_table(ctx.clone());

    let config = JoinConfig::full_outer_join(0, 0)
        .with_algorithm(JoinAlgorithm::Sort);

    let result = join(&left, &right, &config).unwrap();

    // Full outer join should have 5 rows (union of both tables: 1, 2, 3, 4, 5)
    assert_eq!(result.rows(), 5, "Sort full outer join should have 5 rows");
    assert_eq!(result.columns(), 4, "Result should have 4 columns");
}

#[test]
fn test_sort_join_int32() {
    let ctx = Arc::new(CylonContext::new(false));
    let (left, right) = create_int32_tables(ctx);

    let config = JoinConfig::inner_join(0, 0)
        .with_algorithm(JoinAlgorithm::Sort);

    let result = join(&left, &right, &config).unwrap();

    // Inner join on Int32: should match ids 3, 4, 5
    assert_eq!(result.rows(), 3, "Sort inner join on Int32 should have 3 rows");

    let batch = result.batch(0).unwrap();
    let result_ids = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();

    let mut found_ids: Vec<i32> = (0..result_ids.len())
        .map(|i| result_ids.value(i))
        .collect();
    found_ids.sort();

    assert_eq!(found_ids, vec![3, 4, 5], "Sort inner join should match ids 3, 4, 5");
}

#[test]
fn test_sort_join_float64() {
    let ctx = Arc::new(CylonContext::new(false));
    let (left, right) = create_float64_tables(ctx);

    let config = JoinConfig::inner_join(0, 0)
        .with_algorithm(JoinAlgorithm::Sort);

    let result = join(&left, &right, &config).unwrap();

    // Inner join on Float64: should match ids 2.0, 3.0, 4.0
    assert_eq!(result.rows(), 3, "Sort inner join on Float64 should have 3 rows");

    let batch = result.batch(0).unwrap();
    let result_ids = batch.column(0).as_any().downcast_ref::<Float64Array>().unwrap();

    let mut found_ids: Vec<i64> = (0..result_ids.len())
        .map(|i| result_ids.value(i) as i64)
        .collect();
    found_ids.sort();

    assert_eq!(found_ids, vec![2, 3, 4], "Sort inner join should match float ids 2.0, 3.0, 4.0");
}

#[test]
fn test_sort_multi_column_join() {
    let ctx = Arc::new(CylonContext::new(false));

    // Create tables with composite keys
    let left_key1 = Int64Array::from(vec![1, 1, 2, 2]);
    let left_key2 = Int64Array::from(vec![10, 20, 10, 20]);
    let left_values = StringArray::from(vec!["A", "B", "C", "D"]);

    let left_schema = Arc::new(Schema::new(vec![
        Field::new("key1", DataType::Int64, false),
        Field::new("key2", DataType::Int64, false),
        Field::new("value", DataType::Utf8, false),
    ]));

    let left_batch = RecordBatch::try_new(
        left_schema,
        vec![Arc::new(left_key1), Arc::new(left_key2), Arc::new(left_values)],
    ).unwrap();

    let left = Table::from_record_batch(ctx.clone(), left_batch).unwrap();

    let right_key1 = Int64Array::from(vec![1, 2, 2, 3]);
    let right_key2 = Int64Array::from(vec![10, 20, 30, 10]);
    let right_scores = Int64Array::from(vec![100, 200, 300, 400]);

    let right_schema = Arc::new(Schema::new(vec![
        Field::new("key1", DataType::Int64, false),
        Field::new("key2", DataType::Int64, false),
        Field::new("score", DataType::Int64, false),
    ]));

    let right_batch = RecordBatch::try_new(
        right_schema,
        vec![Arc::new(right_key1), Arc::new(right_key2), Arc::new(right_scores)],
    ).unwrap();

    let right = Table::from_record_batch(ctx.clone(), right_batch).unwrap();

    // Join on both key1 and key2 using sort join
    let config = JoinConfig::inner_join_multi(vec![0, 1], vec![0, 1])
        .unwrap()
        .with_algorithm(JoinAlgorithm::Sort);

    let result = join(&left, &right, &config).unwrap();

    // Should match: (1, 10) and (2, 20) - 2 rows
    assert_eq!(result.rows(), 2, "Sort multi-column inner join should have 2 rows");
}

#[test]
fn test_sort_join_empty_result() {
    let ctx = Arc::new(CylonContext::new(false));

    // Create two tables with no overlapping IDs
    let left_ids = Int64Array::from(vec![1, 2, 3]);
    let left_values = StringArray::from(vec!["A", "B", "C"]);

    let left_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("value", DataType::Utf8, false),
    ]));

    let left_batch = RecordBatch::try_new(
        left_schema,
        vec![Arc::new(left_ids), Arc::new(left_values)],
    ).unwrap();

    let left = Table::from_record_batch(ctx.clone(), left_batch).unwrap();

    let right_ids = Int64Array::from(vec![4, 5, 6]);
    let right_scores = Int64Array::from(vec![40, 50, 60]);

    let right_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("score", DataType::Int64, false),
    ]));

    let right_batch = RecordBatch::try_new(
        right_schema,
        vec![Arc::new(right_ids), Arc::new(right_scores)],
    ).unwrap();

    let right = Table::from_record_batch(ctx.clone(), right_batch).unwrap();

    let config = JoinConfig::inner_join(0, 0)
        .with_algorithm(JoinAlgorithm::Sort);

    let result = join(&left, &right, &config).unwrap();

    // Inner join with no matching keys should return 0 rows
    assert_eq!(result.rows(), 0, "Sort inner join with no matches should have 0 rows");
}

#[test]
fn test_sort_join_with_duplicates() {
    let ctx = Arc::new(CylonContext::new(false));

    // Left table with duplicate keys
    let left_ids = Int64Array::from(vec![1, 1, 2, 2, 3]);
    let left_vals = StringArray::from(vec!["A1", "A2", "B1", "B2", "C"]);

    let left_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("value", DataType::Utf8, false),
    ]));

    let left_batch = RecordBatch::try_new(
        left_schema,
        vec![Arc::new(left_ids), Arc::new(left_vals)],
    ).unwrap();

    let left = Table::from_record_batch(ctx.clone(), left_batch).unwrap();

    // Right table with duplicate keys
    let right_ids = Int64Array::from(vec![1, 2, 2, 3, 3]);
    let right_vals = Int64Array::from(vec![10, 20, 21, 30, 31]);

    let right_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("score", DataType::Int64, false),
    ]));

    let right_batch = RecordBatch::try_new(
        right_schema,
        vec![Arc::new(right_ids), Arc::new(right_vals)],
    ).unwrap();

    let right = Table::from_record_batch(ctx.clone(), right_batch).unwrap();

    let config = JoinConfig::inner_join(0, 0)
        .with_algorithm(JoinAlgorithm::Sort);

    let result = join(&left, &right, &config).unwrap();

    // Cartesian product for each matching key:
    // id=1: 2 left * 1 right = 2 rows
    // id=2: 2 left * 2 right = 4 rows
    // id=3: 1 left * 2 right = 2 rows
    // Total: 8 rows
    assert_eq!(result.rows(), 8, "Sort inner join with duplicates should produce cartesian product");
}

#[test]
fn test_sort_join_unsorted_input() {
    let ctx = Arc::new(CylonContext::new(false));

    // Deliberately unsorted input
    let left_ids = Int64Array::from(vec![4, 1, 3, 2]);
    let left_vals = StringArray::from(vec!["D", "A", "C", "B"]);

    let left_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("value", DataType::Utf8, false),
    ]));

    let left_batch = RecordBatch::try_new(
        left_schema,
        vec![Arc::new(left_ids), Arc::new(left_vals)],
    ).unwrap();

    let left = Table::from_record_batch(ctx.clone(), left_batch).unwrap();

    let right_ids = Int64Array::from(vec![5, 2, 4, 3]);
    let right_vals = Int64Array::from(vec![50, 20, 40, 30]);

    let right_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("score", DataType::Int64, false),
    ]));

    let right_batch = RecordBatch::try_new(
        right_schema,
        vec![Arc::new(right_ids), Arc::new(right_vals)],
    ).unwrap();

    let right = Table::from_record_batch(ctx.clone(), right_batch).unwrap();

    let config = JoinConfig::inner_join(0, 0)
        .with_algorithm(JoinAlgorithm::Sort);

    let result = join(&left, &right, &config).unwrap();

    // Should still produce correct results: ids 2, 3, 4 match
    assert_eq!(result.rows(), 3, "Sort join should handle unsorted input");

    let batch = result.batch(0).unwrap();
    let result_ids = batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();

    let mut found_ids: Vec<i64> = (0..result_ids.len())
        .map(|i| result_ids.value(i))
        .collect();
    found_ids.sort();

    assert_eq!(found_ids, vec![2, 3, 4], "Sort join should correctly match ids from unsorted input");
}
