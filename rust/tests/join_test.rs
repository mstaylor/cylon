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

//! Join operation tests

use std::sync::Arc;
use cylon::ctx::CylonContext;
use cylon::table::Table;
use cylon::join::{join, JoinConfig};
use arrow::array::{Int64Array, StringArray, Array};
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

#[test]
fn test_inner_join() {
    let ctx = Arc::new(CylonContext::new(false));
    let left = create_left_table(ctx.clone());
    let right = create_right_table(ctx.clone());

    let config = JoinConfig::inner_join(0, 0); // Join on id column (index 0)

    let result = join(&left, &right, &config).unwrap();

    // Inner join should have 3 rows (ids 2, 3, 4)
    assert_eq!(result.rows(), 3, "Inner join should have 3 rows");

    // Check result has 4 columns (id, value, id, score)
    assert_eq!(result.columns(), 4, "Result should have 4 columns");

    // Verify the joined data
    let batch = result.batch(0).unwrap();
    let result_ids = batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
    let result_values = batch.column(1).as_any().downcast_ref::<StringArray>().unwrap();
    let result_scores = batch.column(3).as_any().downcast_ref::<Int64Array>().unwrap();

    // Collect results into vectors for easier verification
    let mut found_ids: Vec<i64> = (0..result_ids.len())
        .map(|i| result_ids.value(i))
        .collect();
    found_ids.sort();

    assert_eq!(found_ids, vec![2, 3, 4], "Inner join should contain ids 2, 3, 4");
}

#[test]
fn test_left_join() {
    let ctx = Arc::new(CylonContext::new(false));
    let left = create_left_table(ctx.clone());
    let right = create_right_table(ctx.clone());

    let config = JoinConfig::left_join(0, 0);

    let result = join(&left, &right, &config).unwrap();

    // Left join should have 4 rows (all left table rows)
    assert_eq!(result.rows(), 4, "Left join should have 4 rows");

    // Check result has 4 columns
    assert_eq!(result.columns(), 4, "Result should have 4 columns");

    let batch = result.batch(0).unwrap();
    let result_ids = batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();

    // Collect all IDs
    let mut found_ids: Vec<i64> = (0..result_ids.len())
        .map(|i| result_ids.value(i))
        .collect();
    found_ids.sort();

    // Should contain all left table IDs: 1, 2, 3, 4
    assert_eq!(found_ids, vec![1, 2, 3, 4], "Left join should contain all left table ids");
}

#[test]
fn test_right_join() {
    let ctx = Arc::new(CylonContext::new(false));
    let left = create_left_table(ctx.clone());
    let right = create_right_table(ctx.clone());

    let config = JoinConfig::right_join(0, 0);

    let result = join(&left, &right, &config).unwrap();

    // Right join should have 4 rows (all right table rows)
    assert_eq!(result.rows(), 4, "Right join should have 4 rows");

    // Check result has 4 columns
    assert_eq!(result.columns(), 4, "Result should have 4 columns");

    let batch = result.batch(0).unwrap();
    let result_right_ids = batch.column(2).as_any().downcast_ref::<Int64Array>().unwrap();

    // Collect all right IDs
    let mut found_ids: Vec<i64> = (0..result_right_ids.len())
        .map(|i| result_right_ids.value(i))
        .collect();
    found_ids.sort();

    // Should contain all right table IDs: 2, 3, 4, 5
    assert_eq!(found_ids, vec![2, 3, 4, 5], "Right join should contain all right table ids");
}

#[test]
fn test_full_outer_join() {
    let ctx = Arc::new(CylonContext::new(false));
    let left = create_left_table(ctx.clone());
    let right = create_right_table(ctx.clone());

    let config = JoinConfig::full_outer_join(0, 0);

    let result = join(&left, &right, &config).unwrap();

    // Full outer join should have 5 rows (union of both tables: 1, 2, 3, 4, 5)
    assert_eq!(result.rows(), 5, "Full outer join should have 5 rows");

    // Check result has 4 columns
    assert_eq!(result.columns(), 4, "Result should have 4 columns");
}

#[test]
fn test_multi_column_join() {
    let ctx = Arc::new(CylonContext::new(false));

    // Create tables with composite keys
    // Left table: key1, key2, value
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

    // Right table: key1, key2, score
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

    // Join on both key1 and key2
    let config = JoinConfig::inner_join_multi(vec![0, 1], vec![0, 1]).unwrap();

    let result = join(&left, &right, &config).unwrap();

    // Should match: (1, 10) and (2, 20) - 2 rows
    assert_eq!(result.rows(), 2, "Multi-column inner join should have 2 rows");
}

#[test]
fn test_join_with_suffixes() {
    let ctx = Arc::new(CylonContext::new(false));
    let left = create_left_table(ctx.clone());
    let right = create_right_table(ctx.clone());

    let config = JoinConfig::inner_join(0, 0)
        .with_suffixes("_left".to_string(), "_right".to_string());

    let result = join(&left, &right, &config).unwrap();

    // Check that column names have suffixes
    let col_names = result.column_names();

    // Both tables have "id" column, so they should have suffixes
    assert!(col_names.contains(&"id_left".to_string()) || col_names.contains(&"id_right".to_string()),
            "Column names should have suffixes for duplicate names");
}

#[test]
fn test_empty_join_result() {
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

    let config = JoinConfig::inner_join(0, 0);

    let result = join(&left, &right, &config).unwrap();

    // Inner join with no matching keys should return 0 rows
    assert_eq!(result.rows(), 0, "Inner join with no matches should have 0 rows");
}
