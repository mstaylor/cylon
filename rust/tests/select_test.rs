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

//! Select (row filtering) operation tests
//! Corresponds to C++ Select operation (table.cpp:892)

use std::sync::Arc;
use cylon::ctx::CylonContext;
use cylon::table::Table;
use arrow::array::{Int64Array, StringArray, Float64Array, BooleanArray, Array};

fn create_test_table() -> Table {
    let ctx = Arc::new(CylonContext::new(false));

    let ids = Int64Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    let names = StringArray::from(vec![
        "Alice", "Bob", "Charlie", "David", "Eve",
        "Frank", "Grace", "Henry", "Iris", "Jack"
    ]);
    let scores = Float64Array::from(vec![
        95.5, 87.3, 92.1, 78.9, 88.7,
        91.2, 85.5, 89.3, 94.7, 82.1
    ]);

    let schema = Arc::new(arrow::datatypes::Schema::new(vec![
        arrow::datatypes::Field::new("id", arrow::datatypes::DataType::Int64, false),
        arrow::datatypes::Field::new("name", arrow::datatypes::DataType::Utf8, false),
        arrow::datatypes::Field::new("score", arrow::datatypes::DataType::Float64, false),
    ]));

    let batch = arrow::record_batch::RecordBatch::try_new(
        schema,
        vec![Arc::new(ids), Arc::new(names), Arc::new(scores)],
    ).unwrap();

    Table::from_record_batch(ctx, batch).unwrap()
}

#[test]
fn test_select_all_rows() {
    let table = create_test_table();

    // Create mask that selects all rows (all true)
    let mask = BooleanArray::from(vec![true; 10]);

    let selected = table.select(&mask).unwrap();

    assert_eq!(selected.rows(), 10);
    assert_eq!(selected.columns(), 3);
}

#[test]
fn test_select_no_rows() {
    let table = create_test_table();

    // Create mask that selects no rows (all false)
    let mask = BooleanArray::from(vec![false; 10]);

    let selected = table.select(&mask).unwrap();

    assert_eq!(selected.rows(), 0);
    assert_eq!(selected.columns(), 3);

    // Verify schema is preserved
    let col_names = selected.column_names();
    assert_eq!(col_names[0], "id");
    assert_eq!(col_names[1], "name");
    assert_eq!(col_names[2], "score");
}

#[test]
fn test_select_even_rows() {
    let table = create_test_table();

    // Select rows at even indices (0, 2, 4, 6, 8)
    let mask = BooleanArray::from(vec![
        true, false, true, false, true,
        false, true, false, true, false
    ]);

    let selected = table.select(&mask).unwrap();

    assert_eq!(selected.rows(), 5);
    assert_eq!(selected.columns(), 3);

    // Verify the selected rows
    let batch = selected.batch(0).unwrap();
    let ids = batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();

    assert_eq!(ids.value(0), 1);  // Row 0
    assert_eq!(ids.value(1), 3);  // Row 2
    assert_eq!(ids.value(2), 5);  // Row 4
    assert_eq!(ids.value(3), 7);  // Row 6
    assert_eq!(ids.value(4), 9);  // Row 8
}

#[test]
fn test_select_first_half() {
    let table = create_test_table();

    // Select first 5 rows
    let mask = BooleanArray::from(vec![
        true, true, true, true, true,
        false, false, false, false, false
    ]);

    let selected = table.select(&mask).unwrap();

    assert_eq!(selected.rows(), 5);

    let batch = selected.batch(0).unwrap();
    let names = batch.column(1).as_any().downcast_ref::<StringArray>().unwrap();

    assert_eq!(names.value(0), "Alice");
    assert_eq!(names.value(1), "Bob");
    assert_eq!(names.value(2), "Charlie");
    assert_eq!(names.value(3), "David");
    assert_eq!(names.value(4), "Eve");
}

#[test]
fn test_select_specific_rows() {
    let table = create_test_table();

    // Select rows 1, 3, 7 (indices)
    let mask = BooleanArray::from(vec![
        false, true, false, true, false,
        false, false, true, false, false
    ]);

    let selected = table.select(&mask).unwrap();

    assert_eq!(selected.rows(), 3);

    let batch = selected.batch(0).unwrap();
    let ids = batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();

    assert_eq!(ids.value(0), 2);  // Row 1
    assert_eq!(ids.value(1), 4);  // Row 3
    assert_eq!(ids.value(2), 8);  // Row 7
}

#[test]
fn test_select_invalid_mask_length() {
    let table = create_test_table();

    // Mask with wrong length
    let mask = BooleanArray::from(vec![true, true, true]);

    let result = table.select(&mask);
    assert!(result.is_err(), "Should fail with mask length mismatch");
}

#[test]
fn test_select_preserves_column_types() {
    let table = create_test_table();

    // Select some rows
    let mask = BooleanArray::from(vec![
        true, false, true, false, false,
        false, false, false, false, false
    ]);

    let selected = table.select(&mask).unwrap();

    let batch = selected.batch(0).unwrap();

    // Verify column types are preserved
    let ids = batch.column(0).as_any().downcast_ref::<Int64Array>();
    assert!(ids.is_some(), "ID column should be Int64Array");

    let names = batch.column(1).as_any().downcast_ref::<StringArray>();
    assert!(names.is_some(), "Name column should be StringArray");

    let scores = batch.column(2).as_any().downcast_ref::<Float64Array>();
    assert!(scores.is_some(), "Score column should be Float64Array");
}

#[test]
fn test_select_preserves_data_integrity() {
    let table = create_test_table();

    // Select rows where we can verify the data
    let mask = BooleanArray::from(vec![
        true, false, false, false, true,
        false, false, false, true, false
    ]);

    let selected = table.select(&mask).unwrap();

    assert_eq!(selected.rows(), 3);

    let batch = selected.batch(0).unwrap();
    let ids = batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
    let names = batch.column(1).as_any().downcast_ref::<StringArray>().unwrap();
    let scores = batch.column(2).as_any().downcast_ref::<Float64Array>().unwrap();

    // Row 0: Alice
    assert_eq!(ids.value(0), 1);
    assert_eq!(names.value(0), "Alice");
    assert!((scores.value(0) - 95.5).abs() < 0.001);

    // Row 4: Eve
    assert_eq!(ids.value(1), 5);
    assert_eq!(names.value(1), "Eve");
    assert!((scores.value(1) - 88.7).abs() < 0.001);

    // Row 8: Iris
    assert_eq!(ids.value(2), 9);
    assert_eq!(names.value(2), "Iris");
    assert!((scores.value(2) - 94.7).abs() < 0.001);
}

#[test]
fn test_select_with_arrow_compute() {
    let table = create_test_table();

    // Use manual comparison to create mask: scores >= 90.0
    let batch = table.batch(0).unwrap();
    let scores = batch.column(2).as_any().downcast_ref::<Float64Array>().unwrap();

    // Create mask manually
    let mask_values: Vec<bool> = (0..scores.len())
        .map(|i| scores.value(i) >= 90.0)
        .collect();
    let mask = BooleanArray::from(mask_values);

    let selected = table.select(&mask).unwrap();

    // Should have rows with scores >= 90.0 (95.5, 92.1, 91.2, 94.7)
    assert_eq!(selected.rows(), 4);

    let result_batch = selected.batch(0).unwrap();
    let result_scores = result_batch.column(2).as_any().downcast_ref::<Float64Array>().unwrap();

    // Verify all scores are >= 90.0
    for i in 0..result_scores.len() {
        assert!(result_scores.value(i) >= 90.0);
    }
}

#[test]
fn test_select_with_string_filter() {
    let table = create_test_table();

    // Select rows where name == "Alice" or "Eve"
    let batch = table.batch(0).unwrap();
    let names = batch.column(1).as_any().downcast_ref::<StringArray>().unwrap();

    // Create mask manually
    let mask_values: Vec<bool> = (0..names.len())
        .map(|i| {
            let name = names.value(i);
            name == "Alice" || name == "Eve"
        })
        .collect();
    let combined_mask = BooleanArray::from(mask_values);

    let selected = table.select(&combined_mask).unwrap();

    assert_eq!(selected.rows(), 2);

    let result_batch = selected.batch(0).unwrap();
    let result_names = result_batch.column(1).as_any().downcast_ref::<StringArray>().unwrap();

    assert_eq!(result_names.value(0), "Alice");
    assert_eq!(result_names.value(1), "Eve");
}

#[test]
fn test_select_complex_predicate() {
    let table = create_test_table();

    // Select rows where 85.0 < score < 93.0
    let batch = table.batch(0).unwrap();
    let scores = batch.column(2).as_any().downcast_ref::<Float64Array>().unwrap();

    // Create mask manually with complex predicate
    let mask_values: Vec<bool> = (0..scores.len())
        .map(|i| {
            let score = scores.value(i);
            score > 85.0 && score < 93.0
        })
        .collect();
    let mask = BooleanArray::from(mask_values);

    let selected = table.select(&mask).unwrap();

    // Should select: 87.3, 92.1, 88.7, 91.2, 85.5, 89.3
    assert_eq!(selected.rows(), 6);

    let result_batch = selected.batch(0).unwrap();
    let result_scores = result_batch.column(2).as_any().downcast_ref::<Float64Array>().unwrap();

    // Verify all scores are in range
    for i in 0..result_scores.len() {
        let score = result_scores.value(i);
        assert!(score > 85.0 && score < 93.0);
    }
}

#[test]
fn test_select_single_row() {
    let table = create_test_table();

    // Select only one row (index 5)
    let mask = BooleanArray::from(vec![
        false, false, false, false, false,
        true, false, false, false, false
    ]);

    let selected = table.select(&mask).unwrap();

    assert_eq!(selected.rows(), 1);
    assert_eq!(selected.columns(), 3);

    let batch = selected.batch(0).unwrap();
    let names = batch.column(1).as_any().downcast_ref::<StringArray>().unwrap();

    assert_eq!(names.value(0), "Frank");
}

#[test]
fn test_select_empty_table() {
    let ctx = Arc::new(CylonContext::new(false));

    let ids = Int64Array::from(Vec::<i64>::new());
    let schema = Arc::new(arrow::datatypes::Schema::new(vec![
        arrow::datatypes::Field::new("id", arrow::datatypes::DataType::Int64, false),
    ]));

    let batch = arrow::record_batch::RecordBatch::try_new(
        schema,
        vec![Arc::new(ids)],
    ).unwrap();

    let table = Table::from_record_batch(ctx, batch).unwrap();

    // Empty mask for empty table
    let mask = BooleanArray::from(Vec::<bool>::new());

    let selected = table.select(&mask).unwrap();
    assert_eq!(selected.rows(), 0);
}

#[test]
fn test_select_chaining() {
    let table = create_test_table();

    // First select: scores > 85.0
    let batch1 = table.batch(0).unwrap();
    let scores1 = batch1.column(2).as_any().downcast_ref::<Float64Array>().unwrap();
    let mask1_values: Vec<bool> = (0..scores1.len())
        .map(|i| scores1.value(i) > 85.0)
        .collect();
    let mask1 = BooleanArray::from(mask1_values);
    let selected1 = table.select(&mask1).unwrap();

    // Second select on result: scores < 92.0
    let batch2 = selected1.batch(0).unwrap();
    let scores2 = batch2.column(2).as_any().downcast_ref::<Float64Array>().unwrap();
    let mask2_values: Vec<bool> = (0..scores2.len())
        .map(|i| scores2.value(i) < 92.0)
        .collect();
    let mask2 = BooleanArray::from(mask2_values);
    let selected2 = selected1.select(&mask2).unwrap();

    // Final result should have scores in range (85.0, 92.0)
    let result_batch = selected2.batch(0).unwrap();
    let result_scores = result_batch.column(2).as_any().downcast_ref::<Float64Array>().unwrap();

    for i in 0..result_scores.len() {
        let score = result_scores.value(i);
        assert!(score > 85.0 && score < 92.0);
    }
}
