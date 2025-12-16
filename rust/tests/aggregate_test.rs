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

//! Aggregate tests - mirrors cpp/test/aggregate_test.cpp

use std::sync::Arc;
use cylon::ctx::CylonContext;
use cylon::table::Table;
use cylon::compute::{
    ScalarValue, AggregateOptions, VarianceOptions,
    // Local array aggregates
    sum_array, min_array, max_array, count_array, mean_array, variance_array, stddev_array,
    // Context-aware aggregates
    sum, min, max, count, mean, variance, stddev,
    // Table-level aggregates
    sum_column, sum_table, min_table, max_table, count_table, mean_table,
};
use arrow::array::{Int64Array, Float64Array, Array, ArrayRef};
use arrow::datatypes::{Schema, Field, DataType};
use arrow::record_batch::RecordBatch;

/// Create test table with rows values starting from 10.0
/// Mirrors CreateTable from C++ test (column 1 has values 10.0 + i)
fn create_test_table(ctx: Arc<CylonContext>, rows: usize) -> Table {
    let col0: Vec<i64> = (0..rows as i64).collect();
    let col1: Vec<f64> = (0..rows).map(|i| 10.0 + i as f64).collect();

    let schema = Arc::new(Schema::new(vec![
        Field::new("col0", DataType::Int64, false),
        Field::new("col1", DataType::Float64, false),
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(col0)),
            Arc::new(Float64Array::from(col1)),
        ],
    ).unwrap();

    Table::from_record_batch(ctx, batch).unwrap()
}

/// Create simple int64 array for testing
fn create_int64_array(values: &[i64]) -> ArrayRef {
    Arc::new(Int64Array::from(values.to_vec()))
}

/// Create simple float64 array for testing
fn create_float64_array(values: &[f64]) -> ArrayRef {
    Arc::new(Float64Array::from(values.to_vec()))
}

// =============================================================================
// Local Array Aggregate Tests
// =============================================================================

#[test]
fn test_sum_array_i64() {
    let array = create_int64_array(&[1, 2, 3, 4, 5]);
    let options = AggregateOptions::default();
    let result = sum_array(&*array, &options).unwrap();

    match result {
        ScalarValue::Int64(v) => assert_eq!(v, 15),
        ScalarValue::Float64(v) => assert!((v - 15.0).abs() < 1e-10),
        _ => panic!("Unexpected scalar type"),
    }
}

#[test]
fn test_sum_array_f64() {
    let array = create_float64_array(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let options = AggregateOptions::default();
    let result = sum_array(&*array, &options).unwrap();

    match result {
        ScalarValue::Float64(v) => assert!((v - 15.0).abs() < 1e-10),
        _ => panic!("Expected Float64"),
    }
}

#[test]
fn test_min_array() {
    let array = create_float64_array(&[5.0, 2.0, 8.0, 1.0, 9.0]);
    let options = AggregateOptions::default();
    let result = min_array(&*array, &options).unwrap();

    match result {
        ScalarValue::Float64(v) => assert!((v - 1.0).abs() < 1e-10),
        _ => panic!("Expected Float64"),
    }
}

#[test]
fn test_max_array() {
    let array = create_float64_array(&[5.0, 2.0, 8.0, 1.0, 9.0]);
    let options = AggregateOptions::default();
    let result = max_array(&*array, &options).unwrap();

    match result {
        ScalarValue::Float64(v) => assert!((v - 9.0).abs() < 1e-10),
        _ => panic!("Expected Float64"),
    }
}

#[test]
fn test_count_array() {
    let array = create_int64_array(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    let options = AggregateOptions::default();
    let result = count_array(&*array, &options).unwrap();

    match result {
        ScalarValue::Int64(v) => assert_eq!(v, 10),
        _ => panic!("Expected Int64"),
    }
}

#[test]
fn test_mean_array() {
    let array = create_float64_array(&[2.0, 4.0, 6.0, 8.0, 10.0]);
    let options = AggregateOptions::default();
    let result = mean_array(&*array, &options).unwrap();

    match result {
        ScalarValue::Float64(v) => assert!((v - 6.0).abs() < 1e-10),
        _ => panic!("Expected Float64"),
    }
}

#[test]
fn test_variance_array() {
    // Values: 2, 4, 6, 8, 10; Mean = 6
    // Population variance = (16+4+0+4+16)/5 = 8.0
    let array = create_float64_array(&[2.0, 4.0, 6.0, 8.0, 10.0]);
    let options = VarianceOptions { ddof: 0, skip_nulls: true };
    let result = variance_array(&*array, &options).unwrap();

    match result {
        ScalarValue::Float64(v) => assert!((v - 8.0).abs() < 1e-10),
        _ => panic!("Expected Float64"),
    }
}

#[test]
fn test_variance_array_sample() {
    // Values: 2, 4, 6, 8, 10; Mean = 6
    // Sample variance = (40/4) = 10.0
    let array = create_float64_array(&[2.0, 4.0, 6.0, 8.0, 10.0]);
    let options = VarianceOptions { ddof: 1, skip_nulls: true };
    let result = variance_array(&*array, &options).unwrap();

    match result {
        ScalarValue::Float64(v) => assert!((v - 10.0).abs() < 1e-10),
        _ => panic!("Expected Float64"),
    }
}

#[test]
fn test_stddev_array() {
    let array = create_float64_array(&[2.0, 4.0, 6.0, 8.0, 10.0]);
    let options = VarianceOptions { ddof: 0, skip_nulls: true };
    let result = stddev_array(&*array, &options).unwrap();

    match result {
        ScalarValue::Float64(v) => {
            let expected = 8.0_f64.sqrt(); // sqrt of variance
            assert!((v - expected).abs() < 1e-10);
        }
        _ => panic!("Expected Float64"),
    }
}

// =============================================================================
// Table-Level Aggregate Tests (mirrors C++ SECTION tests)
// =============================================================================

#[test]
fn test_table_sum() {
    // Mirrors SECTION("testing sum") from C++
    let ctx = Arc::new(CylonContext::new(false));
    let rows = 12;
    let table = create_test_table(ctx.clone(), rows);

    // Column 1 has values 10.0, 11.0, ..., 21.0
    // Sum = 10 + 11 + ... + 21 = (12 * (10 + 21)) / 2 = 12 * 31 / 2 = 186
    // In C++: ((rows * (rows - 1) / 2.0) + 10.0 * rows) = (12*11/2) + 120 = 66 + 120 = 186
    let options = AggregateOptions::default();
    let result = sum_column(&ctx, &table, 1, &options).unwrap();

    match result {
        ScalarValue::Float64(v) => assert!((v - 186.0).abs() < 1e-10, "Sum should be 186.0, got {}", v),
        _ => panic!("Expected Float64"),
    }
}

#[test]
fn test_table_count() {
    // Mirrors SECTION("testing count") from C++
    let ctx = Arc::new(CylonContext::new(false));
    let rows = 12;
    let table = create_test_table(ctx.clone(), rows);

    let options = AggregateOptions::default();
    let results = count_table(&ctx, &table, &options).unwrap();

    // Count of column 1 (index 1) should be 12
    match &results[1] {
        ScalarValue::Int64(v) => assert_eq!(*v, rows as i64, "Count should be {}", rows),
        _ => panic!("Expected Int64"),
    }
}

#[test]
fn test_table_min() {
    // Mirrors SECTION("testing min") from C++
    let ctx = Arc::new(CylonContext::new(false));
    let rows = 12;
    let table = create_test_table(ctx.clone(), rows);

    // Minimum value in column 1 is 10.0
    let options = AggregateOptions::default();
    let results = min_table(&ctx, &table, &options).unwrap();

    match &results[1] {
        ScalarValue::Float64(v) => assert!((v - 10.0).abs() < 1e-10, "Min should be 10.0, got {}", v),
        _ => panic!("Expected Float64"),
    }
}

#[test]
fn test_table_max() {
    // Mirrors SECTION("testing max") from C++
    let ctx = Arc::new(CylonContext::new(false));
    let rows = 12;
    let table = create_test_table(ctx.clone(), rows);

    // Maximum value in column 1 is 10.0 + (rows - 1) = 10.0 + 11 = 21.0
    let options = AggregateOptions::default();
    let results = max_table(&ctx, &table, &options).unwrap();

    match &results[1] {
        ScalarValue::Float64(v) => assert!((v - 21.0).abs() < 1e-10, "Max should be 21.0, got {}", v),
        _ => panic!("Expected Float64"),
    }
}

#[test]
fn test_table_mean() {
    let ctx = Arc::new(CylonContext::new(false));
    let rows = 12;
    let table = create_test_table(ctx.clone(), rows);

    // Mean = 186 / 12 = 15.5
    let options = AggregateOptions::default();
    let results = mean_table(&ctx, &table, &options).unwrap();

    match &results[1] {
        ScalarValue::Float64(v) => assert!((v - 15.5).abs() < 1e-10, "Mean should be 15.5, got {}", v),
        _ => panic!("Expected Float64"),
    }
}

// =============================================================================
// Context-Aware Aggregate Tests (for distributed support)
// =============================================================================

#[test]
fn test_context_sum() {
    let ctx = Arc::new(CylonContext::new(false));
    let array = create_float64_array(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let options = AggregateOptions::default();

    let result = sum(&ctx, &*array, &options).unwrap();

    match result {
        ScalarValue::Float64(v) => assert!((v - 15.0).abs() < 1e-10),
        _ => panic!("Expected Float64"),
    }
}

#[test]
fn test_context_min() {
    let ctx = Arc::new(CylonContext::new(false));
    let array = create_float64_array(&[5.0, 2.0, 8.0, 1.0, 9.0]);
    let options = AggregateOptions::default();

    let result = min(&ctx, &*array, &options).unwrap();

    match result {
        ScalarValue::Float64(v) => assert!((v - 1.0).abs() < 1e-10),
        _ => panic!("Expected Float64"),
    }
}

#[test]
fn test_context_max() {
    let ctx = Arc::new(CylonContext::new(false));
    let array = create_float64_array(&[5.0, 2.0, 8.0, 1.0, 9.0]);
    let options = AggregateOptions::default();

    let result = max(&ctx, &*array, &options).unwrap();

    match result {
        ScalarValue::Float64(v) => assert!((v - 9.0).abs() < 1e-10),
        _ => panic!("Expected Float64"),
    }
}

#[test]
fn test_context_count() {
    let ctx = Arc::new(CylonContext::new(false));
    let array = create_int64_array(&[1, 2, 3, 4, 5]);
    let options = AggregateOptions::default();

    let result = count(&ctx, &*array, &options).unwrap();

    match result {
        ScalarValue::Int64(v) => assert_eq!(v, 5),
        _ => panic!("Expected Int64"),
    }
}

#[test]
fn test_context_mean() {
    let ctx = Arc::new(CylonContext::new(false));
    let array = create_float64_array(&[2.0, 4.0, 6.0, 8.0, 10.0]);
    let options = AggregateOptions::default();

    let result = mean(&ctx, &*array, &options).unwrap();

    match result {
        ScalarValue::Float64(v) => assert!((v - 6.0).abs() < 1e-10),
        _ => panic!("Expected Float64"),
    }
}

#[test]
fn test_context_variance() {
    let ctx = Arc::new(CylonContext::new(false));
    let array = create_float64_array(&[2.0, 4.0, 6.0, 8.0, 10.0]);
    let options = VarianceOptions { ddof: 0, skip_nulls: true };

    let result = variance(&ctx, &*array, &options).unwrap();

    match result {
        ScalarValue::Float64(v) => assert!((v - 8.0).abs() < 1e-10),
        _ => panic!("Expected Float64"),
    }
}

#[test]
fn test_context_stddev() {
    let ctx = Arc::new(CylonContext::new(false));
    let array = create_float64_array(&[2.0, 4.0, 6.0, 8.0, 10.0]);
    let options = VarianceOptions { ddof: 0, skip_nulls: true };

    let result = stddev(&ctx, &*array, &options).unwrap();

    match result {
        ScalarValue::Float64(v) => {
            let expected = 8.0_f64.sqrt();
            assert!((v - expected).abs() < 1e-10);
        }
        _ => panic!("Expected Float64"),
    }
}

// =============================================================================
// Edge Case Tests
// =============================================================================

#[test]
fn test_sum_empty_array() {
    let array: ArrayRef = Arc::new(Float64Array::from(Vec::<f64>::new()));
    let options = AggregateOptions::default();
    let result = sum_array(&*array, &options);

    // Empty array sum should return null (no values to sum)
    match result {
        Ok(ScalarValue::Null) => {} // Expected for empty
        Ok(ScalarValue::Float64(v)) => assert!((v - 0.0).abs() < 1e-10), // Or zero
        Ok(ScalarValue::Int64(v)) => assert_eq!(v, 0),
        Err(_) => {} // Also acceptable
        _ => panic!("Unexpected result type"),
    }
}

#[test]
fn test_min_single_value() {
    let array = create_float64_array(&[42.0]);
    let options = AggregateOptions::default();
    let result = min_array(&*array, &options).unwrap();

    match result {
        ScalarValue::Float64(v) => assert!((v - 42.0).abs() < 1e-10),
        _ => panic!("Expected Float64"),
    }
}

#[test]
fn test_max_single_value() {
    let array = create_float64_array(&[42.0]);
    let options = AggregateOptions::default();
    let result = max_array(&*array, &options).unwrap();

    match result {
        ScalarValue::Float64(v) => assert!((v - 42.0).abs() < 1e-10),
        _ => panic!("Expected Float64"),
    }
}

#[test]
fn test_count_with_nulls() {
    // Create array with nulls
    let array: ArrayRef = Arc::new(
        Float64Array::from(vec![Some(1.0), None, Some(3.0), None, Some(5.0)])
    );
    let options = AggregateOptions::default(); // skip_nulls = true by default
    let result = count_array(&*array, &options).unwrap();

    match result {
        ScalarValue::Int64(v) => assert_eq!(v, 3, "Count should exclude nulls"),
        _ => panic!("Expected Int64"),
    }
}

#[test]
fn test_sum_with_nulls() {
    let array: ArrayRef = Arc::new(
        Float64Array::from(vec![Some(1.0), None, Some(3.0), None, Some(5.0)])
    );
    let options = AggregateOptions::default();
    let result = sum_array(&*array, &options).unwrap();

    match result {
        ScalarValue::Float64(v) => assert!((v - 9.0).abs() < 1e-10, "Sum should be 9.0 (1+3+5)"),
        _ => panic!("Expected Float64"),
    }
}

// =============================================================================
// Multi-column Table Tests
// =============================================================================

#[test]
fn test_table_all_columns_sum() {
    let ctx = Arc::new(CylonContext::new(false));

    let col0 = Int64Array::from(vec![1, 2, 3, 4, 5]);
    let col1 = Float64Array::from(vec![10.0, 20.0, 30.0, 40.0, 50.0]);

    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int64, false),
        Field::new("b", DataType::Float64, false),
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(col0), Arc::new(col1)],
    ).unwrap();

    let table = Table::from_record_batch(ctx.clone(), batch).unwrap();
    let options = AggregateOptions::default();

    // Test sum on all columns
    let results = sum_table(&ctx, &table, &options).unwrap();

    // First column (Int64): 1+2+3+4+5 = 15
    match &results[0] {
        ScalarValue::Int64(v) => assert_eq!(*v, 15),
        ScalarValue::Float64(v) => assert!((v - 15.0).abs() < 1e-10),
        _ => panic!("Unexpected type for column 0"),
    }

    // Second column (Float64): 10+20+30+40+50 = 150
    match &results[1] {
        ScalarValue::Float64(v) => assert!((v - 150.0).abs() < 1e-10),
        _ => panic!("Expected Float64 for column 1"),
    }
}
