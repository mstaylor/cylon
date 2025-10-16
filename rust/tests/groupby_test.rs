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

//! GroupBy tests - mirrors cpp/test/groupby_test.cpp

use std::sync::Arc;
use cylon::ctx::CylonContext;
use cylon::table::Table;
use cylon::mapreduce::{mapred_hash_groupby, AggregationOpId};
use arrow::array::{Int64Array, Float64Array, Array};
use arrow::datatypes::{Schema, Field, DataType};
use arrow::record_batch::RecordBatch;

/// Create test table with pattern: [0, 0, 1, 1, 2, 2, 3, 3]
/// Mirrors create_table() from C++ test
fn create_test_table_i64(ctx: Arc<CylonContext>) -> Table {
    let col0 = Int64Array::from(vec![0, 0, 1, 1, 2, 2, 3, 3]);
    let col1 = Int64Array::from(vec![0, 0, 1, 1, 2, 2, 3, 3]);

    let schema = Arc::new(Schema::new(vec![
        Field::new("col0", DataType::Int64, false),
        Field::new("col1", DataType::Int64, false),
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(col0), Arc::new(col1)],
    ).unwrap();

    Table::from_record_batch(ctx, batch).unwrap()
}

fn create_test_table_f64(ctx: Arc<CylonContext>) -> Table {
    let col0 = Int64Array::from(vec![0, 0, 1, 1, 2, 2, 3, 3]);
    let col1 = Float64Array::from(vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);

    let schema = Arc::new(Schema::new(vec![
        Field::new("col0", DataType::Int64, false),
        Field::new("col1", DataType::Float64, false),
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(col0), Arc::new(col1)],
    ).unwrap();

    Table::from_record_batch(ctx, batch).unwrap()
}

/// Helper to sum a column - mirrors compute::Sum from C++
fn sum_column_i64(table: &Table, col_idx: usize) -> i64 {
    let batch = table.batch(0).unwrap();
    let array = batch.column(col_idx).as_any().downcast_ref::<Int64Array>().unwrap();

    let mut sum = 0i64;
    for i in 0..array.len() {
        if !array.is_null(i) {
            sum += array.value(i);
        }
    }
    sum
}

fn sum_column_f64(table: &Table, col_idx: usize) -> f64 {
    let batch = table.batch(0).unwrap();
    let array = batch.column(col_idx).as_any().downcast_ref::<Float64Array>().unwrap();

    let mut sum = 0.0f64;
    for i in 0..array.len() {
        if !array.is_null(i) {
            sum += array.value(i);
        }
    }
    sum
}

#[test]
fn test_hash_groupby_sum_i64() {
    // Mirrors SECTION("testing hash group by sum") from C++
    let ctx = Arc::new(CylonContext::new(false));
    let table = create_test_table_i64(ctx.clone());

    let mut output = None;
    mapred_hash_groupby(&table, &[0], &[(1, AggregationOpId::Sum)], &mut output).unwrap();

    let result = output.unwrap();

    // Check: sum of group column should be 6 (0+1+2+3)
    let group_sum = sum_column_i64(&result, 0);
    assert_eq!(group_sum, 6, "Sum of group column should be 6");

    // Check: sum of aggregated values
    // Each group has 2 values of the same number, so: 0*2 + 1*2 + 2*2 + 3*2 = 12
    // In C++: T(2 * 6 * ctx->GetWorldSize()) where world_size=1, so 2*6*1 = 12
    let value_sum = sum_column_i64(&result, 1);
    assert_eq!(value_sum, 12, "Sum of aggregated values should be 12");
}

#[test]
fn test_hash_groupby_count() {
    // Mirrors SECTION("testing hash group by count") from C++
    let ctx = Arc::new(CylonContext::new(false));
    let table = create_test_table_i64(ctx.clone());

    let mut output = None;
    mapred_hash_groupby(&table, &[0], &[(1, AggregationOpId::Count)], &mut output).unwrap();

    let result = output.unwrap();

    // Check: sum of group column should be 6
    let group_sum = sum_column_i64(&result, 0);
    assert_eq!(group_sum, 6);

    // Check: sum of counts
    // Each group has 2 rows, 4 groups total: 2*4 = 8
    // In C++: int64_t(4 * 2 * ctx->GetWorldSize()) where world_size=1, so 4*2*1 = 8
    let count_sum = sum_column_i64(&result, 1);
    assert_eq!(count_sum, 8, "Sum of counts should be 8");
}

#[test]
fn test_hash_groupby_mean_i64() {
    // Mirrors SECTION("testing hash group by mean") from C++
    let ctx = Arc::new(CylonContext::new(false));
    let table = create_test_table_i64(ctx.clone());

    let mut output = None;
    mapred_hash_groupby(&table, &[0], &[(1, AggregationOpId::Mean)], &mut output).unwrap();

    let result = output.unwrap();

    // Check: sum of group column should be 6
    let group_sum = sum_column_i64(&result, 0);
    assert_eq!(group_sum, 6);

    // Check: sum of means
    // Each group: group 0 mean=0, group 1 mean=1, group 2 mean=2, group 3 mean=3
    // Sum = 0+1+2+3 = 6
    // In C++: T(6)
    let mean_sum = sum_column_i64(&result, 1);
    assert_eq!(mean_sum, 6, "Sum of means should be 6");
}

#[test]
fn test_hash_groupby_mean_f64() {
    // Additional test for floating point mean
    let ctx = Arc::new(CylonContext::new(false));
    let table = create_test_table_f64(ctx.clone());

    let mut output = None;
    mapred_hash_groupby(&table, &[0], &[(1, AggregationOpId::Mean)], &mut output).unwrap();

    let result = output.unwrap();

    let group_sum = sum_column_i64(&result, 0);
    assert_eq!(group_sum, 6);

    let mean_sum = sum_column_f64(&result, 1);
    assert!((mean_sum - 6.0).abs() < 1e-10, "Sum of means should be 6.0");
}

#[test]
fn test_hash_groupby_var() {
    // Mirrors SECTION("testing hash group by var") from C++
    let ctx = Arc::new(CylonContext::new(false));
    let table = create_test_table_f64(ctx.clone());

    let mut output = None;
    mapred_hash_groupby(&table, &[0], &[(1, AggregationOpId::Var)], &mut output).unwrap();

    let result = output.unwrap();

    // Check: sum of group column should be 6
    let group_sum = sum_column_i64(&result, 0);
    assert_eq!(group_sum, 6);

    // Check: sum of variances
    // Each group has identical values (0,0), (1,1), (2,2), (3,3), so variance is 0
    // In C++: double(0)
    let var_sum = sum_column_f64(&result, 1);
    assert!((var_sum - 0.0).abs() < 1e-10, "Sum of variances should be 0");
}

#[test]
fn test_hash_groupby_stddev() {
    // Mirrors SECTION("testing hash group by stddev") from C++
    let ctx = Arc::new(CylonContext::new(false));
    let table = create_test_table_f64(ctx.clone());

    let mut output = None;
    mapred_hash_groupby(&table, &[0], &[(1, AggregationOpId::Stddev)], &mut output).unwrap();

    let result = output.unwrap();

    // Check: sum of group column should be 6
    let group_sum = sum_column_i64(&result, 0);
    assert_eq!(group_sum, 6);

    // Check: sum of standard deviations
    // Each group has identical values, so stddev is 0
    // In C++: double(0)
    let stddev_sum = sum_column_f64(&result, 1);
    assert!((stddev_sum - 0.0).abs() < 1e-10, "Sum of standard deviations should be 0");
}

#[test]
fn test_hash_groupby_min() {
    let ctx = Arc::new(CylonContext::new(false));
    let table = create_test_table_i64(ctx.clone());

    let mut output = None;
    mapred_hash_groupby(&table, &[0], &[(1, AggregationOpId::Min)], &mut output).unwrap();

    let result = output.unwrap();

    let group_sum = sum_column_i64(&result, 0);
    assert_eq!(group_sum, 6);

    // Min values: 0, 1, 2, 3 -> sum = 6
    let min_sum = sum_column_i64(&result, 1);
    assert_eq!(min_sum, 6);
}

#[test]
fn test_hash_groupby_max() {
    let ctx = Arc::new(CylonContext::new(false));
    let table = create_test_table_i64(ctx.clone());

    let mut output = None;
    mapred_hash_groupby(&table, &[0], &[(1, AggregationOpId::Max)], &mut output).unwrap();

    let result = output.unwrap();

    let group_sum = sum_column_i64(&result, 0);
    assert_eq!(group_sum, 6);

    // Max values: 0, 1, 2, 3 -> sum = 6
    let max_sum = sum_column_i64(&result, 1);
    assert_eq!(max_sum, 6);
}
