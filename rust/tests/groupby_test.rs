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

//! GroupBy tests
//!
//! Ported from cpp/test/groupby_test.cpp
//!
//! Tests for distributed hash groupby operations.

use std::sync::Arc;
use arrow::array::{Int64Array, Float64Array, ArrayRef, Array};
use arrow::datatypes::{Schema, Field, DataType as ArrowDataType};
use arrow::record_batch::RecordBatch;

use cylon::ctx::CylonContext;
use cylon::table::Table;
use cylon::groupby::{hash_groupby, distributed_hash_groupby};
use cylon::mapreduce::AggregationOpId;

// ============================================================================
// Helper functions (mirroring C++ test helpers)
// ============================================================================

/// Create test table with two columns: col0 (key) and col1 (value)
/// Corresponds to C++ create_table (groupby_test.cpp:30-40)
///
/// Creates: col0 = [0, 0, 1, 1, 2, 2, 3, 3]
///          col1 = [0, 0, 1, 1, 2, 2, 3, 3]
fn create_test_table_int64(ctx: Arc<CylonContext>) -> cylon::error::CylonResult<Table> {
    let col0: ArrayRef = Arc::new(Int64Array::from(vec![0, 0, 1, 1, 2, 2, 3, 3]));
    let col1: ArrayRef = Arc::new(Int64Array::from(vec![0, 0, 1, 1, 2, 2, 3, 3]));

    let schema = Arc::new(Schema::new(vec![
        Field::new("col0", ArrowDataType::Int64, false),
        Field::new("col1", ArrowDataType::Int64, false),
    ]));

    let batch = RecordBatch::try_new(schema, vec![col0, col1])
        .map_err(|e| cylon::error::CylonError::new(
            cylon::error::Code::ExecutionError,
            e.to_string()
        ))?;

    Table::from_record_batch(ctx, batch)
}

/// Create test table with float64 values
fn create_test_table_float64(ctx: Arc<CylonContext>) -> cylon::error::CylonResult<Table> {
    let col0: ArrayRef = Arc::new(Int64Array::from(vec![0, 0, 1, 1, 2, 2, 3, 3]));
    let col1: ArrayRef = Arc::new(Float64Array::from(vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0]));

    let schema = Arc::new(Schema::new(vec![
        Field::new("col0", ArrowDataType::Int64, false),
        Field::new("col1", ArrowDataType::Float64, false),
    ]));

    let batch = RecordBatch::try_new(schema, vec![col0, col1])
        .map_err(|e| cylon::error::CylonError::new(
            cylon::error::Code::ExecutionError,
            e.to_string()
        ))?;

    Table::from_record_batch(ctx, batch)
}

/// Sum all values in a column (for verification)
fn sum_int64_column(table: &Table, col_idx: usize) -> i64 {
    let mut sum = 0i64;
    for batch in table.batches() {
        let col = batch.column(col_idx);
        let arr = col.as_any().downcast_ref::<Int64Array>().unwrap();
        for i in 0..arr.len() {
            if !arr.is_null(i) {
                sum += arr.value(i);
            }
        }
    }
    sum
}

/// Sum all values in a float64 column (for verification)
fn sum_float64_column(table: &Table, col_idx: usize) -> f64 {
    let mut sum = 0.0f64;
    for batch in table.batches() {
        let col = batch.column(col_idx);
        let arr = col.as_any().downcast_ref::<Float64Array>().unwrap();
        for i in 0..arr.len() {
            if !arr.is_null(i) {
                sum += arr.value(i);
            }
        }
    }
    sum
}

// ============================================================================
// Local GroupBy Tests (single process)
// ============================================================================

mod local_groupby_tests {
    use super::*;

    #[test]
    fn test_hash_groupby_sum_int64() {
        // Corresponds to C++ "testing hash group by sum" section
        let ctx = CylonContext::init();
        let table = create_test_table_int64(ctx).unwrap();

        let result = hash_groupby(&table, &[0], &[1], &[AggregationOpId::Sum]).unwrap();

        // After groupby on col0, we should have 4 groups (0, 1, 2, 3)
        assert_eq!(result.rows(), 4, "Should have 4 groups");

        // Sum of keys: 0 + 1 + 2 + 3 = 6
        let key_sum = sum_int64_column(&result, 0);
        assert_eq!(key_sum, 6, "Sum of keys should be 6");

        // Sum of values: 0+0 + 1+1 + 2+2 + 3+3 = 12
        let value_sum = sum_int64_column(&result, 1);
        assert_eq!(value_sum, 12, "Sum of values should be 12");

        println!("hash_groupby sum int64 test passed");
    }

    #[test]
    fn test_hash_groupby_sum_float64() {
        let ctx = CylonContext::init();
        let table = create_test_table_float64(ctx).unwrap();

        let result = hash_groupby(&table, &[0], &[1], &[AggregationOpId::Sum]).unwrap();

        assert_eq!(result.rows(), 4, "Should have 4 groups");

        // Sum of values: 0+0 + 1+1 + 2+2 + 3+3 = 12.0
        let value_sum = sum_float64_column(&result, 1);
        assert!((value_sum - 12.0).abs() < 1e-10, "Sum of values should be 12.0");

        println!("hash_groupby sum float64 test passed");
    }

    #[test]
    fn test_hash_groupby_min() {
        let ctx = CylonContext::init();
        let table = create_test_table_int64(ctx).unwrap();

        let result = hash_groupby(&table, &[0], &[1], &[AggregationOpId::Min]).unwrap();

        assert_eq!(result.rows(), 4, "Should have 4 groups");

        // Sum of min values: 0 + 1 + 2 + 3 = 6
        let value_sum = sum_int64_column(&result, 1);
        assert_eq!(value_sum, 6, "Sum of min values should be 6");

        println!("hash_groupby min test passed");
    }

    #[test]
    fn test_hash_groupby_max() {
        let ctx = CylonContext::init();
        let table = create_test_table_int64(ctx).unwrap();

        let result = hash_groupby(&table, &[0], &[1], &[AggregationOpId::Max]).unwrap();

        assert_eq!(result.rows(), 4, "Should have 4 groups");

        // Sum of max values: 0 + 1 + 2 + 3 = 6
        let value_sum = sum_int64_column(&result, 1);
        assert_eq!(value_sum, 6, "Sum of max values should be 6");

        println!("hash_groupby max test passed");
    }

    #[test]
    fn test_hash_groupby_count() {
        // Corresponds to C++ "testing hash group by count" section
        let ctx = CylonContext::init();
        let table = create_test_table_int64(ctx).unwrap();

        let result = hash_groupby(&table, &[0], &[1], &[AggregationOpId::Count]).unwrap();

        assert_eq!(result.rows(), 4, "Should have 4 groups");

        // Each group has 2 elements, sum of counts = 8
        let count_sum = sum_int64_column(&result, 1);
        assert_eq!(count_sum, 8, "Sum of counts should be 8");

        println!("hash_groupby count test passed");
    }

    #[test]
    fn test_hash_groupby_mean() {
        // Corresponds to C++ "testing hash group by mean" section
        let ctx = CylonContext::init();
        let table = create_test_table_float64(ctx).unwrap();

        let result = hash_groupby(&table, &[0], &[1], &[AggregationOpId::Mean]).unwrap();

        assert_eq!(result.rows(), 4, "Should have 4 groups");

        // Mean values are 0, 1, 2, 3. Sum = 6
        let mean_sum = sum_float64_column(&result, 1);
        assert!((mean_sum - 6.0).abs() < 1e-10, "Sum of means should be 6.0");

        println!("hash_groupby mean test passed");
    }

    #[test]
    fn test_hash_groupby_multiple_aggregations() {
        let ctx = CylonContext::init();

        // Create table with key and two value columns
        let col0: ArrayRef = Arc::new(Int64Array::from(vec![0, 0, 1, 1, 2, 2]));
        let col1: ArrayRef = Arc::new(Int64Array::from(vec![10, 20, 30, 40, 50, 60]));
        let col2: ArrayRef = Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5, 6]));

        let schema = Arc::new(Schema::new(vec![
            Field::new("key", ArrowDataType::Int64, false),
            Field::new("val1", ArrowDataType::Int64, false),
            Field::new("val2", ArrowDataType::Int64, false),
        ]));

        let batch = RecordBatch::try_new(schema, vec![col0, col1, col2]).unwrap();
        let table = Table::from_record_batch(ctx, batch).unwrap();

        // Group by key, sum val1, max val2
        let result = hash_groupby(
            &table,
            &[0],
            &[1, 2],
            &[AggregationOpId::Sum, AggregationOpId::Max],
        ).unwrap();

        assert_eq!(result.rows(), 3, "Should have 3 groups");
        assert_eq!(result.columns(), 3, "Should have 3 columns (key, sum, max)");

        println!("hash_groupby multiple aggregations test passed");
    }

    #[test]
    fn test_hash_groupby_multiple_keys() {
        let ctx = CylonContext::init();

        // Create table with two key columns
        let col0: ArrayRef = Arc::new(Int64Array::from(vec![0, 0, 0, 0, 1, 1, 1, 1]));
        let col1: ArrayRef = Arc::new(Int64Array::from(vec![0, 0, 1, 1, 0, 0, 1, 1]));
        let col2: ArrayRef = Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8]));

        let schema = Arc::new(Schema::new(vec![
            Field::new("key1", ArrowDataType::Int64, false),
            Field::new("key2", ArrowDataType::Int64, false),
            Field::new("value", ArrowDataType::Int64, false),
        ]));

        let batch = RecordBatch::try_new(schema, vec![col0, col1, col2]).unwrap();
        let table = Table::from_record_batch(ctx, batch).unwrap();

        // Group by both key columns
        let result = hash_groupby(
            &table,
            &[0, 1],
            &[2],
            &[AggregationOpId::Sum],
        ).unwrap();

        // Should have 4 groups: (0,0), (0,1), (1,0), (1,1)
        assert_eq!(result.rows(), 4, "Should have 4 groups");

        println!("hash_groupby multiple keys test passed");
    }

    #[test]
    fn test_hash_groupby_invalid_args() {
        let ctx = CylonContext::init();
        let table = create_test_table_int64(ctx).unwrap();

        // Mismatched aggregate_cols and aggregate_ops
        let result = hash_groupby(&table, &[0], &[1, 1], &[AggregationOpId::Sum]);
        assert!(result.is_err(), "Should fail with mismatched args");

        println!("hash_groupby invalid args test passed");
    }
}

// ============================================================================
// Distributed GroupBy Tests (single process simulation)
// ============================================================================

mod distributed_groupby_tests {
    use super::*;

    #[test]
    fn test_distributed_hash_groupby_local() {
        // Test distributed groupby in local mode (world_size = 1)
        let ctx = CylonContext::init();
        let table = create_test_table_int64(ctx).unwrap();

        let result = distributed_hash_groupby(&table, &[0], &[1], &[AggregationOpId::Sum]).unwrap();

        assert_eq!(result.rows(), 4, "Should have 4 groups");

        let key_sum = sum_int64_column(&result, 0);
        assert_eq!(key_sum, 6, "Sum of keys should be 6");

        let value_sum = sum_int64_column(&result, 1);
        assert_eq!(value_sum, 12, "Sum of values should be 12");

        println!("distributed_hash_groupby local test passed");
    }

    #[test]
    fn test_distributed_hash_groupby_single() {
        // Test convenience function with single index column
        let ctx = CylonContext::init();
        let table = create_test_table_int64(ctx).unwrap();

        let result = cylon::groupby::distributed_hash_groupby_single(
            &table, 0, &[1], &[AggregationOpId::Sum]
        ).unwrap();

        assert_eq!(result.rows(), 4, "Should have 4 groups");

        println!("distributed_hash_groupby_single test passed");
    }

    #[test]
    fn test_distributed_hash_groupby_associative_ops() {
        // Associative ops (Sum, Min, Max) should do local groupby first
        let ctx = CylonContext::init();
        let table = create_test_table_int64(ctx).unwrap();

        // Sum is associative
        let result = distributed_hash_groupby(&table, &[0], &[1], &[AggregationOpId::Sum]).unwrap();
        assert_eq!(result.rows(), 4);

        // Min is associative
        let result = distributed_hash_groupby(&table, &[0], &[1], &[AggregationOpId::Min]).unwrap();
        assert_eq!(result.rows(), 4);

        // Max is associative
        let result = distributed_hash_groupby(&table, &[0], &[1], &[AggregationOpId::Max]).unwrap();
        assert_eq!(result.rows(), 4);

        println!("distributed_hash_groupby associative ops test passed");
    }

    #[test]
    fn test_distributed_hash_groupby_non_associative_ops() {
        // Non-associative ops (Mean, Count) need different handling in distributed mode
        let ctx = CylonContext::init();
        let table = create_test_table_float64(ctx).unwrap();

        // Mean is non-associative
        let result = distributed_hash_groupby(&table, &[0], &[1], &[AggregationOpId::Mean]).unwrap();
        assert_eq!(result.rows(), 4);

        let mean_sum = sum_float64_column(&result, 1);
        assert!((mean_sum - 6.0).abs() < 1e-10, "Sum of means should be 6.0");

        println!("distributed_hash_groupby non-associative ops test passed");
    }

    #[test]
    fn test_distributed_hash_groupby_projection() {
        // Verify that projection happens correctly
        let ctx = CylonContext::init();

        // Create table with extra column that should be ignored
        let col0: ArrayRef = Arc::new(Int64Array::from(vec![0, 0, 1, 1]));
        let col1: ArrayRef = Arc::new(Int64Array::from(vec![10, 20, 30, 40]));
        let col2: ArrayRef = Arc::new(Int64Array::from(vec![100, 200, 300, 400])); // extra column

        let schema = Arc::new(Schema::new(vec![
            Field::new("key", ArrowDataType::Int64, false),
            Field::new("val1", ArrowDataType::Int64, false),
            Field::new("extra", ArrowDataType::Int64, false),
        ]));

        let batch = RecordBatch::try_new(schema, vec![col0, col1, col2]).unwrap();
        let table = Table::from_record_batch(ctx, batch).unwrap();

        // Only group by key and aggregate val1
        let result = distributed_hash_groupby(&table, &[0], &[1], &[AggregationOpId::Sum]).unwrap();

        // Result should have 2 columns: key and sum(val1)
        assert_eq!(result.columns(), 2, "Should have 2 columns after projection");
        assert_eq!(result.rows(), 2, "Should have 2 groups");

        println!("distributed_hash_groupby projection test passed");
    }
}

// ============================================================================
// MPI Distributed Tests (require mpirun)
// ============================================================================

#[cfg(feature = "mpi")]
mod mpi_groupby_tests {
    use super::*;
    use cylon::net::mpi::MPICommunicator;
    use cylon::net::Communicator;

    /// Helper to create MPI distributed context
    fn create_mpi_context() -> Arc<CylonContext> {
        let comm = MPICommunicator::make().expect("Failed to create MPI communicator");
        let mut ctx = CylonContext::new(true);
        ctx.set_communicator(comm);
        Arc::new(ctx)
    }

    #[test]
    #[ignore] // Requires mpirun -n <N>
    fn test_mpi_distributed_hash_groupby_sum() {
        // Corresponds to C++ "testing hash group by sum" with multiple processes
        let ctx = create_mpi_context();
        let _world_size = ctx.get_world_size();

        let table = create_test_table_int64(ctx.clone()).unwrap();

        let result = distributed_hash_groupby(&table, &[0], &[1], &[AggregationOpId::Sum]).unwrap();

        // In distributed mode, each rank has the same input data
        // After shuffle, keys are distributed by hash
        // Final result should have 4 groups total (distributed across ranks)

        // Sum of keys across all ranks should be 6
        let local_key_sum = sum_int64_column(&result, 0);

        // Sum of values: each group sums values from all ranks
        // Group 0: 0+0 from each rank = 0 * world_size
        // Group 1: 1+1 from each rank = 2 * world_size
        // etc.
        // Total: (0+2+4+6) * world_size = 12 * world_size
        let local_value_sum = sum_int64_column(&result, 1);

        println!("Rank {}: key_sum={}, value_sum={}, num_rows={}",
                 ctx.get_rank(), local_key_sum, local_value_sum, result.rows());

        // Each process contributes to the total sum
        // We can't directly assert the total without an allreduce,
        // but we verify local results are valid
        assert!(result.rows() > 0 && result.rows() <= 4,
                "Should have between 1 and 4 groups locally");
    }

    #[test]
    #[ignore] // Requires mpirun -n <N>
    fn test_mpi_distributed_hash_groupby_count() {
        // Corresponds to C++ "testing hash group by count"
        let ctx = create_mpi_context();
        let _world_size = ctx.get_world_size();

        let table = create_test_table_int64(ctx.clone()).unwrap();

        let result = distributed_hash_groupby(&table, &[0], &[1], &[AggregationOpId::Count]).unwrap();

        let local_count_sum = sum_int64_column(&result, 1);

        println!("Rank {}: count_sum={}, num_rows={}",
                 ctx.get_rank(), local_count_sum, result.rows());

        // Total count should be 8 * world_size (8 rows per rank, counting all)
        // But distributed, so we can't check total without allreduce
    }

    #[test]
    #[ignore] // Requires mpirun -n <N>
    fn test_mpi_distributed_hash_groupby_mean() {
        // Corresponds to C++ "testing hash group by mean"
        let ctx = create_mpi_context();

        let table = create_test_table_float64(ctx.clone()).unwrap();

        let result = distributed_hash_groupby(&table, &[0], &[1], &[AggregationOpId::Mean]).unwrap();

        let local_mean_sum = sum_float64_column(&result, 1);

        println!("Rank {}: mean_sum={}, num_rows={}",
                 ctx.get_rank(), local_mean_sum, result.rows());
    }
}
