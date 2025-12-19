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

//! Tests for MPI Column and Scalar operations
//!
//! These tests verify the MpiAllReduceImpl and MpiAllgatherImpl implementations.
//!
//! To run distributed tests:
//! ```
//! OMPI_CC=gcc CC=gcc mpirun -n 4 cargo test --features mpi --test mpi_column_scalar_test -- --ignored
//! ```

#[cfg(feature = "mpi")]
mod mpi_column_scalar_tests {
    use std::sync::Arc;
    use arrow::array::{Int64Array, Float64Array, Array, ArrayRef};

    use cylon::table::Column;
    use cylon::scalar::Scalar;
    use cylon::net::comm_operations::ReduceOp;
    use cylon::net::ops::base_ops::{AllReduceImpl, AllGatherImpl};

    // =========================================================================
    // Unit Tests (single process, no MPI required)
    // =========================================================================

    #[test]
    fn test_column_creation() {
        // Test that we can create Column objects for use in operations
        let array: ArrayRef = Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5]));
        let column = Column::new(array);

        assert_eq!(column.length(), 5);

        println!("✓ Column creation test passed");
    }

    #[test]
    fn test_scalar_creation() {
        // Test that we can create Scalar objects for use in operations
        let array: ArrayRef = Arc::new(Int64Array::from(vec![42]));
        let scalar = Scalar::new(array);

        assert!(scalar.is_valid());

        println!("✓ Scalar creation test passed");
    }

    #[test]
    fn test_column_from_different_types() {
        // Test Column creation with different Arrow types

        // Int64
        let int_array: ArrayRef = Arc::new(Int64Array::from(vec![1, 2, 3]));
        let int_col = Column::new(int_array);
        assert_eq!(int_col.length(), 3);

        // Float64
        let float_array: ArrayRef = Arc::new(Float64Array::from(vec![1.1, 2.2, 3.3]));
        let float_col = Column::new(float_array);
        assert_eq!(float_col.length(), 3);

        println!("✓ Column type variants test passed");
    }

    #[test]
    fn test_reduce_op_variants() {
        // Test that all ReduceOp variants are available
        let ops = vec![
            ReduceOp::Sum,
            ReduceOp::Min,
            ReduceOp::Max,
            ReduceOp::Prod,
            ReduceOp::Land,
            ReduceOp::Lor,
            ReduceOp::Band,
            ReduceOp::Bor,
        ];

        assert_eq!(ops.len(), 8);
        println!("✓ ReduceOp variants test passed");
    }

    // =========================================================================
    // MPI Implementation Structure Tests
    // =========================================================================

    #[test]
    fn test_mpi_allreduce_impl_creation() {
        use mpi::environment::Universe;
        use std::sync::Mutex;
        use cylon::net::mpi::MpiAllReduceImpl;

        // Create with null universe (won't actually work, but tests struct creation)
        let universe: Arc<Mutex<Option<Universe>>> = Arc::new(Mutex::new(None));
        let _impl = MpiAllReduceImpl::new(universe);

        println!("✓ MpiAllReduceImpl creation test passed");
    }

    #[test]
    fn test_mpi_allgather_impl_creation() {
        use mpi::environment::Universe;
        use std::sync::Mutex;
        use cylon::net::mpi::MpiAllgatherImpl;

        // Create with null universe (won't actually work, but tests struct creation)
        let universe: Arc<Mutex<Option<Universe>>> = Arc::new(Mutex::new(None));
        let _impl = MpiAllgatherImpl::new(universe, 4);

        println!("✓ MpiAllgatherImpl creation test passed");
    }

    #[test]
    fn test_mpi_impl_trait_bounds() {
        use cylon::net::mpi::{MpiAllReduceImpl, MpiAllgatherImpl};

        // Compile-time test that our implementations satisfy the traits
        fn assert_allreduce<T: AllReduceImpl>() {}
        fn assert_allgather<T: AllGatherImpl>() {}

        assert_allreduce::<MpiAllReduceImpl>();
        assert_allgather::<MpiAllgatherImpl>();

        println!("✓ MPI trait bounds test passed");
    }

    // =========================================================================
    // MPI Distributed Tests (require mpirun)
    // =========================================================================

    #[test]
    #[ignore] // Run with: mpirun -n 4 cargo test --features mpi -- --ignored
    fn test_mpi_allreduce_column_sum() {
        use cylon::net::mpi::MPICommunicator;
        use cylon::net::Communicator;

        // Initialize MPI communicator
        let comm = MPICommunicator::make().expect("Failed to create MPI communicator");
        let rank = comm.get_rank();
        let world_size = comm.get_world_size();

        // Each process contributes [rank, rank, rank]
        let array: ArrayRef = Arc::new(Int64Array::from(vec![rank as i64; 3]));
        let column = Column::new(array);

        // AllReduce with SUM
        let result = comm.all_reduce_column(&column, ReduceOp::Sum)
            .expect("AllReduce failed");

        // Expected sum = 0 + 1 + 2 + ... + (world_size-1) = world_size * (world_size-1) / 2
        let expected_sum = (world_size * (world_size - 1) / 2) as i64;

        // Verify result
        let result_array = result.data();
        let int_array = result_array.as_any().downcast_ref::<Int64Array>().unwrap();

        for i in 0..int_array.len() {
            assert_eq!(int_array.value(i), expected_sum,
                "Rank {}: AllReduce SUM mismatch at index {}", rank, i);
        }

        println!("Rank {}: ✓ MPI AllReduce Column SUM test passed", rank);
    }

    #[test]
    #[ignore] // Run with: mpirun -n 4 cargo test --features mpi -- --ignored
    fn test_mpi_allreduce_column_max() {
        use cylon::net::mpi::MPICommunicator;
        use cylon::net::Communicator;

        let comm = MPICommunicator::make().expect("Failed to create MPI communicator");
        let rank = comm.get_rank();
        let world_size = comm.get_world_size();

        // Each process contributes [rank * 10]
        let array: ArrayRef = Arc::new(Int64Array::from(vec![rank as i64 * 10]));
        let column = Column::new(array);

        // AllReduce with MAX
        let result = comm.all_reduce_column(&column, ReduceOp::Max)
            .expect("AllReduce MAX failed");

        // Expected max = (world_size - 1) * 10
        let expected_max = (world_size - 1) as i64 * 10;

        let result_array = result.data();
        let int_array = result_array.as_any().downcast_ref::<Int64Array>().unwrap();

        assert_eq!(int_array.value(0), expected_max,
            "Rank {}: AllReduce MAX mismatch", rank);

        println!("Rank {}: ✓ MPI AllReduce Column MAX test passed", rank);
    }

    #[test]
    #[ignore] // Run with: mpirun -n 4 cargo test --features mpi -- --ignored
    fn test_mpi_allreduce_scalar_sum() {
        use cylon::net::mpi::MPICommunicator;
        use cylon::net::Communicator;

        let comm = MPICommunicator::make().expect("Failed to create MPI communicator");
        let rank = comm.get_rank();
        let world_size = comm.get_world_size();

        // Each process contributes its rank as a scalar
        let array: ArrayRef = Arc::new(Int64Array::from(vec![rank as i64]));
        let scalar = Scalar::new(array);

        // AllReduce with SUM
        let result = comm.all_reduce_scalar(&scalar, ReduceOp::Sum)
            .expect("AllReduce Scalar failed");

        // Expected sum = 0 + 1 + 2 + ... + (world_size-1)
        let expected_sum = (world_size * (world_size - 1) / 2) as i64;

        let result_array = result.data();
        let int_array = result_array.as_any().downcast_ref::<Int64Array>().unwrap();

        assert_eq!(int_array.value(0), expected_sum,
            "Rank {}: AllReduce Scalar SUM mismatch", rank);

        println!("Rank {}: ✓ MPI AllReduce Scalar SUM test passed", rank);
    }

    #[test]
    #[ignore] // Run with: mpirun -n 4 cargo test --features mpi -- --ignored
    fn test_mpi_allgather_column() {
        use cylon::net::mpi::MPICommunicator;
        use cylon::net::Communicator;

        let comm = MPICommunicator::make().expect("Failed to create MPI communicator");
        let rank = comm.get_rank();
        let world_size = comm.get_world_size();

        // Each process contributes a column with 2 elements: [rank*10, rank*10+1]
        let array: ArrayRef = Arc::new(Int64Array::from(vec![rank as i64 * 10, rank as i64 * 10 + 1]));
        let column = Column::new(array);

        // Allgather
        let results = comm.allgather_column(&column)
            .expect("Allgather Column failed");

        // Should get world_size columns back
        assert_eq!(results.len(), world_size as usize,
            "Rank {}: Expected {} columns, got {}", rank, world_size, results.len());

        // Verify each column
        for (i, col) in results.iter().enumerate() {
            let col_array = col.data();
            let int_array = col_array.as_any().downcast_ref::<Int64Array>().unwrap();

            let expected_0 = i as i64 * 10;
            let expected_1 = i as i64 * 10 + 1;

            assert_eq!(int_array.value(0), expected_0,
                "Rank {}: Column {} value 0 mismatch", rank, i);
            assert_eq!(int_array.value(1), expected_1,
                "Rank {}: Column {} value 1 mismatch", rank, i);
        }

        println!("Rank {}: ✓ MPI Allgather Column test passed", rank);
    }

    #[test]
    #[ignore] // Run with: mpirun -n 4 cargo test --features mpi -- --ignored
    fn test_mpi_allgather_scalar() {
        use cylon::net::mpi::MPICommunicator;
        use cylon::net::Communicator;

        let comm = MPICommunicator::make().expect("Failed to create MPI communicator");
        let rank = comm.get_rank();
        let world_size = comm.get_world_size();

        // Each process contributes its rank as a scalar
        let array: ArrayRef = Arc::new(Int64Array::from(vec![rank as i64 * 100]));
        let scalar = Scalar::new(array);

        // Allgather - returns a Column with all scalars
        let result = comm.allgather_scalar(&scalar)
            .expect("Allgather Scalar failed");

        // Result should be a column with world_size elements
        assert_eq!(result.length(), world_size as i64,
            "Rank {}: Expected {} elements, got {}", rank, world_size, result.length());

        let result_array = result.data();
        let int_array = result_array.as_any().downcast_ref::<Int64Array>().unwrap();

        // Verify each element
        for i in 0..world_size as usize {
            let expected = i as i64 * 100;
            assert_eq!(int_array.value(i), expected,
                "Rank {}: Element {} mismatch", rank, i);
        }

        println!("Rank {}: ✓ MPI Allgather Scalar test passed", rank);
    }

    #[test]
    #[ignore] // Run with: mpirun -n 4 cargo test --features mpi -- --ignored
    fn test_mpi_allreduce_float64() {
        use cylon::net::mpi::MPICommunicator;
        use cylon::net::Communicator;

        let comm = MPICommunicator::make().expect("Failed to create MPI communicator");
        let rank = comm.get_rank();
        let world_size = comm.get_world_size();

        // Each process contributes [rank as f64]
        let array: ArrayRef = Arc::new(Float64Array::from(vec![rank as f64]));
        let column = Column::new(array);

        // AllReduce with SUM
        let result = comm.all_reduce_column(&column, ReduceOp::Sum)
            .expect("AllReduce Float64 failed");

        // Expected sum
        let expected_sum = (0..world_size).map(|r| r as f64).sum::<f64>();

        let result_array = result.data();
        let float_array = result_array.as_any().downcast_ref::<Float64Array>().unwrap();

        let diff = (float_array.value(0) - expected_sum).abs();
        assert!(diff < 1e-10, "Rank {}: Float64 AllReduce mismatch: got {}, expected {}",
            rank, float_array.value(0), expected_sum);

        println!("Rank {}: ✓ MPI AllReduce Float64 test passed", rank);
    }
}

#[cfg(not(feature = "mpi"))]
mod mpi_disabled {
    #[test]
    fn mpi_feature_not_enabled() {
        println!("MPI Column/Scalar tests skipped - feature not enabled");
        println!("To run: cargo test --features mpi --test mpi_column_scalar_test");
    }
}
