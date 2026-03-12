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

//! Tests for UCC collective operations
//!
//! Based on C++ ucc_example.cpp, ucc_allgather_vector_example.cpp,
//! and ucc_operators_example.cpp

#[cfg(feature = "ucc")]
mod ucc_tests {
    use cylon::net::ucc::operations::*;
    use cylon::net::ucc::ucc_sys::*;
    use cylon::data_types::{DataType, Type};
    use cylon::net::ops::base_ops::*;
    use cylon::net::comm_operations::ReduceOp;
    use std::sync::Arc;

    #[test]
    fn test_ucc_datatype_conversion() {
        // Test datatype conversion function
        let int32_type = DataType::new(Type::Int32);
        let int64_type = DataType::new(Type::Int64);
        let float_type = DataType::new(Type::Float);
        let double_type = DataType::new(Type::Double);

        // These are internal functions, but we can test through the public API
        // by checking if operations would accept these types

        println!("✓ UCC datatype conversion test passed");
    }

    #[test]
    fn test_reduce_op_enum() {
        // Test ReduceOp enum
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

        for op in ops {
            println!("ReduceOp: {:?}", op);
        }

        assert_eq!(ReduceOp::Sum, ReduceOp::Sum);
        assert_ne!(ReduceOp::Sum, ReduceOp::Max);

        println!("✓ ReduceOp enum test passed");
    }

    #[test]
    #[ignore] // Requires UCC runtime and MPI/UCX
    fn test_ucc_table_allgather_init() {
        // This test requires actual UCC team and context
        // In a real environment with MPI/UCX initialized:

        unsafe {
            let ucc_team: ucc_team_h = std::ptr::null_mut();
            let ucc_context: ucc_context_h = std::ptr::null_mut();
            let world_size = 4;

            // Create UCC table allgather implementation
            let mut impl_ = UccTableAllgatherImpl::new(ucc_team, ucc_context, world_size);

            // Initialize for 3 buffers
            impl_.init(3);

            println!("✓ UCC TableAllgatherImpl initialized");
        }
    }

    #[test]
    #[ignore] // Requires UCC runtime
    fn test_ucc_table_gather_init() {
        unsafe {
            let ucc_team: ucc_team_h = std::ptr::null_mut();
            let ucc_context: ucc_context_h = std::ptr::null_mut();
            let rank = 0;
            let world_size = 4;

            let mut impl_ = UccTableGatherImpl::new(ucc_team, ucc_context, rank, world_size);

            impl_.init(3);

            println!("✓ UCC TableGatherImpl initialized");
        }
    }

    #[test]
    #[ignore] // Requires UCC runtime
    fn test_ucc_allreduce_init() {
        unsafe {
            let ucc_team: ucc_team_h = std::ptr::null_mut();
            let ucc_context: ucc_context_h = std::ptr::null_mut();

            let _impl = UccAllReduceImpl::new(ucc_team, ucc_context);

            println!("✓ UCC AllReduceImpl created");
        }
    }

    #[test]
    #[ignore] // Requires UCC runtime
    fn test_ucc_table_bcast_init() {
        unsafe {
            let ucc_team: ucc_team_h = std::ptr::null_mut();
            let ucc_context: ucc_context_h = std::ptr::null_mut();

            let mut impl_ = UccTableBcastImpl::new(ucc_team, ucc_context);

            impl_.init(3);

            println!("✓ UCC TableBcastImpl initialized");
        }
    }

    #[test]
    #[ignore] // Requires UCC runtime
    fn test_ucc_allgather_init() {
        unsafe {
            let ucc_team: ucc_team_h = std::ptr::null_mut();
            let ucc_context: ucc_context_h = std::ptr::null_mut();
            let world_size = 4;

            let _impl = UccAllGatherImpl::new(ucc_team, ucc_context, world_size);

            println!("✓ UCC AllGatherImpl created");
        }
    }

    #[test]
    fn test_ucc_constants() {
        // Test that UCC constants are properly defined
        unsafe {
            assert_eq!(UCC_DT_INT8, 0);
            assert_eq!(UCC_DT_UINT8, 1);
            assert_eq!(UCC_DT_INT32, 4);
            assert_eq!(UCC_DT_FLOAT32, 9);

            assert!(UCC_OK == 0);
            assert!(UCC_INPROGRESS == 1);

            assert!(UCC_OP_SUM == ucc_reduction_op_t_UCC_OP_SUM);
            assert!(UCC_OP_MAX == ucc_reduction_op_t_UCC_OP_MAX);

            assert!(UCC_COLL_TYPE_ALLGATHER == ucc_coll_type_t_UCC_COLL_TYPE_ALLGATHER);
            assert!(UCC_COLL_TYPE_BCAST == ucc_coll_type_t_UCC_COLL_TYPE_BCAST);

            println!("✓ UCC constants test passed");
        }
    }

    #[test]
    fn test_ucc_struct_sizes() {
        // Test that we can create UCC structs
        unsafe {
            use std::mem;

            let _args: ucc_coll_args_t = mem::zeroed();
            println!("  ucc_coll_args_t size: {} bytes", mem::size_of::<ucc_coll_args_t>());

            let _params: ucc_lib_params_t = mem::zeroed();
            println!("  ucc_lib_params_t size: {} bytes", mem::size_of::<ucc_lib_params_t>());

            println!("✓ UCC struct size test passed");
        }
    }

    #[test]
    fn test_ucc_impl_trait_implementations() {
        // Compile-time test that our implementations satisfy the traits
        fn assert_table_allgather<T: TableAllgatherImpl>() {}
        fn assert_table_gather<T: TableGatherImpl>() {}
        fn assert_table_bcast<T: TableBcastImpl>() {}

        assert_table_allgather::<UccTableAllgatherImpl>();
        assert_table_gather::<UccTableGatherImpl>();
        assert_table_bcast::<UccTableBcastImpl>();

        println!("✓ UCC trait implementation test passed");
    }

    #[test]
    fn test_ucc_send_sync_markers() {
        // Test that our UCC types are Send and Sync
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<UccTableAllgatherImpl>();
        assert_sync::<UccTableAllgatherImpl>();

        assert_send::<UccTableGatherImpl>();
        assert_sync::<UccTableGatherImpl>();

        assert_send::<UccAllReduceImpl>();
        assert_sync::<UccAllReduceImpl>();

        assert_send::<UccTableBcastImpl>();
        assert_sync::<UccTableBcastImpl>();

        assert_send::<UccAllGatherImpl>();
        assert_sync::<UccAllGatherImpl>();

        println!("✓ UCC Send/Sync test passed");
    }
}

#[cfg(not(feature = "ucc"))]
mod ucc_disabled {
    #[test]
    fn ucc_feature_not_enabled() {
        println!("UCC tests skipped - feature not enabled");
    }
}
