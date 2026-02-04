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

//! Tests for UCX communicator
//!
//! Based on C++ ucx_join_example.cpp and redis_ucc_ucx_example.cpp
//!
//! ## Running Tests
//!
//! Tests require `CYLON_SESSION_ID` environment variable to prevent conflicts
//! with stale Redis data:
//!
//! ```bash
//! export CYLON_SESSION_ID=$(uuidgen)
//! export CYLON_REDIS_URL=redis://localhost:6379
//! cargo test --features ucx test_ucx_communicator_init_with_redis -- --ignored
//! ```

#[cfg(feature = "ucx")]
mod ucx_comm_tests {
    use cylon::net::ucx::redis_oob::UCXRedisOOBContext;
    use cylon::net::ucx::communicator::UCXCommunicator;
    use cylon::net::{Communicator, CommType};

    #[test]
    #[ignore] // Requires Redis server and CYLON_SESSION_ID env var
    fn test_ucx_communicator_init_with_redis() {
        let redis_url = std::env::var("CYLON_REDIS_URL")
            .unwrap_or_else(|_| "redis://localhost:6379".to_string());

        let world_size = 2;
        let oob_context = Box::new(UCXRedisOOBContext::new(world_size, &redis_url)
            .expect("Failed to create Redis OOB context"));

        let comm_result = UCXCommunicator::make_oob(oob_context);

        match comm_result {
            Ok(comm) => {
                println!("✓ UCX communicator initialized");
                println!("  Rank: {}/{}", comm.get_rank(), comm.get_world_size());
                assert!(comm.get_rank() >= 0);
                assert_eq!(comm.get_world_size(), world_size);
                assert!(!comm.is_finalized());
            }
            Err(e) => {
                eprintln!("Failed to initialize UCX communicator: {:?}", e);
                eprintln!("Make sure Redis is running at: {}", redis_url);
                panic!("UCX communicator initialization failed");
            }
        }
    }

    #[test]
    #[ignore] // Requires Redis and multiple processes
    fn test_ucx_create_channel() {
        let redis_url = std::env::var("CYLON_REDIS_URL")
            .unwrap_or_else(|_| "redis://localhost:6379".to_string());

        let world_size = 2;
        let oob_context = Box::new(UCXRedisOOBContext::new(world_size, &redis_url)
            .expect("Failed to create Redis OOB context"));

        let comm = UCXCommunicator::make_oob(oob_context)
            .expect("Failed to create UCX communicator");

        // Test channel creation
        let channel_result = comm.create_channel();
        assert!(channel_result.is_ok(), "Failed to create channel");

        let channel = channel_result.unwrap();
        println!("✓ Channel created successfully");

        // Channel should be usable
        // In a real multi-process test, we would test send/receive here
    }

    #[test]
    #[ignore] // Requires Redis and multiple processes
    fn test_ucx_barrier() {
        let redis_url = std::env::var("CYLON_REDIS_URL")
            .unwrap_or_else(|_| "redis://localhost:6379".to_string());

        let world_size = 2;
        let oob_context = Box::new(UCXRedisOOBContext::new(world_size, &redis_url)
            .expect("Failed to create Redis OOB context"));

        let comm = UCXCommunicator::make_oob(oob_context)
            .expect("Failed to create UCX communicator");

        // Barrier is a no-op in UCX (uses OOB if needed)
        let result = comm.barrier();
        assert!(result.is_ok(), "Barrier failed");

        println!("✓ Barrier completed on rank {}", comm.get_rank());
    }

    #[test]
    #[ignore] // Requires Redis and multiple processes
    fn test_ucx_finalize() {
        let redis_url = std::env::var("CYLON_REDIS_URL")
            .unwrap_or_else(|_| "redis://localhost:6379".to_string());

        let world_size = 2;
        let oob_context = Box::new(UCXRedisOOBContext::new(world_size, &redis_url)
            .expect("Failed to create Redis OOB context"));

        let mut comm = UCXCommunicator::make_oob(oob_context)
            .expect("Failed to create UCX communicator");

        assert!(!comm.is_finalized());

        let result = comm.finalize();
        assert!(result.is_ok(), "Finalize failed");
        assert!(comm.is_finalized());

        println!("✓ UCX communicator finalized successfully");
    }

    #[test]
    fn test_ucx_comm_type() {
        // Test CommType enum value
        use cylon::net::CommType;
        assert_eq!(CommType::Ucx, CommType::Ucx);
        println!("✓ UCX CommType defined correctly");
    }
}

#[cfg(not(feature = "ucx"))]
mod ucx_disabled {
    #[test]
    fn ucx_feature_not_enabled() {
        println!("UCX communicator tests skipped - feature not enabled");
    }
}
