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

//! Tests for UCX channel operations
//!
//! Based on C++ UCX examples and usage patterns
//!
//! ## Running Tests
//!
//! Tests require `CYLON_SESSION_ID` environment variable:
//!
//! ```bash
//! export CYLON_SESSION_ID=$(uuidgen)
//! export CYLON_REDIS_URL=redis://localhost:6379
//! cargo test --features ucx test_ucx_channel_create -- --ignored
//! ```

#[cfg(feature = "ucx")]
mod ucx_channel_tests {
    use cylon::net::ucx::redis_oob::UCXRedisOOBContext;
    use cylon::net::ucx::communicator::UCXCommunicator;
    use cylon::net::Communicator;

    #[test]
    #[ignore] // Requires Redis, CYLON_SESSION_ID, and multiple processes
    fn test_ucx_channel_create() {
        // This test requires a Redis server and should be run with multiple processes
        // Similar to C++ redis_ucc_ucx_example.cpp pattern

        let redis_url = std::env::var("CYLON_REDIS_URL")
            .unwrap_or_else(|_| "redis://localhost:6379".to_string());

        let world_size: i32 = std::env::var("CYLON_WORLD_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(2);

        let oob_context = Box::new(UCXRedisOOBContext::new(world_size, &redis_url)
            .expect("Failed to create Redis OOB context"));

        let comm_result = UCXCommunicator::make_oob(oob_context);
        assert!(comm_result.is_ok(), "Failed to create UCX communicator");

        let comm = comm_result.unwrap();
        assert_eq!(comm.get_world_size(), world_size);
        assert!(comm.get_rank() >= 0 && comm.get_rank() < world_size);

        // Create channel
        let channel_result = comm.create_channel();
        assert!(channel_result.is_ok(), "Failed to create UCX channel");

        println!("UCX channel created successfully on rank {}/{}",
                 comm.get_rank(), comm.get_world_size());
    }

    #[test]
    fn test_ucx_channel_interface() {
        // Test that UCXChannel implements the Channel trait correctly
        // This is a compile-time check
        use cylon::net::Channel;

        fn assert_channel<T: Channel>() {}

        // This will fail to compile if UCXChannel doesn't implement Channel
        assert_channel::<cylon::net::ucx::channel::UCXChannel>();
    }
}

#[cfg(not(feature = "ucx"))]
mod ucx_disabled {
    #[test]
    fn ucx_feature_not_enabled() {
        println!("UCX tests skipped - feature not enabled");
    }
}
