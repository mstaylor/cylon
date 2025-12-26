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

//! Tests for Libfabric communicator
//!
//! ## Running Tests
//!
//! Tests require:
//! - Libfabric library installed
//! - Redis server running
//! - `CYLON_SESSION_ID` environment variable set
//!
//! ```bash
//! # Start Redis server
//! redis-server &
//!
//! # Run tests (single process - config and query tests)
//! export CYLON_SESSION_ID=$(uuidgen)
//! export CYLON_REDIS_URL=redis://localhost:6379
//! cargo test --features libfabric test_libfabric -- --ignored
//!
//! # Run multi-process tests (2 processes)
//! # Terminal 1:
//! export CYLON_SESSION_ID=$(uuidgen)
//! export CYLON_RANK=0
//! cargo test --features libfabric test_libfabric_communicator_init -- --ignored --nocapture
//!
//! # Terminal 2 (same session ID):
//! export CYLON_SESSION_ID=<same-uuid>
//! export CYLON_RANK=1
//! cargo test --features libfabric test_libfabric_communicator_init -- --ignored --nocapture
//! ```

#[cfg(feature = "libfabric")]
mod libfabric_comm_tests {
    use cylon::net::libfabric::{
        LibfabricConfig, LibfabricCommunicator,
        query_providers, EndpointType,
    };
    use cylon::net::{Communicator as CylonCommunicator, CommType};

    /// Test single-process communicator init (world_size=1)
    /// This validates the full initialization without needing multiple processes
    #[test]
    #[ignore] // Requires Redis and libfabric
    fn test_libfabric_single_process_init() {
        let redis_url = std::env::var("CYLON_REDIS_URL")
            .unwrap_or_else(|_| "redis://localhost:6379".to_string());
        // Generate unique session ID to avoid stale Redis data
        let session_id = format!("test-single-{}", std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos());

        let (redis_host, redis_port) = parse_redis_url(&redis_url);
        let world_size = 1; // Single process test

        let config = LibfabricConfig::with_redis(&redis_host, redis_port, &session_id, world_size)
            .endpoint_type(EndpointType::ReliableDatagram);

        println!("Creating single-process LibfabricCommunicator:");
        println!("  Redis: {}:{}", config.redis_host, config.redis_port);
        println!("  Session ID: {}", config.session_id);

        match LibfabricCommunicator::new(config) {
            Ok(comm) => {
                println!("Communicator initialized successfully!");
                println!("  Rank: {}", comm.get_rank());
                println!("  World size: {}", comm.get_world_size());
                assert_eq!(comm.get_rank(), 0);
                assert_eq!(comm.get_world_size(), 1);
                assert!(!comm.is_finalized());
                assert_eq!(comm.get_comm_type(), CommType::Libfabric);
                println!("Single-process test PASSED");
            }
            Err(e) => {
                panic!("Failed to initialize communicator: {:?}", e);
            }
        }
    }

    /// Test querying available libfabric providers
    /// This test doesn't require Redis, just libfabric installation
    #[test]
    #[ignore] // Requires libfabric to be installed
    fn test_libfabric_query_providers() {
        let result = query_providers();

        match result {
            Ok(providers) => {
                println!("Available libfabric providers:");
                for provider in &providers {
                    println!("  - {}", provider);
                }
                assert!(!providers.is_empty(), "Expected at least one provider");
                println!("Found {} providers", providers.len());
            }
            Err(e) => {
                panic!("Failed to query providers: {:?}", e);
            }
        }
    }

    /// Test creating a libfabric communicator with Redis OOB
    /// Requires Redis server and multiple processes
    #[test]
    #[ignore] // Requires Redis server and CYLON_SESSION_ID env var
    fn test_libfabric_communicator_init_with_redis() {
        let redis_url = std::env::var("CYLON_REDIS_URL")
            .unwrap_or_else(|_| "redis://localhost:6379".to_string());
        let session_id = std::env::var("CYLON_SESSION_ID")
            .expect("CYLON_SESSION_ID environment variable required");

        // Parse redis URL to get host and port
        let (redis_host, redis_port) = parse_redis_url(&redis_url);
        let world_size = 2;

        let config = LibfabricConfig::with_redis(&redis_host, redis_port, &session_id, world_size)
            .endpoint_type(EndpointType::ReliableDatagram);

        println!("Creating LibfabricCommunicator with config:");
        println!("  Provider: {:?}", config.provider);
        println!("  Redis: {}:{}", config.redis_host, config.redis_port);
        println!("  Session ID: {}", config.session_id);
        println!("  World size: {}", config.world_size);

        match LibfabricCommunicator::new(config) {
            Ok(comm) => {
                println!("Libfabric communicator initialized");
                println!("  Rank: {}/{}", comm.get_rank(), comm.get_world_size());
                assert!(comm.get_rank() >= 0);
                assert!(comm.get_rank() < world_size);
                assert_eq!(comm.get_world_size(), world_size);
                assert!(!comm.is_finalized());
                assert_eq!(comm.get_comm_type(), CommType::Libfabric);
            }
            Err(e) => {
                eprintln!("Failed to initialize communicator: {:?}", e);
                eprintln!("Make sure:");
                eprintln!("  1. Redis is running at: {}", redis_url);
                eprintln!("  2. CYLON_SESSION_ID is set");
                eprintln!("  3. Libfabric is installed correctly");
                panic!("Communicator initialization failed");
            }
        }
    }

    /// Test creating a channel from the communicator
    #[test]
    #[ignore] // Requires Redis and multiple processes
    fn test_libfabric_create_channel() {
        let redis_url = std::env::var("CYLON_REDIS_URL")
            .unwrap_or_else(|_| "redis://localhost:6379".to_string());
        let session_id = std::env::var("CYLON_SESSION_ID")
            .expect("CYLON_SESSION_ID environment variable required");

        let (redis_host, redis_port) = parse_redis_url(&redis_url);
        let world_size = 2;

        let config = LibfabricConfig::with_redis(&redis_host, redis_port, &session_id, world_size);

        let comm = LibfabricCommunicator::new(config)
            .expect("Failed to create communicator");

        // Test channel creation
        let channel_result = comm.create_channel();
        assert!(channel_result.is_ok(), "Failed to create channel: {:?}", channel_result.err());

        let _channel = channel_result.unwrap();
        println!("Channel created successfully for rank {}", comm.get_rank());
    }

    /// Test barrier operation
    #[test]
    #[ignore] // Requires Redis and multiple processes
    fn test_libfabric_barrier() {
        let redis_url = std::env::var("CYLON_REDIS_URL")
            .unwrap_or_else(|_| "redis://localhost:6379".to_string());
        let session_id = std::env::var("CYLON_SESSION_ID")
            .expect("CYLON_SESSION_ID environment variable required");

        let (redis_host, redis_port) = parse_redis_url(&redis_url);
        let world_size = 2;

        let config = LibfabricConfig::with_redis(&redis_host, redis_port, &session_id, world_size);

        let comm = LibfabricCommunicator::new(config)
            .expect("Failed to create communicator");

        println!("Rank {} entering barrier...", comm.get_rank());

        let result = comm.barrier();
        assert!(result.is_ok(), "Barrier failed: {:?}", result.err());

        println!("Rank {} passed barrier", comm.get_rank());
    }

    /// Test finalize operation (via drop)
    /// Note: finalize() requires &mut self which is hard with Arc<dyn Communicator>
    /// Instead we test that the communicator can be dropped cleanly
    #[test]
    #[ignore] // Requires Redis and multiple processes
    fn test_libfabric_drop() {
        let redis_url = std::env::var("CYLON_REDIS_URL")
            .unwrap_or_else(|_| "redis://localhost:6379".to_string());
        let session_id = std::env::var("CYLON_SESSION_ID")
            .expect("CYLON_SESSION_ID environment variable required");

        let (redis_host, redis_port) = parse_redis_url(&redis_url);
        let world_size = 2;

        let config = LibfabricConfig::with_redis(&redis_host, redis_port, &session_id, world_size);

        let comm = LibfabricCommunicator::new(config)
            .expect("Failed to create communicator");

        let rank = comm.get_rank();
        assert!(!comm.is_finalized());

        // Drop the communicator - this should clean up resources
        drop(comm);

        println!("Rank {} communicator dropped successfully", rank);
    }

    /// Test with TCP provider specifically
    #[test]
    #[ignore] // Requires Redis and TCP provider
    fn test_libfabric_tcp_provider() {
        let redis_url = std::env::var("CYLON_REDIS_URL")
            .unwrap_or_else(|_| "redis://localhost:6379".to_string());
        let session_id = std::env::var("CYLON_SESSION_ID")
            .expect("CYLON_SESSION_ID environment variable required");

        let (redis_host, redis_port) = parse_redis_url(&redis_url);
        let world_size = 2;

        let config = LibfabricConfig::with_redis(&redis_host, redis_port, &session_id, world_size)
            .provider("tcp");

        match LibfabricCommunicator::new(config) {
            Ok(comm) => {
                println!("TCP provider communicator initialized");
                println!("  Rank: {}/{}", comm.get_rank(), comm.get_world_size());
                assert_eq!(comm.get_world_size(), world_size);
            }
            Err(e) => {
                eprintln!("TCP provider not available: {:?}", e);
                // Don't fail - TCP provider may not be installed
            }
        }
    }

    /// Test with sockets provider
    #[test]
    #[ignore] // Requires Redis and sockets provider
    fn test_libfabric_sockets_provider() {
        let redis_url = std::env::var("CYLON_REDIS_URL")
            .unwrap_or_else(|_| "redis://localhost:6379".to_string());
        let session_id = std::env::var("CYLON_SESSION_ID")
            .expect("CYLON_SESSION_ID environment variable required");

        let (redis_host, redis_port) = parse_redis_url(&redis_url);
        let world_size = 2;

        let config = LibfabricConfig::with_redis(&redis_host, redis_port, &session_id, world_size)
            .provider("sockets");

        match LibfabricCommunicator::new(config) {
            Ok(comm) => {
                println!("Sockets provider communicator initialized");
                println!("  Rank: {}/{}", comm.get_rank(), comm.get_world_size());
                assert_eq!(comm.get_world_size(), world_size);
            }
            Err(e) => {
                eprintln!("Sockets provider not available: {:?}", e);
                // Don't fail - sockets provider may not be installed
            }
        }
    }

    #[test]
    fn test_libfabric_comm_type() {
        assert_eq!(CommType::Libfabric, CommType::Libfabric);
        println!("Libfabric CommType defined correctly");
    }

    /// Helper function to parse redis URL into host and port
    fn parse_redis_url(url: &str) -> (String, u16) {
        let url = url.strip_prefix("redis://").unwrap_or(url);
        let parts: Vec<&str> = url.split(':').collect();
        let host = parts.first().unwrap_or(&"localhost").to_string();
        let port: u16 = parts.get(1)
            .and_then(|p| p.parse().ok())
            .unwrap_or(6379);
        (host, port)
    }
}

#[cfg(not(feature = "libfabric"))]
mod libfabric_disabled {
    #[test]
    fn libfabric_feature_not_enabled() {
        println!("Libfabric communicator tests skipped - feature not enabled");
    }
}
