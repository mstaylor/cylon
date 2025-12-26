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

//! Tests for Libfabric channel operations
//!
//! These tests verify channel creation and basic lifecycle operations.
//! The Channel trait uses a callback-based API for actual send/receive operations.
//!
//! ## Running Tests
//!
//! Tests require two processes running simultaneously:
//!
//! ```bash
//! # Start Redis server
//! redis-server &
//!
//! # Terminal 1 (Rank 0):
//! export CYLON_SESSION_ID=$(uuidgen)
//! export CYLON_REDIS_URL=redis://localhost:6379
//! cargo test --features libfabric test_libfabric_channel -- --ignored --nocapture
//!
//! # Terminal 2 (Rank 1, same session):
//! export CYLON_SESSION_ID=<same-uuid-as-terminal1>
//! export CYLON_REDIS_URL=redis://localhost:6379
//! cargo test --features libfabric test_libfabric_channel -- --ignored --nocapture
//! ```

#[cfg(feature = "libfabric")]
mod libfabric_channel_tests {
    use cylon::net::libfabric::{LibfabricConfig, LibfabricCommunicator};
    use cylon::net::Communicator as CylonCommunicator;

    /// Test creating a channel from the communicator
    #[test]
    #[ignore] // Requires Redis and libfabric
    fn test_libfabric_channel_create() {
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

        // Test channel creation
        let channel_result = comm.create_channel();
        assert!(channel_result.is_ok(), "Failed to create channel: {:?}", channel_result.err());

        let _channel = channel_result.unwrap();
        println!("Rank {}: Channel created successfully", rank);
    }

    /// Test channel close
    #[test]
    #[ignore] // Requires Redis and libfabric
    fn test_libfabric_channel_close() {
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
        let mut channel = comm.create_channel().expect("Failed to create channel");

        // Close should succeed (returns ())
        channel.close();

        println!("Rank {}: Channel closed successfully", rank);
    }

    /// Test creating multiple channels
    #[test]
    #[ignore] // Requires Redis and libfabric
    fn test_libfabric_multiple_channels() {
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

        // Create multiple channels
        let mut channels = Vec::new();
        for i in 0..3 {
            let channel = comm.create_channel()
                .expect(&format!("Failed to create channel {}", i));
            channels.push(channel);
        }

        println!("Rank {}: Created {} channels", rank, channels.len());
        assert_eq!(channels.len(), 3);

        // Close all channels
        for mut channel in channels {
            channel.close();
        }

        println!("Rank {}: All channels closed", rank);
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
        println!("Libfabric channel tests skipped - feature not enabled");
    }
}
