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

//! Tests for Redis FMI channel backend
//!
//! ## Test Categories
//!
//! 1. **Unit Tests** - Test configuration, types (no Redis required)
//! 2. **Integration Tests** - Test Redis operations (requires Redis server)
//!
//! ## Running Integration Tests
//!
//! Integration tests require:
//! - Redis server running (default: localhost:6379)
//!
//! ```bash
//! # Start Redis server (in separate terminal)
//! docker run -p 6379:6379 redis:latest
//!
//! # Run tests
//! cargo test --features "fmi,redis" redis_channel_test -- --ignored
//! ```
//!
//! ## Environment Variables
//!
//! - `FMI_TEST_REDIS_HOST` - Redis host (default: localhost)
//! - `FMI_TEST_REDIS_PORT` - Redis port (default: 6379)

// ============================================================================
// Unit Tests - No Redis required
// ============================================================================

#[cfg(all(feature = "fmi", feature = "redis"))]
mod redis_unit_tests {
    use cylon::net::fmi::{RedisBackend, BackendType};

    #[test]
    fn test_redis_backend_defaults() {
        let backend = RedisBackend::new();

        assert_eq!(backend.get_host(), "localhost");
        assert_eq!(backend.get_port(), 6379);
        assert_eq!(backend.get_timeout(), 100);
        assert_eq!(backend.get_max_timeout(), 30000);
        assert_eq!(backend.get_backend_type(), BackendType::Redis);
        assert_eq!(backend.get_name(), "redis");
        assert!(!backend.is_enabled());

        println!("✓ RedisBackend defaults are correct");
    }

    #[test]
    fn test_redis_backend_builder() {
        let backend = RedisBackend::new()
            .set_enabled(true)
            .with_host("redis.example.com")
            .with_port(6380)
            .with_timeout(200)
            .with_max_timeout(60000);

        assert_eq!(backend.get_host(), "redis.example.com");
        assert_eq!(backend.get_port(), 6380);
        assert_eq!(backend.get_timeout(), 200);
        assert_eq!(backend.get_max_timeout(), 60000);
        assert!(backend.is_enabled());

        println!("✓ RedisBackend builder pattern works correctly");
    }
}

// ============================================================================
// Integration Tests - Requires Redis server
// ============================================================================

#[cfg(all(feature = "fmi", feature = "redis"))]
mod redis_integration_tests {
    use std::env;
    use std::sync::Arc;

    use cylon::net::fmi::{RedisBackend, RedisStorage, new_redis_channel, ChannelData};
    use cylon::net::fmi::client_server::StorageBackend;

    /// Get test configuration from environment
    struct RedisTestConfig {
        host: String,
        port: i32,
    }

    impl RedisTestConfig {
        fn new() -> Self {
            Self {
                host: env::var("FMI_TEST_REDIS_HOST").unwrap_or_else(|_| "localhost".to_string()),
                port: env::var("FMI_TEST_REDIS_PORT")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(6379),
            }
        }

        fn backend(&self) -> RedisBackend {
            RedisBackend::new()
                .set_enabled(true)
                .with_host(&self.host)
                .with_port(self.port)
                .with_timeout(100)
                .with_max_timeout(30000)
        }
    }

    // -------------------------------------------------------------------------
    // Blocking Operations Tests
    // -------------------------------------------------------------------------

    #[test]
    #[ignore] // Requires Redis server
    fn test_redis_upload_download_small_data() {
        let config = RedisTestConfig::new();
        let backend = config.backend();
        let storage = RedisStorage::new(&backend).expect("Failed to create Redis storage");

        let test_data = b"hello redis world";
        let key = "fmi_rust_test/small_key_1";

        // Upload
        storage.upload_object(test_data, key).expect("Upload failed");

        // Download
        let mut download_buf = vec![0u8; test_data.len()];
        let found = storage.download_object(&mut download_buf, key).expect("Download failed");

        assert!(found, "Key not found after upload");
        assert_eq!(&download_buf, test_data);

        // Cleanup
        storage.delete_object(key).expect("Delete failed");

        println!("✓ Redis upload and download small data works correctly");
    }

    #[test]
    #[ignore] // Requires Redis server
    fn test_redis_upload_download_binary_data() {
        let config = RedisTestConfig::new();
        let backend = config.backend();
        let storage = RedisStorage::new(&backend).expect("Failed to create Redis storage");

        let binary_data: Vec<u8> = vec![0x00, 0x01, 0x02, 0xFF, 0xFE, 0x00, 0x10];
        let key = "fmi_rust_test/binary_key";

        storage.upload_object(&binary_data, key).expect("Upload failed");

        let mut download_buf = vec![0u8; binary_data.len()];
        let found = storage.download_object(&mut download_buf, key).expect("Download failed");

        assert!(found);
        assert_eq!(download_buf, binary_data);

        storage.delete_object(key).expect("Delete failed");

        println!("✓ Redis upload and download binary data works correctly");
    }

    #[test]
    #[ignore] // Requires Redis server
    fn test_redis_download_nonexistent_key() {
        let config = RedisTestConfig::new();
        let backend = config.backend();
        let storage = RedisStorage::new(&backend).expect("Failed to create Redis storage");

        let mut download_buf = vec![0u8; 64];
        let found = storage.download_object(&mut download_buf, "fmi_rust_test/nonexistent_xyz123")
            .expect("Download should not fail for missing key");

        assert!(!found, "Nonexistent key should return false");

        println!("✓ Redis download nonexistent key returns false");
    }

    #[test]
    #[ignore] // Requires Redis server
    fn test_redis_upload_larger_data() {
        let config = RedisTestConfig::new();
        let backend = config.backend();
        let storage = RedisStorage::new(&backend).expect("Failed to create Redis storage");

        // Create 1KB of test data
        let large_data: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
        let key = "fmi_rust_test/large_key";

        storage.upload_object(&large_data, key).expect("Upload failed");

        let mut download_buf = vec![0u8; large_data.len()];
        let found = storage.download_object(&mut download_buf, key).expect("Download failed");

        assert!(found);
        assert_eq!(download_buf, large_data);

        storage.delete_object(key).expect("Delete failed");

        println!("✓ Redis upload larger data (1KB) works correctly");
    }

    #[test]
    #[ignore] // Requires Redis server
    fn test_redis_delete_object() {
        let config = RedisTestConfig::new();
        let backend = config.backend();
        let storage = RedisStorage::new(&backend).expect("Failed to create Redis storage");

        let test_data = b"to be deleted";
        let key = "fmi_rust_test/delete_test";

        storage.upload_object(test_data, key).expect("Upload failed");

        // Verify it exists
        let mut download_buf = vec![0u8; test_data.len()];
        assert!(storage.download_object(&mut download_buf, key).expect("Download failed"));

        // Delete
        storage.delete_object(key).expect("Delete failed");

        // Verify it's gone
        assert!(!storage.download_object(&mut download_buf, key).expect("Download failed"));

        println!("✓ Redis delete object works correctly");
    }

    #[test]
    #[ignore] // Requires Redis server
    fn test_redis_get_object_names() {
        let config = RedisTestConfig::new();
        let backend = config.backend();
        let storage = RedisStorage::new(&backend).expect("Failed to create Redis storage");

        // Upload a few objects with unique prefix
        let test_data = b"test";
        let key1 = "fmi_rust_test_list/key_1";
        let key2 = "fmi_rust_test_list/key_2";

        storage.upload_object(test_data, key1).expect("Upload 1 failed");
        storage.upload_object(test_data, key2).expect("Upload 2 failed");

        let names = storage.get_object_names().expect("Get names failed");
        let found1 = names.iter().any(|s| s == key1);
        let found2 = names.iter().any(|s| s == key2);

        assert!(found1, "Key 1 not found in object names");
        assert!(found2, "Key 2 not found in object names");

        // Cleanup
        storage.delete_object(key1).expect("Delete 1 failed");
        storage.delete_object(key2).expect("Delete 2 failed");

        println!("✓ Redis get object names works correctly");
    }

    // -------------------------------------------------------------------------
    // Channel Tests
    // -------------------------------------------------------------------------

    #[test]
    #[ignore] // Requires Redis server
    fn test_redis_channel_creation() {
        let config = RedisTestConfig::new();
        let backend = config.backend();

        let channel = new_redis_channel(&backend);
        assert!(channel.is_ok(), "Failed to create Redis channel: {:?}", channel.err());

        println!("✓ Redis channel creation works correctly");
    }

    // -------------------------------------------------------------------------
    // Async Operations Tests (using mutable storage)
    // -------------------------------------------------------------------------

    #[test]
    #[ignore] // Requires Redis server
    fn test_redis_async_upload() {
        use cylon::net::fmi::redis_channel::RedisStorageMut;
        use cylon::net::fmi::EventProcessStatus;
        use std::sync::atomic::{AtomicBool, Ordering};

        let config = RedisTestConfig::new();
        let backend = config.backend();
        let mut storage = RedisStorageMut::new(&backend).expect("Failed to create Redis storage");

        let test_data = b"async redis hello";
        let data = Arc::new(ChannelData::from_slice(test_data));
        let key = "fmi_rust_test/async_key".to_string();

        let callback_called = Arc::new(AtomicBool::new(false));
        let callback_called_clone = callback_called.clone();

        let callback = Arc::new(move |status: cylon::net::fmi::NbxStatus, _msg: &str, _ctx: &cylon::net::fmi::FmiContext| {
            callback_called_clone.store(true, Ordering::SeqCst);
            assert_eq!(status, cylon::net::fmi::NbxStatus::Success);
        });

        storage.upload_object_async_mut(data, key.clone(), None, Some(callback))
            .expect("Async upload failed");

        // Process until complete
        let mut iterations = 0;
        while storage.has_pending_operations() && iterations < 300 {
            storage.process_pending_operations_mut();
            std::thread::sleep(std::time::Duration::from_millis(10));
            iterations += 1;
        }

        assert!(callback_called.load(Ordering::SeqCst), "Callback was not called");

        // Cleanup using sync storage
        let sync_storage = RedisStorage::new(&backend).expect("Failed to create sync storage");
        sync_storage.delete_object(&key).expect("Delete failed");

        println!("✓ Redis async upload works correctly");
    }

    #[test]
    #[ignore] // Requires Redis server
    fn test_redis_async_download() {
        use cylon::net::fmi::redis_channel::RedisStorageMut;
        use std::sync::atomic::{AtomicBool, Ordering};

        let config = RedisTestConfig::new();
        let backend = config.backend();

        // First upload some data using sync storage
        let sync_storage = RedisStorage::new(&backend).expect("Failed to create sync storage");
        let test_data = b"async download test data";
        let key = "fmi_rust_test/async_download_key";
        sync_storage.upload_object(test_data, key).expect("Upload failed");

        // Now download async
        let mut storage = RedisStorageMut::new(&backend).expect("Failed to create Redis storage");
        let download_buf = Arc::new(ChannelData::with_capacity(test_data.len()));

        let callback_called = Arc::new(AtomicBool::new(false));
        let callback_called_clone = callback_called.clone();

        let callback = Arc::new(move |status: cylon::net::fmi::NbxStatus, _msg: &str, _ctx: &cylon::net::fmi::FmiContext| {
            callback_called_clone.store(true, Ordering::SeqCst);
            assert_eq!(status, cylon::net::fmi::NbxStatus::Success);
        });

        storage.download_object_async_mut(download_buf.clone(), key.to_string(), None, Some(callback))
            .expect("Async download failed");

        // Process until complete
        let mut iterations = 0;
        while storage.has_pending_operations() && iterations < 300 {
            storage.process_pending_operations_mut();
            std::thread::sleep(std::time::Duration::from_millis(10));
            iterations += 1;
        }

        assert!(callback_called.load(Ordering::SeqCst), "Callback was not called");

        // Verify data
        let data = download_buf.as_slice();
        assert_eq!(&data[..test_data.len()], test_data);

        // Cleanup
        sync_storage.delete_object(key).expect("Delete failed");

        println!("✓ Redis async download works correctly");
    }
}