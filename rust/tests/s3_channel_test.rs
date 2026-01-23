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

//! Tests for S3 FMI channel backend
//!
//! ## Test Categories
//!
//! 1. **Unit Tests** - Test configuration, types (no S3 required)
//! 2. **Integration Tests** - Test S3 operations (requires S3 or MinIO)
//!
//! ## Running Integration Tests
//!
//! Integration tests require:
//! - AWS credentials configured (via environment variables or ~/.aws/credentials)
//! - An existing S3 bucket for testing, or MinIO running locally
//!
//! ```bash
//! # Option 1: Use real S3
//! export FMI_TEST_S3_BUCKET=your-test-bucket
//! export FMI_TEST_S3_REGION=us-east-1
//!
//! # Option 2: Use MinIO (local S3-compatible storage)
//! docker run -p 9000:9000 -p 9001:9001 minio/minio server /data --console-address ":9001"
//! export FMI_TEST_S3_BUCKET=test-bucket
//! export FMI_TEST_S3_REGION=us-east-1
//! export FMI_TEST_S3_ENDPOINT=http://localhost:9000
//! export AWS_ACCESS_KEY_ID=minioadmin
//! export AWS_SECRET_ACCESS_KEY=minioadmin
//!
//! # Run tests
//! cargo test --features "fmi,s3" s3_channel_test -- --ignored
//! ```
//!
//! ## Environment Variables
//!
//! - `FMI_TEST_S3_BUCKET` - S3 bucket name (required for integration tests)
//! - `FMI_TEST_S3_REGION` - AWS region (required for integration tests)
//! - `FMI_TEST_S3_ENDPOINT` - Custom endpoint URL for MinIO/LocalStack (optional)

// ============================================================================
// Unit Tests - No S3 required
// ============================================================================

#[cfg(all(feature = "fmi", feature = "s3"))]
mod s3_unit_tests {
    use cylon::net::fmi::{S3Backend, BackendType};

    #[test]
    fn test_s3_backend_defaults() {
        let backend = S3Backend::new();

        assert_eq!(backend.get_bucket_name(), "");
        assert_eq!(backend.get_region(), "us-east-1");
        assert_eq!(backend.get_endpoint(), None);
        assert_eq!(backend.get_timeout(), 100);
        assert_eq!(backend.get_max_timeout(), 30000);
        assert_eq!(backend.get_backend_type(), BackendType::S3);
        assert_eq!(backend.get_name(), "s3");
        assert!(!backend.is_enabled());

        println!("✓ S3Backend defaults are correct");
    }

    #[test]
    fn test_s3_backend_builder() {
        let backend = S3Backend::new()
            .set_enabled(true)
            .with_bucket_name("my-bucket")
            .with_region("eu-west-1")
            .with_endpoint("http://localhost:9000")
            .with_timeout(200)
            .with_max_timeout(60000);

        assert_eq!(backend.get_bucket_name(), "my-bucket");
        assert_eq!(backend.get_region(), "eu-west-1");
        assert_eq!(backend.get_endpoint(), Some("http://localhost:9000"));
        assert_eq!(backend.get_timeout(), 200);
        assert_eq!(backend.get_max_timeout(), 60000);
        assert!(backend.is_enabled());

        println!("✓ S3Backend builder pattern works correctly");
    }
}

// ============================================================================
// Integration Tests - Requires S3/MinIO
// ============================================================================

#[cfg(all(feature = "fmi", feature = "s3"))]
mod s3_integration_tests {
    use std::env;
    use std::sync::Arc;

    use cylon::net::fmi::{S3Backend, S3Storage, new_s3_channel, ChannelData};
    use cylon::net::fmi::client_server::StorageBackend;

    /// Get test configuration from environment
    struct S3TestConfig {
        bucket: String,
        region: String,
        endpoint: Option<String>,
        valid: bool,
    }

    impl S3TestConfig {
        fn new() -> Self {
            let bucket = env::var("FMI_TEST_S3_BUCKET").ok();
            let region = env::var("FMI_TEST_S3_REGION").ok();
            let endpoint = env::var("FMI_TEST_S3_ENDPOINT").ok();

            match (bucket, region) {
                (Some(b), Some(r)) => Self {
                    bucket: b,
                    region: r,
                    endpoint,
                    valid: true,
                },
                _ => Self {
                    bucket: String::new(),
                    region: String::new(),
                    endpoint: None,
                    valid: false,
                },
            }
        }

        fn backend(&self) -> S3Backend {
            let mut backend = S3Backend::new()
                .set_enabled(true)
                .with_bucket_name(&self.bucket)
                .with_region(&self.region)
                .with_timeout(100)
                .with_max_timeout(30000);

            if let Some(ref ep) = self.endpoint {
                backend = backend.with_endpoint(ep);
            }

            backend
        }

        fn skip_message(&self) -> &'static str {
            "S3 tests skipped - set FMI_TEST_S3_BUCKET and FMI_TEST_S3_REGION"
        }
    }

    // -------------------------------------------------------------------------
    // Blocking Operations Tests
    // -------------------------------------------------------------------------

    #[test]
    #[ignore] // Requires S3/MinIO
    fn test_s3_upload_download_small_data() {
        let config = S3TestConfig::new();
        if !config.valid {
            println!("{}", config.skip_message());
            return;
        }

        let backend = config.backend();
        let storage = S3Storage::new(&backend).expect("Failed to create S3 storage");

        let test_data = b"hello s3 world";
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

        println!("✓ S3 upload and download small data works correctly");
    }

    #[test]
    #[ignore] // Requires S3/MinIO
    fn test_s3_upload_download_binary_data() {
        let config = S3TestConfig::new();
        if !config.valid {
            println!("{}", config.skip_message());
            return;
        }

        let backend = config.backend();
        let storage = S3Storage::new(&backend).expect("Failed to create S3 storage");

        let binary_data: Vec<u8> = vec![0x00, 0x01, 0x02, 0xFF, 0xFE, 0x00, 0x10];
        let key = "fmi_rust_test/binary_key";

        storage.upload_object(&binary_data, key).expect("Upload failed");

        let mut download_buf = vec![0u8; binary_data.len()];
        let found = storage.download_object(&mut download_buf, key).expect("Download failed");

        assert!(found);
        assert_eq!(download_buf, binary_data);

        storage.delete_object(key).expect("Delete failed");

        println!("✓ S3 upload and download binary data works correctly");
    }

    #[test]
    #[ignore] // Requires S3/MinIO
    fn test_s3_download_nonexistent_key() {
        let config = S3TestConfig::new();
        if !config.valid {
            println!("{}", config.skip_message());
            return;
        }

        let backend = config.backend();
        let storage = S3Storage::new(&backend).expect("Failed to create S3 storage");

        let mut download_buf = vec![0u8; 64];
        let found = storage.download_object(&mut download_buf, "fmi_rust_test/nonexistent_xyz123")
            .expect("Download should not fail for missing key");

        assert!(!found, "Nonexistent key should return false");

        println!("✓ S3 download nonexistent key returns false");
    }

    #[test]
    #[ignore] // Requires S3/MinIO
    fn test_s3_upload_larger_data() {
        let config = S3TestConfig::new();
        if !config.valid {
            println!("{}", config.skip_message());
            return;
        }

        let backend = config.backend();
        let storage = S3Storage::new(&backend).expect("Failed to create S3 storage");

        // Create 1KB of test data
        let large_data: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
        let key = "fmi_rust_test/large_key";

        storage.upload_object(&large_data, key).expect("Upload failed");

        let mut download_buf = vec![0u8; large_data.len()];
        let found = storage.download_object(&mut download_buf, key).expect("Download failed");

        assert!(found);
        assert_eq!(download_buf, large_data);

        storage.delete_object(key).expect("Delete failed");

        println!("✓ S3 upload larger data (1KB) works correctly");
    }

    #[test]
    #[ignore] // Requires S3/MinIO
    fn test_s3_delete_object() {
        let config = S3TestConfig::new();
        if !config.valid {
            println!("{}", config.skip_message());
            return;
        }

        let backend = config.backend();
        let storage = S3Storage::new(&backend).expect("Failed to create S3 storage");

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

        println!("✓ S3 delete object works correctly");
    }

    #[test]
    #[ignore] // Requires S3/MinIO
    fn test_s3_get_object_names() {
        let config = S3TestConfig::new();
        if !config.valid {
            println!("{}", config.skip_message());
            return;
        }

        let backend = config.backend();
        let storage = S3Storage::new(&backend).expect("Failed to create S3 storage");

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

        println!("✓ S3 get object names works correctly");
    }

    // -------------------------------------------------------------------------
    // Channel Tests
    // -------------------------------------------------------------------------

    #[test]
    #[ignore] // Requires S3/MinIO
    fn test_s3_channel_creation() {
        let config = S3TestConfig::new();
        if !config.valid {
            println!("{}", config.skip_message());
            return;
        }

        let backend = config.backend();
        let channel = new_s3_channel(&backend);
        assert!(channel.is_ok(), "Failed to create S3 channel: {:?}", channel.err());

        println!("✓ S3 channel creation works correctly");
    }

    // -------------------------------------------------------------------------
    // Async Operations Tests
    // -------------------------------------------------------------------------

    #[test]
    #[ignore] // Requires S3/MinIO
    fn test_s3_async_upload() {
        use cylon::net::fmi::s3_channel::S3StorageMut;
        use std::sync::atomic::{AtomicBool, Ordering};

        let config = S3TestConfig::new();
        if !config.valid {
            println!("{}", config.skip_message());
            return;
        }

        let backend = config.backend();
        let mut storage = S3StorageMut::new(&backend).expect("Failed to create S3 storage");
        storage.init().expect("Failed to initialize S3 client");

        let test_data = b"async s3 hello";
        let data = Arc::new(ChannelData::from_slice(test_data));
        let key = "fmi_rust_test/async_key".to_string();

        let callback_called = Arc::new(AtomicBool::new(false));
        let callback_called_clone = callback_called.clone();

        let callback = Arc::new(move |status: cylon::net::fmi::NbxStatus, _msg: &str, _ctx: &mut cylon::net::fmi::FmiContext| {
            callback_called_clone.store(true, Ordering::SeqCst);
            assert_eq!(status, cylon::net::fmi::NbxStatus::Success);
        });

        storage.upload_object_async_mut(data, key.clone(), Some(callback))
            .expect("Async upload failed");

        // Process until complete (S3 operations can take longer)
        let mut iterations = 0;
        while storage.has_pending_operations() && iterations < 300 {
            storage.process_pending_operations_mut();
            std::thread::sleep(std::time::Duration::from_millis(100));
            iterations += 1;
        }

        assert!(callback_called.load(Ordering::SeqCst), "Callback was not called");

        // Cleanup using sync storage
        let sync_storage = S3Storage::new(&backend).expect("Failed to create sync storage");
        sync_storage.delete_object(&key).expect("Delete failed");

        println!("✓ S3 async upload works correctly");
    }

    #[test]
    #[ignore] // Requires S3/MinIO
    fn test_s3_async_download() {
        use cylon::net::fmi::s3_channel::S3StorageMut;
        use std::sync::atomic::{AtomicBool, Ordering};

        let config = S3TestConfig::new();
        if !config.valid {
            println!("{}", config.skip_message());
            return;
        }

        let backend = config.backend();

        // First upload some data using sync storage
        let sync_storage = S3Storage::new(&backend).expect("Failed to create sync storage");
        let test_data = b"async download test data";
        let key = "fmi_rust_test/async_download_key";
        sync_storage.upload_object(test_data, key).expect("Upload failed");

        // Now download async
        let mut storage = S3StorageMut::new(&backend).expect("Failed to create S3 storage");
        storage.init().expect("Failed to initialize S3 client");
        let download_buf = Arc::new(ChannelData::with_capacity(test_data.len()));

        let callback_called = Arc::new(AtomicBool::new(false));
        let callback_called_clone = callback_called.clone();

        let callback = Arc::new(move |status: cylon::net::fmi::NbxStatus, _msg: &str, _ctx: &mut cylon::net::fmi::FmiContext| {
            callback_called_clone.store(true, Ordering::SeqCst);
            assert_eq!(status, cylon::net::fmi::NbxStatus::Success);
        });

        storage.download_object_async_mut(download_buf.clone(), key.to_string(), Some(callback))
            .expect("Async download failed");

        // Process until complete
        let mut iterations = 0;
        while storage.has_pending_operations() && iterations < 300 {
            storage.process_pending_operations_mut();
            std::thread::sleep(std::time::Duration::from_millis(100));
            iterations += 1;
        }

        assert!(callback_called.load(Ordering::SeqCst), "Callback was not called");

        // Verify data
        let data = download_buf.as_slice();
        assert_eq!(&data[..test_data.len()], test_data);

        // Cleanup
        sync_storage.delete_object(key).expect("Delete failed");

        println!("✓ S3 async download works correctly");
    }
}