//! Checkpoint storage backend implementations.
//!
//! Provides storage backends for persisting checkpoint data.
//!
//! # Available Backends
//!
//! - [`FileSystemStorage`] - Local or shared filesystem (Lustre, NFS, GPFS)
//! - [`S3Storage`] - Amazon S3 or S3-compatible storage (requires `s3` feature)

use async_trait::async_trait;
use std::path::{Path, PathBuf};
use tokio::fs;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use crate::error::{Code, CylonError, CylonResult};

use super::serializer::JsonSerializer;
use super::traits::CheckpointStorage;
use super::types::{CheckpointMetadata, CheckpointStatus, WorkerId};

/// Filesystem-based checkpoint storage.
///
/// Stores checkpoints on a local or shared filesystem (e.g., Lustre, NFS, GPFS).
/// Uses a staging directory for atomic commits via rename.
///
/// Directory structure:
/// ```text
/// {base_path}/{job_id}/
/// ├── metadata/
/// │   ├── checkpoint_000001.json
/// │   ├── checkpoint_000002.json
/// │   └── latest.json
/// ├── staging/
/// │   └── checkpoint_000003/
/// │       ├── rank_0/
/// │       │   └── table_orders.arrow
/// │       └── rank_1/
/// │           └── table_orders.arrow
/// └── committed/
///     ├── checkpoint_000001/
///     │   └── ...
///     └── checkpoint_000002/
///         └── ...
/// ```
pub struct FileSystemStorage {
    /// Base path for all checkpoints
    base_path: PathBuf,
    /// Job identifier
    job_id: String,
    /// JSON serializer for metadata
    json_serializer: JsonSerializer,
}

impl FileSystemStorage {
    /// Create a new filesystem storage backend
    pub fn new(base_path: impl Into<PathBuf>, job_id: impl Into<String>) -> Self {
        Self {
            base_path: base_path.into(),
            job_id: job_id.into(),
            json_serializer: JsonSerializer::new(),
        }
    }

    /// Get the job directory path
    fn job_path(&self) -> PathBuf {
        self.base_path.join(&self.job_id)
    }

    /// Get the metadata directory path
    fn metadata_path(&self) -> PathBuf {
        self.job_path().join("metadata")
    }

    /// Get the staging directory path
    fn staging_path(&self) -> PathBuf {
        self.job_path().join("staging")
    }

    /// Get the committed directory path
    fn committed_path(&self) -> PathBuf {
        self.job_path().join("committed")
    }

    /// Get the path for a checkpoint's metadata file
    fn checkpoint_metadata_path(&self, checkpoint_id: u64) -> PathBuf {
        self.metadata_path()
            .join(format!("checkpoint_{:06}.json", checkpoint_id))
    }

    /// Get the staging path for a worker's checkpoint data
    fn worker_staging_path(&self, checkpoint_id: u64, worker_id: &WorkerId) -> PathBuf {
        self.staging_path()
            .join(format!("checkpoint_{:06}", checkpoint_id))
            .join(worker_id.to_path_string())
    }

    /// Get the committed path for a worker's checkpoint data
    fn worker_committed_path(&self, checkpoint_id: u64, worker_id: &WorkerId) -> PathBuf {
        self.committed_path()
            .join(format!("checkpoint_{:06}", checkpoint_id))
            .join(worker_id.to_path_string())
    }

    /// Initialize directory structure
    pub async fn initialize(&self) -> CylonResult<()> {
        fs::create_dir_all(self.metadata_path()).await?;
        fs::create_dir_all(self.staging_path()).await?;
        fs::create_dir_all(self.committed_path()).await?;
        Ok(())
    }

    /// Format checkpoint ID for display
    fn format_checkpoint_id(checkpoint_id: u64) -> String {
        format!("checkpoint_{:06}", checkpoint_id)
    }

    /// Parse checkpoint ID from directory name
    fn parse_checkpoint_id(name: &str) -> Option<u64> {
        name.strip_prefix("checkpoint_")
            .and_then(|s| s.strip_suffix(".json").unwrap_or(s).parse().ok())
    }
}

#[async_trait]
impl CheckpointStorage for FileSystemStorage {
    async fn write(
        &self,
        checkpoint_id: u64,
        worker_id: &WorkerId,
        key: &str,
        data: &[u8],
    ) -> CylonResult<String> {
        let worker_path = self.worker_staging_path(checkpoint_id, worker_id);
        fs::create_dir_all(&worker_path).await?;

        let file_path = worker_path.join(key);
        let mut file = fs::File::create(&file_path).await?;
        file.write_all(data).await?;
        file.sync_all().await?;

        Ok(file_path.to_string_lossy().to_string())
    }

    async fn read(
        &self,
        checkpoint_id: u64,
        worker_id: &WorkerId,
        key: &str,
    ) -> CylonResult<Vec<u8>> {
        // Try committed first, then staging
        let committed_path = self.worker_committed_path(checkpoint_id, worker_id).join(key);
        let staging_path = self.worker_staging_path(checkpoint_id, worker_id).join(key);

        let file_path = if committed_path.exists() {
            committed_path
        } else if staging_path.exists() {
            staging_path
        } else {
            return Err(CylonError::new(
                Code::NotFound,
                format!(
                    "Checkpoint data not found: checkpoint={}, worker={}, key={}",
                    checkpoint_id, worker_id, key
                ),
            ));
        };

        let mut file = fs::File::open(&file_path).await?;
        let mut data = Vec::new();
        file.read_to_end(&mut data).await?;

        Ok(data)
    }

    async fn exists(
        &self,
        checkpoint_id: u64,
        worker_id: &WorkerId,
        key: &str,
    ) -> CylonResult<bool> {
        let committed_path = self.worker_committed_path(checkpoint_id, worker_id).join(key);
        let staging_path = self.worker_staging_path(checkpoint_id, worker_id).join(key);

        Ok(committed_path.exists() || staging_path.exists())
    }

    async fn list_keys(
        &self,
        checkpoint_id: u64,
        worker_id: &WorkerId,
    ) -> CylonResult<Vec<String>> {
        let mut keys = Vec::new();

        // Check committed directory first
        let committed_path = self.worker_committed_path(checkpoint_id, worker_id);
        if committed_path.exists() {
            let mut entries = fs::read_dir(&committed_path).await?;
            while let Some(entry) = entries.next_entry().await? {
                if let Some(name) = entry.file_name().to_str() {
                    keys.push(name.to_string());
                }
            }
        }

        // Also check staging if committed is empty
        if keys.is_empty() {
            let staging_path = self.worker_staging_path(checkpoint_id, worker_id);
            if staging_path.exists() {
                let mut entries = fs::read_dir(&staging_path).await?;
                while let Some(entry) = entries.next_entry().await? {
                    if let Some(name) = entry.file_name().to_str() {
                        keys.push(name.to_string());
                    }
                }
            }
        }

        Ok(keys)
    }

    async fn delete(&self, checkpoint_id: u64) -> CylonResult<()> {
        let checkpoint_name = Self::format_checkpoint_id(checkpoint_id);

        // Delete from committed
        let committed_dir = self.committed_path().join(&checkpoint_name);
        if committed_dir.exists() {
            fs::remove_dir_all(&committed_dir).await?;
        }

        // Delete from staging
        let staging_dir = self.staging_path().join(&checkpoint_name);
        if staging_dir.exists() {
            fs::remove_dir_all(&staging_dir).await?;
        }

        // Delete metadata
        let metadata_file = self.checkpoint_metadata_path(checkpoint_id);
        if metadata_file.exists() {
            fs::remove_file(&metadata_file).await?;
        }

        Ok(())
    }

    async fn list_checkpoints(&self) -> CylonResult<Vec<u64>> {
        let mut checkpoints = Vec::new();

        let metadata_path = self.metadata_path();
        if !metadata_path.exists() {
            return Ok(checkpoints);
        }

        let mut entries = fs::read_dir(&metadata_path).await?;
        while let Some(entry) = entries.next_entry().await? {
            if let Some(name) = entry.file_name().to_str() {
                if let Some(id) = Self::parse_checkpoint_id(name) {
                    checkpoints.push(id);
                }
            }
        }

        // Sort newest first
        checkpoints.sort_by(|a, b| b.cmp(a));

        Ok(checkpoints)
    }

    async fn commit_write(
        &self,
        checkpoint_id: u64,
        worker_id: &WorkerId,
    ) -> CylonResult<()> {
        let staging_path = self.worker_staging_path(checkpoint_id, worker_id);
        let committed_path = self.worker_committed_path(checkpoint_id, worker_id);

        if !staging_path.exists() {
            return Err(CylonError::new(
                Code::NotFound,
                format!(
                    "Staging data not found for commit: checkpoint={}, worker={}",
                    checkpoint_id, worker_id
                ),
            ));
        }

        // Create parent directory for committed path
        if let Some(parent) = committed_path.parent() {
            fs::create_dir_all(parent).await?;
        }

        // Atomic rename from staging to committed
        fs::rename(&staging_path, &committed_path).await?;

        Ok(())
    }

    async fn write_metadata(
        &self,
        checkpoint_id: u64,
        metadata: &CheckpointMetadata,
    ) -> CylonResult<()> {
        fs::create_dir_all(self.metadata_path()).await?;

        let metadata_path = self.checkpoint_metadata_path(checkpoint_id);
        let data = self.json_serializer.serialize_json(metadata)?;

        let mut file = fs::File::create(&metadata_path).await?;
        file.write_all(&data).await?;
        file.sync_all().await?;

        // Also update "latest" pointer if this is a committed checkpoint
        if metadata.status == CheckpointStatus::Committed {
            let latest_path = self.metadata_path().join("latest.json");
            let latest_data = self.json_serializer.serialize_json(&checkpoint_id)?;
            let mut latest_file = fs::File::create(&latest_path).await?;
            latest_file.write_all(&latest_data).await?;
            latest_file.sync_all().await?;
        }

        Ok(())
    }

    async fn read_metadata(&self, checkpoint_id: u64) -> CylonResult<CheckpointMetadata> {
        let metadata_path = self.checkpoint_metadata_path(checkpoint_id);

        if !metadata_path.exists() {
            return Err(CylonError::new(
                Code::NotFound,
                format!("Checkpoint metadata not found: checkpoint={}", checkpoint_id),
            ));
        }

        let mut file = fs::File::open(&metadata_path).await?;
        let mut data = Vec::new();
        file.read_to_end(&mut data).await?;

        self.json_serializer.deserialize_json(&data)
    }

    fn base_path(&self) -> &str {
        self.base_path.to_str().unwrap_or("")
    }
}

// =============================================================================
// S3 Storage Backend
// =============================================================================

#[cfg(feature = "s3")]
mod s3_storage {
    use super::*;
    use aws_sdk_s3::primitives::ByteStream;
    use aws_sdk_s3::Client;
    use std::sync::Arc;

    /// Configuration for S3 storage backend.
    #[derive(Clone, Debug)]
    pub struct S3StorageConfig {
        /// S3 bucket name
        pub bucket: String,
        /// Key prefix for all checkpoint data (e.g., "checkpoints/my-job")
        pub prefix: String,
        /// Optional custom endpoint URL (for S3-compatible services like MinIO)
        pub endpoint_url: Option<String>,
        /// AWS region
        pub region: Option<String>,
        /// Whether to use path-style addressing (required for some S3-compatible services)
        pub force_path_style: bool,
    }

    impl S3StorageConfig {
        /// Create a new S3 storage config with bucket and prefix
        pub fn new(bucket: impl Into<String>, prefix: impl Into<String>) -> Self {
            Self {
                bucket: bucket.into(),
                prefix: prefix.into(),
                endpoint_url: None,
                region: None,
                force_path_style: false,
            }
        }

        /// Set custom endpoint URL (for MinIO, LocalStack, etc.)
        pub fn with_endpoint(mut self, endpoint_url: impl Into<String>) -> Self {
            self.endpoint_url = Some(endpoint_url.into());
            self
        }

        /// Set AWS region
        pub fn with_region(mut self, region: impl Into<String>) -> Self {
            self.region = Some(region.into());
            self
        }

        /// Enable path-style addressing
        pub fn with_path_style(mut self, enabled: bool) -> Self {
            self.force_path_style = enabled;
            self
        }
    }

    /// S3-based checkpoint storage.
    ///
    /// Stores checkpoints in Amazon S3 or S3-compatible storage (MinIO, LocalStack, etc.).
    /// Uses a two-phase commit approach:
    /// 1. Write to staging prefix
    /// 2. Copy to committed prefix on commit
    ///
    /// # Key Structure
    ///
    /// ```text
    /// {prefix}/
    /// ├── metadata/
    /// │   ├── checkpoint_000001.json
    /// │   ├── checkpoint_000002.json
    /// │   └── latest.json
    /// ├── staging/
    /// │   └── checkpoint_000003/
    /// │       ├── rank_0/
    /// │       │   └── table_orders.arrow
    /// │       └── rank_1/
    /// │           └── table_orders.arrow
    /// └── committed/
    ///     ├── checkpoint_000001/
    ///     │   └── ...
    ///     └── checkpoint_000002/
    ///         └── ...
    /// ```
    pub struct S3Storage {
        /// S3 client
        client: Client,
        /// Bucket name
        bucket: String,
        /// Key prefix
        prefix: String,
        /// JSON serializer for metadata
        json_serializer: JsonSerializer,
        /// Base URI for display
        base_uri: String,
    }

    impl S3Storage {
        /// Create a new S3 storage backend with the provided configuration
        pub async fn new(config: S3StorageConfig) -> CylonResult<Self> {
            let sdk_config = Self::build_sdk_config(&config).await?;
            let client = Client::new(&sdk_config);

            let base_uri = if let Some(ref endpoint) = config.endpoint_url {
                format!("{}/{}/{}", endpoint, config.bucket, config.prefix)
            } else {
                format!("s3://{}/{}", config.bucket, config.prefix)
            };

            Ok(Self {
                client,
                bucket: config.bucket,
                prefix: config.prefix,
                json_serializer: JsonSerializer::new(),
                base_uri,
            })
        }

        /// Create S3 storage with an existing client (for testing or custom configuration)
        pub fn with_client(
            client: Client,
            bucket: impl Into<String>,
            prefix: impl Into<String>,
        ) -> Self {
            let bucket = bucket.into();
            let prefix = prefix.into();
            let base_uri = format!("s3://{}/{}", bucket, prefix);

            Self {
                client,
                bucket,
                prefix,
                json_serializer: JsonSerializer::new(),
                base_uri,
            }
        }

        async fn build_sdk_config(
            config: &S3StorageConfig,
        ) -> CylonResult<aws_config::SdkConfig> {
            let mut loader = aws_config::defaults(aws_config::BehaviorVersion::latest());

            if let Some(ref region) = config.region {
                loader = loader.region(aws_config::Region::new(region.clone()));
            }

            if let Some(ref endpoint) = config.endpoint_url {
                loader = loader.endpoint_url(endpoint.clone());
            }

            Ok(loader.load().await)
        }

        /// Get the metadata key prefix
        fn metadata_prefix(&self) -> String {
            format!("{}/metadata", self.prefix)
        }

        /// Get the staging key prefix
        fn staging_prefix(&self) -> String {
            format!("{}/staging", self.prefix)
        }

        /// Get the committed key prefix
        fn committed_prefix(&self) -> String {
            format!("{}/committed", self.prefix)
        }

        /// Get the key for a checkpoint's metadata
        fn checkpoint_metadata_key(&self, checkpoint_id: u64) -> String {
            format!(
                "{}/checkpoint_{:06}.json",
                self.metadata_prefix(),
                checkpoint_id
            )
        }

        /// Get the staging key for a worker's checkpoint data
        fn worker_staging_key(
            &self,
            checkpoint_id: u64,
            worker_id: &WorkerId,
            key: &str,
        ) -> String {
            format!(
                "{}/checkpoint_{:06}/{}/{}",
                self.staging_prefix(),
                checkpoint_id,
                worker_id.to_path_string(),
                key
            )
        }

        /// Get the committed key for a worker's checkpoint data
        fn worker_committed_key(
            &self,
            checkpoint_id: u64,
            worker_id: &WorkerId,
            key: &str,
        ) -> String {
            format!(
                "{}/committed/checkpoint_{:06}/{}/{}",
                self.prefix,
                checkpoint_id,
                worker_id.to_path_string(),
                key
            )
        }

        /// Get the staging prefix for a worker
        fn worker_staging_prefix(&self, checkpoint_id: u64, worker_id: &WorkerId) -> String {
            format!(
                "{}/checkpoint_{:06}/{}/",
                self.staging_prefix(),
                checkpoint_id,
                worker_id.to_path_string()
            )
        }

        /// Get the committed prefix for a worker
        fn worker_committed_prefix(&self, checkpoint_id: u64, worker_id: &WorkerId) -> String {
            format!(
                "{}/committed/checkpoint_{:06}/{}/",
                self.prefix,
                checkpoint_id,
                worker_id.to_path_string()
            )
        }

        /// Format checkpoint ID for display
        fn format_checkpoint_id(checkpoint_id: u64) -> String {
            format!("checkpoint_{:06}", checkpoint_id)
        }

        /// Parse checkpoint ID from key name
        fn parse_checkpoint_id(name: &str) -> Option<u64> {
            name.strip_prefix("checkpoint_")
                .and_then(|s| s.strip_suffix(".json").unwrap_or(s).parse().ok())
        }

        /// List objects with a given prefix
        async fn list_objects_with_prefix(&self, prefix: &str) -> CylonResult<Vec<String>> {
            let mut keys = Vec::new();
            let mut continuation_token: Option<String> = None;

            loop {
                let mut request = self
                    .client
                    .list_objects_v2()
                    .bucket(&self.bucket)
                    .prefix(prefix);

                if let Some(token) = continuation_token {
                    request = request.continuation_token(token);
                }

                let response = request.send().await.map_err(|e| {
                    CylonError::new(
                        Code::IoError,
                        format!("Failed to list S3 objects: {}", e),
                    )
                })?;

                if let Some(contents) = response.contents {
                    for object in contents {
                        if let Some(key) = object.key {
                            keys.push(key);
                        }
                    }
                }

                if response.is_truncated == Some(true) {
                    continuation_token = response.next_continuation_token;
                } else {
                    break;
                }
            }

            Ok(keys)
        }

        /// Delete all objects with a given prefix
        async fn delete_objects_with_prefix(&self, prefix: &str) -> CylonResult<()> {
            let keys = self.list_objects_with_prefix(prefix).await?;

            // Delete in batches of 1000 (S3 limit)
            for chunk in keys.chunks(1000) {
                let objects: Vec<_> = chunk
                    .iter()
                    .map(|key| {
                        aws_sdk_s3::types::ObjectIdentifier::builder()
                            .key(key)
                            .build()
                            .unwrap()
                    })
                    .collect();

                if !objects.is_empty() {
                    let delete = aws_sdk_s3::types::Delete::builder()
                        .set_objects(Some(objects))
                        .build()
                        .map_err(|e| {
                            CylonError::new(
                                Code::IoError,
                                format!("Failed to build delete request: {}", e),
                            )
                        })?;

                    self.client
                        .delete_objects()
                        .bucket(&self.bucket)
                        .delete(delete)
                        .send()
                        .await
                        .map_err(|e| {
                            CylonError::new(
                                Code::IoError,
                                format!("Failed to delete S3 objects: {}", e),
                            )
                        })?;
                }
            }

            Ok(())
        }

        /// Copy an object from one key to another
        async fn copy_object(&self, source_key: &str, dest_key: &str) -> CylonResult<()> {
            let copy_source = format!("{}/{}", self.bucket, source_key);

            self.client
                .copy_object()
                .bucket(&self.bucket)
                .copy_source(&copy_source)
                .key(dest_key)
                .send()
                .await
                .map_err(|e| {
                    CylonError::new(Code::IoError, format!("Failed to copy S3 object: {}", e))
                })?;

            Ok(())
        }
    }

    #[async_trait]
    impl CheckpointStorage for S3Storage {
        async fn write(
            &self,
            checkpoint_id: u64,
            worker_id: &WorkerId,
            key: &str,
            data: &[u8],
        ) -> CylonResult<String> {
            let s3_key = self.worker_staging_key(checkpoint_id, worker_id, key);

            self.client
                .put_object()
                .bucket(&self.bucket)
                .key(&s3_key)
                .body(ByteStream::from(data.to_vec()))
                .send()
                .await
                .map_err(|e| {
                    CylonError::new(
                        Code::IoError,
                        format!("Failed to write to S3: {}", e),
                    )
                })?;

            Ok(format!("s3://{}/{}", self.bucket, s3_key))
        }

        async fn read(
            &self,
            checkpoint_id: u64,
            worker_id: &WorkerId,
            key: &str,
        ) -> CylonResult<Vec<u8>> {
            // Try committed first, then staging
            let committed_key = self.worker_committed_key(checkpoint_id, worker_id, key);
            let staging_key = self.worker_staging_key(checkpoint_id, worker_id, key);

            // Try committed location first
            let result = self
                .client
                .get_object()
                .bucket(&self.bucket)
                .key(&committed_key)
                .send()
                .await;

            let response = match result {
                Ok(resp) => resp,
                Err(_) => {
                    // Try staging location
                    self.client
                        .get_object()
                        .bucket(&self.bucket)
                        .key(&staging_key)
                        .send()
                        .await
                        .map_err(|e| {
                            CylonError::new(
                                Code::NotFound,
                                format!(
                                    "Checkpoint data not found in S3: checkpoint={}, worker={}, key={}: {}",
                                    checkpoint_id, worker_id, key, e
                                ),
                            )
                        })?
                }
            };

            let data = response
                .body
                .collect()
                .await
                .map_err(|e| {
                    CylonError::new(Code::IoError, format!("Failed to read S3 body: {}", e))
                })?
                .into_bytes()
                .to_vec();

            Ok(data)
        }

        async fn exists(
            &self,
            checkpoint_id: u64,
            worker_id: &WorkerId,
            key: &str,
        ) -> CylonResult<bool> {
            let committed_key = self.worker_committed_key(checkpoint_id, worker_id, key);
            let staging_key = self.worker_staging_key(checkpoint_id, worker_id, key);

            // Check committed first
            let committed_exists = self
                .client
                .head_object()
                .bucket(&self.bucket)
                .key(&committed_key)
                .send()
                .await
                .is_ok();

            if committed_exists {
                return Ok(true);
            }

            // Check staging
            let staging_exists = self
                .client
                .head_object()
                .bucket(&self.bucket)
                .key(&staging_key)
                .send()
                .await
                .is_ok();

            Ok(staging_exists)
        }

        async fn list_keys(
            &self,
            checkpoint_id: u64,
            worker_id: &WorkerId,
        ) -> CylonResult<Vec<String>> {
            // Try committed first
            let committed_prefix = self.worker_committed_prefix(checkpoint_id, worker_id);
            let mut keys = self.list_objects_with_prefix(&committed_prefix).await?;

            // If no committed keys, try staging
            if keys.is_empty() {
                let staging_prefix = self.worker_staging_prefix(checkpoint_id, worker_id);
                keys = self.list_objects_with_prefix(&staging_prefix).await?;
            }

            // Extract just the key names (strip the prefix)
            let key_names: Vec<String> = keys
                .into_iter()
                .filter_map(|k| {
                    k.rsplit('/').next().map(|s| s.to_string())
                })
                .collect();

            Ok(key_names)
        }

        async fn delete(&self, checkpoint_id: u64) -> CylonResult<()> {
            let checkpoint_name = Self::format_checkpoint_id(checkpoint_id);

            // Delete from committed
            let committed_prefix = format!(
                "{}/committed/{}/",
                self.prefix, checkpoint_name
            );
            self.delete_objects_with_prefix(&committed_prefix).await?;

            // Delete from staging
            let staging_prefix = format!(
                "{}/staging/{}/",
                self.prefix, checkpoint_name
            );
            self.delete_objects_with_prefix(&staging_prefix).await?;

            // Delete metadata
            let metadata_key = self.checkpoint_metadata_key(checkpoint_id);
            let _ = self
                .client
                .delete_object()
                .bucket(&self.bucket)
                .key(&metadata_key)
                .send()
                .await;

            Ok(())
        }

        async fn list_checkpoints(&self) -> CylonResult<Vec<u64>> {
            let metadata_prefix = format!("{}/", self.metadata_prefix());
            let keys = self.list_objects_with_prefix(&metadata_prefix).await?;

            let mut checkpoints: Vec<u64> = keys
                .into_iter()
                .filter_map(|key| {
                    let name = key.rsplit('/').next()?;
                    if name == "latest.json" {
                        return None;
                    }
                    Self::parse_checkpoint_id(name)
                })
                .collect();

            // Sort newest first
            checkpoints.sort_by(|a, b| b.cmp(a));

            Ok(checkpoints)
        }

        async fn commit_write(
            &self,
            checkpoint_id: u64,
            worker_id: &WorkerId,
        ) -> CylonResult<()> {
            let staging_prefix = self.worker_staging_prefix(checkpoint_id, worker_id);
            let committed_prefix = self.worker_committed_prefix(checkpoint_id, worker_id);

            // List all staging objects for this worker
            let staging_keys = self.list_objects_with_prefix(&staging_prefix).await?;

            if staging_keys.is_empty() {
                return Err(CylonError::new(
                    Code::NotFound,
                    format!(
                        "Staging data not found for commit: checkpoint={}, worker={}",
                        checkpoint_id, worker_id
                    ),
                ));
            }

            // Copy each object from staging to committed
            for staging_key in &staging_keys {
                let key_name = staging_key
                    .strip_prefix(&staging_prefix)
                    .unwrap_or(staging_key);
                let committed_key = format!("{}{}", committed_prefix, key_name);

                self.copy_object(staging_key, &committed_key).await?;
            }

            // Delete staging objects after successful copy
            self.delete_objects_with_prefix(&staging_prefix).await?;

            Ok(())
        }

        async fn write_metadata(
            &self,
            checkpoint_id: u64,
            metadata: &CheckpointMetadata,
        ) -> CylonResult<()> {
            let metadata_key = self.checkpoint_metadata_key(checkpoint_id);
            let data = self.json_serializer.serialize_json(metadata)?;

            self.client
                .put_object()
                .bucket(&self.bucket)
                .key(&metadata_key)
                .body(ByteStream::from(data))
                .content_type("application/json")
                .send()
                .await
                .map_err(|e| {
                    CylonError::new(
                        Code::IoError,
                        format!("Failed to write metadata to S3: {}", e),
                    )
                })?;

            // Update "latest" pointer if this is a committed checkpoint
            if metadata.status == CheckpointStatus::Committed {
                let latest_key = format!("{}/latest.json", self.metadata_prefix());
                let latest_data = self.json_serializer.serialize_json(&checkpoint_id)?;

                self.client
                    .put_object()
                    .bucket(&self.bucket)
                    .key(&latest_key)
                    .body(ByteStream::from(latest_data))
                    .content_type("application/json")
                    .send()
                    .await
                    .map_err(|e| {
                        CylonError::new(
                            Code::IoError,
                            format!("Failed to write latest pointer to S3: {}", e),
                        )
                    })?;
            }

            Ok(())
        }

        async fn read_metadata(&self, checkpoint_id: u64) -> CylonResult<CheckpointMetadata> {
            let metadata_key = self.checkpoint_metadata_key(checkpoint_id);

            let response = self
                .client
                .get_object()
                .bucket(&self.bucket)
                .key(&metadata_key)
                .send()
                .await
                .map_err(|e| {
                    CylonError::new(
                        Code::NotFound,
                        format!(
                            "Checkpoint metadata not found in S3: checkpoint={}: {}",
                            checkpoint_id, e
                        ),
                    )
                })?;

            let data = response
                .body
                .collect()
                .await
                .map_err(|e| {
                    CylonError::new(
                        Code::IoError,
                        format!("Failed to read metadata from S3: {}", e),
                    )
                })?
                .into_bytes()
                .to_vec();

            self.json_serializer.deserialize_json(&data)
        }

        fn base_path(&self) -> &str {
            &self.base_uri
        }
    }
}

#[cfg(feature = "s3")]
pub use s3_storage::{S3Storage, S3StorageConfig};
