//! Checkpoint storage backend implementations.
//!
//! Provides storage backends for persisting checkpoint data.

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
