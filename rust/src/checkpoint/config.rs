//! Configuration types for the checkpointing system.

use std::path::PathBuf;
use std::time::Duration;

use super::incremental::IncrementalConfig;

/// Main configuration for the checkpoint system.
#[derive(Clone, Debug)]
pub struct CheckpointConfig {
    /// Job identifier
    pub job_id: String,
    /// Storage configuration
    pub storage: StorageConfig,
    /// Trigger configuration
    pub trigger: TriggerConfig,
    /// Retention/pruning policy
    pub retention: PrunePolicy,
    /// Whether to enable async checkpoint I/O
    pub async_io: bool,
    /// Whether to enable compression
    pub compression: Option<CompressionConfig>,
    /// Whether to enable incremental checkpoints (simple flag, deprecated - use incremental_config)
    pub incremental: bool,
    /// Detailed incremental checkpoint configuration
    pub incremental_config: IncrementalConfig,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            job_id: "default".to_string(),
            storage: StorageConfig::default(),
            trigger: TriggerConfig::default(),
            retention: PrunePolicy::default(),
            async_io: true,
            compression: None,
            incremental: false,
            incremental_config: IncrementalConfig::default(),
        }
    }
}

impl CheckpointConfig {
    /// Create a new config with the given job ID
    pub fn new(job_id: impl Into<String>) -> Self {
        Self {
            job_id: job_id.into(),
            ..Default::default()
        }
    }

    /// Set storage configuration
    pub fn with_storage(mut self, storage: StorageConfig) -> Self {
        self.storage = storage;
        self
    }

    /// Set trigger configuration
    pub fn with_trigger(mut self, trigger: TriggerConfig) -> Self {
        self.trigger = trigger;
        self
    }

    /// Set retention policy
    pub fn with_retention(mut self, retention: PrunePolicy) -> Self {
        self.retention = retention;
        self
    }

    /// Enable async I/O
    pub fn with_async_io(mut self, enabled: bool) -> Self {
        self.async_io = enabled;
        self
    }

    /// Enable compression
    pub fn with_compression(mut self, compression: CompressionConfig) -> Self {
        self.compression = Some(compression);
        self
    }

    /// Enable incremental checkpoints
    pub fn with_incremental(mut self, enabled: bool) -> Self {
        self.incremental = enabled;
        if enabled {
            self.incremental_config.enabled = true;
        }
        self
    }

    /// Set detailed incremental configuration
    pub fn with_incremental_config(mut self, config: IncrementalConfig) -> Self {
        self.incremental = config.enabled;
        self.incremental_config = config;
        self
    }
}

/// Storage backend configuration.
#[derive(Clone, Debug)]
pub enum StorageConfig {
    /// Local/shared filesystem storage
    FileSystem {
        /// Base path for checkpoints
        base_path: PathBuf,
    },
    /// S3-compatible object storage
    #[cfg(feature = "s3")]
    S3 {
        /// S3 bucket name
        bucket: String,
        /// Prefix within the bucket
        prefix: String,
        /// AWS region
        region: Option<String>,
        /// Custom endpoint (for MinIO, etc.)
        endpoint: Option<String>,
        /// Whether to use path-style addressing (for MinIO, etc.)
        force_path_style: bool,
    },
}

impl Default for StorageConfig {
    fn default() -> Self {
        StorageConfig::FileSystem {
            base_path: PathBuf::from("/tmp/cylon_checkpoints"),
        }
    }
}

impl StorageConfig {
    /// Create filesystem storage config
    pub fn filesystem(base_path: impl Into<PathBuf>) -> Self {
        StorageConfig::FileSystem {
            base_path: base_path.into(),
        }
    }

    /// Create S3 storage config
    #[cfg(feature = "s3")]
    pub fn s3(bucket: impl Into<String>, prefix: impl Into<String>) -> Self {
        StorageConfig::S3 {
            bucket: bucket.into(),
            prefix: prefix.into(),
            region: None,
            endpoint: None,
            force_path_style: false,
        }
    }

    /// Create S3 storage config with region
    #[cfg(feature = "s3")]
    pub fn s3_with_region(
        bucket: impl Into<String>,
        prefix: impl Into<String>,
        region: impl Into<String>,
    ) -> Self {
        StorageConfig::S3 {
            bucket: bucket.into(),
            prefix: prefix.into(),
            region: Some(region.into()),
            endpoint: None,
            force_path_style: false,
        }
    }

    /// Create S3-compatible storage config (MinIO, LocalStack, etc.)
    #[cfg(feature = "s3")]
    pub fn s3_compatible(
        bucket: impl Into<String>,
        prefix: impl Into<String>,
        endpoint: impl Into<String>,
    ) -> Self {
        StorageConfig::S3 {
            bucket: bucket.into(),
            prefix: prefix.into(),
            region: None,
            endpoint: Some(endpoint.into()),
            force_path_style: true,
        }
    }
}

/// Trigger configuration for when to checkpoint.
#[derive(Clone, Debug)]
pub struct TriggerConfig {
    /// Checkpoint after this many operations
    pub operation_threshold: Option<u64>,
    /// Checkpoint after processing this many bytes
    pub bytes_threshold: Option<u64>,
    /// Checkpoint at this interval
    pub interval: Option<Duration>,
    /// For serverless: checkpoint when remaining time drops below this
    pub time_budget_threshold: Option<Duration>,
    /// For serverless: total time budget
    pub total_time_budget: Option<Duration>,
}

impl Default for TriggerConfig {
    fn default() -> Self {
        Self {
            operation_threshold: Some(100),
            bytes_threshold: Some(100 * 1024 * 1024), // 100MB
            interval: Some(Duration::from_secs(300)),  // 5 minutes
            time_budget_threshold: None,
            total_time_budget: None,
        }
    }
}

impl TriggerConfig {
    /// Create a new trigger config
    pub fn new() -> Self {
        Self {
            operation_threshold: None,
            bytes_threshold: None,
            interval: None,
            time_budget_threshold: None,
            total_time_budget: None,
        }
    }

    /// Set operation threshold
    pub fn with_operation_threshold(mut self, threshold: u64) -> Self {
        self.operation_threshold = Some(threshold);
        self
    }

    /// Set bytes threshold
    pub fn with_bytes_threshold(mut self, threshold: u64) -> Self {
        self.bytes_threshold = Some(threshold);
        self
    }

    /// Set interval
    pub fn with_interval(mut self, interval: Duration) -> Self {
        self.interval = Some(interval);
        self
    }

    /// Set time budget threshold (for serverless)
    pub fn with_time_budget(mut self, threshold: Duration, total: Duration) -> Self {
        self.time_budget_threshold = Some(threshold);
        self.total_time_budget = Some(total);
        self
    }

    /// Create config for serverless environments
    pub fn serverless(time_budget: Duration, reserve_time: Duration) -> Self {
        Self {
            operation_threshold: None,
            bytes_threshold: None,
            interval: None,
            time_budget_threshold: Some(reserve_time),
            total_time_budget: Some(time_budget),
        }
    }

    /// Create config for HPC/MPI environments
    pub fn hpc(operations: u64, bytes: u64) -> Self {
        Self {
            operation_threshold: Some(operations),
            bytes_threshold: Some(bytes),
            interval: None,
            time_budget_threshold: None,
            total_time_budget: None,
        }
    }
}

/// Pruning policy for old checkpoints.
#[derive(Clone, Debug)]
pub struct PrunePolicy {
    /// Maximum number of checkpoints to retain
    pub max_checkpoints: usize,
    /// Maximum age of checkpoints to retain
    pub max_age: Option<Duration>,
    /// Always keep at least this many recent checkpoints
    pub min_retain: usize,
    /// Only prune checkpoints with status Committed
    pub only_prune_committed: bool,
}

impl Default for PrunePolicy {
    fn default() -> Self {
        Self {
            max_checkpoints: 10,
            max_age: Some(Duration::from_secs(7 * 24 * 60 * 60)), // 7 days
            min_retain: 3,
            only_prune_committed: true,
        }
    }
}

impl PrunePolicy {
    /// Create a new prune policy
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum checkpoints to keep
    pub fn with_max_checkpoints(mut self, max: usize) -> Self {
        self.max_checkpoints = max;
        self
    }

    /// Set maximum age
    pub fn with_max_age(mut self, max_age: Duration) -> Self {
        self.max_age = Some(max_age);
        self
    }

    /// Set minimum checkpoints to retain
    pub fn with_min_retain(mut self, min: usize) -> Self {
        self.min_retain = min;
        self
    }

    /// Disable pruning
    pub fn disabled() -> Self {
        Self {
            max_checkpoints: usize::MAX,
            max_age: None,
            min_retain: usize::MAX,
            only_prune_committed: true,
        }
    }
}

/// Compression configuration for checkpoints.
///
/// Uses Arrow IPC native compression which supports LZ4 and ZSTD codecs.
/// Compression is applied at the Arrow buffer level for efficient storage.
#[derive(Clone, Debug)]
pub struct CompressionConfig {
    /// Compression algorithm
    pub algorithm: CompressionAlgorithm,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            algorithm: CompressionAlgorithm::Lz4,
        }
    }
}

impl CompressionConfig {
    /// Create a new compression config with the given algorithm
    pub fn new(algorithm: CompressionAlgorithm) -> Self {
        Self { algorithm }
    }

    /// Create LZ4 compression config (fast compression)
    ///
    /// LZ4 provides very fast compression and decompression speeds
    /// with moderate compression ratios. Recommended for most use cases.
    pub fn lz4() -> Self {
        Self {
            algorithm: CompressionAlgorithm::Lz4,
        }
    }

    /// Create Zstd compression config (good compression ratio)
    ///
    /// Zstandard provides excellent compression ratios with good speed.
    /// Recommended when storage space is at a premium.
    pub fn zstd() -> Self {
        Self {
            algorithm: CompressionAlgorithm::Zstd,
        }
    }

    /// Create Zstd compression config with specified level
    ///
    /// Note: Arrow IPC uses default Zstd compression level internally.
    /// This method is provided for API compatibility but the level is ignored.
    #[deprecated(note = "Arrow IPC uses default Zstd level; use zstd() instead")]
    pub fn zstd_with_level(_level: i32) -> Self {
        Self {
            algorithm: CompressionAlgorithm::Zstd,
        }
    }

    /// Create Snappy compression config
    ///
    /// Note: Snappy is not supported by Arrow IPC. This will fall back to LZ4.
    #[deprecated(note = "Snappy not supported by Arrow IPC; use lz4() instead")]
    pub fn snappy() -> Self {
        Self {
            algorithm: CompressionAlgorithm::Lz4,
        }
    }

    /// Create config with no compression
    pub fn none() -> Self {
        Self {
            algorithm: CompressionAlgorithm::None,
        }
    }
}

/// Supported compression algorithms for Arrow IPC.
///
/// Arrow IPC supports LZ4 and ZSTD compression at the buffer level.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CompressionAlgorithm {
    /// No compression
    #[default]
    None,
    /// LZ4 frame compression (fast)
    Lz4,
    /// Zstandard compression (good compression ratio)
    Zstd,
}
