# Checkpointing Design for Rust Cylon

## Executive Summary

This document outlines a checkpointing system for Rust Cylon, inspired by Twister2's implementation but modernized for distributed storage backends and improved coordination mechanisms.

### Twister2's Approach (Reference)

- Master-worker coordination with `CheckpointManager` on job master
- Frequency-based checkpointing (every N task executions)
- Barrier synchronization via message stream
- Storage backends: HDFS and Local filesystem
- Full snapshots (no incremental)
- Family-based versioning (minimum across all workers)

### Rust Cylon's Current State

- No fault tolerance mechanisms
- Existing Arrow IPC serialization for shuffle operations
- MPI-based synchronous collective operations
- Clear checkpoint boundaries at shuffle/join operations

### Proposed Design Highlights

- Distributed coordination (no single master bottleneck)
- Operation-based checkpointing (at shuffle boundaries)
- MPI collective barriers for synchronization
- Modern storage: Object storage (S3/MinIO/Ceph), HDFS, parallel filesystems (Lustre/GPFS)
- Support for incremental checkpoints
- Async checkpointing option

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Application Layer                                │
│   Table::join() → Table::shuffle() → Table::aggregate()                 │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────────┐
│                    Checkpoint Manager (per rank)                         │
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────────┐     │
│  │ CheckpointCtx   │  │ OperationTracker │  │ RecoveryManager     │     │
│  │ - enabled       │  │ - op_id          │  │ - find_checkpoint() │     │
│  │ - strategy      │  │ - dependencies   │  │ - restore_tables()  │     │
│  │ - storage       │  │ - metadata       │  │ - validate()        │     │
│  └─────────────────┘  └──────────────────┘  └─────────────────────┘     │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────────┐
│                 Coordination Layer (Distributed)                         │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │  MPI-based barrier sync  │  Consistent snapshot coordination  │     │
│  │  allreduce for version   │  Two-phase commit for completion   │     │
│  └────────────────────────────────────────────────────────────────┘     │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────────┐
│                    Storage Abstraction Layer                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐  │
│  │ ObjectStore  │  │ ParallelFS   │  │ HDFS         │  │ LocalFS     │  │
│  │ S3/MinIO     │  │ Lustre/GPFS  │  │ (legacy)     │  │ (testing)   │  │
│  │ Ceph         │  │ BeeGFS       │  │              │  │             │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └─────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Storage Trait (`src/checkpoint/storage.rs`)

```rust
use async_trait::async_trait;

/// Storage backend for checkpoints
#[async_trait]
pub trait CheckpointStorage: Send + Sync {
    /// Write checkpoint data
    async fn write(&self, path: &str, data: &[u8]) -> CylonResult<()>;

    /// Read checkpoint data
    async fn read(&self, path: &str) -> CylonResult<Vec<u8>>;

    /// Check if checkpoint exists
    async fn exists(&self, path: &str) -> CylonResult<bool>;

    /// Delete checkpoint
    async fn delete(&self, path: &str) -> CylonResult<()>;

    /// List checkpoints matching prefix
    async fn list(&self, prefix: &str) -> CylonResult<Vec<String>>;

    /// Atomic rename (for commit protocol)
    async fn rename(&self, from: &str, to: &str) -> CylonResult<()>;
}

/// Object storage implementation (S3, MinIO, Ceph)
pub struct ObjectStorage {
    client: Box<dyn object_store::ObjectStore>,
    bucket: String,
}

/// Parallel filesystem implementation (Lustre, GPFS, BeeGFS)
pub struct ParallelFsStorage {
    base_path: PathBuf,
    stripe_count: Option<u32>,  // Lustre striping
}

/// HDFS implementation (for compatibility)
pub struct HdfsStorage {
    client: hdfs::HdfsClient,
    base_path: String,
}
```

### 2. Checkpoint Metadata (`src/checkpoint/metadata.rs`)

```rust
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct CheckpointMetadata {
    /// Unique checkpoint ID
    pub checkpoint_id: u64,

    /// Operation that created this checkpoint
    pub operation_id: u64,
    pub operation_name: String,

    /// Distributed context
    pub rank: i32,
    pub world_size: i32,

    /// Timing
    pub timestamp: u64,
    pub duration_ms: u64,

    /// Data statistics
    pub table_rows: u64,
    pub table_columns: u32,
    pub data_size_bytes: u64,

    /// Schema for validation
    pub schema_hash: u64,

    /// Dependencies (for incremental checkpoints)
    pub parent_checkpoint_id: Option<u64>,
    pub is_incremental: bool,

    /// Status
    pub status: CheckpointStatus,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum CheckpointStatus {
    InProgress,
    Committed,
    Failed,
    Expired,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct GlobalCheckpointState {
    pub checkpoint_id: u64,
    pub all_ranks_committed: bool,
    pub committed_ranks: Vec<i32>,
    pub min_version: u64,
}
```

### 3. Checkpoint Strategy (`src/checkpoint/strategy.rs`)

```rust
/// When to trigger checkpoints
pub enum CheckpointTrigger {
    /// After every N operations
    OperationCount(u64),

    /// After N bytes processed
    DataVolume(u64),

    /// Time-based interval
    TimeInterval(Duration),

    /// Manual trigger only
    Manual,

    /// At specific operation types
    OperationType(Vec<OperationType>),
}

#[derive(Clone, Debug)]
pub enum OperationType {
    Shuffle,
    DistributedJoin,
    DistributedSort,
    SetOperation,
    Aggregation,
}

/// Checkpoint strategy configuration
pub struct CheckpointStrategy {
    pub trigger: CheckpointTrigger,
    pub async_write: bool,           // Non-blocking writes
    pub incremental: bool,           // Delta checkpoints
    pub compression: Compression,    // Compress data
    pub retention: RetentionPolicy,  // How long to keep
}

pub enum Compression {
    None,
    Lz4,
    Zstd { level: i32 },
    Snappy,
}

pub struct RetentionPolicy {
    pub max_checkpoints: usize,
    pub max_age: Option<Duration>,
    pub keep_latest_n: usize,
}
```

### 4. Checkpoint Context (`src/checkpoint/context.rs`)

```rust
pub struct CheckpointContext {
    /// Storage backend
    storage: Arc<dyn CheckpointStorage>,

    /// Configuration
    strategy: CheckpointStrategy,
    job_id: String,

    /// Distributed context
    communicator: Arc<dyn Communicator>,
    rank: i32,
    world_size: i32,

    /// State tracking
    current_checkpoint_id: AtomicU64,
    operation_counter: AtomicU64,
    last_checkpoint_time: Mutex<Instant>,

    /// Async checkpoint handle
    pending_checkpoint: Mutex<Option<JoinHandle<CylonResult<()>>>>,
}

impl CheckpointContext {
    pub fn new(
        storage: Arc<dyn CheckpointStorage>,
        strategy: CheckpointStrategy,
        communicator: Arc<dyn Communicator>,
        job_id: String,
    ) -> Self { ... }

    /// Check if checkpoint should be triggered
    pub fn should_checkpoint(&self, op_type: OperationType) -> bool { ... }

    /// Create checkpoint for table(s)
    pub async fn checkpoint(
        &self,
        tables: &[(&str, &Table)],
        operation_id: u64,
        operation_name: &str,
    ) -> CylonResult<u64> { ... }

    /// Synchronous checkpoint with barrier
    pub fn checkpoint_sync(
        &self,
        tables: &[(&str, &Table)],
        operation_id: u64,
        operation_name: &str,
    ) -> CylonResult<u64> { ... }

    /// Restore from latest checkpoint
    pub async fn restore_latest(
        &self,
        ctx: Arc<CylonContext>,
    ) -> CylonResult<HashMap<String, Table>> { ... }

    /// Restore from specific checkpoint
    pub async fn restore(
        &self,
        ctx: Arc<CylonContext>,
        checkpoint_id: u64,
    ) -> CylonResult<HashMap<String, Table>> { ... }
}
```

### 5. Distributed Coordination (`src/checkpoint/coordinator.rs`)

```rust
/// Two-phase commit for consistent distributed checkpoints
pub struct CheckpointCoordinator {
    communicator: Arc<dyn Communicator>,
    rank: i32,
    world_size: i32,
}

impl CheckpointCoordinator {
    /// Phase 1: All ranks prepare checkpoint, vote commit/abort
    pub fn prepare_checkpoint(&self, checkpoint_id: u64) -> CylonResult<bool> {
        // Each rank writes checkpoint to temp location
        // Vote via MPI_Allreduce (AND of all votes)
        let local_vote: i32 = 1; // 1 = prepared, 0 = failed
        let mut global_vote: i32 = 0;

        self.communicator.allreduce(
            &local_vote,
            &mut global_vote,
            MpiOp::Land, // Logical AND
        )?;

        Ok(global_vote == 1)
    }

    /// Phase 2: Commit or abort based on vote
    pub fn commit_checkpoint(&self, checkpoint_id: u64) -> CylonResult<()> {
        // Rename temp checkpoint to final location
        // Barrier to ensure all ranks complete
        self.communicator.barrier()?;
        Ok(())
    }

    /// Find latest consistent checkpoint across all ranks
    pub fn find_latest_checkpoint(&self) -> CylonResult<Option<u64>> {
        // Each rank reports its latest checkpoint
        // Use MPI_Allreduce with MIN to find common checkpoint
        let local_latest = self.get_local_latest()?;
        let mut global_latest: u64 = 0;

        self.communicator.allreduce(
            &local_latest,
            &mut global_latest,
            MpiOp::Min,
        )?;

        if global_latest == 0 {
            Ok(None)
        } else {
            Ok(Some(global_latest))
        }
    }
}
```

### 6. Serialization (`src/checkpoint/serialize.rs`)

```rust
/// Checkpoint data format
pub struct CheckpointData {
    pub metadata: CheckpointMetadata,
    pub tables: HashMap<String, Vec<u8>>,  // Serialized Arrow IPC
}

impl CheckpointData {
    /// Serialize checkpoint to bytes
    pub fn serialize(&self) -> CylonResult<Vec<u8>> {
        // Header: metadata length (4 bytes) + metadata (JSON)
        // Body: for each table:
        //   - name length (4 bytes) + name
        //   - data length (8 bytes) + Arrow IPC data
        ...
    }

    /// Deserialize checkpoint from bytes
    pub fn deserialize(data: &[u8]) -> CylonResult<Self> { ... }

    /// Serialize with compression
    pub fn serialize_compressed(
        &self,
        compression: Compression,
    ) -> CylonResult<Vec<u8>> { ... }
}

/// Incremental checkpoint using Arrow dictionaries
pub struct IncrementalCheckpoint {
    pub base_checkpoint_id: u64,
    pub changed_batches: Vec<RecordBatch>,
    pub deleted_indices: Vec<u64>,
}
```

---

## Integration Points

### 1. CylonContext Extension (`src/ctx.rs`)

```rust
impl CylonContext {
    /// Enable checkpointing with configuration
    pub fn enable_checkpointing(
        &mut self,
        storage: Arc<dyn CheckpointStorage>,
        strategy: CheckpointStrategy,
    ) -> CylonResult<()> { ... }

    /// Get checkpoint context if enabled
    pub fn checkpoint_ctx(&self) -> Option<&CheckpointContext> { ... }

    /// Convenience method to checkpoint
    pub async fn checkpoint(
        &self,
        tables: &[(&str, &Table)],
    ) -> CylonResult<u64> { ... }

    /// Restore from checkpoint
    pub async fn restore(
        &self,
        checkpoint_id: Option<u64>,
    ) -> CylonResult<HashMap<String, Table>> { ... }
}
```

### 2. Distributed Operations (`src/ops/shuffle.rs`)

```rust
pub fn shuffle_with_checkpoint(
    table: &Table,
    hash_columns: &[usize],
) -> CylonResult<Table> {
    let ctx = table.context();

    // Check if checkpoint should be triggered
    if let Some(ckpt_ctx) = ctx.checkpoint_ctx() {
        if ckpt_ctx.should_checkpoint(OperationType::Shuffle) {
            // Checkpoint pre-shuffle state
            ckpt_ctx.checkpoint_sync(
                &[("pre_shuffle", table)],
                ctx.next_operation_id(),
                "shuffle_input",
            )?;
        }
    }

    // Perform shuffle
    let result = shuffle(table, hash_columns)?;

    // Checkpoint post-shuffle state
    if let Some(ckpt_ctx) = ctx.checkpoint_ctx() {
        ckpt_ctx.checkpoint_sync(
            &[("post_shuffle", &result)],
            ctx.next_operation_id(),
            "shuffle_output",
        )?;
    }

    Ok(result)
}
```

---

## Storage Backend Implementations

### Object Storage (S3/MinIO/Ceph)

```rust
use object_store::{ObjectStore, aws::AmazonS3Builder};

impl ObjectStorage {
    pub fn s3(bucket: &str, region: &str) -> CylonResult<Self> {
        let client = AmazonS3Builder::new()
            .with_bucket_name(bucket)
            .with_region(region)
            .build()?;
        Ok(Self { client: Box::new(client), bucket: bucket.to_string() })
    }

    pub fn minio(endpoint: &str, bucket: &str) -> CylonResult<Self> {
        let client = AmazonS3Builder::new()
            .with_endpoint(endpoint)
            .with_bucket_name(bucket)
            .with_allow_http(true)
            .build()?;
        Ok(Self { client: Box::new(client), bucket: bucket.to_string() })
    }
}

#[async_trait]
impl CheckpointStorage for ObjectStorage {
    async fn write(&self, path: &str, data: &[u8]) -> CylonResult<()> {
        let location = object_store::path::Path::from(path);
        self.client.put(&location, data.into()).await?;
        Ok(())
    }

    async fn read(&self, path: &str) -> CylonResult<Vec<u8>> {
        let location = object_store::path::Path::from(path);
        let result = self.client.get(&location).await?;
        Ok(result.bytes().await?.to_vec())
    }
    // ... other methods
}
```

### Parallel Filesystem (Lustre)

```rust
impl ParallelFsStorage {
    pub fn lustre(base_path: &str, stripe_count: u32) -> CylonResult<Self> {
        // Set Lustre striping for better parallel I/O
        let path = PathBuf::from(base_path);
        std::fs::create_dir_all(&path)?;

        // Use lfs setstripe for optimal striping
        #[cfg(target_os = "linux")]
        {
            use std::process::Command;
            Command::new("lfs")
                .args(["setstripe", "-c", &stripe_count.to_string()])
                .arg(&path)
                .output()?;
        }

        Ok(Self { base_path: path, stripe_count: Some(stripe_count) })
    }
}

#[async_trait]
impl CheckpointStorage for ParallelFsStorage {
    async fn write(&self, path: &str, data: &[u8]) -> CylonResult<()> {
        let full_path = self.base_path.join(path);
        if let Some(parent) = full_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        tokio::fs::write(&full_path, data).await?;
        Ok(())
    }
    // ... other methods
}
```

---

## Recovery Process

```rust
impl CheckpointContext {
    /// Full recovery flow
    pub async fn recover(&self, ctx: Arc<CylonContext>) -> CylonResult<RecoveryResult> {
        // 1. Coordinate to find latest consistent checkpoint
        let checkpoint_id = self.coordinator.find_latest_checkpoint()?
            .ok_or(CylonError::new(Code::NotFound, "No checkpoint found"))?;

        // 2. Validate checkpoint exists on all ranks
        let valid = self.validate_checkpoint(checkpoint_id).await?;
        if !valid {
            return Err(CylonError::new(Code::Invalid, "Checkpoint validation failed"));
        }

        // 3. Load checkpoint data
        let data = self.load_checkpoint(checkpoint_id).await?;

        // 4. Deserialize tables
        let mut tables = HashMap::new();
        for (name, bytes) in data.tables {
            let table = deserialize_table(ctx.clone(), &bytes)?;
            tables.insert(name, table);
        }

        // 5. Barrier to ensure all ranks recovered
        self.communicator.barrier()?;

        Ok(RecoveryResult {
            checkpoint_id,
            tables,
            metadata: data.metadata,
        })
    }
}
```

---

## Checkpoint Directory Structure

```
{storage_root}/
├── {job_id}/
│   ├── metadata/
│   │   ├── checkpoint_000001.json
│   │   ├── checkpoint_000002.json
│   │   └── latest -> checkpoint_000002.json
│   ├── data/
│   │   ├── checkpoint_000001/
│   │   │   ├── rank_0/
│   │   │   │   ├── table_pre_shuffle.arrow
│   │   │   │   └── table_post_shuffle.arrow
│   │   │   ├── rank_1/
│   │   │   │   └── ...
│   │   │   └── _SUCCESS  (marker file for committed checkpoint)
│   │   └── checkpoint_000002/
│   │       └── ...
│   └── _temp/
│       └── checkpoint_000003/  (in-progress checkpoint)
│           └── ...
```

---

## Key Differences from Twister2

| Aspect | Twister2 | Rust Cylon (Proposed) |
|--------|----------|----------------------|
| **Coordination** | Centralized CheckpointManager | Distributed MPI collectives |
| **Trigger** | Frequency-based (N executions) | Operation-based + configurable |
| **Storage** | HDFS, Local | Object store, Parallel FS, HDFS, Local |
| **Format** | Custom binary | Arrow IPC (native) |
| **Compression** | None | LZ4, Zstd, Snappy |
| **Incremental** | No | Yes (optional) |
| **Async Write** | No | Yes (optional) |
| **Consistency** | Family-based minimum | Two-phase commit |
| **Recovery** | Per-family | Per-checkpoint with validation |

---

## Implementation Phases

### Phase 1: Core Infrastructure
- Storage trait and local filesystem implementation
- Checkpoint metadata and serialization
- Basic synchronous checkpointing
- Unit tests

### Phase 2: Distributed Coordination
- Two-phase commit protocol
- MPI-based barrier synchronization
- Consistent checkpoint discovery
- Integration tests with multiple ranks

### Phase 3: Storage Backends
- Object storage (S3/MinIO) using `object_store` crate
- Parallel filesystem support (Lustre striping)
- HDFS compatibility layer (optional)

### Phase 4: Advanced Features
- Async checkpointing with background threads
- Incremental checkpoints (delta encoding)
- Compression support (LZ4, Zstd)
- Automatic cleanup/retention policies

### Phase 5: Integration
- Automatic checkpointing in distributed operations
- High-level recovery API
- Monitoring and metrics
- Documentation and examples

---

## Dependencies

Add to `Cargo.toml`:

```toml
[dependencies]
# Async support
async-trait = "0.1"
tokio = { version = "1", features = ["full"] }

# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# Compression
lz4 = "1.24"
zstd = "0.13"

# Object storage
object_store = { version = "0.10", features = ["aws", "gcp", "azure"] }

# Optional: HDFS
hdfs = { version = "0.1", optional = true }

[features]
default = []
hdfs = ["dep:hdfs"]
checkpoint = []  # Enable checkpointing feature
```

---

## Example Usage

```rust
use cylon::prelude::*;
use cylon::checkpoint::{
    CheckpointContext, CheckpointStrategy, CheckpointTrigger,
    ObjectStorage, Compression, RetentionPolicy,
};

fn main() -> CylonResult<()> {
    // Initialize Cylon context
    let mut ctx = CylonContext::new_distributed()?;

    // Configure checkpointing
    let storage = ObjectStorage::minio("http://localhost:9000", "checkpoints")?;
    let strategy = CheckpointStrategy {
        trigger: CheckpointTrigger::OperationType(vec![
            OperationType::Shuffle,
            OperationType::DistributedJoin,
        ]),
        async_write: false,
        incremental: false,
        compression: Compression::Zstd { level: 3 },
        retention: RetentionPolicy {
            max_checkpoints: 10,
            max_age: Some(Duration::from_hours(24)),
            keep_latest_n: 3,
        },
    };

    ctx.enable_checkpointing(Arc::new(storage), strategy)?;

    // Load data
    let left_table = Table::from_csv(&ctx, "left.csv")?;
    let right_table = Table::from_csv(&ctx, "right.csv")?;

    // Distributed join (automatic checkpoint at shuffle boundaries)
    let result = left_table.distributed_join(
        &right_table,
        &["key"],
        &["key"],
        JoinType::Inner,
    )?;

    // Manual checkpoint if needed
    ctx.checkpoint(&[("final_result", &result)])?;

    Ok(())
}

// Recovery example
fn recover_from_failure() -> CylonResult<()> {
    let mut ctx = CylonContext::new_distributed()?;

    // Configure same storage
    let storage = ObjectStorage::minio("http://localhost:9000", "checkpoints")?;
    ctx.enable_checkpointing(Arc::new(storage), default_strategy())?;

    // Restore from latest checkpoint
    let tables = ctx.restore(None)?; // None = latest

    let result = tables.get("final_result")
        .ok_or(CylonError::new(Code::NotFound, "Table not found"))?;

    // Continue processing...
    Ok(())
}
```

---

## References

- [Twister2 Checkpointing Implementation](https://github.com/DSC-SPIDAL/twister2/tree/master/twister2/checkpointing)
- [Apache Arrow IPC Format](https://arrow.apache.org/docs/format/Columnar.html#ipc-file-format)
- [Object Store Crate](https://docs.rs/object_store/latest/object_store/)
- [MPI Two-Phase Commit Patterns](https://www.mpi-forum.org/docs/)
