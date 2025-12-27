# Unified Checkpointing Design for Rust Cylon

## Overview

This document describes a unified checkpointing infrastructure that supports both traditional HPC/MPI environments and serverless environments (Lambda, Fargate, ECS) through a common API with pluggable backends.

## Design Goals

1. **Single API** - Application code uses the same checkpoint API regardless of environment
2. **Pluggable Backends** - Swap coordination and storage implementations based on deployment
3. **Environment-Aware** - Each backend optimizes for its environment's constraints
4. **Incremental Adoption** - Can start with one environment and add others later
5. **Testable** - Local/mock implementations for testing

---

## Architecture

![Unified Checkpointing Architecture](checkpointing_architecture.png)

---

## Core Traits

### 1. CheckpointCoordinator

Handles distributed coordination - when and how to synchronize checkpoints across workers.

```rust
use async_trait::async_trait;

/// Distributed coordination for checkpoints
#[async_trait]
pub trait CheckpointCoordinator: Send + Sync {
    /// Get this worker's identity
    fn worker_id(&self) -> WorkerId;

    /// Total number of workers (may be dynamic in serverless)
    fn world_size(&self) -> usize;

    /// Should we trigger a checkpoint now?
    /// - MPI: checks operation count, data volume
    /// - Serverless: checks remaining time budget
    fn should_checkpoint(&self, context: &CheckpointContext) -> bool;

    /// Begin a checkpoint - coordinate with other workers
    /// - MPI: barrier + vote via allreduce
    /// - Serverless: Redis SETNX for distributed lock + pub/sub
    async fn begin_checkpoint(&self, checkpoint_id: u64) -> CylonResult<CheckpointDecision>;

    /// Commit a checkpoint after all workers have written data
    /// - MPI: barrier to confirm all wrote successfully
    /// - Serverless: Update Redis checkpoint status, verify all shards complete
    async fn commit_checkpoint(&self, checkpoint_id: u64) -> CylonResult<()>;

    /// Abort a checkpoint (rollback)
    async fn abort_checkpoint(&self, checkpoint_id: u64) -> CylonResult<()>;

    /// Find the latest checkpoint that all workers agree on
    /// - MPI: allreduce with MIN
    /// - Serverless: Query Redis for min completed version across workers
    async fn find_latest_checkpoint(&self) -> CylonResult<Option<u64>>;

    /// Register this worker as alive (heartbeat)
    /// - MPI: no-op (MPI handles failure detection)
    /// - Serverless: Update Redis heartbeat with TTL
    async fn heartbeat(&self) -> CylonResult<()>;

    /// Claim a work unit (Serverless-specific, no-op for MPI)
    async fn claim_work(&self, work_unit_id: &str) -> CylonResult<bool> {
        Ok(true) // Default: always succeeds (MPI has static assignment)
    }
}

#[derive(Clone, Debug)]
pub enum WorkerId {
    /// Stable MPI rank
    Rank(i32),
    /// Ephemeral serverless worker identifier
    Serverless { worker_id: String },
}

/// Priority level for checkpoint decisions (more granular than binary urgent/not-urgent)
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum CheckpointPriority {
    /// No checkpoint needed
    None,
    /// Checkpoint when convenient (can be deferred)
    Low,
    /// Should checkpoint soon
    Medium,
    /// Should checkpoint now
    High,
    /// Must checkpoint immediately (e.g., serverless time budget critical)
    Critical,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum CheckpointDecision {
    /// Proceed with checkpoint at the given priority
    Proceed(CheckpointPriority),
    /// Skip this checkpoint (e.g., another worker vetoed)
    Skip,
    /// Defer checkpoint (with reason)
    Defer(String),
}

impl CheckpointDecision {
    /// Helper for urgent checkpoints (Critical priority)
    pub fn urgent() -> Self {
        CheckpointDecision::Proceed(CheckpointPriority::Critical)
    }

    /// Helper for normal checkpoints (Medium priority)
    pub fn proceed() -> Self {
        CheckpointDecision::Proceed(CheckpointPriority::Medium)
    }

    /// Check if this decision allows proceeding
    pub fn should_proceed(&self) -> bool {
        matches!(self, CheckpointDecision::Proceed(_))
    }

    /// Get the priority if proceeding, None if skipping/deferring
    pub fn priority(&self) -> Option<CheckpointPriority> {
        match self {
            CheckpointDecision::Proceed(p) => Some(*p),
            CheckpointDecision::Skip | CheckpointDecision::Defer(_) => None,
        }
    }
}
```

### 2. CheckpointStorage

Handles reading and writing checkpoint data.

```rust
#[async_trait]
pub trait CheckpointStorage: Send + Sync {
    /// Write checkpoint data
    async fn write(
        &self,
        checkpoint_id: u64,
        worker_id: &WorkerId,
        key: &str,
        data: &[u8],
    ) -> CylonResult<String>; // Returns path/URI

    /// Read checkpoint data
    async fn read(
        &self,
        checkpoint_id: u64,
        worker_id: &WorkerId,
        key: &str,
    ) -> CylonResult<Vec<u8>>;

    /// Check if checkpoint exists
    async fn exists(
        &self,
        checkpoint_id: u64,
        worker_id: &WorkerId,
        key: &str,
    ) -> CylonResult<bool>;

    /// List all keys for a checkpoint
    async fn list_keys(
        &self,
        checkpoint_id: u64,
        worker_id: &WorkerId,
    ) -> CylonResult<Vec<String>>;

    /// Delete a checkpoint
    async fn delete(&self, checkpoint_id: u64) -> CylonResult<()>;

    /// List all available checkpoints for the job
    /// Returns checkpoint IDs sorted by creation time (newest first)
    async fn list_checkpoints(&self) -> CylonResult<Vec<u64>>;

    /// Atomic move (for commit protocol)
    /// Moves from staging to final location
    async fn commit_write(
        &self,
        checkpoint_id: u64,
        worker_id: &WorkerId,
    ) -> CylonResult<()>;

    /// Write checkpoint metadata
    async fn write_metadata(
        &self,
        checkpoint_id: u64,
        metadata: &CheckpointMetadata,
    ) -> CylonResult<()>;

    /// Read checkpoint metadata
    async fn read_metadata(
        &self,
        checkpoint_id: u64,
    ) -> CylonResult<CheckpointMetadata>;

    /// Get the base path/URI for this storage
    fn base_path(&self) -> &str;
}
```

### 3. CheckpointSerializer

Handles serialization of tables and state.

```rust
pub trait CheckpointSerializer: Send + Sync {
    /// Serialize a table to bytes
    fn serialize_table(&self, table: &Table) -> CylonResult<Vec<u8>>;

    /// Deserialize bytes to a table
    fn deserialize_table(
        &self,
        data: &[u8],
        ctx: Arc<CylonContext>,
    ) -> CylonResult<Table>;

    /// Serialize arbitrary state
    fn serialize_state<T: Serialize>(&self, state: &T) -> CylonResult<Vec<u8>>;

    /// Deserialize arbitrary state
    fn deserialize_state<T: DeserializeOwned>(&self, data: &[u8]) -> CylonResult<T>;

    /// Get the format identifier (for compatibility checking)
    fn format_id(&self) -> &str;
}
```

### 4. CheckpointTrigger

Determines when to checkpoint (environment-specific strategies).

```rust
pub trait CheckpointTrigger: Send + Sync {
    /// Update trigger state after an operation
    fn record_operation(&self, op_type: OperationType, bytes_processed: u64);

    /// Check if we should checkpoint now
    fn should_checkpoint(&self) -> bool;

    /// Force a checkpoint (e.g., before shutdown)
    fn force_checkpoint(&self);

    /// Reset trigger state after successful checkpoint
    fn reset(&self);

    /// Get urgency level
    fn urgency(&self) -> CheckpointUrgency;

    /// Get current checkpoint context for decision making
    fn get_context(&self) -> CheckpointContext;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum CheckpointUrgency {
    /// No checkpoint needed
    None,
    /// Checkpoint when convenient
    Low,
    /// Should checkpoint soon
    Medium,
    /// Must checkpoint now (serverless timeout approaching)
    Critical,
}
```

---

## Protocol Details and Guarantees

### Two-Phase Commit (2PC) Protocol

The checkpointing system uses a Two-Phase Commit protocol to ensure global consistency across all workers. This section clarifies the implementation details and failure modes.

#### Protocol Phases

**Phase 1: Prepare (Vote)**
- Each worker votes on whether it's ready to checkpoint
- MPI: Uses `allreduce` with AND operation - all workers must agree
- Serverless: Uses Redis `SETNX` for distributed locking + pub/sub for vote collection

**Phase 2: Commit**
- After all workers successfully write checkpoint data, they confirm commitment
- MPI: Uses `barrier` to ensure all ranks have written successfully
- Serverless: Updates Redis checkpoint status and verifies all shards complete via atomic operations

![2PC Protocol Flow](2pc_protocol.png)

#### MPI Coordinator Role

In MPI environments, there is no single coordinator node. Instead, coordination is achieved through collective operations:
- `allreduce` aggregates votes from all ranks
- `barrier` ensures synchronization at commit
- All ranks participate equally in the protocol

#### Serverless/Redis Coordinator Role

In serverless environments, Redis acts as the coordinator through atomic operations:
- Each worker registers its status via Redis hash sets
- Redis's single-threaded execution model provides atomicity
- Pub/sub enables real-time coordination between workers
- A "checkpoint complete" status requires all shards to be marked committed

### Handling 2PC Blocking Scenarios

The classic 2PC protocol is blocking - if a coordinator or participant fails mid-protocol, other participants may be stuck. Here's how each backend handles this:

**MPI Backend:**
- If any rank fails, MPI runtime detects and aborts the job
- No need for timeout handling - MPI's fault detection is authoritative
- On restart, all ranks query for the latest complete checkpoint

**Serverless Backend (Redis):**
- Workers set a TTL on their "IN_PROGRESS" status entries using `SETEX`
- If a worker times out without committing, its entry expires automatically
- Other workers can detect incomplete checkpoints via Redis hash queries
- Recovery: Query Redis to determine if checkpoint should be committed or rolled back

```rust
/// Redis-based recovery from uncertain state
impl RedisCoordinator {
    /// Check if a checkpoint is in uncertain state and resolve it
    async fn resolve_uncertain_checkpoint(&self, checkpoint_id: u64) -> CylonResult<CheckpointResolution> {
        let mut conn = self.redis.get_async_connection().await?;

        // Get all worker statuses for this checkpoint from Redis hash
        let statuses: HashMap<String, String> = conn.hgetall(
            format!("job:{}:ckpt:{}:status", self.job_id, checkpoint_id)
        ).await?;

        let committed_count = statuses.values().filter(|s| *s == "COMMITTED").count();
        let in_progress_count = statuses.values().filter(|s| *s == "IN_PROGRESS").count();
        let expected_workers = self.world_size();

        if committed_count == expected_workers {
            // All committed - checkpoint is valid
            Ok(CheckpointResolution::Committed)
        } else if in_progress_count > 0 {
            // Some workers still in progress - check if TTL expired
            // Redis TTL handles expiration automatically, so if IN_PROGRESS
            // entries still exist, they haven't timed out yet
            Ok(CheckpointResolution::InProgress)
        } else {
            // Partial commit - needs manual resolution or rollback
            // This is the PartiallyCommitted state
            Ok(CheckpointResolution::PartiallyCommitted { committed_count, expected: expected_workers })
        }
    }
}
```

### Atomicity Guarantees by Storage Backend

#### POSIX Filesystem (Lustre, GPFS, local)
- **Guarantee**: Atomic via `rename()` system call
- The `commit_write` operation uses `rename()` to move from staging to committed directory
- `rename()` is atomic on POSIX-compliant filesystems
- If crash occurs mid-rename, the file remains in staging (uncommitted)

#### Amazon S3
- **Guarantee**: Pseudo-atomic (NOT truly atomic)
- S3 does not have a native atomic rename operation
- `commit_write` performs: `CopyObject` → `DeleteObject`
- **Failure Window**: If crash occurs between copy and delete:
  - Data exists in both staging and committed locations
  - Recovery must check for and clean up duplicates

```rust
/// S3 commit with recovery metadata
impl S3Storage {
    async fn commit_write(
        &self,
        checkpoint_id: u64,
        worker_id: &WorkerId,
    ) -> CylonResult<()> {
        // Write a "commit intent" marker first
        self.write_commit_intent(checkpoint_id, worker_id).await?;

        // Copy from staging to committed
        let objects = self.list_staging_objects(checkpoint_id, worker_id).await?;
        for obj in &objects {
            let final_key = obj.replace("/staging/", "/committed/");
            self.s3_client.copy_object()
                .bucket(&self.bucket)
                .key(&final_key)
                .copy_source(format!("{}/{}", self.bucket, obj))
                .send()
                .await?;
        }

        // Delete staging objects
        for obj in objects {
            self.s3_client.delete_object()
                .bucket(&self.bucket)
                .key(&obj)
                .send()
                .await?;
        }

        // Remove commit intent marker
        self.delete_commit_intent(checkpoint_id, worker_id).await?;

        Ok(())
    }

    /// On startup, check for incomplete commits and complete them
    async fn recover_incomplete_commits(&self) -> CylonResult<()> {
        let intents = self.list_commit_intents().await?;
        for intent in intents {
            // Complete the commit that was interrupted
            self.complete_interrupted_commit(&intent).await?;
        }
        Ok(())
    }
}
```

### CheckpointTrigger vs CheckpointCoordinator: Role Clarification

These two traits have distinct responsibilities that should not be confused:

**`CheckpointTrigger`** - Local Decision Maker
- Runs on a **single worker**
- Answers: "Based on MY local state, do I think we should checkpoint?"
- Factors: operation count, bytes processed, time budget, memory pressure
- Does NOT communicate with other workers
- Returns a simple boolean or urgency level

**`CheckpointCoordinator`** - Global Decision Maker
- Orchestrates across **all workers**
- Answers: "Given that workers want to checkpoint, do we ALL agree to do so NOW?"
- Aggregates proposals from individual triggers
- Handles the 2PC protocol (prepare/commit/abort)
- Communicates with other workers (MPI collective or Redis)

**Interaction Flow:**

![Checkpoint Decision Flow](checkpoint_decision_flow.png)

### Recovery from PartiallyCommitted State

The `CheckpointStatus::PartiallyCommitted` state occurs when some workers committed but others failed. This is a critical failure mode that requires explicit recovery.

**Recovery Strategies:**

1. **Roll Forward (Recommended for small failures)**
   - If most workers committed successfully
   - Restart failed workers and have them re-attempt their portion
   - Requires checkpoint data to still be in staging for failed workers

2. **Roll Back (Recommended for large failures)**
   - If many workers failed or data is inconsistent
   - Delete the partially committed checkpoint entirely
   - Restore from the previous complete checkpoint

```rust
/// Recovery protocol for PartiallyCommitted state
pub struct CheckpointRecovery {
    coordinator: Arc<dyn CheckpointCoordinator>,
    storage: Arc<dyn CheckpointStorage>,
}

impl CheckpointRecovery {
    /// Analyze and recover from a partially committed checkpoint
    pub async fn recover(&self, checkpoint_id: u64) -> CylonResult<RecoveryOutcome> {
        let metadata = self.storage.read_metadata(checkpoint_id).await?;

        if metadata.status != CheckpointStatus::PartiallyCommitted {
            return Ok(RecoveryOutcome::NotNeeded);
        }

        let committed_workers: Vec<_> = metadata.workers.iter()
            .filter(|w| w.status == WorkerCheckpointStatus::Committed)
            .collect();

        let failed_workers: Vec<_> = metadata.workers.iter()
            .filter(|w| w.status != WorkerCheckpointStatus::Committed)
            .collect();

        let total = metadata.workers.len();
        let committed = committed_workers.len();

        // Decision: Roll forward if >50% committed and staging data exists
        if committed > total / 2 {
            // Check if staging data exists for failed workers
            let can_roll_forward = self.can_roll_forward(&failed_workers, checkpoint_id).await?;

            if can_roll_forward {
                return self.roll_forward(checkpoint_id, &failed_workers).await;
            }
        }

        // Otherwise, roll back
        self.roll_back(checkpoint_id).await
    }

    async fn roll_forward(
        &self,
        checkpoint_id: u64,
        failed_workers: &[&WorkerCheckpointInfo]
    ) -> CylonResult<RecoveryOutcome> {
        for worker in failed_workers {
            // Move staging data to committed for each failed worker
            self.storage.commit_write(checkpoint_id, &worker.worker_id).await?;

            // Update metadata to mark worker as committed
            self.update_worker_status(checkpoint_id, &worker.worker_id, WorkerCheckpointStatus::Committed).await?;
        }

        // Update overall checkpoint status
        self.update_checkpoint_status(checkpoint_id, CheckpointStatus::Committed).await?;

        Ok(RecoveryOutcome::RolledForward { checkpoint_id })
    }

    async fn roll_back(&self, checkpoint_id: u64) -> CylonResult<RecoveryOutcome> {
        // Delete all data for this checkpoint
        self.storage.delete(checkpoint_id).await?;

        // Find the previous complete checkpoint
        let checkpoints = self.storage.list_checkpoints().await?;
        let previous = checkpoints.into_iter()
            .filter(|id| *id < checkpoint_id)
            .max();

        Ok(RecoveryOutcome::RolledBack {
            deleted_checkpoint: checkpoint_id,
            restore_from: previous,
        })
    }
}

#[derive(Debug)]
pub enum RecoveryOutcome {
    NotNeeded,
    RolledForward { checkpoint_id: u64 },
    RolledBack { deleted_checkpoint: u64, restore_from: Option<u64> },
}
```

---

## Checkpoint Pruning

Old checkpoints consume storage and should be cleaned up according to a configurable retention policy. This is especially important for:
- **HPC environments**: Scratch filesystems often have storage quotas and automatic purge policies
- **Cloud storage**: S3 costs accumulate with stored data
- **Failed checkpoints**: Incomplete or partially committed checkpoints should be cleaned up

### Pruning Policy

```rust
#[derive(Clone, Debug)]
pub struct PrunePolicy {
    /// Maximum number of checkpoints to retain
    pub max_checkpoints: usize,

    /// Maximum age of checkpoints to retain
    pub max_age: Option<Duration>,

    /// Always keep at least this many recent checkpoints
    /// (even if they exceed max_age)
    pub min_retain: usize,

    /// Only prune checkpoints with status Committed
    /// (never prune InProgress or PartiallyCommitted)
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
```

### CheckpointStorage Pruning Extension

```rust
#[async_trait]
pub trait CheckpointStorageExt: CheckpointStorage {
    /// Prune old checkpoints according to policy
    async fn prune(&self, policy: &PrunePolicy) -> CylonResult<PruneResult>;
}

#[async_trait]
impl<T: CheckpointStorage> CheckpointStorageExt for T {
    async fn prune(&self, policy: &PrunePolicy) -> CylonResult<PruneResult> {
        let mut checkpoints = self.list_checkpoints().await?;
        let mut deleted = Vec::new();
        let mut retained = Vec::new();
        let now = SystemTime::now();

        // Sort by ID (oldest first for deletion consideration)
        checkpoints.sort();

        for (i, checkpoint_id) in checkpoints.iter().enumerate() {
            let metadata = self.read_metadata(*checkpoint_id).await?;

            // Never prune non-committed if policy says so
            if policy.only_prune_committed && metadata.status != CheckpointStatus::Committed {
                retained.push(*checkpoint_id);
                continue;
            }

            // Always retain min_retain most recent
            let remaining = checkpoints.len() - i;
            if remaining <= policy.min_retain {
                retained.push(*checkpoint_id);
                continue;
            }

            // Check max_checkpoints
            if retained.len() >= policy.max_checkpoints {
                self.delete(*checkpoint_id).await?;
                deleted.push(*checkpoint_id);
                continue;
            }

            // Check max_age
            if let Some(max_age) = policy.max_age {
                let checkpoint_time = UNIX_EPOCH + Duration::from_secs(metadata.timestamp);
                if let Ok(age) = now.duration_since(checkpoint_time) {
                    if age > max_age && retained.len() >= policy.min_retain {
                        self.delete(*checkpoint_id).await?;
                        deleted.push(*checkpoint_id);
                        continue;
                    }
                }
            }

            retained.push(*checkpoint_id);
        }

        Ok(PruneResult { deleted, retained })
    }
}

#[derive(Debug)]
pub struct PruneResult {
    pub deleted: Vec<u64>,
    pub retained: Vec<u64>,
}
```

### Automatic Pruning in CheckpointManager

```rust
impl CheckpointManager {
    /// Checkpoint and then prune old checkpoints
    pub async fn checkpoint_and_prune(
        &self,
        tables: &[(&str, &Table)],
    ) -> CylonResult<CheckpointResult> {
        let result = self.checkpoint(tables).await?;

        // Only prune on rank 0 / leader to avoid races
        if self.is_leader() {
            if let Err(e) = self.storage.prune(&self.config.retention).await {
                // Log but don't fail - pruning is best-effort
                log::warn!("Failed to prune old checkpoints: {}", e);
            }
        }

        result
    }
}
```

---

## Incremental Checkpoints

The `parent_checkpoint_id` field enables incremental checkpointing, where only changed data is written.

### Incremental Checkpoint Strategy

```rust
/// Tracks changes since last checkpoint for incremental support
pub struct ChangeTracker {
    /// Tables that have been modified since last checkpoint
    modified_tables: HashSet<String>,

    /// For row-level tracking: modified row ranges per table
    modified_ranges: HashMap<String, Vec<RowRange>>,

    /// Parent checkpoint this is based on
    parent_checkpoint_id: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RowRange {
    /// Start row index (inclusive)
    pub start: u64,
    /// End row index (exclusive)
    pub end: u64,
    /// Type of change
    pub change_type: RowChangeType,
}

/// Type of row change
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RowChangeType {
    /// New rows appended
    Append,
    /// Existing rows updated
    Update,
    /// Rows deleted
    Delete,
}

impl ChangeTracker {
    pub fn new(parent_checkpoint_id: Option<u64>) -> Self {
        Self {
            modified_tables: HashSet::new(),
            modified_ranges: HashMap::new(),
            parent_checkpoint_id,
        }
    }

    /// Mark a table as modified
    pub fn mark_table_modified(&mut self, table_name: &str) {
        self.modified_tables.insert(table_name.to_string());
    }

    /// Mark specific rows as modified (for fine-grained incremental)
    pub fn mark_rows_modified(&mut self, table_name: &str, range: RowRange) {
        self.modified_tables.insert(table_name.to_string());
        self.modified_ranges
            .entry(table_name.to_string())
            .or_default()
            .push(range);
    }

    /// Check if table needs to be checkpointed
    pub fn needs_checkpoint(&self, table_name: &str) -> bool {
        self.modified_tables.contains(table_name)
    }
}
```

### Incremental Checkpoint Metadata

```rust
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct IncrementalCheckpointInfo {
    /// The checkpoint ID this is based on
    pub parent_checkpoint_id: u64,

    /// Tables that are unchanged (reference parent)
    pub unchanged_tables: Vec<String>,

    /// Tables that have deltas
    pub delta_tables: Vec<DeltaTableInfo>,

    /// Tables that are fully rewritten
    pub full_tables: Vec<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct DeltaTableInfo {
    pub name: String,
    /// Path to delta file (contains only changed rows)
    pub delta_path: String,
    /// Operation type
    pub delta_type: DeltaType,
    /// Rows affected
    pub affected_rows: u64,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeltaType {
    /// Append new rows to the table
    Append,
    /// Update existing rows (includes row indices)
    Update,
    /// Delete rows (includes row indices to remove)
    Delete,
    /// Mixed operations (append + update + delete)
    Mixed,
    /// Full table rewrite (no delta, complete replacement)
    Full,
}
```

### Restoring from Incremental Checkpoints

```rust
impl CheckpointManager {
    /// Restore from an incremental checkpoint by applying deltas
    pub async fn restore_incremental(
        &self,
        ctx: Arc<CylonContext>,
        checkpoint_id: u64,
    ) -> CylonResult<RestoreResult> {
        let metadata = self.storage.read_metadata(checkpoint_id).await?;

        // Build the chain of checkpoints to apply
        let mut checkpoint_chain = vec![checkpoint_id];
        let mut current = &metadata;

        while let Some(parent_id) = current.parent_checkpoint_id {
            checkpoint_chain.push(parent_id);
            // Note: In practice, cache metadata to avoid repeated reads
            current = &self.storage.read_metadata(parent_id).await?;
        }

        // Reverse to apply from oldest to newest
        checkpoint_chain.reverse();

        // Start with base checkpoint
        let base_id = checkpoint_chain[0];
        let mut tables = self.restore_base(ctx.clone(), base_id).await?;

        // Apply each delta in order
        for delta_checkpoint_id in &checkpoint_chain[1..] {
            self.apply_deltas(&mut tables, *delta_checkpoint_id).await?;
        }

        Ok(RestoreResult {
            checkpoint_id,
            tables,
            metadata,
        })
    }

    async fn apply_deltas(
        &self,
        tables: &mut HashMap<String, Table>,
        checkpoint_id: u64,
    ) -> CylonResult<()> {
        let metadata = self.storage.read_metadata(checkpoint_id).await?;

        if let Some(incremental_info) = &metadata.incremental_info {
            for delta_info in &incremental_info.delta_tables {
                let delta_data = self.storage.read(
                    checkpoint_id,
                    &self.coordinator.worker_id(),
                    &format!("{}.delta", delta_info.name),
                ).await?;

                let delta_table = self.serializer.deserialize_table(&delta_data, self.ctx.clone())?;

                let table = tables.get_mut(&delta_info.name)
                    .ok_or_else(|| CylonError::new(Code::NotFound, "Table not found for delta"))?;

                match delta_info.delta_type {
                    DeltaType::Append => {
                        *table = table.concat(&delta_table)?;
                    }
                    // Other delta types would require more complex merge logic
                    _ => {
                        // Fall back to full replacement for complex deltas
                        *table = delta_table;
                    }
                }
            }
        }

        Ok(())
    }
}
```

---

## Checkpoint Integration with Cylon Operations

Checkpointing integrates automatically with Cylon's synchronous execution model using a callback architecture, similar to the channel API's non-blocking I/O pattern.

### Design Principles

1. **Transparent to application code** - Operations trigger checkpoints automatically
2. **Synchronous execution** - Cylon operations remain synchronous; checkpoint I/O blocks when needed
3. **Callback-based** - Application registers a checkpoint handler, similar to channel callbacks
4. **Cooperative** - Checkpoints happen at operation boundaries, not mid-operation
5. **Crash-safe** - Checkpoints are durable once staged, even if worker dies before commit

### CheckpointController

The `CheckpointController` is embedded in `CylonContext` and intercepts operations:

```rust
pub struct CheckpointController {
    /// Trigger determines when to checkpoint
    trigger: Arc<dyn CheckpointTrigger>,

    /// Coordinator handles distributed consensus
    coordinator: Arc<dyn CheckpointCoordinator>,

    /// Storage backend
    storage: Arc<dyn CheckpointStorage>,

    /// Serializer for table data
    serializer: Arc<dyn CheckpointSerializer>,

    /// Tables registered for checkpointing
    tables: RwLock<HashMap<String, TableRef>>,

    /// Tokio runtime for async checkpoint I/O
    runtime: tokio::runtime::Runtime,

    /// Whether checkpointing is enabled
    enabled: AtomicBool,
}

/// Reference to a table that can be checkpointed
pub struct TableRef {
    /// Weak reference to avoid preventing table cleanup
    table: Weak<RwLock<Table>>,
    /// Last checkpointed version (for incremental)
    last_checkpoint_version: AtomicU64,
}
```

### Registration API

Applications register tables for automatic checkpointing:

```rust
impl CylonContext {
    /// Register a table for automatic checkpointing
    pub fn register_checkpoint_table(&self, name: &str, table: Arc<RwLock<Table>>) {
        self.checkpoint_controller.register(name, table);
    }

    /// Unregister a table (e.g., when done with it)
    pub fn unregister_checkpoint_table(&self, name: &str) {
        self.checkpoint_controller.unregister(name);
    }

    /// Configure checkpoint callback for custom handling
    pub fn on_checkpoint<F>(&self, callback: F)
    where
        F: Fn(CheckpointEvent) -> CheckpointAction + Send + Sync + 'static
    {
        self.checkpoint_controller.set_callback(callback);
    }
}

/// Event passed to checkpoint callback
pub struct CheckpointEvent {
    pub checkpoint_id: u64,
    pub trigger_reason: TriggerReason,
    pub tables: Vec<String>,
    pub urgency: CheckpointPriority,
}

/// Action returned from callback
pub enum CheckpointAction {
    /// Proceed with checkpoint
    Proceed,
    /// Skip this checkpoint
    Skip,
    /// Proceed and then exit (for serverless timeout)
    ProceedAndExit,
}

pub enum TriggerReason {
    OperationThreshold,
    BytesThreshold,
    TimeBudget,
    Manual,
}
```

### Operation Hooks

Each Cylon operation calls the checkpoint controller at boundaries:

```rust
impl Table {
    pub fn join(&self, ctx: &CylonContext, other: &Table) -> CylonResult<Table> {
        // Pre-operation: check if we need to checkpoint BEFORE starting
        // (important for time-budget triggers)
        ctx.checkpoint_controller.before_operation()?;

        // Execute the actual join (synchronous)
        let result = self.join_impl(other)?;

        // Post-operation: record metrics and maybe trigger checkpoint
        ctx.checkpoint_controller.after_operation(
            OperationType::Join,
            result.size_bytes(),
        )?;

        Ok(result)
    }
}
```

### Controller Implementation

```rust
impl CheckpointController {
    /// Called before each operation
    pub fn before_operation(&self) -> CylonResult<()> {
        if !self.enabled.load(Ordering::Relaxed) {
            return Ok(());
        }

        // For time-budget triggers, check if we're about to run out of time
        if self.trigger.urgency() == CheckpointPriority::Critical {
            self.do_checkpoint(TriggerReason::TimeBudget)?;
        }

        Ok(())
    }

    /// Called after each operation completes
    pub fn after_operation(
        &self,
        op_type: OperationType,
        bytes: u64
    ) -> CylonResult<()> {
        if !self.enabled.load(Ordering::Relaxed) {
            return Ok(());
        }

        // Record the operation
        self.trigger.record_operation(op_type, bytes);

        // Check if we should checkpoint
        if self.trigger.should_checkpoint() {
            self.do_checkpoint(TriggerReason::from_trigger(&self.trigger))?;
        }

        Ok(())
    }

    /// Execute checkpoint (blocks until staged - see crash safety below)
    fn do_checkpoint(&self, reason: TriggerReason) -> CylonResult<()> {
        let tables = self.tables.read().unwrap();
        let table_names: Vec<String> = tables.keys().cloned().collect();

        let checkpoint_id = self.next_checkpoint_id();
        let urgency = self.trigger.urgency();

        // Invoke callback if registered
        if let Some(callback) = &self.callback {
            let event = CheckpointEvent {
                checkpoint_id,
                trigger_reason: reason,
                tables: table_names.clone(),
                urgency,
            };

            match callback(event) {
                CheckpointAction::Skip => {
                    self.trigger.reset();
                    return Ok(());
                }
                CheckpointAction::ProceedAndExit => {
                    self.execute_checkpoint(checkpoint_id, &tables)?;
                    return Err(CylonError::checkpoint_exit());
                }
                CheckpointAction::Proceed => {}
            }
        }

        // Execute the checkpoint (blocks on async I/O)
        self.execute_checkpoint(checkpoint_id, &tables)?;

        // Reset trigger counters
        self.trigger.reset();

        Ok(())
    }
}
```

### Async Non-Blocking Checkpoint I/O

To avoid blocking Cylon's processing during checkpoint uploads, the checkpoint controller uses **async I/O with table snapshots**. This allows processing to continue while checkpoint data uploads in the background.

#### Design Principles

1. **Snapshot-before-write**: Clone/snapshot table data before starting async upload
2. **Single in-flight checkpoint**: Only one checkpoint can be in progress at a time
3. **Wait-on-next**: If a new checkpoint triggers while one is in flight, wait for previous to complete
4. **Crash-safe**: Snapshots are immutable; in-flight checkpoints can be recovered

#### Extended Controller with Async Support

```rust
pub struct CheckpointController {
    /// Trigger determines when to checkpoint
    trigger: Arc<dyn CheckpointTrigger>,

    /// Coordinator handles distributed consensus
    coordinator: Arc<dyn CheckpointCoordinator>,

    /// Storage backend
    storage: Arc<dyn CheckpointStorage>,

    /// Serializer for table data
    serializer: Arc<dyn CheckpointSerializer>,

    /// Tables registered for checkpointing
    tables: RwLock<HashMap<String, TableRef>>,

    /// Tokio runtime for async checkpoint I/O
    runtime: tokio::runtime::Runtime,

    /// Whether checkpointing is enabled
    enabled: AtomicBool,

    /// Handle to in-flight checkpoint (if any)
    in_flight: Mutex<Option<InFlightCheckpoint>>,
}

/// Tracks an in-progress async checkpoint
struct InFlightCheckpoint {
    /// Checkpoint ID
    checkpoint_id: u64,
    /// Join handle for the async task
    handle: tokio::task::JoinHandle<CylonResult<()>>,
    /// When the checkpoint started
    started_at: Instant,
}

/// Immutable snapshot of table data for checkpointing
struct TableSnapshot {
    /// Table name
    name: String,
    /// Serialized data (Arrow IPC bytes)
    data: Bytes,
    /// Version number for incremental checkpointing
    version: u64,
}
```

#### Async Checkpoint Execution

```rust
impl CheckpointController {
    /// Execute checkpoint asynchronously - returns immediately after snapshot
    fn do_checkpoint(&self, reason: TriggerReason) -> CylonResult<()> {
        // First, wait for any in-flight checkpoint to complete
        self.wait_for_in_flight()?;

        let tables = self.tables.read().unwrap();
        let table_names: Vec<String> = tables.keys().cloned().collect();

        let checkpoint_id = self.next_checkpoint_id();
        let urgency = self.trigger.urgency();

        // Invoke callback if registered
        if let Some(callback) = &self.callback {
            let event = CheckpointEvent {
                checkpoint_id,
                trigger_reason: reason,
                tables: table_names.clone(),
                urgency,
            };

            match callback(event) {
                CheckpointAction::Skip => {
                    self.trigger.reset();
                    return Ok(());
                }
                CheckpointAction::ProceedAndExit => {
                    // Critical: must block for exit case
                    self.execute_checkpoint_blocking(checkpoint_id, &tables)?;
                    return Err(CylonError::checkpoint_exit());
                }
                CheckpointAction::Proceed => {}
            }
        }

        // SNAPSHOT PHASE (synchronous, fast)
        // Create immutable snapshots of all registered tables
        let snapshots = self.create_snapshots(&tables)?;

        // UPLOAD PHASE (asynchronous, slow)
        // Spawn async task to upload snapshots - Cylon continues processing
        self.spawn_checkpoint_upload(checkpoint_id, snapshots)?;

        // Reset trigger counters immediately - don't wait for upload
        self.trigger.reset();

        Ok(())
    }

    /// Create immutable snapshots of all tables (fast, synchronous)
    fn create_snapshots(
        &self,
        tables: &HashMap<String, TableRef>,
    ) -> CylonResult<Vec<TableSnapshot>> {
        let mut snapshots = Vec::with_capacity(tables.len());

        for (name, table_ref) in tables.iter() {
            if let Some(table) = table_ref.table.upgrade() {
                let table_guard = table.read().unwrap();

                // Serialize to Arrow IPC bytes (in-memory, fast)
                let data = self.serializer.serialize(&table_guard)?;

                snapshots.push(TableSnapshot {
                    name: name.clone(),
                    data,
                    version: table_ref.last_checkpoint_version.load(Ordering::SeqCst) + 1,
                });
            }
        }

        Ok(snapshots)
    }

    /// Spawn async task to upload checkpoint data
    fn spawn_checkpoint_upload(
        &self,
        checkpoint_id: u64,
        snapshots: Vec<TableSnapshot>,
    ) -> CylonResult<()> {
        let storage = self.storage.clone();
        let coordinator = self.coordinator.clone();
        let job_id = self.job_id.clone();

        let handle = self.runtime.spawn(async move {
            // Update state to WRITING
            coordinator.update_state(checkpoint_id, CheckpointState::Writing).await?;

            // Upload each table snapshot to staging
            for snapshot in &snapshots {
                let key = format!(
                    "{}/staging/{}/{}",
                    job_id, checkpoint_id, snapshot.name
                );
                storage.put(&key, snapshot.data.clone()).await?;
            }

            // Write manifest
            let manifest = CheckpointManifest {
                checkpoint_id,
                tables: snapshots.iter().map(|s| s.name.clone()).collect(),
                timestamp: chrono::Utc::now(),
            };
            let manifest_key = format!("{}/staging/{}/manifest.json", job_id, checkpoint_id);
            storage.put(&manifest_key, serde_json::to_vec(&manifest)?).await?;

            // DURABILITY POINT: Update state to STAGED
            coordinator.update_state(checkpoint_id, CheckpointState::Staged).await?;

            // Commit phase: move staging → committed
            coordinator.update_state(checkpoint_id, CheckpointState::Committing).await?;

            for snapshot in &snapshots {
                let src = format!("{}/staging/{}/{}", job_id, checkpoint_id, snapshot.name);
                let dst = format!("{}/committed/{}/{}", job_id, checkpoint_id, snapshot.name);
                storage.copy(&src, &dst).await?;
            }

            // Copy manifest
            let src_manifest = format!("{}/staging/{}/manifest.json", job_id, checkpoint_id);
            let dst_manifest = format!("{}/committed/{}/manifest.json", job_id, checkpoint_id);
            storage.copy(&src_manifest, &dst_manifest).await?;

            // Mark committed
            coordinator.update_state(checkpoint_id, CheckpointState::Committed).await?;

            // Cleanup staging
            storage.delete_prefix(&format!("{}/staging/{}/", job_id, checkpoint_id)).await?;

            Ok(())
        });

        // Store the in-flight checkpoint handle
        let mut in_flight = self.in_flight.lock().unwrap();
        *in_flight = Some(InFlightCheckpoint {
            checkpoint_id,
            handle,
            started_at: Instant::now(),
        });

        Ok(())
    }

    /// Wait for in-flight checkpoint to complete (if any)
    fn wait_for_in_flight(&self) -> CylonResult<()> {
        let mut in_flight = self.in_flight.lock().unwrap();

        if let Some(checkpoint) = in_flight.take() {
            // Block on the async task
            let result = self.runtime.block_on(checkpoint.handle)
                .map_err(|e| CylonError::checkpoint_error(format!("Task panicked: {}", e)))?;

            // Propagate any checkpoint errors
            result?;
        }

        Ok(())
    }

    /// Check if there's an in-flight checkpoint (non-blocking)
    pub fn has_in_flight_checkpoint(&self) -> bool {
        self.in_flight.lock().unwrap().is_some()
    }

    /// Wait for all checkpoints to complete (call before shutdown)
    pub fn flush(&self) -> CylonResult<()> {
        self.wait_for_in_flight()
    }
}
```

#### Timing Diagram: Async Checkpoint Pipeline

![Async Checkpoint Timing](async_checkpoint_timing.png)

#### Edge Cases and Safety

**1. Checkpoint triggers while previous is in flight:**
```rust
// In do_checkpoint():
// Wait for previous checkpoint before starting new one
self.wait_for_in_flight()?;
```

**2. Worker shutdown with in-flight checkpoint:**
```rust
impl Drop for CheckpointController {
    fn drop(&mut self) {
        // Best-effort: wait for in-flight checkpoint
        // If we're being killed, the staged checkpoint protocol handles recovery
        let _ = self.flush();
    }
}
```

**3. Critical priority (timeout approaching):**
```rust
// For ProceedAndExit, we use blocking checkpoint
// because we need to ensure it completes before exit
CheckpointAction::ProceedAndExit => {
    self.execute_checkpoint_blocking(checkpoint_id, &tables)?;
    return Err(CylonError::checkpoint_exit());
}
```

**4. Snapshot memory overhead:**
```rust
// Snapshots are Arrow IPC bytes - typically smaller than in-memory representation
// For very large tables, consider streaming serialization:
impl CheckpointSerializer {
    /// Stream serialize to avoid full copy
    fn serialize_streaming(&self, table: &Table) -> impl Stream<Item = Bytes>;
}
```

#### Configuration

```rust
pub struct CheckpointConfig {
    // ... existing fields ...

    /// Enable async (non-blocking) checkpoint uploads
    /// Default: true
    pub async_upload: bool,

    /// Maximum time to wait for in-flight checkpoint during flush
    /// Default: 30 seconds
    pub flush_timeout: Duration,

    /// Whether to block on checkpoint for critical priority
    /// Default: true (recommended for serverless)
    pub block_on_critical: bool,
}
```

### Usage Example

```rust
use cylon::prelude::*;

fn main() -> CylonResult<()> {
    // Initialize context with checkpoint support
    let ctx = CylonContext::builder()
        .with_checkpointing(CheckpointConfig {
            job_id: "job_123".into(),
            ..Default::default()
        })
        .build()?;

    // Load or restore table
    let table = Arc::new(RwLock::new(
        match ctx.restore_checkpoint() {
            Ok(restored) => restored.tables.remove("orders").unwrap(),
            Err(_) => Table::from_csv(&ctx, "orders.csv")?,
        }
    ));

    // Register for automatic checkpointing
    ctx.register_checkpoint_table("orders", table.clone());

    // Optional: customize checkpoint behavior
    ctx.on_checkpoint(|event| {
        println!("Checkpoint triggered: {:?}", event.trigger_reason);
        if event.urgency == CheckpointPriority::Critical {
            CheckpointAction::ProceedAndExit
        } else {
            CheckpointAction::Proceed
        }
    });

    // Normal processing - checkpoints happen automatically
    let mut t = table.write().unwrap();
    *t = t.join(&ctx, &other_table)?;      // May checkpoint after
    *t = t.filter(&ctx, predicate)?;        // May checkpoint after
    *t = t.aggregate(&ctx, &agg_config)?;   // May checkpoint after

    // Write final output
    t.to_csv("output.csv")?;

    Ok(())
}
```

---

## Serverless Crash Safety

In serverless environments (Lambda, Fargate, ECS), the runtime can terminate the worker at any time after the deadline. We need to ensure checkpoints are durable even if the worker dies mid-checkpoint.

### The Problem

![Serverless Crash Problem](serverless_crash_problem.png)

### Solution: Staged Checkpoint Protocol

The key insight is that **the checkpoint is "safe" once data reaches S3 staging** - we don't need the worker to survive the commit phase. The commit (moving staging → committed) can be completed by the orchestrator or next worker.

#### Checkpoint States

```rust
/// Checkpoint states tracked in Redis
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CheckpointState {
    /// Worker is writing data to S3 staging
    Writing,
    /// Data is in S3 staging, ready to commit (DURABILITY POINT)
    Staged,
    /// Commit in progress (copying staging → committed)
    Committing,
    /// Fully committed and verified
    Committed,
    /// Failed and cleaned up
    Failed,
}
```

#### Protocol Flow

![Staged Checkpoint Protocol](staged_checkpoint_protocol.png)

#### Implementation

```rust
impl CheckpointController {
    /// Execute checkpoint with crash-safe staging
    fn execute_checkpoint(
        &self,
        checkpoint_id: u64,
        tables: &HashMap<String, TableRef>,
    ) -> CylonResult<()> {
        self.runtime.block_on(async {
            // 1. Begin - mark as WRITING
            self.coordinator.begin_checkpoint(checkpoint_id).await?;

            // 2. Write all table data to S3 staging
            for (name, table_ref) in tables {
                if let Some(table) = table_ref.table.upgrade() {
                    let table_guard = table.read().unwrap();
                    let data = self.serializer.serialize_table(&table_guard)?;

                    self.storage.write_to_staging(
                        checkpoint_id,
                        &self.coordinator.worker_id(),
                        name,
                        &data,
                    ).await?;
                }
            }

            // 3. CRITICAL: Mark as STAGED - this is the durability point
            // Even if we die after this, the checkpoint is recoverable
            self.coordinator.mark_staged(checkpoint_id).await?;

            // 4-7. Try to complete commit (best effort - can be done by others)
            if let Err(e) = self.try_complete_commit(checkpoint_id).await {
                // Log but don't fail - orchestrator will complete it
                log::warn!("Worker couldn't complete commit: {}", e);
            }

            Ok(())
        })
    }

    /// Attempt to complete the commit (may be interrupted)
    async fn try_complete_commit(&self, checkpoint_id: u64) -> CylonResult<()> {
        // Mark as committing
        self.coordinator.set_state(checkpoint_id, CheckpointState::Committing).await?;

        // Move from staging to committed
        self.storage.commit_write(
            checkpoint_id,
            &self.coordinator.worker_id(),
        ).await?;

        // Mark as fully committed
        self.coordinator.set_state(checkpoint_id, CheckpointState::Committed).await?;

        Ok(())
    }
}
```

#### Redis Coordinator with State Tracking

```rust
impl RedisCoordinator {
    /// Mark checkpoint as staged (durability point)
    pub async fn mark_staged(&self, checkpoint_id: u64) -> CylonResult<()> {
        let mut conn = self.redis.get_async_connection().await?;

        // Atomic update with timestamp
        let key = format!("job:{}:ckpt:{}:worker:{}",
            self.job_id, checkpoint_id, self.worker_id);

        redis::pipe()
            .hset(&key, "state", "STAGED")
            .hset(&key, "staged_at", now_timestamp())
            .expire(&key, 86400)  // 24 hour TTL
            .query_async(&mut conn)
            .await?;

        Ok(())
    }

    /// Get all checkpoints in a given state
    pub async fn get_checkpoints_in_state(
        &self,
        state: CheckpointState,
    ) -> CylonResult<Vec<StagedCheckpoint>> {
        let mut conn = self.redis.get_async_connection().await?;

        // Scan for checkpoint keys
        let pattern = format!("job:{}:ckpt:*:worker:*", self.job_id);
        let keys: Vec<String> = conn.keys(&pattern).await?;

        let mut results = Vec::new();
        for key in keys {
            let info: HashMap<String, String> = conn.hgetall(&key).await?;
            if info.get("state").map(|s| s.as_str()) == Some(state.as_str()) {
                results.push(StagedCheckpoint::from_redis(&key, &info)?);
            }
        }

        Ok(results)
    }
}

#[derive(Debug)]
pub struct StagedCheckpoint {
    pub checkpoint_id: u64,
    pub worker_id: String,
    pub staged_at: u64,
    pub state: CheckpointState,
}
```

#### Orchestrator Recovery

The orchestrator monitors for orphaned (STAGED but not COMMITTED) checkpoints:

```rust
impl JobOrchestrator {
    /// Main orchestration loop
    pub async fn run(&self) -> CylonResult<JobResult> {
        // Initialize job
        self.initialize_job().await?;
        self.launch_workers(0..self.world_size).await?;

        loop {
            tokio::time::sleep(Duration::from_secs(5)).await;

            // 1. Check for orphaned staged checkpoints
            self.recover_staged_checkpoints().await?;

            // 2. Check for dead workers
            self.replace_dead_workers().await?;

            // 3. Check if job is complete
            if self.is_job_complete().await? {
                return Ok(JobResult::Success);
            }
        }
    }

    /// Find and complete any STAGED checkpoints from dead workers
    async fn recover_staged_checkpoints(&self) -> CylonResult<()> {
        let staged = self.coordinator
            .get_checkpoints_in_state(CheckpointState::Staged)
            .await?;

        for checkpoint in staged {
            // Check if the worker that staged this is still alive
            let worker_alive = self.is_worker_alive(&checkpoint.worker_id).await?;

            if !worker_alive {
                log::info!(
                    "Completing orphaned checkpoint {} from dead worker {}",
                    checkpoint.checkpoint_id,
                    checkpoint.worker_id
                );

                self.complete_staged_checkpoint(&checkpoint).await?;
            } else {
                // Worker is alive - check if it's stuck
                let staged_duration = now_timestamp() - checkpoint.staged_at;
                if staged_duration > COMMIT_TIMEOUT_SECS {
                    log::warn!(
                        "Checkpoint {} stuck in STAGED for {}s, completing",
                        checkpoint.checkpoint_id,
                        staged_duration
                    );
                    self.complete_staged_checkpoint(&checkpoint).await?;
                }
            }
        }

        Ok(())
    }

    /// Complete a staged checkpoint (move staging → committed)
    async fn complete_staged_checkpoint(
        &self,
        checkpoint: &StagedCheckpoint,
    ) -> CylonResult<()> {
        // Update state to COMMITTING
        self.coordinator.set_state(
            checkpoint.checkpoint_id,
            CheckpointState::Committing
        ).await?;

        // Move from staging to committed in S3
        self.storage.commit_write(
            checkpoint.checkpoint_id,
            &WorkerId::Serverless { worker_id: checkpoint.worker_id.clone() },
        ).await?;

        // Update state to COMMITTED
        self.coordinator.set_state(
            checkpoint.checkpoint_id,
            CheckpointState::Committed
        ).await?;

        // Update latest checkpoint pointer if this is newer
        self.coordinator.maybe_update_latest(checkpoint.checkpoint_id).await?;

        Ok(())
    }

    /// Check if a worker is still alive via heartbeat
    async fn is_worker_alive(&self, worker_id: &str) -> CylonResult<bool> {
        let mut conn = self.redis.get_async_connection().await?;

        let key = format!("job:{}:worker:{}:heartbeat", self.job_id, worker_id);
        let exists: bool = conn.exists(&key).await?;

        Ok(exists)  // Key expires via TTL if worker stops heartbeating
    }
}
```

#### Worker Startup Recovery

When a new worker starts, it checks for incomplete checkpoints:

```rust
impl CheckpointController {
    /// Called during worker initialization
    pub async fn recover_on_startup(&self) -> CylonResult<()> {
        // Find any STAGED checkpoints for this worker that weren't committed
        let staged = self.coordinator
            .get_staged_checkpoints_for_worker(&self.worker_id)
            .await?;

        for checkpoint in staged {
            log::info!("Recovering staged checkpoint {}", checkpoint.checkpoint_id);
            self.try_complete_commit(checkpoint.checkpoint_id).await?;
        }

        Ok(())
    }
}
```

### Serverless Usage Example

```rust
#[tokio::main]
async fn main() -> CylonResult<()> {
    let job_id = std::env::var("JOB_ID")?;
    let deadline = Instant::now() + Duration::from_secs(
        std::env::var("TIMEOUT_SECS")?.parse()?
    );

    // Initialize with time-budget trigger
    let ctx = CylonContext::builder()
        .with_checkpointing(CheckpointConfig {
            job_id: job_id.clone(),
            environment: Some(EnvironmentType::Serverless),
            serverless: ServerlessCheckpointConfig {
                s3_bucket: std::env::var("S3_BUCKET")?,
                redis_url: std::env::var("REDIS_URL")?,
                checkpoint_reserve_secs: 60,
                safety_buffer_secs: 30,
            },
            ..Default::default()
        })
        .with_time_budget(deadline)
        .build()?;

    // Recover any incomplete checkpoints from previous run
    ctx.checkpoint_controller.recover_on_startup().await?;

    // Restore from latest committed checkpoint
    let table = Arc::new(RwLock::new(
        ctx.restore_or_load("orders", "s3://bucket/input.parquet").await?
    ));

    ctx.register_checkpoint_table("orders", table.clone());

    // Exit cleanly on critical checkpoint
    ctx.on_checkpoint(|event| {
        if event.urgency == CheckpointPriority::Critical {
            CheckpointAction::ProceedAndExit
        } else {
            CheckpointAction::Proceed
        }
    });

    // Process - will automatically checkpoint when needed
    // If killed after STAGED, orchestrator completes the checkpoint
    let mut t = table.write().unwrap();
    for batch in work_batches {
        *t = t.process_batch(&ctx, &batch)?;
        ctx.heartbeat().await?;
    }

    // Finished all work
    t.to_s3("s3://bucket/output.parquet").await?;
    ctx.mark_worker_complete().await?;

    Ok(())
}
```

### Checkpoint State Diagram

![Checkpoint State Diagram](checkpoint_state_diagram.png)

---

## Configuration and Factory Pattern

### Unified Configuration

```rust
/// Main configuration for the checkpoint system
#[derive(Clone, Debug)]
pub struct CheckpointConfig {
    /// Job identifier
    pub job_id: String,

    /// Storage configuration (filesystem or S3)
    pub storage: StorageConfig,

    /// Trigger configuration (when to checkpoint)
    pub trigger: TriggerConfig,

    /// Retention/pruning policy
    pub retention: PrunePolicy,

    /// Whether to enable async checkpoint I/O
    pub async_io: bool,

    /// Compression configuration (optional)
    pub compression: Option<CompressionConfig>,

    /// Whether to enable incremental checkpoints (simple flag)
    pub incremental: bool,

    /// Detailed incremental checkpoint configuration
    pub incremental_config: IncrementalConfig,
}

/// Storage backend configuration
#[derive(Clone, Debug)]
pub enum StorageConfig {
    /// Local/shared filesystem storage
    FileSystem {
        /// Base path for checkpoints
        base_path: PathBuf,
    },
    /// S3-compatible object storage (requires "s3" feature)
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
        /// Whether to use path-style addressing
        force_path_style: bool,
    },
}

/// Trigger configuration for when to checkpoint
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

/// Configuration for incremental checkpoint behavior
#[derive(Clone, Debug)]
pub struct IncrementalConfig {
    /// Enable incremental checkpoints
    pub enabled: bool,
    /// Enable row-level change tracking (more overhead, smaller deltas)
    pub track_rows: bool,
    /// Maximum chain depth before forcing a full checkpoint
    /// This prevents restore from becoming too slow
    pub max_chain_depth: u32,
    /// Force full checkpoint if savings ratio is below this threshold
    /// (i.e., if most tables are modified, just do a full checkpoint)
    pub min_savings_ratio: f64,
}

impl Default for IncrementalConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            track_rows: false,
            max_chain_depth: 10,
            min_savings_ratio: 0.2, // At least 20% unchanged tables to use incremental
        }
    }
}
```

### Builder Pattern

The `CheckpointManagerBuilder` provides a flexible way to construct checkpoint managers
with specific build methods for different deployment scenarios.

```rust
/// Builder for constructing CheckpointManager
pub struct CheckpointManagerBuilder {
    config: CheckpointConfig,
    ctx: Option<Arc<CylonContext>>,
}

impl CheckpointManagerBuilder {
    /// Create a new builder with default config
    pub fn new() -> Self {
        Self {
            config: CheckpointConfig::default(),
            ctx: None,
        }
    }

    /// Set the configuration
    pub fn with_config(mut self, config: CheckpointConfig) -> Self {
        self.config = config;
        self
    }

    /// Set the Cylon context
    pub fn with_context(mut self, ctx: Arc<CylonContext>) -> Self {
        self.ctx = Some(ctx);
        self
    }

    /// Set the job ID
    pub fn with_job_id(mut self, job_id: impl Into<String>) -> Self {
        self.config.job_id = job_id.into();
        self
    }

    /// Set the storage configuration
    pub fn with_storage(mut self, storage: StorageConfig) -> Self {
        self.config.storage = storage;
        self
    }

    /// Build a local (single-worker) checkpoint manager with filesystem storage
    pub async fn build_local(self) -> CylonResult<CheckpointManager<
        LocalCoordinator,
        FileSystemStorage,
        ArrowIpcSerializer,
        CompositeTrigger,
    >> {
        let ctx = self.ctx.ok_or_else(|| {
            CylonError::new(Code::InvalidArgument, "Context is required")
        })?;

        let base_path = match &self.config.storage {
            StorageConfig::FileSystem { base_path } => base_path.clone(),
            #[cfg(feature = "s3")]
            _ => return Err(CylonError::new(
                Code::InvalidArgument,
                "Local manager requires filesystem storage",
            )),
        };

        let storage = FileSystemStorage::new(base_path, &self.config.job_id);
        storage.initialize().await?;

        let coordinator = LocalCoordinator::new();
        let serializer = ArrowIpcSerializer::new();
        let trigger = CompositeTrigger::from_config(&self.config.trigger);

        Ok(CheckpointManager::new(
            ctx,
            Arc::new(coordinator),
            Arc::new(storage),
            Arc::new(serializer),
            Arc::new(trigger),
            self.config,
        ))
    }

    /// Build a local checkpoint manager with S3 storage
    #[cfg(feature = "s3")]
    pub async fn build_local_s3(self) -> CylonResult<CheckpointManager<
        LocalCoordinator,
        S3Storage,
        ArrowIpcSerializer,
        CompositeTrigger,
    >> {
        // ... S3-specific initialization
    }

    /// Build an MPI-based distributed checkpoint manager
    #[cfg(feature = "mpi")]
    pub async fn build_mpi(self) -> CylonResult<CheckpointManager<
        MpiCoordinator,
        FileSystemStorage,
        ArrowIpcSerializer,
        CompositeTrigger,
    >> {
        // ... MPI-specific initialization
    }

    /// Build a distributed checkpoint manager with Redis coordination
    #[cfg(feature = "redis")]
    pub async fn build_distributed(self) -> CylonResult<CheckpointManager<
        DistributedCoordinator,
        FileSystemStorage,
        ArrowIpcSerializer,
        CompositeTrigger,
    >> {
        // ... Redis-coordinated initialization
    }
}
```

### Usage Examples

```rust
// Local checkpoint manager with filesystem storage
let manager = CheckpointManagerBuilder::new()
    .with_context(ctx.clone())
    .with_job_id("my_job")
    .with_storage(StorageConfig::filesystem("/tmp/checkpoints"))
    .build_local()
    .await?;

// MPI-based distributed manager
#[cfg(feature = "mpi")]
let manager = CheckpointManagerBuilder::new()
    .with_context(ctx.clone())
    .with_job_id("distributed_job")
    .build_mpi()
    .await?;

// S3-backed checkpoint manager
#[cfg(feature = "s3")]
let manager = CheckpointManagerBuilder::new()
    .with_context(ctx.clone())
    .with_job_id("cloud_job")
    .with_storage(StorageConfig::s3("my-bucket", "checkpoints"))
    .build_local_s3()
    .await?;
```

---

## Shared Components

### CheckpointMetadata

```rust
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct CheckpointMetadata {
    /// Unique checkpoint ID (monotonically increasing)
    pub checkpoint_id: u64,

    /// Job identifier
    pub job_id: String,

    /// Timestamp when checkpoint started
    pub timestamp: u64,

    /// Duration to create checkpoint (ms)
    pub duration_ms: u64,

    /// Environment type
    pub environment: EnvironmentType,

    /// Workers that participated
    pub workers: Vec<WorkerCheckpointInfo>,

    /// Status
    pub status: CheckpointStatus,

    /// Schema hash for validation
    pub schema_hash: u64,

    /// Custom application state
    pub app_state: Option<Vec<u8>>,

    /// Parent checkpoint (for incremental)
    pub parent_checkpoint_id: Option<u64>,

    /// Incremental checkpoint details (if this is an incremental checkpoint)
    pub incremental_info: Option<IncrementalCheckpointInfo>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct WorkerCheckpointInfo {
    pub worker_id: WorkerId,
    pub tables: Vec<TableCheckpointInfo>,
    pub rows_total: u64,
    pub bytes_written: u64,
    pub status: WorkerCheckpointStatus,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TableCheckpointInfo {
    pub name: String,
    pub rows: u64,
    pub columns: u32,
    pub storage_path: String,
    /// Schema version for backward/forward compatibility during restore
    /// Increment when schema changes require migration logic
    pub schema_version: u32,
    /// Arrow schema fingerprint for validation
    pub schema_fingerprint: Option<String>,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
pub enum CheckpointStatus {
    InProgress,
    Committed,
    Failed,
    PartiallyCommitted, // Some workers committed, others failed
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
pub enum EnvironmentType {
    Mpi,
    Serverless,
    Local, // For testing
}
```

### CheckpointManager

The main entry point that orchestrates everything:

```rust
pub struct CheckpointManager {
    /// Distributed coordination
    coordinator: Arc<dyn CheckpointCoordinator>,

    /// Storage backend
    storage: Arc<dyn CheckpointStorage>,

    /// Serialization
    serializer: Arc<dyn CheckpointSerializer>,

    /// Checkpoint trigger
    trigger: Arc<dyn CheckpointTrigger>,

    /// Configuration
    config: CheckpointConfig,

    /// Current checkpoint ID
    current_checkpoint_id: AtomicU64,

    /// Job identifier
    job_id: String,
}

impl CheckpointManager {
    /// Create a new checkpoint manager with the given backends
    pub fn new(
        coordinator: Arc<dyn CheckpointCoordinator>,
        storage: Arc<dyn CheckpointStorage>,
        serializer: Arc<dyn CheckpointSerializer>,
        trigger: Arc<dyn CheckpointTrigger>,
        config: CheckpointConfig,
    ) -> Self { ... }

    /// Create with default backends for MPI environment
    pub fn for_mpi(
        communicator: Arc<dyn Communicator>,
        storage_path: &str,
        config: CheckpointConfig,
    ) -> CylonResult<Self> { ... }

    /// Create with default backends for serverless environment
    pub fn for_serverless(
        s3_bucket: &str,
        redis_url: &str,
        config: CheckpointConfig,
    ) -> CylonResult<Self> { ... }

    /// Record an operation (updates trigger)
    pub fn record_operation(&self, op_type: OperationType, bytes: u64) {
        self.trigger.record_operation(op_type, bytes);
    }

    /// Check if checkpoint should happen now
    pub fn should_checkpoint(&self) -> bool {
        self.trigger.should_checkpoint()
    }

    /// Get checkpoint urgency
    pub fn urgency(&self) -> CheckpointUrgency {
        self.trigger.urgency()
    }

    /// Checkpoint tables (main API)
    pub async fn checkpoint(
        &self,
        tables: &[(&str, &Table)],
    ) -> CylonResult<CheckpointResult> {
        self.checkpoint_with_state(tables, None::<()>).await
    }

    /// Checkpoint tables with custom application state
    pub async fn checkpoint_with_state<S: Serialize>(
        &self,
        tables: &[(&str, &Table)],
        app_state: Option<S>,
    ) -> CylonResult<CheckpointResult> {
        let checkpoint_id = self.current_checkpoint_id.fetch_add(1, Ordering::SeqCst);

        // 1. Coordinate: begin checkpoint
        let decision = self.coordinator.begin_checkpoint(checkpoint_id).await?;

        match decision {
            CheckpointDecision::Skip => {
                return Ok(CheckpointResult::Skipped);
            }
            CheckpointDecision::Proceed | CheckpointDecision::Urgent => {}
        }

        // 2. Serialize and write tables
        let worker_id = self.coordinator.worker_id();
        let mut table_infos = Vec::new();

        for (name, table) in tables {
            let data = self.serializer.serialize_table(table)?;
            let path = self.storage.write(
                checkpoint_id,
                &worker_id,
                name,
                &data,
            ).await?;

            table_infos.push(TableCheckpointInfo {
                name: name.to_string(),
                rows: table.num_rows() as u64,
                columns: table.num_columns() as u32,
                storage_path: path,
            });
        }

        // 3. Commit write (move from staging)
        self.storage.commit_write(checkpoint_id, &worker_id).await?;

        // 4. Coordinate: commit checkpoint
        self.coordinator.commit_checkpoint(checkpoint_id).await?;

        // 5. Reset trigger
        self.trigger.reset();

        Ok(CheckpointResult::Committed {
            checkpoint_id,
            tables: table_infos,
        })
    }

    /// Restore from the latest checkpoint
    pub async fn restore(
        &self,
        ctx: Arc<CylonContext>,
    ) -> CylonResult<RestoreResult> {
        self.restore_from(ctx, None).await
    }

    /// Restore from a specific checkpoint
    pub async fn restore_from(
        &self,
        ctx: Arc<CylonContext>,
        checkpoint_id: Option<u64>,
    ) -> CylonResult<RestoreResult> {
        // 1. Find checkpoint to restore
        let checkpoint_id = match checkpoint_id {
            Some(id) => id,
            None => self.coordinator.find_latest_checkpoint().await?
                .ok_or_else(|| CylonError::new(Code::NotFound, "No checkpoint found"))?,
        };

        // 2. Read metadata
        let metadata = self.storage.read_metadata(checkpoint_id).await?;

        // 3. Find our worker's checkpoint info
        let worker_id = self.coordinator.worker_id();
        let worker_info = self.find_worker_info(&metadata, &worker_id)?;

        // 4. Read and deserialize tables
        let mut tables = HashMap::new();
        for table_info in &worker_info.tables {
            let data = self.storage.read(
                checkpoint_id,
                &worker_id,
                &table_info.name,
            ).await?;

            let table = self.serializer.deserialize_table(&data, ctx.clone())?;
            tables.insert(table_info.name.clone(), table);
        }

        // 5. Update current checkpoint ID
        self.current_checkpoint_id.store(checkpoint_id + 1, Ordering::SeqCst);

        Ok(RestoreResult {
            checkpoint_id,
            tables,
            metadata,
        })
    }
}

#[derive(Debug)]
pub enum CheckpointResult {
    Committed {
        checkpoint_id: u64,
        tables: Vec<TableCheckpointInfo>,
    },
    Skipped,
}

#[derive(Debug)]
pub struct RestoreResult {
    pub checkpoint_id: u64,
    pub tables: HashMap<String, Table>,
    pub metadata: CheckpointMetadata,
}
```

---

## MPI Backend Implementation

### MpiCoordinator

```rust
pub struct MpiCoordinator {
    communicator: Arc<dyn Communicator>,
    rank: i32,
    world_size: i32,
}

#[async_trait]
impl CheckpointCoordinator for MpiCoordinator {
    fn worker_id(&self) -> WorkerId {
        WorkerId::Rank(self.rank)
    }

    fn world_size(&self) -> usize {
        self.world_size as usize
    }

    fn should_checkpoint(&self, context: &CheckpointContext) -> bool {
        // MPI: delegate to trigger (operation-based)
        context.trigger.should_checkpoint()
    }

    async fn begin_checkpoint(&self, checkpoint_id: u64) -> CylonResult<CheckpointDecision> {
        // Phase 1 of 2PC: Vote
        // Each rank votes 1 (ready) or 0 (not ready)
        let local_vote: i32 = 1;
        let mut global_vote: i32 = 0;

        // Allreduce with AND - all must agree
        self.communicator.allreduce_i32(
            &local_vote,
            &mut global_vote,
            MpiOp::Land,
        )?;

        if global_vote == 1 {
            Ok(CheckpointDecision::Proceed)
        } else {
            Ok(CheckpointDecision::Skip)
        }
    }

    async fn commit_checkpoint(&self, _checkpoint_id: u64) -> CylonResult<()> {
        // Phase 2 of 2PC: Commit
        // Barrier ensures all ranks have written before any considers it committed
        self.communicator.barrier()?;
        Ok(())
    }

    async fn abort_checkpoint(&self, _checkpoint_id: u64) -> CylonResult<()> {
        // Barrier to synchronize abort
        self.communicator.barrier()?;
        Ok(())
    }

    async fn find_latest_checkpoint(&self) -> CylonResult<Option<u64>> {
        // Each rank reports its latest checkpoint
        // Use MIN to find common checkpoint all ranks have
        let local_latest = self.get_local_latest()?;
        let mut global_latest: u64 = 0;

        self.communicator.allreduce_u64(
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

    async fn heartbeat(&self) -> CylonResult<()> {
        // No-op for MPI - failure detection is handled by MPI runtime
        Ok(())
    }
}
```

### MpiTrigger (Operation-Based)

```rust
pub struct MpiTrigger {
    /// Checkpoint after N operations
    operation_threshold: u64,
    current_operations: AtomicU64,

    /// Checkpoint after N bytes
    bytes_threshold: u64,
    current_bytes: AtomicU64,

    /// Force checkpoint flag
    force: AtomicBool,
}

impl CheckpointTrigger for MpiTrigger {
    fn record_operation(&self, op_type: OperationType, bytes: u64) {
        self.current_operations.fetch_add(1, Ordering::Relaxed);
        self.current_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    fn should_checkpoint(&self) -> bool {
        if self.force.load(Ordering::Relaxed) {
            return true;
        }

        let ops = self.current_operations.load(Ordering::Relaxed);
        let bytes = self.current_bytes.load(Ordering::Relaxed);

        ops >= self.operation_threshold || bytes >= self.bytes_threshold
    }

    fn force_checkpoint(&self) {
        self.force.store(true, Ordering::Relaxed);
    }

    fn reset(&self) {
        self.current_operations.store(0, Ordering::Relaxed);
        self.current_bytes.store(0, Ordering::Relaxed);
        self.force.store(false, Ordering::Relaxed);
    }

    fn urgency(&self) -> CheckpointUrgency {
        if self.force.load(Ordering::Relaxed) {
            CheckpointUrgency::Critical
        } else if self.should_checkpoint() {
            CheckpointUrgency::Medium
        } else {
            CheckpointUrgency::None
        }
    }
}
```

### ParallelFsStorage (for MPI)

```rust
pub struct ParallelFsStorage {
    base_path: PathBuf,
    staging_path: PathBuf,
}

#[async_trait]
impl CheckpointStorage for ParallelFsStorage {
    async fn write(
        &self,
        checkpoint_id: u64,
        worker_id: &WorkerId,
        key: &str,
        data: &[u8],
    ) -> CylonResult<String> {
        let rank = match worker_id {
            WorkerId::Rank(r) => *r,
            _ => return Err(CylonError::new(Code::InvalidArgument, "Expected MPI rank")),
        };

        // Write to staging directory first
        let staging_path = self.staging_path
            .join(format!("checkpoint_{}", checkpoint_id))
            .join(format!("rank_{}", rank))
            .join(key);

        if let Some(parent) = staging_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        tokio::fs::write(&staging_path, data).await?;

        Ok(staging_path.to_string_lossy().to_string())
    }

    async fn commit_write(
        &self,
        checkpoint_id: u64,
        worker_id: &WorkerId,
    ) -> CylonResult<()> {
        let rank = match worker_id {
            WorkerId::Rank(r) => *r,
            _ => return Err(CylonError::new(Code::InvalidArgument, "Expected MPI rank")),
        };

        // Move from staging to final location
        let staging = self.staging_path
            .join(format!("checkpoint_{}", checkpoint_id))
            .join(format!("rank_{}", rank));

        let final_path = self.base_path
            .join(format!("checkpoint_{}", checkpoint_id))
            .join(format!("rank_{}", rank));

        if let Some(parent) = final_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        tokio::fs::rename(&staging, &final_path).await?;

        Ok(())
    }

    async fn read(
        &self,
        checkpoint_id: u64,
        worker_id: &WorkerId,
        key: &str,
    ) -> CylonResult<Vec<u8>> {
        let rank = match worker_id {
            WorkerId::Rank(r) => *r,
            _ => return Err(CylonError::new(Code::InvalidArgument, "Expected MPI rank")),
        };

        let path = self.base_path
            .join(format!("checkpoint_{}", checkpoint_id))
            .join(format!("rank_{}", rank))
            .join(key);

        Ok(tokio::fs::read(&path).await?)
    }

    // ... other methods
}
```

---

## Serverless Backend Implementation

### RedisCoordinator

```rust
pub struct RedisCoordinator {
    redis: redis::Client,
    job_id: String,
    worker_id: String,
    world_size: usize,
    time_budget: TimeBudget,
}

pub struct TimeBudget {
    deadline: Instant,
    checkpoint_reserve: Duration,
    safety_buffer: Duration,
}

impl TimeBudget {
    pub fn new(deadline: Instant, checkpoint_reserve_secs: u64, safety_buffer_secs: u64) -> Self {
        Self {
            deadline,
            checkpoint_reserve: Duration::from_secs(checkpoint_reserve_secs),
            safety_buffer: Duration::from_secs(safety_buffer_secs),
        }
    }

    pub fn remaining_work_time(&self) -> Duration {
        let reserve = self.checkpoint_reserve + self.safety_buffer;
        self.deadline
            .checked_duration_since(Instant::now())
            .unwrap_or(Duration::ZERO)
            .saturating_sub(reserve)
    }

    pub fn is_critical(&self) -> bool {
        self.remaining_work_time() == Duration::ZERO
    }
}

impl RedisCoordinator {
    pub fn from_env() -> CylonResult<Self> {
        let redis_url = std::env::var("CHECKPOINT_REDIS_URL")
            .unwrap_or_else(|_| "redis://localhost:6379".to_string());
        let job_id = std::env::var("JOB_ID")?;
        let worker_id = std::env::var("WORKER_ID")
            .unwrap_or_else(|_| uuid::Uuid::new_v4().to_string());
        let world_size: usize = std::env::var("WORLD_SIZE")?.parse()?;

        Ok(Self {
            redis: redis::Client::open(redis_url)?,
            job_id,
            worker_id,
            world_size,
            time_budget: TimeBudget::new(
                Instant::now() + Duration::from_secs(300), // Default 5 min
                60, 30
            ),
        })
    }
}

#[async_trait]
impl CheckpointCoordinator for RedisCoordinator {
    fn worker_id(&self) -> WorkerId {
        WorkerId::Serverless {
            worker_id: self.worker_id.clone()
        }
    }

    fn world_size(&self) -> usize {
        self.world_size
    }

    fn should_checkpoint(&self, _context: &CheckpointContext) -> bool {
        self.time_budget.remaining_work_time() < Duration::from_secs(120)
    }

    async fn begin_checkpoint(&self, checkpoint_id: u64) -> CylonResult<CheckpointDecision> {
        if self.time_budget.is_critical() {
            return Ok(CheckpointDecision::urgent());
        }

        let mut conn = self.redis.get_async_connection().await?;

        // Register checkpoint intent using Redis hash
        let key = format!("job:{}:ckpt:{}:status", self.job_id, checkpoint_id);
        let _: () = conn.hset(&key, &self.worker_id, "IN_PROGRESS").await?;

        // Set TTL for automatic cleanup of stale checkpoints
        let _: () = conn.expire(&key, 3600).await?; // 1 hour TTL

        Ok(CheckpointDecision::proceed())
    }

    async fn commit_checkpoint(&self, checkpoint_id: u64) -> CylonResult<()> {
        let mut conn = self.redis.get_async_connection().await?;

        // Update status to COMMITTED
        let key = format!("job:{}:ckpt:{}:status", self.job_id, checkpoint_id);
        let _: () = conn.hset(&key, &self.worker_id, "COMMITTED").await?;

        // Check if all workers committed
        let statuses: HashMap<String, String> = conn.hgetall(&key).await?;
        let committed_count = statuses.values().filter(|s| *s == "COMMITTED").count();

        if committed_count == self.world_size {
            // Mark checkpoint as globally complete
            let complete_key = format!("job:{}:latest_checkpoint", self.job_id);
            let _: () = conn.set(&complete_key, checkpoint_id).await?;
        }

        Ok(())
    }

    async fn find_latest_checkpoint(&self) -> CylonResult<Option<u64>> {
        let mut conn = self.redis.get_async_connection().await?;

        let key = format!("job:{}:latest_checkpoint", self.job_id);
        let result: Option<u64> = conn.get(&key).await?;

        Ok(result)
    }

    async fn heartbeat(&self) -> CylonResult<()> {
        let mut conn = self.redis.get_async_connection().await?;

        // Set heartbeat with TTL
        let key = format!("job:{}:worker:{}:heartbeat", self.job_id, self.worker_id);
        let _: () = conn.set_ex(&key, now_timestamp(), 60).await?; // 60 second TTL

        Ok(())
    }

    async fn claim_work(&self, work_unit_id: &str) -> CylonResult<bool> {
        let mut conn = self.redis.get_async_connection().await?;

        // Atomic claim using SETNX
        let key = format!("job:{}:work:{}:owner", self.job_id, work_unit_id);
        let result: bool = conn.set_nx(&key, &self.worker_id).await?;

        if result {
            // Set TTL for automatic release
            let _: () = conn.expire(&key, 300).await?; // 5 min lease
        }

        Ok(result)
    }
}
```

### TimeBudgetTrigger

```rust
pub struct TimeBudgetTrigger {
    time_budget: TimeBudget,
    force: AtomicBool,
}

impl CheckpointTrigger for TimeBudgetTrigger {
    fn record_operation(&self, _op_type: OperationType, _bytes: u64) {
        // Time-based trigger, not operation-based
    }

    fn should_checkpoint(&self) -> bool {
        self.force.load(Ordering::Relaxed) || self.time_budget.is_critical()
    }

    fn force_checkpoint(&self) {
        self.force.store(true, Ordering::Relaxed);
    }

    fn reset(&self) {
        self.force.store(false, Ordering::Relaxed);
    }

    fn urgency(&self) -> CheckpointPriority {
        let remaining = self.time_budget.remaining_work_time();

        if self.force.load(Ordering::Relaxed) || remaining == Duration::ZERO {
            CheckpointPriority::Critical
        } else if remaining < Duration::from_secs(120) {
            CheckpointPriority::High
        } else if remaining < Duration::from_secs(300) {
            CheckpointPriority::Medium
        } else {
            CheckpointPriority::None
        }
    }
}
```

### S3Storage

```rust
pub struct S3Storage {
    s3_client: aws_sdk_s3::Client,
    bucket: String,
    prefix: String,
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
        let worker_str = worker_id.to_string();

        // Write to staging prefix
        let s3_key = format!(
            "{}/staging/checkpoint_{}/{}/{}",
            self.prefix, checkpoint_id, worker_str, key
        );

        self.s3_client.put_object()
            .bucket(&self.bucket)
            .key(&s3_key)
            .body(data.to_vec().into())
            .send()
            .await?;

        Ok(format!("s3://{}/{}", self.bucket, s3_key))
    }

    async fn commit_write(
        &self,
        checkpoint_id: u64,
        worker_id: &WorkerId,
    ) -> CylonResult<()> {
        let worker_str = worker_id.to_string();

        // List all objects in staging
        let staging_prefix = format!(
            "{}/staging/checkpoint_{}/{}/",
            self.prefix, checkpoint_id, worker_str
        );

        let objects = self.list_objects(&staging_prefix).await?;

        // Copy each to final location
        for obj in &objects {
            let final_key = obj.replace("/staging/", "/committed/");

            self.s3_client.copy_object()
                .bucket(&self.bucket)
                .key(&final_key)
                .copy_source(format!("{}/{}", self.bucket, obj))
                .send()
                .await?;
        }

        // Delete staging objects
        for obj in objects {
            self.s3_client.delete_object()
                .bucket(&self.bucket)
                .key(&obj)
                .send()
                .await?;
        }

        Ok(())
    }

    async fn read(
        &self,
        checkpoint_id: u64,
        worker_id: &WorkerId,
        key: &str,
    ) -> CylonResult<Vec<u8>> {
        let worker_str = worker_id.to_string();

        let s3_key = format!(
            "{}/committed/checkpoint_{}/{}/{}",
            self.prefix, checkpoint_id, worker_str, key
        );

        let result = self.s3_client.get_object()
            .bucket(&self.bucket)
            .key(&s3_key)
            .send()
            .await?;

        let data = result.body.collect().await?.to_vec();
        Ok(data)
    }

    // ... other methods
}
```

---

## Shared Serializer Implementation

```rust
pub struct ArrowSerializer {
    compression: Compression,
}

impl CheckpointSerializer for ArrowSerializer {
    fn serialize_table(&self, table: &Table) -> CylonResult<Vec<u8>> {
        let mut buffer = Vec::new();

        // Use Arrow IPC format
        let schema = table.schema();
        let mut writer = arrow_ipc::writer::FileWriter::try_new(
            &mut buffer,
            &schema,
        )?;

        for batch in table.batches() {
            writer.write(batch)?;
        }

        writer.finish()?;

        // Apply compression if configured
        match self.compression {
            Compression::None => Ok(buffer),
            Compression::Lz4 => Ok(lz4::compress(&buffer)?),
            Compression::Zstd { level } => Ok(zstd::encode_all(&buffer[..], level)?),
        }
    }

    fn deserialize_table(
        &self,
        data: &[u8],
        ctx: Arc<CylonContext>,
    ) -> CylonResult<Table> {
        // Decompress if needed
        let data = match self.compression {
            Compression::None => data.to_vec(),
            Compression::Lz4 => lz4::decompress(data)?,
            Compression::Zstd { .. } => zstd::decode_all(data)?,
        };

        // Read Arrow IPC
        let reader = arrow_ipc::reader::FileReader::try_new(
            std::io::Cursor::new(&data),
            None,
        )?;

        let batches: Vec<RecordBatch> = reader.collect::<Result<_, _>>()?;

        Table::from_record_batches(ctx, batches)
    }

    fn serialize_state<T: Serialize>(&self, state: &T) -> CylonResult<Vec<u8>> {
        Ok(bincode::serialize(state)?)
    }

    fn deserialize_state<T: DeserializeOwned>(&self, data: &[u8]) -> CylonResult<T> {
        Ok(bincode::deserialize(data)?)
    }

    fn format_id(&self) -> &str {
        "arrow-ipc-v1"
    }
}
```

---

## Configuration

```rust
#[derive(Clone, Debug)]
pub struct CheckpointConfig {
    /// Job identifier
    pub job_id: String,

    /// Environment type (auto-detected if not specified)
    pub environment: Option<EnvironmentType>,

    /// Compression for checkpoint data
    pub compression: Compression,

    /// Retention policy
    pub retention: RetentionPolicy,

    /// MPI-specific settings
    pub mpi: MpiCheckpointConfig,

    /// Serverless-specific settings
    pub serverless: ServerlessCheckpointConfig,
}

#[derive(Clone, Debug)]
pub struct MpiCheckpointConfig {
    /// Checkpoint after N operations
    pub operation_threshold: u64,

    /// Checkpoint after N bytes processed
    pub bytes_threshold: u64,

    /// Storage path (filesystem)
    pub storage_path: String,
}

#[derive(Clone, Debug)]
pub struct ServerlessCheckpointConfig {
    /// S3 bucket for checkpoints
    pub s3_bucket: String,

    /// Redis URL for coordination
    pub redis_url: String,

    /// Reserved time for checkpoint before timeout (seconds)
    pub checkpoint_reserve_secs: u64,

    /// Safety buffer before deadline (seconds)
    pub safety_buffer_secs: u64,
}

#[derive(Clone, Debug)]
pub struct RetentionPolicy {
    /// Maximum number of checkpoints to keep
    pub max_checkpoints: usize,

    /// Maximum age of checkpoints
    pub max_age: Option<Duration>,

    /// Always keep the N most recent
    pub keep_latest_n: usize,
}

#[derive(Clone, Copy, Debug)]
pub enum Compression {
    None,
    Lz4,
    Zstd { level: i32 },
}
```

---

## Usage Examples

### MPI Environment

```rust
use cylon::prelude::*;
use cylon::checkpoint::{CheckpointManager, CheckpointConfig, MpiCheckpointConfig};

fn main() -> CylonResult<()> {
    // Initialize MPI context
    let ctx = CylonContext::new_mpi()?;

    // Configure checkpointing
    let config = CheckpointConfig {
        job_id: "job_12345".to_string(),
        environment: Some(EnvironmentType::Mpi),
        compression: Compression::Zstd { level: 3 },
        mpi: MpiCheckpointConfig {
            operation_threshold: 100,
            bytes_threshold: 1_000_000_000, // 1GB
            storage_path: "/scratch/checkpoints".to_string(),
        },
        ..Default::default()
    };

    // Create checkpoint manager
    let checkpoint_mgr = CheckpointManager::for_mpi(
        ctx.communicator().clone(),
        &config.mpi.storage_path,
        config,
    )?;

    // Load data
    let mut table = Table::from_csv(&ctx, "input.csv")?;

    // Process with checkpointing
    for i in 0..1000 {
        table = process_iteration(&table)?;

        // Record operation
        checkpoint_mgr.record_operation(OperationType::Transform, table.size_bytes());

        // Check if we should checkpoint
        if checkpoint_mgr.should_checkpoint() {
            checkpoint_mgr.checkpoint(&[("table", &table)]).await?;
        }
    }

    Ok(())
}
```

### Serverless Environment (Fargate/ECS)

```rust
use cylon::prelude::*;
use cylon::checkpoint::{CheckpointManager, CheckpointConfig, ServerlessCheckpointConfig};

#[tokio::main]
async fn main() -> CylonResult<()> {
    // Get configuration from environment
    let job_id = std::env::var("JOB_ID").expect("JOB_ID required");
    let s3_bucket = std::env::var("S3_BUCKET").expect("S3_BUCKET required");
    let redis_url = std::env::var("REDIS_URL").expect("REDIS_URL required");

    // Initialize Cylon context
    let ctx = CylonContext::new_local()?;

    // Configure checkpointing
    let config = CheckpointConfig {
        job_id: job_id.clone(),
        environment: Some(EnvironmentType::Serverless),
        compression: Compression::Zstd { level: 3 },
        serverless: ServerlessCheckpointConfig {
            s3_bucket: s3_bucket.clone(),
            redis_url: redis_url.clone(),
            checkpoint_reserve_secs: 60,
            safety_buffer_secs: 30,
        },
        ..Default::default()
    };

    // Create checkpoint manager (auto-detects environment)
    let checkpoint_mgr = CheckpointManager::auto(&job_id)?;

    // Try to restore from previous checkpoint
    let mut table = match checkpoint_mgr.restore(ctx.clone()).await {
        Ok(result) => result.tables.remove("table").unwrap(),
        Err(_) => Table::from_s3(&ctx, &std::env::var("INPUT_PATH")?).await?,
    };

    // Process until time runs out or done
    while !is_done(&table) {
        // Check urgency BEFORE each operation
        match checkpoint_mgr.urgency() {
            CheckpointPriority::Critical => {
                // Must checkpoint NOW
                checkpoint_mgr.checkpoint(&[("table", &table)]).await?;
                // Exit cleanly - orchestrator will restart
                std::process::exit(0);
            }
            CheckpointPriority::High => {
                // Checkpoint and continue if time permits
                checkpoint_mgr.checkpoint(&[("table", &table)]).await?;
            }
            _ => {}
        }

        // Process one batch
        table = process_batch(&table)?;

        // Heartbeat
        checkpoint_mgr.coordinator().heartbeat().await?;
    }

    // Done - write final output
    table.to_s3(&std::env::var("OUTPUT_PATH")?).await?;

    Ok(())
}
```

### Testing with Local Backend

```rust
#[cfg(test)]
mod tests {
    use cylon::checkpoint::{CheckpointManager, LocalCoordinator, LocalStorage};

    #[tokio::test]
    async fn test_checkpoint_restore() {
        let temp_dir = tempfile::tempdir().unwrap();

        let config = CheckpointConfig {
            job_id: "test_job".to_string(),
            environment: Some(EnvironmentType::Local),
            ..Default::default()
        };

        // Use local implementations for testing
        let coordinator = Arc::new(LocalCoordinator::new());
        let storage = Arc::new(LocalStorage::new(temp_dir.path()));
        let serializer = Arc::new(ArrowSerializer::default());
        let trigger = Arc::new(ManualTrigger::new());

        let mgr = CheckpointManager::new(
            coordinator,
            storage,
            serializer,
            trigger,
            config,
        );

        // Create test table
        let ctx = CylonContext::new_local().unwrap();
        let table = create_test_table(&ctx);

        // Checkpoint
        let result = mgr.checkpoint(&[("test_table", &table)]).await.unwrap();
        assert!(matches!(result, CheckpointResult::Committed { .. }));

        // Restore
        let restored = mgr.restore(ctx.clone()).await.unwrap();
        let restored_table = restored.tables.get("test_table").unwrap();

        assert_eq!(table.num_rows(), restored_table.num_rows());
    }
}
```

---

## Directory/Storage Layout

### MPI (Parallel Filesystem)

```
/scratch/checkpoints/{job_id}/
├── metadata/
│   ├── checkpoint_000001.json
│   ├── checkpoint_000002.json
│   └── latest.json -> checkpoint_000002.json
├── staging/
│   └── checkpoint_000003/     # In-progress
│       ├── rank_0/
│       │   └── table.arrow
│       └── rank_1/
│           └── table.arrow
└── committed/
    ├── checkpoint_000001/
    │   ├── rank_0/
    │   │   └── table.arrow
    │   └── rank_1/
    │       └── table.arrow
    └── checkpoint_000002/
        └── ...
```

### Serverless (S3 + Redis)

**S3:**
```
s3://checkpoint-bucket/{job_id}/
├── metadata/
│   ├── checkpoint_000001.json
│   └── checkpoint_000002.json
├── staging/
│   └── checkpoint_000003/
│       ├── worker_abc123/
│       │   └── table.parquet
│       └── worker_def456/
│           └── table.parquet
└── committed/
    ├── checkpoint_000001/
    │   └── ...
    └── checkpoint_000002/
        └── ...
```

**Redis Keys:**
```
# Checkpoint status (hash)
job:{job_id}:ckpt:{checkpoint_id}:status
  worker_abc -> "COMMITTED"
  worker_def -> "COMMITTED"

# Latest complete checkpoint
job:{job_id}:latest_checkpoint -> 2

# Worker heartbeats (with TTL)
job:{job_id}:worker:{worker_id}:heartbeat -> timestamp  (TTL: 60s)

# Work unit claims (with TTL)
job:{job_id}:work:{partition_id}:owner -> worker_id  (TTL: 300s)
```

---

## Feature Comparison

| Feature | MPI Backend | Serverless Backend |
|---------|-------------|-------------------|
| Coordination | MPI barriers + allreduce | Redis atomic operations |
| Trigger | Operation count / bytes | Time budget |
| Storage | Parallel FS (Lustre/GPFS) | S3 |
| Worker ID | Stable rank | Ephemeral worker ID |
| Heartbeat | Not needed (MPI handles) | Required (Redis TTL) |
| Work claiming | Static assignment | Dynamic (Redis SETNX) |
| Failure detection | MPI runtime | Heartbeat timeout |
| Commit protocol | 2PC with barriers | S3 copy + Redis update |

---

## Implementation Phases

### Phase 1: Core Traits & Local Backend
- Define all traits
- Implement local/mock backends for testing
- Basic CheckpointManager
- Unit tests

### Phase 2: Shared Components
- ArrowSerializer with compression
- CheckpointMetadata
- Retention policy implementation

### Phase 3: MPI Backend
- MpiCoordinator with 2PC
- ParallelFsStorage
- MpiTrigger (operation-based)
- Integration tests

### Phase 4: Serverless Backend
- RedisCoordinator with atomic operations
- S3Storage with staging/commit
- TimeBudgetTrigger (time-based)
- Redis-based heartbeat and work claiming

### Phase 5: Advanced Features
- Incremental checkpoints
- Streaming writes (multipart S3)
- Cross-environment restore (read MPI checkpoint in serverless)
- Monitoring & metrics

---

## Dependencies

```toml
[dependencies]
# Core
async-trait = "0.1"
tokio = { version = "1", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }

# Serialization
arrow = "57"
arrow-ipc = "57"
bincode = "1.3"
lz4 = "1.24"
zstd = "0.13"

# MPI (optional)
mpi = { version = "0.8", optional = true }

# AWS/Serverless (optional)
aws-config = { version = "1.0", optional = true }
aws-sdk-s3 = { version = "1.0", optional = true }
redis = { version = "0.24", features = ["tokio-comp"], optional = true }

[features]
default = []
mpi = ["dep:mpi"]
serverless = ["dep:aws-config", "dep:aws-sdk-s3", "dep:redis"]
full = ["mpi", "serverless"]
```

---

## Deployment Environments

### UVA Rivanna (On-Premise HPC)

Rivanna uses **Lustre** parallel filesystem mounted at `/scratch` with **InfiniBand** networking.

**Configuration:**
```rust
let config = CheckpointConfig {
    job_id: format!("job_{}", std::env::var("SLURM_JOB_ID").unwrap()),
    environment: Some(EnvironmentType::Mpi),
    compression: Compression::Zstd { level: 3 },
    mpi: MpiCheckpointConfig {
        operation_threshold: 100,
        bytes_threshold: 10_000_000_000, // 10GB - Lustre can handle large writes
        storage_path: format!("/scratch/{}/checkpoints",
            std::env::var("USER").unwrap()),
    },
    ..Default::default()
};
```

**Key characteristics:**
- Lustre provides high-bandwidth parallel I/O (multiple OSTs)
- InfiniBand provides low-latency MPI communication
- All ranks can read/write concurrently to different files
- `/scratch` has storage quotas and purge policies (typically 90 days)

### AWS HPC (EC2 + FSx for Lustre + EFA)

For non-serverless HPC workloads on AWS, use:
- **FSx for Lustre**: Managed Lustre filesystem (up to 1200 Gbps throughput)
- **EFA (Elastic Fabric Adapter)**: OS-bypass networking for MPI (via libfabric)

**Architecture:**

![AWS ParallelCluster Architecture](aws_parallelcluster.png)

**libfabric Integration:**
- OpenMPI and MPICH use libfabric as the transport abstraction layer
- EFA provides a libfabric provider (`efa`) for OS-bypass networking
- No code changes needed - libfabric is used automatically when MPI is configured

**Configuration:**
```rust
let config = CheckpointConfig {
    job_id: format!("job_{}", std::env::var("AWS_BATCH_JOB_ID")
        .or_else(|_| std::env::var("SLURM_JOB_ID"))
        .unwrap_or_else(|_| uuid::Uuid::new_v4().to_string())),
    environment: Some(EnvironmentType::Mpi),
    compression: Compression::Zstd { level: 3 },
    mpi: MpiCheckpointConfig {
        operation_threshold: 100,
        bytes_threshold: 50_000_000_000, // 50GB - FSx can handle very large writes
        storage_path: "/fsx/checkpoints".to_string(),  // FSx mount point
    },
    ..Default::default()
};
```

**FSx for Lustre Tiers:**
| Tier | Throughput | Use Case |
|------|------------|----------|
| Scratch | 200 MB/s per TiB | Short-term, cost-effective |
| Persistent | Up to 1000 MB/s per TiB | Longer-running jobs |

**AWS ParallelCluster Setup:**
```yaml
# parallelcluster config snippet
SharedStorage:
  - Name: FsxLustre
    MountDir: /fsx
    StorageType: FsxLustre
    FsxLustreSettings:
      StorageCapacity: 1200  # GiB
      DeploymentType: SCRATCH_2
      ImportPath: s3://my-bucket/input/
      ExportPath: s3://my-bucket/output/

Scheduling:
  Scheduler: slurm
  SlurmQueues:
    - Name: hpc
      ComputeResources:
        - Name: c6i
          InstanceType: c6i.8xlarge
          Efa:
            Enabled: true
```

### AWS Lambda / Fargate (Serverless with Hole Punching)

For parallel serverless workloads with direct P2P communication:
- **S3**: Checkpoint data storage
- **Redis (ElastiCache)**: Coordination, heartbeats, job state, checkpoint metadata
- **TCP Hole Punching**: Direct worker-to-worker communication
- **Step Functions / ECS**: Worker lifecycle management

---

## Serverless Orchestration Architecture

When running parallel Lambdas or Fargate tasks, you need an **orchestrator** to:
1. Launch initial workers
2. Monitor worker health
3. Replace workers that timeout or fail
4. Track job completion

![Serverless Architecture](serverless_architecture.png)

### Orchestrator Implementation (Step Functions)

```json
{
  "Comment": "Cylon Parallel Job Orchestrator",
  "StartAt": "InitializeJob",
  "States": {
    "InitializeJob": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:...:InitializeJobFunction",
      "Parameters": {
        "job_id.$": "$.job_id",
        "world_size.$": "$.world_size",
        "input_path.$": "$.input_path"
      },
      "Next": "LaunchWorkers"
    },

    "LaunchWorkers": {
      "Type": "Map",
      "ItemsPath": "$.worker_configs",
      "MaxConcurrency": 0,
      "Iterator": {
        "StartAt": "InvokeWorker",
        "States": {
          "InvokeWorker": {
            "Type": "Task",
            "Resource": "arn:aws:lambda:...:CylonWorkerFunction",
            "End": true
          }
        }
      },
      "Next": "MonitorAndReplace"
    },

    "MonitorAndReplace": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:...:MonitorWorkersFunction",
      "Parameters": {
        "job_id.$": "$.job_id",
        "world_size.$": "$.world_size"
      },
      "Next": "CheckCompletion"
    },

    "CheckCompletion": {
      "Type": "Choice",
      "Choices": [
        {
          "Variable": "$.status",
          "StringEquals": "COMPLETED",
          "Next": "Success"
        },
        {
          "Variable": "$.status",
          "StringEquals": "NEEDS_REPLACEMENT",
          "Next": "ReplaceWorkers"
        }
      ],
      "Default": "MonitorAndReplace"
    },

    "ReplaceWorkers": {
      "Type": "Map",
      "ItemsPath": "$.failed_workers",
      "Iterator": {
        "StartAt": "InvokeReplacement",
        "States": {
          "InvokeReplacement": {
            "Type": "Task",
            "Resource": "arn:aws:lambda:...:CylonWorkerFunction",
            "End": true
          }
        }
      },
      "Next": "MonitorAndReplace"
    },

    "Success": {
      "Type": "Succeed"
    }
  }
}
```

### Orchestrator Lambda (Alternative to Step Functions)

```rust
use aws_sdk_lambda::Client as LambdaClient;
use redis::AsyncCommands;

pub struct JobOrchestrator {
    ecs_client: aws_sdk_ecs::Client,
    redis: redis::Client,
    job_id: String,
    world_size: usize,
    task_definition_arn: String,
}

impl JobOrchestrator {
    /// Main orchestration loop
    pub async fn run(&self) -> CylonResult<JobResult> {
        // 1. Initialize job state in Redis
        self.initialize_job().await?;

        // 2. Launch initial workers
        self.launch_workers(0..self.world_size).await?;

        // 3. Monitor and replace loop
        loop {
            tokio::time::sleep(Duration::from_secs(10)).await;

            let status = self.check_job_status().await?;

            match status {
                JobStatus::Completed => {
                    return Ok(JobResult::Success);
                }
                JobStatus::Failed(reason) => {
                    return Ok(JobResult::Failed(reason));
                }
                JobStatus::NeedsReplacement(failed_ranks) => {
                    // Some workers timed out or failed
                    self.replace_workers(&failed_ranks).await?;
                }
                JobStatus::InProgress => {
                    // All workers healthy, continue monitoring
                }
            }
        }
    }

    /// Launch workers for given ranks
    async fn launch_workers(&self, ranks: impl Iterator<Item = usize>) -> CylonResult<()> {
        let futures: Vec<_> = ranks.map(|rank| {
            let payload = serde_json::json!({
                "job_id": self.job_id,
                "rank": rank,
                "world_size": self.world_size,
            });

            self.lambda_client.invoke()
                .function_name(&self.worker_function_arn)
                .invocation_type(aws_sdk_lambda::types::InvocationType::Event) // Async
                .payload(aws_sdk_lambda::primitives::Blob::new(
                    serde_json::to_vec(&payload).unwrap()
                ))
                .send()
        }).collect();

        // Launch all in parallel
        futures::future::try_join_all(futures).await?;

        Ok(())
    }

    /// Check which workers are alive via Redis heartbeats
    async fn check_job_status(&self) -> CylonResult<JobStatus> {
        let mut conn = self.redis.get_async_connection().await?;

        // Check if job is marked complete
        let complete: bool = conn.exists(
            format!("job:{}:complete", self.job_id)
        ).await?;

        if complete {
            return Ok(JobStatus::Completed);
        }

        // Check heartbeats for each rank
        let mut failed_ranks = Vec::new();
        let now = current_timestamp();
        let timeout_threshold = now - 60; // 60 second timeout

        for rank in 0..self.world_size {
            let heartbeat: Option<i64> = conn.get(
                format!("job:{}:heartbeat:{}", self.job_id, rank)
            ).await?;

            match heartbeat {
                None => failed_ranks.push(rank),
                Some(ts) if ts < timeout_threshold => failed_ranks.push(rank),
                Some(_) => {} // Healthy
            }
        }

        if failed_ranks.is_empty() {
            Ok(JobStatus::InProgress)
        } else {
            Ok(JobStatus::NeedsReplacement(failed_ranks))
        }
    }

    /// Replace failed workers
    async fn replace_workers(&self, failed_ranks: &[usize]) -> CylonResult<()> {
        // Mark old workers as replaced (they'll exit gracefully if still running)
        let mut conn = self.redis.get_async_connection().await?;

        for rank in failed_ranks {
            // Increment generation - old workers will see mismatch and exit
            conn.incr(
                format!("job:{}:generation:{}", self.job_id, rank),
                1i64
            ).await?;
        }

        // Launch replacements
        self.launch_workers(failed_ranks.iter().copied()).await?;

        Ok(())
    }
}

#[derive(Debug)]
enum JobStatus {
    InProgress,
    Completed,
    Failed(String),
    NeedsReplacement(Vec<usize>),
}
```

### Worker with Redis Coordination

```rust
/// Redis-based coordinator for hole-punched workers
pub struct RedisCoordinator {
    redis: redis::Client,
    job_id: String,
    rank: i32,
    generation: i64,  // Detects if we've been replaced
    world_size: usize,
    time_budget: TimeBudget,
    peers: RwLock<HashMap<i32, PeerConnection>>,
}

impl RedisCoordinator {
    /// Bootstrap: register and establish P2P connections
    pub async fn bootstrap(&self) -> CylonResult<()> {
        let mut conn = self.redis.get_async_connection().await?;

        // 1. Get our public endpoint for hole punching
        let my_endpoint = discover_public_endpoint().await?;

        // 2. Register with Redis
        conn.hset(
            format!("job:{}:endpoints", self.job_id),
            self.rank.to_string(),
            serde_json::to_string(&my_endpoint)?
        ).await?;

        // 3. Update heartbeat
        conn.set_ex(
            format!("job:{}:heartbeat:{}", self.job_id, self.rank),
            current_timestamp(),
            60  // 60 second TTL
        ).await?;

        // 4. Subscribe to peer announcements
        let mut pubsub = self.redis.get_async_connection().await?.into_pubsub();
        pubsub.subscribe(format!("job:{}:peers", self.job_id)).await?;

        // 5. Announce ourselves
        conn.publish(
            format!("job:{}:peers", self.job_id),
            serde_json::to_string(&PeerAnnouncement {
                rank: self.rank,
                endpoint: my_endpoint.clone(),
                generation: self.generation,
            })?
        ).await?;

        // 6. Wait for all peers and establish connections
        let deadline = Instant::now() + Duration::from_secs(60);

        while self.peers.read().await.len() < self.world_size - 1 {
            if Instant::now() > deadline {
                return Err(CylonError::new(Code::DeadlineExceeded, "Peer discovery timeout"));
            }

            // Also poll for announcements from peers
            if let Ok(Some(msg)) = tokio::time::timeout(
                Duration::from_secs(1),
                pubsub.on_message().next()
            ).await {
                if let Ok(announcement) = serde_json::from_str::<PeerAnnouncement>(
                    &msg.get_payload::<String>()?
                ) {
                    if announcement.rank != self.rank {
                        // Establish hole-punched connection
                        let conn = establish_connection(&announcement.endpoint).await?;
                        self.peers.write().await.insert(announcement.rank, conn);
                    }
                }
            }
        }

        Ok(())
    }

    /// Check if we've been replaced (orchestrator launched a new worker for our rank)
    pub async fn check_replaced(&self) -> CylonResult<bool> {
        let mut conn = self.redis.get_async_connection().await?;
        let current_gen: i64 = conn.get(
            format!("job:{}:generation:{}", self.job_id, self.rank)
        ).await.unwrap_or(0);

        Ok(current_gen > self.generation)
    }

    /// Heartbeat - must call regularly
    pub async fn heartbeat(&self) -> CylonResult<()> {
        // Check if we've been replaced
        if self.check_replaced().await? {
            return Err(CylonError::new(Code::Aborted, "Worker replaced by orchestrator"));
        }

        let mut conn = self.redis.get_async_connection().await?;
        conn.set_ex(
            format!("job:{}:heartbeat:{}", self.job_id, self.rank),
            current_timestamp(),
            60
        ).await?;

        Ok(())
    }
}

#[async_trait]
impl CheckpointCoordinator for RedisCoordinator {
    fn worker_id(&self) -> WorkerId {
        WorkerId::Rank(self.rank)  // Use rank even for Lambda
    }

    fn world_size(&self) -> usize {
        self.world_size
    }

    async fn begin_checkpoint(&self, checkpoint_id: u64) -> CylonResult<CheckpointDecision> {
        // Collective vote over P2P connections
        let my_remaining = self.time_budget.remaining_work_time();
        let my_vote = if my_remaining < Duration::from_secs(120) {
            CheckpointVote::Urgent
        } else if my_remaining < Duration::from_secs(300) {
            CheckpointVote::Yes
        } else {
            CheckpointVote::Continue
        };

        // Exchange votes with all peers
        let global_vote = self.allreduce_vote(my_vote).await?;

        match global_vote {
            CheckpointVote::Urgent => Ok(CheckpointDecision::Urgent),
            CheckpointVote::Yes => Ok(CheckpointDecision::Proceed),
            CheckpointVote::Continue => Ok(CheckpointDecision::Skip),
        }
    }

    async fn commit_checkpoint(&self, checkpoint_id: u64) -> CylonResult<()> {
        // Barrier over P2P connections
        self.barrier().await?;

        // Update Redis with committed checkpoint
        let mut conn = self.redis.get_async_connection().await?;
        conn.hset(
            format!("job:{}:checkpoints", self.job_id),
            format!("{}:{}", checkpoint_id, self.rank),
            "COMMITTED"
        ).await?;

        Ok(())
    }

    async fn find_latest_checkpoint(&self) -> CylonResult<Option<u64>> {
        // Query Redis for latest checkpoint where all ranks committed
        let mut conn = self.redis.get_async_connection().await?;
        let checkpoints: HashMap<String, String> = conn.hgetall(
            format!("job:{}:checkpoints", self.job_id)
        ).await?;

        // Find highest checkpoint_id where all ranks have committed
        let mut checkpoint_counts: HashMap<u64, usize> = HashMap::new();
        for key in checkpoints.keys() {
            if let Some(id_str) = key.split(':').next() {
                if let Ok(id) = id_str.parse::<u64>() {
                    *checkpoint_counts.entry(id).or_insert(0) += 1;
                }
            }
        }

        // Find the highest checkpoint where count == world_size
        Ok(checkpoint_counts.into_iter()
            .filter(|(_, count)| *count >= self.world_size)
            .map(|(id, _)| id)
            .max())
    }

    async fn heartbeat(&self) -> CylonResult<()> {
        RedisCoordinator::heartbeat(self).await
    }
}
```

### Worker Main Loop

```rust
/// Lambda or Fargate worker entry point
pub async fn worker_main(config: WorkerConfig) -> CylonResult<WorkerResult> {
    // 1. Initialize coordinator
    let coordinator = RedisCoordinator::new(
        &config.redis_url,
        &config.job_id,
        config.rank,
        config.world_size,
        TimeBudget::from_remaining(config.time_limit),
    );

    // 2. Bootstrap - register and establish P2P connections
    coordinator.bootstrap().await?;

    // 3. Create checkpoint manager
    let storage = Arc::new(S3Storage::new(&config.s3_bucket, &config.job_id));
    let serializer = Arc::new(ArrowSerializer::default());
    let trigger = Arc::new(CollectiveLambdaTrigger::new(
        coordinator.time_budget.clone(),
        Arc::new(coordinator.clone()),
    ));

    let checkpoint_mgr = CheckpointManager::new(
        Arc::new(coordinator.clone()),
        storage,
        serializer,
        trigger,
        config.checkpoint_config,
    );

    // 4. Restore from checkpoint or load initial data
    let ctx = CylonContext::new_local()?;
    let mut table = match checkpoint_mgr.restore(ctx.clone()).await {
        Ok(result) => {
            log::info!("Restored from checkpoint {}", result.checkpoint_id);
            result.tables.remove("data").unwrap()
        }
        Err(_) => {
            log::info!("No checkpoint found, loading initial data");
            Table::from_s3(&ctx, &config.input_path).await?
        }
    };

    // 5. Processing loop
    loop {
        // Check if we've been replaced
        if coordinator.check_replaced().await? {
            log::info!("Replaced by orchestrator, exiting gracefully");
            return Ok(WorkerResult::Replaced);
        }

        // Check checkpoint urgency
        match checkpoint_mgr.urgency() {
            CheckpointUrgency::Critical => {
                log::info!("Critical: checkpointing before timeout");
                checkpoint_mgr.checkpoint(&[("data", &table)]).await?;
                return Ok(WorkerResult::Checkpointed);
            }
            CheckpointUrgency::High => {
                log::info!("High urgency: checkpointing");
                checkpoint_mgr.checkpoint(&[("data", &table)]).await?;
            }
            _ => {}
        }

        // Check if processing is complete
        if is_processing_complete(&table) {
            break;
        }

        // Process one batch
        table = process_batch(&table)?;

        // Heartbeat
        coordinator.heartbeat().await?;
    }

    // 6. Mark job complete (rank 0 only)
    if coordinator.worker_id() == WorkerId::Rank(0) {
        let mut conn = coordinator.redis.get_async_connection().await?;
        conn.set(format!("job:{}:complete", config.job_id), "true").await?;
    }

    // Final barrier to ensure all workers finished
    coordinator.barrier().await?;

    Ok(WorkerResult::Completed)
}

#[derive(Debug)]
pub enum WorkerResult {
    Completed,
    Checkpointed,  // Exited normally after checkpoint, needs continuation
    Replaced,      // Orchestrator launched replacement
}
```

### Fargate vs Lambda Choice

![Lambda vs Fargate](lambda_vs_fargate.png)

### Cost Comparison

| Service | Use Case | Cost Model |
|---------|----------|------------|
| Lambda | Short bursts | ~$0.0000167/GB-sec |
| Fargate | Long running | ~$0.04/vCPU-hour + $0.004/GB-hour |
| Fargate Spot | Fault tolerant | 50-70% discount |
| ECS on EC2 | Maximum control | EC2 pricing |

For a 4-worker job running 1 hour:
- Lambda (10GB, max duration): ~$36/hour
- Fargate (4 vCPU, 8GB): ~$0.19/hour
- Fargate Spot: ~$0.08/hour

**Recommendation**: Use Lambda for jobs completing in a few checkpoint cycles. Use Fargate (especially Spot) for longer jobs.

---

## Container Restart Strategies

When workers checkpoint and exit, something must restart them to continue processing. The strategy differs by compute platform.

### Lambda: Step Functions Restart Loop

Lambda workers are restarted by Step Functions automatically:

![Step Functions State Machine](step_functions.png)

See the Step Functions definition in the Orchestrator section above.

### Fargate: Step Functions with ECS Tasks (Recommended)

For Fargate, use Step Functions with the `ecs:runTask.sync` integration. This gives you explicit restart control:

```json
{
  "Comment": "Cylon Fargate Job with Checkpoint Restart",
  "StartAt": "LaunchWorkers",
  "States": {
    "LaunchWorkers": {
      "Type": "Map",
      "ItemsPath": "$.worker_ranks",
      "MaxConcurrency": 0,
      "Iterator": {
        "StartAt": "RunWorkerTask",
        "States": {
          "RunWorkerTask": {
            "Type": "Task",
            "Resource": "arn:aws:states:::ecs:runTask.sync",
            "Parameters": {
              "LaunchType": "FARGATE",
              "Cluster": "${EcsClusterArn}",
              "TaskDefinition": "${TaskDefinitionArn}",
              "NetworkConfiguration": {
                "AwsvpcConfiguration": {
                  "Subnets.$": "$.subnets",
                  "SecurityGroups.$": "$.security_groups",
                  "AssignPublicIp": "ENABLED"
                }
              },
              "Overrides": {
                "ContainerOverrides": [{
                  "Name": "cylon-worker",
                  "Environment": [
                    { "Name": "RANK", "Value.$": "States.Format('{}', $.rank)" },
                    { "Name": "WORLD_SIZE", "Value.$": "States.Format('{}', $.world_size)" },
                    { "Name": "JOB_ID", "Value.$": "$$.Execution.Name" },
                    { "Name": "REDIS_URL", "Value.$": "$.redis_url" },
                    { "Name": "S3_BUCKET", "Value.$": "$.s3_bucket" }
                  ]
                }]
              }
            },
            "ResultPath": "$.task_result",
            "Next": "CheckTaskResult"
          },

          "CheckTaskResult": {
            "Type": "Choice",
            "Choices": [
              {
                "And": [
                  {
                    "Variable": "$.task_result.Containers[0].ExitCode",
                    "NumericEquals": 0
                  }
                ],
                "Next": "CheckJobComplete"
              }
            ],
            "Default": "TaskFailed"
          },

          "CheckJobComplete": {
            "Type": "Task",
            "Resource": "arn:aws:lambda:${Region}:${Account}:function:CheckJobComplete",
            "Parameters": {
              "job_id.$": "$$.Execution.Name",
              "rank.$": "$.rank"
            },
            "ResultPath": "$.completion_check",
            "Next": "DecideRestart"
          },

          "DecideRestart": {
            "Type": "Choice",
            "Choices": [
              {
                "Variable": "$.completion_check.job_complete",
                "BooleanEquals": true,
                "Next": "WorkerComplete"
              },
              {
                "Variable": "$.completion_check.worker_complete",
                "BooleanEquals": true,
                "Next": "WorkerComplete"
              }
            ],
            "Default": "RunWorkerTask"
          },

          "WorkerComplete": {
            "Type": "Succeed"
          },

          "TaskFailed": {
            "Type": "Task",
            "Resource": "arn:aws:lambda:${Region}:${Account}:function:HandleTaskFailure",
            "Next": "RunWorkerTask",
            "Retry": [
              {
                "ErrorEquals": ["States.ALL"],
                "MaxAttempts": 3,
                "BackoffRate": 2
              }
            ]
          }
        }
      },
      "Next": "JobSuccess"
    },

    "JobSuccess": {
      "Type": "Succeed"
    }
  }
}
```

**CheckJobComplete Lambda:**

```rust
use aws_lambda_events::event::stepfunctions::StepFunctionsPayload;
use lambda_runtime::{service_fn, Error, LambdaEvent};
use redis::AsyncCommands;

#[derive(Deserialize)]
struct Input {
    job_id: String,
    rank: i32,
}

#[derive(Serialize)]
struct Output {
    job_complete: bool,
    worker_complete: bool,
}

async fn handler(event: LambdaEvent<Input>) -> Result<Output, Error> {
    let input = event.payload;

    let redis = redis::Client::open(std::env::var("REDIS_URL")?)?;
    let mut conn = redis.get_async_connection().await?;

    // Check if entire job is complete
    let job_complete: bool = conn.exists(
        format!("job:{}:complete", input.job_id)
    ).await?;

    if job_complete {
        return Ok(Output { job_complete: true, worker_complete: true });
    }

    // Check if this specific worker's partition is done
    let worker_complete: bool = conn.sismember(
        format!("job:{}:workers_done", input.job_id),
        input.rank
    ).await?;

    Ok(Output { job_complete: false, worker_complete })
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    lambda_runtime::run(service_fn(handler)).await
}
```

**Fargate Task Definition (Terraform):**

```hcl
resource "aws_ecs_task_definition" "cylon_worker" {
  family                   = "cylon-worker"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = 4096   # 4 vCPU
  memory                   = 8192   # 8 GB
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([{
    name  = "cylon-worker"
    image = "${aws_ecr_repository.cylon.repository_url}:latest"

    essential = true

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.cylon.name
        "awslogs-region"        = var.region
        "awslogs-stream-prefix" = "worker"
      }
    }

    # Environment will be overridden by Step Functions
    environment = []
  }])
}

# IAM role for task (S3, Redis access)
resource "aws_iam_role" "ecs_task" {
  name = "cylon-worker-task-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy" "ecs_task_s3" {
  name = "s3-access"
  role = aws_iam_role.ecs_task.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"]
      Resource = [
        aws_s3_bucket.checkpoints.arn,
        "${aws_s3_bucket.checkpoints.arn}/*"
      ]
    }]
  })
}
```

### EC2: Systemd with Completion Check

For EC2 instances, use systemd with a pre-start check:

```ini
# /etc/systemd/system/cylon-worker.service
[Unit]
Description=Cylon Checkpoint Worker
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=cylon
Group=cylon

# Check if job is complete before starting
ExecStartPre=/usr/local/bin/cylon-check-complete

# Run the worker
ExecStart=/usr/local/bin/cylon-worker

# Restart on clean exit (checkpoint) but not on failure
Restart=on-success
RestartSec=5

# Environment
EnvironmentFile=/etc/cylon/worker.env

# Limits
LimitNOFILE=65535
LimitNPROC=65535

[Install]
WantedBy=multi-user.target
```

```bash
#!/bin/bash
# /usr/local/bin/cylon-check-complete

source /etc/cylon/worker.env

# Check Redis for job completion
COMPLETE=$(redis-cli -u "$REDIS_URL" GET "job:${JOB_ID}:complete" 2>/dev/null)

if [ "$COMPLETE" = "true" ]; then
    echo "Job ${JOB_ID} is complete, stopping worker permanently"
    systemctl disable cylon-worker.service
    exit 1  # Prevents ExecStart from running
fi

# Check if this rank is done
RANK_DONE=$(redis-cli -u "$REDIS_URL" SISMEMBER "job:${JOB_ID}:workers_done" "${RANK}" 2>/dev/null)

if [ "$RANK_DONE" = "1" ]; then
    echo "Worker rank ${RANK} completed, stopping"
    systemctl disable cylon-worker.service
    exit 1
fi

echo "Job in progress, starting worker"
exit 0
```

```bash
# /etc/cylon/worker.env
JOB_ID=job_12345
RANK=0
WORLD_SIZE=4
REDIS_URL=redis://my-redis:6379
S3_BUCKET=my-checkpoint-bucket
```

**EC2 Launch Script (User Data):**

```bash
#!/bin/bash
set -e

# Install dependencies
yum install -y docker redis

# Pull worker image
docker pull my-repo/cylon-worker:latest

# Create environment file
cat > /etc/cylon/worker.env << EOF
JOB_ID=${JOB_ID}
RANK=${RANK}
WORLD_SIZE=${WORLD_SIZE}
REDIS_URL=${REDIS_URL}
S3_BUCKET=${S3_BUCKET}
EOF

# Create check script
cat > /usr/local/bin/cylon-check-complete << 'SCRIPT'
#!/bin/bash
source /etc/cylon/worker.env
COMPLETE=$(redis-cli -u "$REDIS_URL" GET "job:${JOB_ID}:complete" 2>/dev/null)
if [ "$COMPLETE" = "true" ]; then
    systemctl disable cylon-worker.service
    shutdown -h now  # Terminate instance when done
    exit 1
fi
exit 0
SCRIPT
chmod +x /usr/local/bin/cylon-check-complete

# Create worker wrapper
cat > /usr/local/bin/cylon-worker << 'SCRIPT'
#!/bin/bash
source /etc/cylon/worker.env
docker run --rm \
    -e JOB_ID -e RANK -e WORLD_SIZE -e REDIS_URL -e S3_BUCKET \
    --network host \
    my-repo/cylon-worker:latest
SCRIPT
chmod +x /usr/local/bin/cylon-worker

# Enable and start service
systemctl daemon-reload
systemctl enable cylon-worker.service
systemctl start cylon-worker.service
```

### EKS: Argo Workflows (Recommended for Kubernetes)

For Kubernetes, Argo Workflows provides the best checkpoint-restart support:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Workflow
metadata:
  generateName: cylon-job-
spec:
  entrypoint: parallel-workers

  arguments:
    parameters:
    - name: world-size
      value: "4"
    - name: job-id
      value: "{{workflow.name}}"
    - name: redis-url
      value: "redis://redis-master:6379"
    - name: s3-bucket
      value: "cylon-checkpoints"

  templates:
  - name: parallel-workers
    steps:
    - - name: worker
        template: worker-with-restart
        arguments:
          parameters:
          - name: rank
            value: "{{item}}"
        withSequence:
          count: "{{workflow.parameters.world-size}}"

  - name: worker-with-restart
    inputs:
      parameters:
      - name: rank
    steps:
    - - name: run-worker
        template: worker-task
        arguments:
          parameters:
          - name: rank
            value: "{{inputs.parameters.rank}}"
    - - name: check-complete
        template: check-completion
        arguments:
          parameters:
          - name: rank
            value: "{{inputs.parameters.rank}}"
    - - name: decide-restart
        template: restart-decision
        arguments:
          parameters:
          - name: complete
            value: "{{steps.check-complete.outputs.result}}"
          - name: rank
            value: "{{inputs.parameters.rank}}"

  - name: worker-task
    inputs:
      parameters:
      - name: rank
    container:
      image: my-repo/cylon-worker:latest
      env:
      - name: RANK
        value: "{{inputs.parameters.rank}}"
      - name: WORLD_SIZE
        value: "{{workflow.parameters.world-size}}"
      - name: JOB_ID
        value: "{{workflow.parameters.job-id}}"
      - name: REDIS_URL
        value: "{{workflow.parameters.redis-url}}"
      - name: S3_BUCKET
        value: "{{workflow.parameters.s3-bucket}}"
      resources:
        requests:
          memory: "8Gi"
          cpu: "4"
    # Retry on failure (crash), but not on success (checkpoint)
    retryStrategy:
      limit: 3
      retryPolicy: "OnFailure"

  - name: check-completion
    inputs:
      parameters:
      - name: rank
    script:
      image: redis:alpine
      command: [sh]
      source: |
        JOB_COMPLETE=$(redis-cli -u {{workflow.parameters.redis-url}} GET "job:{{workflow.parameters.job-id}}:complete")
        WORKER_DONE=$(redis-cli -u {{workflow.parameters.redis-url}} SISMEMBER "job:{{workflow.parameters.job-id}}:workers_done" {{inputs.parameters.rank}})

        if [ "$JOB_COMPLETE" = "true" ] || [ "$WORKER_DONE" = "1" ]; then
          echo "complete"
        else
          echo "continue"
        fi

  - name: restart-decision
    inputs:
      parameters:
      - name: complete
      - name: rank
    steps:
    - - name: restart-if-needed
        template: worker-with-restart
        arguments:
          parameters:
          - name: rank
            value: "{{inputs.parameters.rank}}"
        when: "{{inputs.parameters.complete}} == continue"
```

### Comparison Summary

| Platform | Restart Method | Completion Detection | Complexity | Cost |
|----------|---------------|---------------------|------------|------|
| Lambda + Step Functions | State machine loop | Built-in | Low | Pay per invocation |
| Fargate + Step Functions | ECS runTask.sync loop | Lambda check | Medium | Step Functions + Fargate |
| EC2 + Systemd | Restart=on-success | Pre-start script | Low | EC2 instances |
| EKS + Argo | Workflow recursion | Script step | Medium-High | EKS + Argo controller |
| ECS Service | Desired count auto-restart | External scale-down | Medium | Fargate continuous |

**Recommendations:**
- **Lambda**: Use Step Functions (built-in restart loop)
- **Fargate**: Use Step Functions with `ecs:runTask.sync`
- **EC2**: Use Systemd with pre-start completion check
- **Kubernetes**: Use Argo Workflows with recursive templates

---

## Environment Detection

```rust
impl CheckpointConfig {
    /// Auto-detect environment based on available indicators
    pub fn detect_environment() -> EnvironmentType {
        // Lambda: AWS_LAMBDA_FUNCTION_NAME is set
        if std::env::var("AWS_LAMBDA_FUNCTION_NAME").is_ok() {
            return EnvironmentType::Lambda;
        }

        // MPI: Check for common MPI environment variables
        if std::env::var("OMPI_COMM_WORLD_RANK").is_ok()
            || std::env::var("PMI_RANK").is_ok()
            || std::env::var("SLURM_PROCID").is_ok()
        {
            return EnvironmentType::Mpi;
        }

        // Default to local for testing
        EnvironmentType::Local
    }

    /// Auto-detect storage path for MPI environments
    pub fn detect_storage_path() -> String {
        // UVA Rivanna
        if Path::new("/scratch").exists() {
            if let Ok(user) = std::env::var("USER") {
                return format!("/scratch/{}/checkpoints", user);
            }
        }

        // AWS FSx for Lustre
        if Path::new("/fsx").exists() {
            return "/fsx/checkpoints".to_string();
        }

        // Fallback to temp directory
        std::env::temp_dir().join("cylon_checkpoints")
            .to_string_lossy().to_string()
    }
}
```

---

## Performance Optimizations

Checkpointing introduces overhead that can impact application throughput. This section analyzes performance bottlenecks and presents optimization strategies.

### Performance Analysis

#### Where Time Goes

![Checkpoint Timeline](checkpoint_timeline.png)

#### Performance Bottlenecks

| Phase | Bottleneck | Impact |
|-------|------------|--------|
| **Serialization** | CPU-bound, memory copies | Blocks processing |
| **S3 Upload** | Network latency, bandwidth | Async but consumes bandwidth |
| **Coordination** | Redis RTT, MPI barriers | Synchronization overhead |
| **Memory** | 2x table size during snapshot | Memory pressure |

#### Quantifying the Cost

For a 1GB table on typical cloud infrastructure:

```
Serialization:     ~200-500ms (Arrow IPC, single-threaded)
S3 Upload:         ~2-8 seconds (depending on network)
Redis State:       ~5-15ms (3-4 round trips)
Memory Overhead:   +1GB during snapshot
```

**With checkpoints every 100 operations:**
- If each operation takes 50ms → 5 seconds of work between checkpoints
- Snapshot blocks for ~300ms → **6% overhead just from serialization**

---

### Optimization 1: Zero-Copy Serialization

The baseline design serializes tables to a new buffer. Arrow supports **zero-copy slicing**:

```rust
/// Zero-copy snapshot using Arrow's immutable buffers
struct ZeroCopySnapshot {
    /// Reference to original Arrow buffers (no copy)
    batches: Vec<Arc<RecordBatch>>,
    /// Epoch number for consistency
    epoch: u64,
}

impl CheckpointController {
    fn create_zero_copy_snapshot(&self, table: &Table) -> ZeroCopySnapshot {
        // Arrow RecordBatches are immutable and reference-counted
        // This is O(1) - just incrementing Arc counters
        ZeroCopySnapshot {
            batches: table.batches().iter().map(Arc::clone).collect(),
            epoch: table.epoch(),
        }
    }
}
```

**Requirement**: Tables must use copy-on-write semantics. When Cylon modifies a table after snapshot, it creates new buffers rather than mutating in place.

**Benefit**: Snapshot phase drops from ~300ms to <1ms.

#### Copy-on-Write Table Implementation

```rust
/// Table with copy-on-write semantics for zero-copy checkpointing
pub struct CowTable {
    /// Immutable record batches (shared via Arc)
    batches: Vec<Arc<RecordBatch>>,
    /// Current epoch (incremented on each mutation)
    epoch: AtomicU64,
    /// Schema
    schema: Arc<Schema>,
}

impl CowTable {
    /// Append new data (creates new batch, doesn't modify existing)
    pub fn append(&mut self, batch: RecordBatch) {
        self.batches.push(Arc::new(batch));
        self.epoch.fetch_add(1, Ordering::SeqCst);
    }

    /// Filter operation - returns new table, original unchanged
    pub fn filter(&self, predicate: &Predicate) -> CylonResult<CowTable> {
        let filtered_batches: Vec<Arc<RecordBatch>> = self.batches
            .iter()
            .map(|batch| Arc::new(filter_batch(batch, predicate)?))
            .collect::<CylonResult<_>>()?;

        Ok(CowTable {
            batches: filtered_batches,
            epoch: AtomicU64::new(0),
            schema: self.schema.clone(),
        })
    }

    /// Create zero-copy snapshot (O(1) operation)
    pub fn snapshot(&self) -> TableSnapshot {
        TableSnapshot {
            batches: self.batches.clone(), // Just clones Arc pointers
            epoch: self.epoch.load(Ordering::SeqCst),
        }
    }
}
```

---

### Optimization 2: Incremental Checkpoints

Only upload changed data since last checkpoint:

```rust
/// Tracks changes for incremental checkpointing
pub struct IncrementalTracker {
    /// Base checkpoint this delta is relative to
    base_checkpoint: u64,
    /// Batches added since base checkpoint
    added_batches: Vec<usize>,
    /// Batches removed since base checkpoint
    removed_batches: Vec<usize>,
    /// Batches modified since base checkpoint
    modified_batches: Vec<(usize, Arc<RecordBatch>)>,
}

pub struct IncrementalCheckpoint {
    /// Base checkpoint ID this is relative to
    base_checkpoint_id: u64,
    /// Changed record batches only
    deltas: Vec<BatchDelta>,
    /// Total size of delta (for metrics)
    delta_bytes: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum BatchDelta {
    /// New batch added at index
    Added { index: usize, data: Vec<u8> },
    /// Batch removed at index
    Removed { index: usize },
    /// Batch at index replaced with new data
    Replaced { index: usize, data: Vec<u8> },
}

impl CowTable {
    /// Track modifications for incremental checkpointing
    pub fn enable_change_tracking(&mut self) {
        self.change_tracker = Some(ChangeTracker::new());
    }

    /// Get delta since last checkpoint
    pub fn get_delta(&self, since_epoch: u64) -> IncrementalCheckpoint {
        self.change_tracker
            .as_ref()
            .map(|t| t.get_delta(since_epoch))
            .unwrap_or_else(|| IncrementalCheckpoint::full(self))
    }
}

impl CheckpointController {
    /// Create incremental checkpoint if beneficial
    fn create_checkpoint_data(&self, table: &CowTable) -> CheckpointData {
        if !self.config.incremental.enabled {
            return CheckpointData::Full(self.serialize_full(table));
        }

        let delta = table.get_delta(self.last_checkpoint_epoch);
        let full_size = table.size_bytes();
        let delta_size = delta.delta_bytes;

        // Use incremental if delta is < threshold of full size
        if delta_size < full_size * self.config.incremental.threshold_ratio {
            CheckpointData::Incremental(delta)
        } else {
            // Delta too large, do full checkpoint
            CheckpointData::Full(self.serialize_full(table))
        }
    }
}
```

**Benefit**: If only 10% of data changes between checkpoints, upload size drops 90%.

#### Incremental Restore

```rust
impl CheckpointManager {
    /// Restore from incremental checkpoint chain
    pub async fn restore_incremental(
        &self,
        checkpoint_id: u64,
    ) -> CylonResult<RestoreResult> {
        // Build chain of checkpoints back to last full
        let mut chain = Vec::new();
        let mut current_id = checkpoint_id;

        loop {
            let metadata = self.storage.read_metadata(current_id).await?;
            chain.push((current_id, metadata.clone()));

            match &metadata.incremental_info {
                Some(info) => {
                    current_id = info.base_checkpoint_id;
                }
                None => break, // Found full checkpoint
            }
        }

        // Apply chain from oldest (full) to newest (incremental)
        chain.reverse();
        let mut tables = HashMap::new();

        for (ckpt_id, metadata) in chain {
            if metadata.incremental_info.is_none() {
                // Full checkpoint - load completely
                tables = self.load_full_checkpoint(ckpt_id).await?;
            } else {
                // Incremental - apply deltas
                self.apply_deltas(ckpt_id, &mut tables).await?;
            }
        }

        Ok(RestoreResult { checkpoint_id, tables, .. })
    }
}
```

---

### Optimization 3: Parallel Serialization

Serialize multiple tables or batches concurrently using Rayon:

```rust
use rayon::prelude::*;

impl CheckpointController {
    /// Serialize tables in parallel
    fn create_snapshots_parallel(
        &self,
        tables: &HashMap<String, TableRef>,
    ) -> CylonResult<Vec<TableSnapshot>> {
        tables.par_iter()
            .map(|(name, table_ref)| {
                let table = table_ref.table.upgrade()
                    .ok_or_else(|| CylonError::new(Code::Internal, "Table dropped"))?;
                let table_guard = table.read().unwrap();
                let data = self.serializer.serialize(&table_guard)?;
                Ok(TableSnapshot {
                    name: name.clone(),
                    data,
                    version: table_ref.last_checkpoint_version.load(Ordering::SeqCst) + 1,
                })
            })
            .collect()
    }

    /// Serialize batches within a table in parallel
    fn serialize_table_parallel(&self, table: &Table) -> CylonResult<Bytes> {
        let serialized_batches: Vec<Bytes> = table.batches()
            .par_iter()
            .map(|batch| self.serializer.serialize_batch(batch))
            .collect::<CylonResult<_>>()?;

        // Combine into single IPC stream
        self.serializer.combine_batches(serialized_batches)
    }
}
```

**Benefit**: With 4 tables and 4 cores, serialization time drops ~4x.

---

### Optimization 4: Compression

Arrow IPC supports LZ4/ZSTD compression:

```rust
/// Compression codec for checkpoint data
#[derive(Clone, Copy, Debug, Default)]
pub enum CompressionCodec {
    #[default]
    None,
    /// LZ4: fast compression, moderate ratio (~2-3x)
    Lz4,
    /// ZSTD: slower compression, better ratio (~4-6x)
    Zstd { level: i32 },
}

impl CompressionCodec {
    /// Compress data
    pub fn compress(&self, data: &[u8]) -> CylonResult<Vec<u8>> {
        match self {
            CompressionCodec::None => Ok(data.to_vec()),
            CompressionCodec::Lz4 => {
                Ok(lz4_flex::compress_prepend_size(data))
            }
            CompressionCodec::Zstd { level } => {
                Ok(zstd::encode_all(data, *level)?)
            }
        }
    }

    /// Decompress data
    pub fn decompress(&self, data: &[u8]) -> CylonResult<Vec<u8>> {
        match self {
            CompressionCodec::None => Ok(data.to_vec()),
            CompressionCodec::Lz4 => {
                Ok(lz4_flex::decompress_size_prepended(data)?)
            }
            CompressionCodec::Zstd { .. } => {
                Ok(zstd::decode_all(data)?)
            }
        }
    }
}

pub struct CheckpointConfig {
    // ... other fields ...

    /// Compression codec for checkpoint data
    /// Default: Lz4 (best balance of speed and compression)
    pub compression: CompressionCodec,
}
```

**Tradeoff Analysis**:

| Codec | Compression Ratio | Compress Speed | Decompress Speed |
|-------|------------------|----------------|------------------|
| None | 1x | N/A | N/A |
| LZ4 | 2-3x | ~500 MB/s | ~1000 MB/s |
| ZSTD-1 | 3-4x | ~300 MB/s | ~800 MB/s |
| ZSTD-9 | 5-7x | ~50 MB/s | ~800 MB/s |

**Recommendation**: LZ4 for most cases - compression is faster than network I/O, so it reduces total checkpoint time.

---

### Optimization 5: Pipelined Upload

Stream data to storage as it's serialized rather than buffering everything:

```rust
use tokio::sync::mpsc;
use futures::Stream;

impl CheckpointController {
    /// Stream checkpoint data to storage without full buffering
    async fn stream_checkpoint(
        &self,
        table: &Table,
        storage: &dyn CheckpointStorage,
        key: &str,
    ) -> CylonResult<()> {
        // Small buffer - only a few batches in memory at once
        let (tx, rx) = mpsc::channel::<Bytes>(4);

        // Producer: serialize batches one at a time
        let serializer = self.serializer.clone();
        let batches = table.batches().clone();
        let producer = tokio::spawn(async move {
            for batch in batches {
                let bytes = serializer.serialize_batch(&batch)?;
                if tx.send(bytes).await.is_err() {
                    break; // Consumer dropped
                }
            }
            Ok::<_, CylonError>(())
        });

        // Consumer: upload chunks as they arrive
        let consumer = storage.put_streaming(key, rx);

        // Run concurrently
        let (prod_result, cons_result) = tokio::join!(producer, consumer);
        prod_result??;
        cons_result?;

        Ok(())
    }
}

/// Extended storage trait for streaming uploads
#[async_trait]
pub trait CheckpointStorage: Send + Sync {
    // ... existing methods ...

    /// Stream upload - receives chunks and uploads incrementally
    async fn put_streaming(
        &self,
        key: &str,
        stream: mpsc::Receiver<Bytes>,
    ) -> CylonResult<()>;
}

/// S3 multipart upload implementation
impl CheckpointStorage for S3Storage {
    async fn put_streaming(
        &self,
        key: &str,
        mut stream: mpsc::Receiver<Bytes>,
    ) -> CylonResult<()> {
        // Start multipart upload
        let upload_id = self.client
            .create_multipart_upload()
            .bucket(&self.bucket)
            .key(key)
            .send()
            .await?
            .upload_id
            .unwrap();

        let mut parts = Vec::new();
        let mut part_number = 1;
        let mut buffer = Vec::new();
        const MIN_PART_SIZE: usize = 5 * 1024 * 1024; // 5MB minimum for S3

        while let Some(chunk) = stream.recv().await {
            buffer.extend_from_slice(&chunk);

            // Upload when buffer reaches minimum part size
            if buffer.len() >= MIN_PART_SIZE {
                let part = self.upload_part(
                    key, &upload_id, part_number, &buffer
                ).await?;
                parts.push(part);
                part_number += 1;
                buffer.clear();
            }
        }

        // Upload remaining data
        if !buffer.is_empty() {
            let part = self.upload_part(
                key, &upload_id, part_number, &buffer
            ).await?;
            parts.push(part);
        }

        // Complete multipart upload
        self.complete_multipart_upload(key, &upload_id, parts).await?;

        Ok(())
    }
}
```

**Benefit**:
- Memory: Only ~4 batches buffered at once instead of entire table
- Latency: Upload starts immediately, doesn't wait for full serialization

---

### Optimization 6: Adaptive Checkpoint Frequency

Adjust checkpoint frequency based on measured overhead:

```rust
/// Adaptive trigger that maintains target overhead ratio
pub struct AdaptiveTrigger {
    /// Target checkpoint overhead (e.g., 0.05 = 5% of runtime)
    target_overhead: f64,

    /// Measured checkpoint duration (exponential moving average)
    avg_checkpoint_ms: AtomicU64,

    /// Measured work time between checkpoints
    avg_work_ms: AtomicU64,

    /// Current operation threshold (adjusted dynamically)
    current_threshold: AtomicU64,

    /// Base threshold (minimum)
    base_threshold: u64,

    /// Maximum threshold (don't checkpoint too rarely)
    max_threshold: u64,

    /// Operations since last checkpoint
    ops_since_checkpoint: AtomicU64,

    /// EMA smoothing factor
    alpha: f64,
}

impl AdaptiveTrigger {
    pub fn new(target_overhead: f64, base_threshold: u64) -> Self {
        Self {
            target_overhead,
            avg_checkpoint_ms: AtomicU64::new(0),
            avg_work_ms: AtomicU64::new(0),
            current_threshold: AtomicU64::new(base_threshold),
            base_threshold,
            max_threshold: base_threshold * 10,
            ops_since_checkpoint: AtomicU64::new(0),
            alpha: 0.3, // EMA smoothing
        }
    }

    /// Record checkpoint completion and adjust threshold
    pub fn record_checkpoint(&self, checkpoint_ms: u64, work_ms: u64) {
        // Update EMAs
        let old_ckpt = self.avg_checkpoint_ms.load(Ordering::Relaxed);
        let new_ckpt = ((1.0 - self.alpha) * old_ckpt as f64
            + self.alpha * checkpoint_ms as f64) as u64;
        self.avg_checkpoint_ms.store(new_ckpt, Ordering::Relaxed);

        let old_work = self.avg_work_ms.load(Ordering::Relaxed);
        let new_work = ((1.0 - self.alpha) * old_work as f64
            + self.alpha * work_ms as f64) as u64;
        self.avg_work_ms.store(new_work, Ordering::Relaxed);

        // Calculate current overhead
        let current_overhead = new_ckpt as f64 / (new_ckpt + new_work) as f64;

        // Adjust threshold to approach target overhead
        let current = self.current_threshold.load(Ordering::Relaxed);
        let new_threshold = if current_overhead > self.target_overhead {
            // Overhead too high - checkpoint less often
            (current as f64 * (current_overhead / self.target_overhead)) as u64
        } else {
            // Overhead below target - can checkpoint more often
            (current as f64 * (current_overhead / self.target_overhead).max(0.5)) as u64
        };

        // Clamp to bounds
        let clamped = new_threshold.clamp(self.base_threshold, self.max_threshold);
        self.current_threshold.store(clamped, Ordering::Relaxed);
    }
}

impl CheckpointTrigger for AdaptiveTrigger {
    fn should_checkpoint(&self) -> bool {
        let ops = self.ops_since_checkpoint.load(Ordering::Relaxed);
        let threshold = self.current_threshold.load(Ordering::Relaxed);
        ops >= threshold
    }

    fn record_operation(&self, _op_type: OperationType, _bytes: u64) {
        self.ops_since_checkpoint.fetch_add(1, Ordering::Relaxed);
    }

    fn reset(&self) {
        self.ops_since_checkpoint.store(0, Ordering::Relaxed);
    }

    fn urgency(&self) -> CheckpointPriority {
        CheckpointPriority::Medium
    }
}
```

**Benefit**: Automatically balances checkpoint frequency against overhead, adapting to workload characteristics.

---

### Optimization 7: Local SSD Staging (HPC)

For HPC with parallel filesystems, write to local NVMe first, then async copy to shared storage:

```rust
/// Two-tier storage: fast local + durable remote
pub struct TieredStorage {
    /// Fast local storage (NVMe, tmpfs, local SSD)
    local: Arc<dyn CheckpointStorage>,
    /// Durable remote storage (Lustre, S3, GPFS)
    remote: Arc<dyn CheckpointStorage>,
    /// Background replication handle
    replication_handle: Mutex<Option<JoinHandle<()>>>,
}

impl TieredStorage {
    pub fn new(local_path: &str, remote: Arc<dyn CheckpointStorage>) -> Self {
        Self {
            local: Arc::new(LocalStorage::new(local_path)),
            remote,
            replication_handle: Mutex::new(None),
        }
    }
}

#[async_trait]
impl CheckpointStorage for TieredStorage {
    async fn put(&self, key: &str, data: Bytes) -> CylonResult<()> {
        // Write to local first (fast, ~10ms for 1GB on NVMe)
        self.local.put(key, data.clone()).await?;

        // Async replicate to remote (slow, but doesn't block application)
        let remote = self.remote.clone();
        let key = key.to_string();
        let handle = tokio::spawn(async move {
            if let Err(e) = remote.put(&key, data).await {
                log::error!("Background replication failed: {}", e);
                // Don't panic - local copy exists for recovery
            }
        });

        // Track handle for cleanup
        *self.replication_handle.lock().unwrap() = Some(handle);

        Ok(())
    }

    async fn get(&self, key: &str) -> CylonResult<Bytes> {
        // Try local first
        match self.local.get(key).await {
            Ok(data) => Ok(data),
            Err(_) => {
                // Fall back to remote
                let data = self.remote.get(key).await?;
                // Cache locally for next time
                let _ = self.local.put(key, data.clone()).await;
                Ok(data)
            }
        }
    }

    /// Wait for all background replications to complete
    async fn sync(&self) -> CylonResult<()> {
        if let Some(handle) = self.replication_handle.lock().unwrap().take() {
            handle.await.map_err(|e|
                CylonError::new(Code::Internal, format!("Replication task failed: {}", e))
            )?;
        }
        Ok(())
    }
}
```

**Benefit**: Checkpoint "completes" in ~10ms (local NVMe) instead of ~2s (network storage). Remote replication happens in background.

**Recovery Consideration**: On restore, check local first, then remote. If node died before replication completed, use previous checkpoint from remote.

---

### Performance Comparison Summary

| Optimization | Snapshot Time | Upload Time | Memory | Complexity |
|--------------|---------------|-------------|--------|------------|
| **Baseline** | 300ms | 4s | 2x table | Low |
| **Zero-copy** | <1ms | 4s | 1x + refs | Medium |
| **Incremental** | 30ms | 400ms* | 1.1x | High |
| **Parallel serialize** | 75ms | 4s | 2x | Low |
| **Compression (LZ4)** | 350ms | 1.5s | 2.3x | Low |
| **Pipelined upload** | N/A | 3s | 0.1x | Medium |
| **Local staging** | 300ms | 10ms† | 2x | Medium |

*Assuming 10% data change
†Remote replication happens async

---

### Recommended Implementation Phases

**Phase 1 (Low-hanging fruit):**
1. **LZ4 compression** - Almost free performance win, reduces upload time ~50%
2. **Parallel serialization** - Simple with Rayon, scales with core count

**Phase 2 (Significant gains):**
3. **Zero-copy snapshots** - Requires COW table implementation
4. **Local SSD staging** - For HPC environments with local NVMe

**Phase 3 (Maximum performance):**
5. **Incremental checkpoints** - Significant complexity, best for large tables with low churn
6. **Adaptive frequency** - Automatically tunes to workload

### Configuration

```rust
pub struct CheckpointConfig {
    // ... existing fields ...

    /// Performance optimization settings
    pub performance: PerformanceConfig,
}

#[derive(Clone, Debug)]
pub struct PerformanceConfig {
    /// Enable LZ4 compression (recommended)
    /// Default: true
    pub compression_enabled: bool,

    /// Compression codec
    /// Default: Lz4
    pub compression_codec: CompressionCodec,

    /// Enable parallel serialization
    /// Default: true
    pub parallel_serialize: bool,

    /// Number of threads for parallel serialization
    /// Default: num_cpus
    pub serialize_threads: usize,

    /// Enable zero-copy snapshots (requires COW tables)
    /// Default: false (opt-in)
    pub zero_copy_snapshots: bool,

    /// Enable incremental checkpoints
    /// Default: false (opt-in)
    pub incremental_enabled: bool,

    /// Incremental threshold: use incremental if delta < threshold * full_size
    /// Default: 0.5 (use incremental if delta is less than 50% of full)
    pub incremental_threshold: f64,

    /// Enable pipelined/streaming upload
    /// Default: true
    pub streaming_upload: bool,

    /// Local staging path (for tiered storage)
    /// Default: None (disabled)
    pub local_staging_path: Option<String>,

    /// Enable adaptive checkpoint frequency
    /// Default: false
    pub adaptive_frequency: bool,

    /// Target overhead ratio for adaptive frequency
    /// Default: 0.05 (5%)
    pub target_overhead: f64,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            compression_enabled: true,
            compression_codec: CompressionCodec::Lz4,
            parallel_serialize: true,
            serialize_threads: num_cpus::get(),
            zero_copy_snapshots: false,
            incremental_enabled: false,
            incremental_threshold: 0.5,
            streaming_upload: true,
            local_staging_path: None,
            adaptive_frequency: false,
            target_overhead: 0.05,
        }
    }
}
```

---

## References

- [Twister2 Checkpointing](https://github.com/DSC-SPIDAL/twister2/tree/master/twister2/checkpointing)
- [Apache Arrow IPC](https://arrow.apache.org/docs/format/Columnar.html#ipc-file-format)
- [Redis Commands](https://redis.io/commands/)
- [AWS FSx for Lustre](https://docs.aws.amazon.com/fsx/latest/LustreGuide/what-is.html)
- [AWS EFA with libfabric](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html)
- [AWS ParallelCluster](https://docs.aws.amazon.com/parallelcluster/)
- [UVA Rivanna](https://www.rc.virginia.edu/userinfo/rivanna/overview/)
