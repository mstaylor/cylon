//! Core types for the checkpointing system.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Worker identifier - supports both MPI ranks and serverless worker IDs.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WorkerId {
    /// Stable MPI rank
    Rank(i32),
    /// Ephemeral serverless worker identifier
    Serverless { worker_id: String },
}

impl WorkerId {
    /// Create a WorkerId from an MPI rank
    pub fn from_rank(rank: i32) -> Self {
        WorkerId::Rank(rank)
    }

    /// Create a WorkerId for serverless environments
    pub fn serverless(id: impl Into<String>) -> Self {
        WorkerId::Serverless {
            worker_id: id.into(),
        }
    }

    /// Get a string representation suitable for use in paths
    pub fn to_path_string(&self) -> String {
        match self {
            WorkerId::Rank(rank) => format!("rank_{}", rank),
            WorkerId::Serverless { worker_id } => format!("worker_{}", worker_id),
        }
    }
}

impl std::fmt::Display for WorkerId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WorkerId::Rank(rank) => write!(f, "Rank({})", rank),
            WorkerId::Serverless { worker_id } => write!(f, "Worker({})", worker_id),
        }
    }
}

/// Status of a checkpoint in its lifecycle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CheckpointStatus {
    /// Checkpoint is being written (not yet durable)
    Writing,
    /// Data is staged/durable but not yet committed
    Staged,
    /// Commit in progress (moving from staging to committed)
    Committing,
    /// Checkpoint is fully committed and valid
    Committed,
    /// Checkpoint failed and should be cleaned up
    Failed,
    /// Some workers committed but others failed
    PartiallyCommitted,
}

impl CheckpointStatus {
    /// Returns true if this checkpoint is safe to restore from
    pub fn is_restorable(&self) -> bool {
        matches!(self, CheckpointStatus::Committed)
    }

    /// Returns true if this checkpoint is in a terminal state
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            CheckpointStatus::Committed | CheckpointStatus::Failed
        )
    }

    /// Returns true if this checkpoint needs recovery/cleanup
    pub fn needs_recovery(&self) -> bool {
        matches!(
            self,
            CheckpointStatus::Writing
                | CheckpointStatus::Staged
                | CheckpointStatus::Committing
                | CheckpointStatus::PartiallyCommitted
        )
    }
}

/// Status of a single worker's checkpoint contribution.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkerCheckpointStatus {
    /// Worker is preparing to checkpoint
    Preparing,
    /// Worker is writing data
    Writing,
    /// Worker has staged its data
    Staged,
    /// Worker has committed its data
    Committed,
    /// Worker failed during checkpoint
    Failed,
}

/// Information about a worker's contribution to a checkpoint.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorkerCheckpointInfo {
    /// The worker's identifier
    pub worker_id: WorkerId,
    /// Current status
    pub status: WorkerCheckpointStatus,
    /// Timestamp when this status was set
    pub timestamp: u64,
    /// Size of data written (bytes)
    pub data_size: u64,
    /// List of tables checkpointed by this worker
    pub tables: Vec<String>,
    /// Optional error message if failed
    pub error: Option<String>,
}

impl WorkerCheckpointInfo {
    /// Create a new WorkerCheckpointInfo
    pub fn new(worker_id: WorkerId) -> Self {
        Self {
            worker_id,
            status: WorkerCheckpointStatus::Preparing,
            timestamp: current_timestamp(),
            data_size: 0,
            tables: Vec::new(),
            error: None,
        }
    }

    /// Update the status
    pub fn set_status(&mut self, status: WorkerCheckpointStatus) {
        self.status = status;
        self.timestamp = current_timestamp();
    }
}

/// Metadata for a checkpoint.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    /// Unique checkpoint identifier
    pub checkpoint_id: u64,
    /// Job identifier
    pub job_id: String,
    /// Timestamp when checkpoint was initiated
    #[serde(with = "system_time_serde")]
    pub timestamp: std::time::SystemTime,
    /// Overall checkpoint status
    pub status: CheckpointStatus,
    /// Workers that participated in this checkpoint
    pub workers: Vec<WorkerId>,
    /// Tables checkpointed
    pub tables: Vec<String>,
    /// Total bytes written
    pub total_bytes: u64,
    /// Format version for compatibility
    pub format_version: String,
    /// Custom application metadata
    pub metadata: HashMap<String, String>,
    /// Parent checkpoint ID for incremental checkpoints (None for full checkpoints)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_checkpoint_id: Option<u64>,
    /// Incremental checkpoint information (None for full checkpoints)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub incremental_info: Option<super::incremental::IncrementalCheckpointInfo>,
}

impl CheckpointMetadata {
    /// Create new checkpoint metadata
    pub fn new(checkpoint_id: u64, job_id: impl Into<String>) -> Self {
        Self {
            checkpoint_id,
            job_id: job_id.into(),
            timestamp: std::time::SystemTime::now(),
            status: CheckpointStatus::Writing,
            workers: Vec::new(),
            tables: Vec::new(),
            total_bytes: 0,
            format_version: "1.0".to_string(),
            metadata: HashMap::new(),
            parent_checkpoint_id: None,
            incremental_info: None,
        }
    }

    /// Create new incremental checkpoint metadata
    pub fn new_incremental(
        checkpoint_id: u64,
        job_id: impl Into<String>,
        parent_checkpoint_id: u64,
    ) -> Self {
        let mut metadata = Self::new(checkpoint_id, job_id);
        metadata.parent_checkpoint_id = Some(parent_checkpoint_id);
        metadata.incremental_info =
            Some(super::incremental::IncrementalCheckpointInfo::new(parent_checkpoint_id));
        metadata
    }

    /// Check if this is an incremental checkpoint
    pub fn is_incremental(&self) -> bool {
        self.parent_checkpoint_id.is_some()
    }

    /// Get the chain depth (1 for full checkpoints, >1 for incremental)
    pub fn chain_depth(&self) -> u32 {
        self.incremental_info
            .as_ref()
            .map(|i| i.chain_depth)
            .unwrap_or(0)
    }
}

/// Custom serde module for SystemTime
mod system_time_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    pub fn serialize<S>(time: &SystemTime, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let duration = time.duration_since(UNIX_EPOCH).unwrap_or(Duration::ZERO);
        duration.as_secs().serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<SystemTime, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs = u64::deserialize(deserializer)?;
        Ok(UNIX_EPOCH + Duration::from_secs(secs))
    }
}

/// Priority level for checkpoint decisions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
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

impl Default for CheckpointPriority {
    fn default() -> Self {
        CheckpointPriority::None
    }
}

/// Decision from checkpoint coordination.
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

/// Urgency level for checkpoint triggers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum CheckpointUrgency {
    /// No checkpoint needed
    None,
    /// Checkpoint when convenient
    Low,
    /// Should checkpoint soon
    Medium,
    /// Should checkpoint now
    High,
    /// Must checkpoint immediately (serverless timeout approaching)
    Critical,
}

impl Default for CheckpointUrgency {
    fn default() -> Self {
        CheckpointUrgency::None
    }
}

impl From<CheckpointUrgency> for CheckpointPriority {
    fn from(urgency: CheckpointUrgency) -> Self {
        match urgency {
            CheckpointUrgency::None => CheckpointPriority::None,
            CheckpointUrgency::Low => CheckpointPriority::Low,
            CheckpointUrgency::Medium => CheckpointPriority::Medium,
            CheckpointUrgency::High => CheckpointPriority::High,
            CheckpointUrgency::Critical => CheckpointPriority::Critical,
        }
    }
}

/// Context passed to checkpoint trigger for decision making.
#[derive(Clone, Debug)]
pub struct CheckpointContext {
    /// Number of operations since last checkpoint
    pub operations_since_checkpoint: u64,
    /// Bytes processed since last checkpoint
    pub bytes_since_checkpoint: u64,
    /// Time since last checkpoint
    pub time_since_checkpoint: Duration,
    /// Remaining time budget (for serverless)
    pub remaining_time_budget: Option<Duration>,
    /// Current memory usage (bytes)
    pub memory_usage: Option<u64>,
    /// Memory limit (bytes)
    pub memory_limit: Option<u64>,
    /// Last checkpoint ID
    pub last_checkpoint_id: Option<u64>,
}

impl Default for CheckpointContext {
    fn default() -> Self {
        Self {
            operations_since_checkpoint: 0,
            bytes_since_checkpoint: 0,
            time_since_checkpoint: Duration::ZERO,
            remaining_time_budget: None,
            memory_usage: None,
            memory_limit: None,
            last_checkpoint_id: None,
        }
    }
}

/// Type of operation for tracking.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperationType {
    /// Join operation
    Join,
    /// Sort operation
    Sort,
    /// Shuffle operation
    Shuffle,
    /// Filter operation
    Filter,
    /// Projection operation
    Project,
    /// Aggregation operation
    Aggregate,
    /// Union operation
    Union,
    /// Intersection operation
    Intersect,
    /// Difference operation
    Difference,
    /// Generic/other operation
    Other,
}

/// Reason for triggering a checkpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TriggerReason {
    /// Reached operation count threshold
    OperationThreshold,
    /// Reached bytes processed threshold
    BytesThreshold,
    /// Time budget running low (serverless)
    TimeBudget,
    /// Memory pressure
    MemoryPressure,
    /// Manual trigger
    Manual,
    /// Periodic interval
    Interval,
}

/// Event emitted during checkpoint lifecycle.
#[derive(Clone, Debug)]
pub enum CheckpointEvent {
    /// Checkpoint has started
    Started { checkpoint_id: u64 },
    /// Checkpoint completed successfully
    Completed { checkpoint_id: u64 },
    /// Checkpoint failed
    Failed { checkpoint_id: u64, error: String },
    /// Restore started
    RestoreStarted { checkpoint_id: u64 },
    /// Restore completed
    RestoreCompleted { checkpoint_id: u64 },
    /// Old checkpoint was pruned
    Pruned { checkpoint_id: u64 },
}

/// Recommended action based on checkpoint trigger state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CheckpointAction {
    /// Continue processing - no checkpoint needed
    Continue,
    /// Should checkpoint soon
    CheckpointSoon,
    /// Should checkpoint immediately
    CheckpointNow,
    /// Checkpoint and exit (for serverless timeout)
    CheckpointAndExit,
}

/// Helper function to get current Unix timestamp
pub fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_secs()
}
