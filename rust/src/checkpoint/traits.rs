//! Core traits for the checkpointing system.
//!
//! The checkpointing system is built on four core traits that can be implemented
//! for different backends:
//!
//! - [`CheckpointCoordinator`] - Distributed coordination
//! - [`CheckpointStorage`] - Reading/writing checkpoint data
//! - [`CheckpointSerializer`] - Serialization of tables and state
//! - [`CheckpointTrigger`] - Determining when to checkpoint

use async_trait::async_trait;
use serde::{de::DeserializeOwned, Serialize};
use std::sync::Arc;

use crate::table::Table;
use crate::ctx::CylonContext;
use crate::error::CylonResult;

use super::types::{
    CheckpointContext, CheckpointDecision, CheckpointMetadata, CheckpointUrgency,
    OperationType, WorkerId,
};

/// Distributed coordination for checkpoints.
///
/// Handles when and how to synchronize checkpoints across workers.
/// Different implementations handle MPI vs serverless environments.
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
    async fn claim_work(&self, _work_unit_id: &str) -> CylonResult<bool> {
        Ok(true) // Default: always succeeds (MPI has static assignment)
    }

    /// Check if this worker is the leader/rank 0
    fn is_leader(&self) -> bool {
        match self.worker_id() {
            WorkerId::Rank(0) => true,
            WorkerId::Rank(_) => false,
            WorkerId::Serverless { .. } => false, // Serverless doesn't have a fixed leader
        }
    }
}

/// Handles reading and writing checkpoint data.
///
/// Abstracts over different storage backends (filesystem, S3, etc.)
#[async_trait]
pub trait CheckpointStorage: Send + Sync {
    /// Write checkpoint data to staging area
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

    /// Check if checkpoint data exists
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

    /// Atomic move from staging to final location (for commit protocol)
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
    async fn read_metadata(&self, checkpoint_id: u64) -> CylonResult<CheckpointMetadata>;

    /// Get the base path/URI for this storage
    fn base_path(&self) -> &str;
}

/// Handles serialization of tables and state.
///
/// Abstracts over different serialization formats (Arrow IPC, Parquet, etc.)
pub trait CheckpointSerializer: Send + Sync {
    /// Serialize a table to bytes
    fn serialize_table(&self, table: &Table) -> CylonResult<Vec<u8>>;

    /// Deserialize bytes to a table
    fn deserialize_table(&self, data: &[u8], ctx: Arc<CylonContext>) -> CylonResult<Table>;

    /// Serialize arbitrary state
    fn serialize_state<T: Serialize>(&self, state: &T) -> CylonResult<Vec<u8>>;

    /// Deserialize arbitrary state
    fn deserialize_state<T: DeserializeOwned>(&self, data: &[u8]) -> CylonResult<T>;

    /// Get the format identifier (for compatibility checking)
    fn format_id(&self) -> &str;
}

/// Determines when to checkpoint (environment-specific strategies).
///
/// Different implementations can trigger based on:
/// - Operation count
/// - Bytes processed
/// - Time intervals
/// - Remaining time budget (serverless)
/// - Memory pressure
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
