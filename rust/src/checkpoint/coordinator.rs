//! Checkpoint coordinator implementations.
//!
//! Provides distributed coordination for checkpoint operations.
//! Works with any communication backend (MPI, UCX, UCC, FMI, Gloo).

use async_trait::async_trait;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::error::{Code, CylonError, CylonResult};
use crate::net::communicator::Communicator;

use super::traits::CheckpointCoordinator;
use super::types::{CheckpointContext, CheckpointDecision, CheckpointPriority, WorkerId};

/// Distributed checkpoint coordinator.
///
/// Uses the Cylon `Communicator` trait for coordination, which means it works
/// with any communication backend: MPI, UCX, UCC, FMI, or Gloo.
///
/// Coordination is achieved through collective operations:
/// - `allgather` for vote collection (simulates allreduce AND)
/// - `barrier` for synchronization
/// - `allgather` for finding latest checkpoint (simulates allreduce MIN)
///
/// This coordinator assumes all workers must participate in checkpoints.
pub struct DistributedCoordinator {
    /// Communicator (works with MPI, UCX, UCC, FMI, Gloo)
    communicator: Arc<dyn Communicator>,
    /// Worker ID (rank)
    worker_id: WorkerId,
    /// World size
    world_size: usize,
    /// Next checkpoint ID to use
    next_checkpoint_id: AtomicU64,
    /// Latest committed checkpoint ID
    latest_committed: AtomicU64,
}

impl DistributedCoordinator {
    /// Create a new distributed coordinator from any Communicator
    pub fn new(communicator: Arc<dyn Communicator>) -> Self {
        let rank = communicator.get_rank();
        let world_size = communicator.get_world_size() as usize;

        Self {
            communicator,
            worker_id: WorkerId::Rank(rank),
            world_size,
            next_checkpoint_id: AtomicU64::new(1),
            latest_committed: AtomicU64::new(0),
        }
    }

    /// Get the underlying communicator
    pub fn communicator(&self) -> &Arc<dyn Communicator> {
        &self.communicator
    }

    /// Perform an allreduce AND operation on a boolean vote using allgather
    fn allreduce_vote(&self, my_vote: bool) -> CylonResult<bool> {
        // Encode vote as u8
        let vote_byte: u8 = if my_vote { 1 } else { 0 };
        let vote_data = vec![vote_byte];

        // Allgather all votes
        let all_votes = self.communicator.allgather(&vote_data)?;

        // AND all votes together - all must vote 1 for result to be true
        let result = all_votes.iter().all(|v| !v.is_empty() && v[0] == 1);

        Ok(result)
    }

    /// Perform an allreduce MIN operation on a u64 value using allgather
    fn allreduce_min(&self, value: u64) -> CylonResult<u64> {
        // Encode as bytes (little-endian)
        let value_bytes = value.to_le_bytes().to_vec();

        // Allgather all values
        let all_values = self.communicator.allgather(&value_bytes)?;

        // Find minimum (treating 0 as "no checkpoint")
        let mut min_value = u64::MAX;
        for bytes in all_values {
            if bytes.len() >= 8 {
                let v = u64::from_le_bytes(bytes[..8].try_into().unwrap_or([0; 8]));
                if v > 0 && v < min_value {
                    min_value = v;
                }
            }
        }

        Ok(if min_value == u64::MAX { 0 } else { min_value })
    }

    /// Perform an allreduce MAX operation on a u64 value using allgather
    fn allreduce_max(&self, value: u64) -> CylonResult<u64> {
        // Encode as bytes (little-endian)
        let value_bytes = value.to_le_bytes().to_vec();

        // Allgather all values
        let all_values = self.communicator.allgather(&value_bytes)?;

        // Find maximum
        let mut max_value = 0u64;
        for bytes in all_values {
            if bytes.len() >= 8 {
                let v = u64::from_le_bytes(bytes[..8].try_into().unwrap_or([0; 8]));
                if v > max_value {
                    max_value = v;
                }
            }
        }

        Ok(max_value)
    }

    /// Get the next checkpoint ID (synchronized across all workers)
    pub fn get_next_checkpoint_id(&self) -> CylonResult<u64> {
        // Each worker proposes its next ID
        let my_next = self.next_checkpoint_id.load(Ordering::SeqCst);

        // Take the maximum to ensure all workers use the same ID
        let global_next = self.allreduce_max(my_next)?;

        // Update local state
        self.next_checkpoint_id
            .store(global_next + 1, Ordering::SeqCst);

        Ok(global_next)
    }
}

#[async_trait]
impl CheckpointCoordinator for DistributedCoordinator {
    fn worker_id(&self) -> WorkerId {
        self.worker_id.clone()
    }

    fn world_size(&self) -> usize {
        self.world_size
    }

    fn should_checkpoint(&self, context: &CheckpointContext) -> bool {
        // Simple heuristic: checkpoint after threshold operations or bytes
        // This is a local decision; actual checkpoint requires global agreement
        context.operations_since_checkpoint >= 100
            || context.bytes_since_checkpoint >= 100 * 1024 * 1024 // 100MB
    }

    async fn begin_checkpoint(&self, checkpoint_id: u64) -> CylonResult<CheckpointDecision> {
        // All workers vote "ready" - in a real scenario, workers might vote
        // based on their local state (memory pressure, etc.)
        let all_ready = self.allreduce_vote(true)?;

        if all_ready {
            // Store the checkpoint ID for future reference
            self.next_checkpoint_id
                .store(checkpoint_id + 1, Ordering::SeqCst);
            Ok(CheckpointDecision::Proceed(CheckpointPriority::Medium))
        } else {
            Ok(CheckpointDecision::Skip)
        }
    }

    async fn commit_checkpoint(&self, checkpoint_id: u64) -> CylonResult<()> {
        // Use barrier to ensure all workers have written their data
        self.communicator.barrier()?;

        // Update latest committed checkpoint
        self.latest_committed.store(checkpoint_id, Ordering::SeqCst);

        Ok(())
    }

    async fn abort_checkpoint(&self, _checkpoint_id: u64) -> CylonResult<()> {
        // Use barrier to synchronize abort across all workers
        self.communicator.barrier()?;
        Ok(())
    }

    async fn find_latest_checkpoint(&self) -> CylonResult<Option<u64>> {
        // Each worker reports its latest committed checkpoint
        let my_latest = self.latest_committed.load(Ordering::SeqCst);

        // Find the minimum across all workers (the checkpoint that all workers have)
        let global_latest = self.allreduce_min(my_latest)?;

        Ok(if global_latest > 0 {
            Some(global_latest)
        } else {
            None
        })
    }

    async fn heartbeat(&self) -> CylonResult<()> {
        // No-op for distributed coordinators - failure detection is handled
        // by the underlying communication runtime (MPI, UCX, etc.)
        Ok(())
    }

    fn is_leader(&self) -> bool {
        matches!(self.worker_id, WorkerId::Rank(0))
    }
}

/// Local/single-worker coordinator for testing and non-distributed use.
///
/// Always agrees to checkpoint, no distributed coordination needed.
pub struct LocalCoordinator {
    /// Worker ID
    worker_id: WorkerId,
    /// Latest checkpoint ID
    latest_checkpoint: AtomicU64,
}

impl LocalCoordinator {
    /// Create a new local coordinator
    pub fn new() -> Self {
        Self {
            worker_id: WorkerId::Rank(0),
            latest_checkpoint: AtomicU64::new(0),
        }
    }

    /// Create a local coordinator with a specific worker ID
    pub fn with_worker_id(worker_id: WorkerId) -> Self {
        Self {
            worker_id,
            latest_checkpoint: AtomicU64::new(0),
        }
    }
}

impl Default for LocalCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl CheckpointCoordinator for LocalCoordinator {
    fn worker_id(&self) -> WorkerId {
        self.worker_id.clone()
    }

    fn world_size(&self) -> usize {
        1
    }

    fn should_checkpoint(&self, context: &CheckpointContext) -> bool {
        context.operations_since_checkpoint >= 100
            || context.bytes_since_checkpoint >= 100 * 1024 * 1024
    }

    async fn begin_checkpoint(&self, checkpoint_id: u64) -> CylonResult<CheckpointDecision> {
        // Always proceed in local mode
        Ok(CheckpointDecision::Proceed(CheckpointPriority::Medium))
    }

    async fn commit_checkpoint(&self, checkpoint_id: u64) -> CylonResult<()> {
        self.latest_checkpoint.store(checkpoint_id, Ordering::SeqCst);
        Ok(())
    }

    async fn abort_checkpoint(&self, _checkpoint_id: u64) -> CylonResult<()> {
        Ok(())
    }

    async fn find_latest_checkpoint(&self) -> CylonResult<Option<u64>> {
        let latest = self.latest_checkpoint.load(Ordering::SeqCst);
        Ok(if latest > 0 { Some(latest) } else { None })
    }

    async fn heartbeat(&self) -> CylonResult<()> {
        Ok(())
    }

    fn is_leader(&self) -> bool {
        true // Local coordinator is always the leader
    }
}

#[cfg(feature = "redis")]
pub mod redis_coordinator {
    //! Redis-based coordinator for serverless environments.
    //!
    //! This module provides coordination for serverless environments where
    //! workers may be ephemeral and traditional collective operations aren't available.

    use super::*;

    /// Redis-based checkpoint coordinator for serverless environments.
    ///
    /// Uses Redis for coordination:
    /// - Pub/Sub for vote collection
    /// - SETNX for distributed locking
    /// - Hash sets for worker status tracking
    /// - TTL for automatic timeout handling
    pub struct RedisCoordinator {
        // TODO: Implement when Redis support is needed
        _private: (),
    }

    impl RedisCoordinator {
        /// Create a new Redis coordinator
        pub fn new(_redis_url: &str, _job_id: &str, _worker_id: &str) -> CylonResult<Self> {
            Err(CylonError::new(
                Code::NotImplemented,
                "Redis coordinator not yet implemented".to_string(),
            ))
        }
    }
}

#[cfg(feature = "redis")]
pub use redis_coordinator::RedisCoordinator;

// Re-export DistributedCoordinator as the main coordinator for backwards compatibility
// with any code that might have referenced MpiCoordinator
pub use DistributedCoordinator as MpiCoordinator;
