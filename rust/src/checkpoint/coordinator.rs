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
    //! Redis-based coordinator for distributed environments.
    //!
    //! This module provides coordination using Redis as a central coordination point.
    //! It works for any distributed environment - HPC clusters, cloud VMs, Kubernetes,
    //! or serverless functions.
    //!
    //! # Fault Tolerance
    //!
    //! This coordinator is designed for environments where workers can drop unexpectedly
    //! (e.g., Lambda with NAT hole-punching). Key features:
    //!
    //! - **Heartbeat-based failure detection**: Workers must maintain heartbeats
    //! - **Participant tracking**: Only workers that join a checkpoint are expected to complete
    //! - **Abort on failure**: If any participant drops during coordination, checkpoint is aborted
    //! - **Retry with survivors**: After abort, surviving workers can retry with new participant set
    //!
    //! # Key Structure
    //!
    //! ```text
    //! cylon:{job_id}:workers                        - Set of registered worker IDs
    //! cylon:{job_id}:worker:{worker_id}:heartbeat   - Worker heartbeat (with TTL)
    //! cylon:{job_id}:checkpoint:{id}:participants   - Set of workers participating in checkpoint
    //! cylon:{job_id}:checkpoint:{id}:votes          - Set of worker votes for checkpoint
    //! cylon:{job_id}:checkpoint:{id}:completed      - Set of workers that completed checkpoint
    //! cylon:{job_id}:checkpoint:{id}:status         - Checkpoint status (active/aborted/committed)
    //! cylon:{job_id}:checkpoint:latest              - Latest committed checkpoint ID
    //! cylon:{job_id}:lock:{resource}                - Distributed lock
    //! ```

    use super::*;
    use redis::aio::MultiplexedConnection;
    use redis::{AsyncCommands, Client};
    use std::collections::HashSet;
    use std::time::Duration;
    use tokio::sync::RwLock;

    /// Status of a checkpoint coordination
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum CheckpointCoordinationStatus {
        /// Checkpoint is actively being coordinated
        Active,
        /// Checkpoint was aborted (worker failure or vote rejection)
        Aborted,
        /// Checkpoint was successfully committed
        Committed,
    }

    impl CheckpointCoordinationStatus {
        /// Convert status to string representation
        pub fn as_str(&self) -> &'static str {
            match self {
                Self::Active => "active",
                Self::Aborted => "aborted",
                Self::Committed => "committed",
            }
        }

        /// Parse status from string
        pub fn from_str(s: &str) -> Option<Self> {
            match s {
                "active" => Some(Self::Active),
                "aborted" => Some(Self::Aborted),
                "committed" => Some(Self::Committed),
                _ => None,
            }
        }
    }

    /// Configuration for Redis coordinator
    #[derive(Clone, Debug)]
    pub struct RedisCoordinatorConfig {
        /// Redis connection URL (e.g., "redis://localhost:6379")
        pub redis_url: String,
        /// Job identifier (used as key prefix)
        pub job_id: String,
        /// This worker's unique ID
        pub worker_id: String,
        /// Expected number of workers (for coordination)
        /// If None, uses the count of workers with active heartbeats
        pub expected_workers: Option<usize>,
        /// Heartbeat interval
        pub heartbeat_interval: Duration,
        /// Heartbeat TTL (worker considered dead if no heartbeat within this time)
        pub heartbeat_ttl: Duration,
        /// Lock TTL for distributed locks
        pub lock_ttl: Duration,
        /// Timeout for waiting on coordination
        pub coordination_timeout: Duration,
        /// Remaining time budget for serverless (triggers early checkpoint)
        pub remaining_time_budget: Option<Duration>,
        /// How often to check for worker failures during coordination
        pub failure_check_interval: Duration,
        /// Whether to abort checkpoint on any worker failure
        pub abort_on_worker_failure: bool,
    }

    impl RedisCoordinatorConfig {
        /// Create a new Redis coordinator config
        ///
        /// # Arguments
        /// * `redis_url` - Redis connection URL (e.g., "redis://localhost:6379")
        /// * `job_id` - Unique identifier for this job (used as key prefix)
        /// * `worker_id` - Unique identifier for this worker
        pub fn new(
            redis_url: impl Into<String>,
            job_id: impl Into<String>,
            worker_id: impl Into<String>,
        ) -> Self {
            Self {
                redis_url: redis_url.into(),
                job_id: job_id.into(),
                worker_id: worker_id.into(),
                expected_workers: None,
                heartbeat_interval: Duration::from_secs(5),
                heartbeat_ttl: Duration::from_secs(30),
                lock_ttl: Duration::from_secs(60),
                coordination_timeout: Duration::from_secs(300),
                remaining_time_budget: None,
                failure_check_interval: Duration::from_millis(500),
                abort_on_worker_failure: true,
            }
        }

        /// Create config optimized for serverless/Lambda environments
        ///
        /// Uses faster failure detection suitable for NAT hole-punching scenarios
        pub fn for_serverless(
            redis_url: impl Into<String>,
            job_id: impl Into<String>,
            worker_id: impl Into<String>,
        ) -> Self {
            Self {
                redis_url: redis_url.into(),
                job_id: job_id.into(),
                worker_id: worker_id.into(),
                expected_workers: None,
                heartbeat_interval: Duration::from_secs(2),
                heartbeat_ttl: Duration::from_secs(10),
                lock_ttl: Duration::from_secs(30),
                coordination_timeout: Duration::from_secs(60),
                remaining_time_budget: None,
                failure_check_interval: Duration::from_millis(200),
                abort_on_worker_failure: true,
            }
        }

        /// Set expected number of workers
        /// If not set, coordination will wait for all workers with active heartbeats
        pub fn with_expected_workers(mut self, count: usize) -> Self {
            self.expected_workers = Some(count);
            self
        }

        /// Set heartbeat interval
        pub fn with_heartbeat_interval(mut self, interval: Duration) -> Self {
            self.heartbeat_interval = interval;
            self
        }

        /// Set heartbeat TTL
        pub fn with_heartbeat_ttl(mut self, ttl: Duration) -> Self {
            self.heartbeat_ttl = ttl;
            self
        }

        /// Set coordination timeout
        pub fn with_coordination_timeout(mut self, timeout: Duration) -> Self {
            self.coordination_timeout = timeout;
            self
        }

        /// Set lock TTL
        pub fn with_lock_ttl(mut self, ttl: Duration) -> Self {
            self.lock_ttl = ttl;
            self
        }

        /// Set remaining time budget (for serverless environments)
        /// When remaining time drops below this, checkpoint will be triggered
        pub fn with_time_budget(mut self, remaining: Duration) -> Self {
            self.remaining_time_budget = Some(remaining);
            self
        }

        /// Set failure check interval during coordination
        pub fn with_failure_check_interval(mut self, interval: Duration) -> Self {
            self.failure_check_interval = interval;
            self
        }

        /// Set whether to abort checkpoint on worker failure
        pub fn with_abort_on_worker_failure(mut self, abort: bool) -> Self {
            self.abort_on_worker_failure = abort;
            self
        }
    }

    /// Redis-based checkpoint coordinator.
    ///
    /// Uses Redis for distributed coordination. This coordinator works for any
    /// distributed environment:
    ///
    /// - HPC clusters (as an alternative to MPI)
    /// - Cloud VMs
    /// - Kubernetes pods
    /// - Serverless functions (Lambda, Cloud Functions)
    ///
    /// # Features
    ///
    /// - Distributed voting for checkpoint decisions
    /// - Worker heartbeats with automatic cleanup
    /// - Distributed locking for safe coordination
    /// - Barrier-like synchronization via Redis
    /// - Time budget awareness for serverless
    pub struct RedisCoordinator {
        /// Redis client
        client: Client,
        /// Multiplexed connection (shared across async tasks)
        connection: RwLock<Option<MultiplexedConnection>>,
        /// Configuration
        config: RedisCoordinatorConfig,
        /// Worker ID as WorkerId type
        worker_id: WorkerId,
        /// Latest known committed checkpoint
        latest_committed: AtomicU64,
    }

    impl RedisCoordinator {
        /// Create a new Redis coordinator
        pub async fn new(config: RedisCoordinatorConfig) -> CylonResult<Self> {
            let client = Client::open(config.redis_url.as_str()).map_err(|e| {
                CylonError::new(
                    Code::IoError,
                    format!("Failed to create Redis client: {}", e),
                )
            })?;

            let connection = client.get_multiplexed_async_connection().await.map_err(|e| {
                CylonError::new(
                    Code::IoError,
                    format!("Failed to connect to Redis: {}", e),
                )
            })?;

            let worker_id = WorkerId::Serverless {
                worker_id: config.worker_id.clone(),
            };

            let coordinator = Self {
                client,
                connection: RwLock::new(Some(connection)),
                config,
                worker_id,
                latest_committed: AtomicU64::new(0),
            };

            // Register this worker
            coordinator.register_worker().await?;

            Ok(coordinator)
        }

        /// Get a Redis connection
        async fn get_connection(&self) -> CylonResult<MultiplexedConnection> {
            let conn = self.connection.read().await;
            if let Some(ref c) = *conn {
                return Ok(c.clone());
            }
            drop(conn);

            // Reconnect
            let mut conn = self.connection.write().await;
            let new_conn = self.client.get_multiplexed_async_connection().await.map_err(|e| {
                CylonError::new(
                    Code::IoError,
                    format!("Failed to reconnect to Redis: {}", e),
                )
            })?;
            *conn = Some(new_conn.clone());
            Ok(new_conn)
        }

        /// Get key prefix for this job
        fn key_prefix(&self) -> String {
            format!("cylon:{}", self.config.job_id)
        }

        /// Register this worker as active
        async fn register_worker(&self) -> CylonResult<()> {
            let mut conn = self.get_connection().await?;
            let prefix = self.key_prefix();
            let worker_key = format!("{}:workers", prefix);

            conn.sadd::<_, _, ()>(&worker_key, &self.config.worker_id)
                .await
                .map_err(|e| {
                    CylonError::new(Code::IoError, format!("Failed to register worker: {}", e))
                })?;

            // Send initial heartbeat
            self.send_heartbeat().await?;

            Ok(())
        }

        /// Send a heartbeat to indicate this worker is alive
        async fn send_heartbeat(&self) -> CylonResult<()> {
            let mut conn = self.get_connection().await?;
            let prefix = self.key_prefix();
            let heartbeat_key = format!("{}:worker:{}:heartbeat", prefix, self.config.worker_id);

            conn.set_ex::<_, _, ()>(
                &heartbeat_key,
                "alive",
                self.config.heartbeat_ttl.as_secs(),
            )
            .await
            .map_err(|e| {
                CylonError::new(Code::IoError, format!("Failed to send heartbeat: {}", e))
            })?;

            Ok(())
        }

        /// Get the count of active workers (with valid heartbeats)
        pub async fn get_active_worker_count(&self) -> CylonResult<usize> {
            let mut conn = self.get_connection().await?;
            let prefix = self.key_prefix();
            let worker_key = format!("{}:workers", prefix);

            // Get all registered workers
            let workers: Vec<String> = conn.smembers(&worker_key).await.map_err(|e| {
                CylonError::new(Code::IoError, format!("Failed to get workers: {}", e))
            })?;

            // Check which ones have valid heartbeats
            let mut active_count = 0;
            for worker in workers {
                let heartbeat_key = format!("{}:worker:{}:heartbeat", prefix, worker);
                let exists: bool = conn.exists(&heartbeat_key).await.unwrap_or(false);
                if exists {
                    active_count += 1;
                }
            }

            Ok(active_count)
        }

        /// Try to acquire a distributed lock
        pub async fn acquire_lock(&self, resource: &str, ttl: Duration) -> CylonResult<bool> {
            let mut conn = self.get_connection().await?;
            let prefix = self.key_prefix();
            let lock_key = format!("{}:lock:{}", prefix, resource);

            // Use SET NX EX for atomic lock acquisition
            let result: Option<String> = redis::cmd("SET")
                .arg(&lock_key)
                .arg(&self.config.worker_id)
                .arg("NX")
                .arg("EX")
                .arg(ttl.as_secs())
                .query_async(&mut conn)
                .await
                .map_err(|e| {
                    CylonError::new(Code::IoError, format!("Failed to acquire lock: {}", e))
                })?;

            Ok(result.is_some())
        }

        /// Release a distributed lock (only if we own it)
        pub async fn release_lock(&self, resource: &str) -> CylonResult<()> {
            let mut conn = self.get_connection().await?;
            let prefix = self.key_prefix();
            let lock_key = format!("{}:lock:{}", prefix, resource);

            // Only release if we own the lock (using Lua script for atomicity)
            let script = redis::Script::new(
                r#"
                if redis.call("get", KEYS[1]) == ARGV[1] then
                    return redis.call("del", KEYS[1])
                else
                    return 0
                end
                "#,
            );

            script
                .key(&lock_key)
                .arg(&self.config.worker_id)
                .invoke_async::<_, i32>(&mut conn)
                .await
                .map_err(|e| {
                    CylonError::new(Code::IoError, format!("Failed to release lock: {}", e))
                })?;

            Ok(())
        }

        // ==================== Participant & Status Management ====================

        /// Register this worker as a participant in a checkpoint
        async fn join_checkpoint(&self, checkpoint_id: u64) -> CylonResult<()> {
            let mut conn = self.get_connection().await?;
            let prefix = self.key_prefix();
            let participants_key = format!("{}:checkpoint:{}:participants", prefix, checkpoint_id);

            conn.sadd::<_, _, ()>(&participants_key, &self.config.worker_id)
                .await
                .map_err(|e| {
                    CylonError::new(Code::IoError, format!("Failed to join checkpoint: {}", e))
                })?;

            // Set expiry
            conn.expire::<_, ()>(&participants_key, self.config.coordination_timeout.as_secs() as i64)
                .await
                .ok();

            Ok(())
        }

        /// Get the set of participants in a checkpoint
        async fn get_participants(&self, checkpoint_id: u64) -> CylonResult<HashSet<String>> {
            let mut conn = self.get_connection().await?;
            let prefix = self.key_prefix();
            let participants_key = format!("{}:checkpoint:{}:participants", prefix, checkpoint_id);

            let participants: Vec<String> = conn.smembers(&participants_key).await.map_err(|e| {
                CylonError::new(Code::IoError, format!("Failed to get participants: {}", e))
            })?;

            Ok(participants.into_iter().collect())
        }

        /// Set the status of a checkpoint
        async fn set_checkpoint_status(
            &self,
            checkpoint_id: u64,
            status: CheckpointCoordinationStatus,
        ) -> CylonResult<()> {
            let mut conn = self.get_connection().await?;
            let prefix = self.key_prefix();
            let status_key = format!("{}:checkpoint:{}:status", prefix, checkpoint_id);

            conn.set_ex::<_, _, ()>(
                &status_key,
                status.as_str(),
                self.config.coordination_timeout.as_secs(),
            )
            .await
            .map_err(|e| {
                CylonError::new(Code::IoError, format!("Failed to set checkpoint status: {}", e))
            })?;

            Ok(())
        }

        /// Get the status of a checkpoint
        pub async fn get_checkpoint_status(
            &self,
            checkpoint_id: u64,
        ) -> CylonResult<Option<CheckpointCoordinationStatus>> {
            let mut conn = self.get_connection().await?;
            let prefix = self.key_prefix();
            let status_key = format!("{}:checkpoint:{}:status", prefix, checkpoint_id);

            let status: Option<String> = conn.get(&status_key).await.unwrap_or(None);
            Ok(status.and_then(|s| CheckpointCoordinationStatus::from_str(&s)))
        }

        /// Signal that checkpoint should be aborted (any worker can call this)
        pub async fn signal_abort(&self, checkpoint_id: u64, reason: &str) -> CylonResult<()> {
            let mut conn = self.get_connection().await?;
            let prefix = self.key_prefix();
            let status_key = format!("{}:checkpoint:{}:status", prefix, checkpoint_id);
            let reason_key = format!("{}:checkpoint:{}:abort_reason", prefix, checkpoint_id);

            // Set status to aborted (only if not already committed)
            let script = redis::Script::new(
                r#"
                local current = redis.call("get", KEYS[1])
                if current ~= "committed" then
                    redis.call("set", KEYS[1], "aborted")
                    redis.call("expire", KEYS[1], ARGV[2])
                    redis.call("set", KEYS[2], ARGV[1])
                    redis.call("expire", KEYS[2], ARGV[2])
                    return 1
                end
                return 0
                "#,
            );

            script
                .key(&status_key)
                .key(&reason_key)
                .arg(reason)
                .arg(self.config.coordination_timeout.as_secs())
                .invoke_async::<_, i32>(&mut conn)
                .await
                .map_err(|e| {
                    CylonError::new(Code::IoError, format!("Failed to signal abort: {}", e))
                })?;

            Ok(())
        }

        // ==================== Worker Failure Detection ====================

        /// Check if a specific worker is alive (has valid heartbeat)
        pub async fn is_worker_alive(&self, worker_id: &str) -> CylonResult<bool> {
            let mut conn = self.get_connection().await?;
            let prefix = self.key_prefix();
            let heartbeat_key = format!("{}:worker:{}:heartbeat", prefix, worker_id);

            let exists: bool = conn.exists(&heartbeat_key).await.map_err(|e| {
                CylonError::new(Code::IoError, format!("Failed to check heartbeat: {}", e))
            })?;

            Ok(exists)
        }

        /// Check if any participants in a checkpoint have failed (no heartbeat)
        /// Returns the list of failed worker IDs
        async fn check_participant_failures(&self, checkpoint_id: u64) -> CylonResult<Vec<String>> {
            let participants = self.get_participants(checkpoint_id).await?;
            let mut failed = Vec::new();

            for worker_id in participants {
                if !self.is_worker_alive(&worker_id).await? {
                    failed.push(worker_id);
                }
            }

            Ok(failed)
        }

        /// Check if checkpoint should be aborted due to worker failures
        /// Returns Err if checkpoint was aborted, Ok(()) if all participants alive
        async fn check_for_failures_and_abort(&self, checkpoint_id: u64) -> CylonResult<()> {
            // First check if already aborted
            if let Some(status) = self.get_checkpoint_status(checkpoint_id).await? {
                if status == CheckpointCoordinationStatus::Aborted {
                    return Err(CylonError::new(
                        Code::ExecutionError,
                        "Checkpoint was aborted".to_string(),
                    ));
                }
            }

            // Check for participant failures if enabled
            if self.config.abort_on_worker_failure {
                let failed = self.check_participant_failures(checkpoint_id).await?;
                if !failed.is_empty() {
                    let reason = format!("Workers failed: {}", failed.join(", "));
                    self.signal_abort(checkpoint_id, &reason).await?;
                    return Err(CylonError::new(
                        Code::ExecutionError,
                        format!("Checkpoint aborted: {}", reason),
                    ));
                }
            }

            Ok(())
        }

        // ==================== Voting ====================

        /// Vote for a checkpoint (also registers as participant)
        async fn vote_for_checkpoint(&self, checkpoint_id: u64, vote: bool) -> CylonResult<()> {
            // First join as participant
            self.join_checkpoint(checkpoint_id).await?;

            // Set status to active if this is the first participant
            self.set_checkpoint_status(checkpoint_id, CheckpointCoordinationStatus::Active)
                .await
                .ok(); // Ignore if already set

            let mut conn = self.get_connection().await?;
            let prefix = self.key_prefix();
            let vote_key = format!("{}:checkpoint:{}:votes", prefix, checkpoint_id);

            let vote_value = format!("{}:{}", self.config.worker_id, if vote { "yes" } else { "no" });
            conn.sadd::<_, _, ()>(&vote_key, &vote_value)
                .await
                .map_err(|e| {
                    CylonError::new(Code::IoError, format!("Failed to record vote: {}", e))
                })?;

            // Set expiry on vote key
            conn.expire::<_, ()>(&vote_key, self.config.coordination_timeout.as_secs() as i64)
                .await
                .ok();

            Ok(())
        }

        /// Get the expected worker count (from config or active heartbeats)
        async fn get_expected_worker_count(&self) -> CylonResult<usize> {
            match self.config.expected_workers {
                Some(count) => Ok(count),
                None => self.get_active_worker_count().await,
            }
        }

        /// Get expected participant count for a checkpoint
        /// Uses participants list if available, otherwise falls back to expected_workers
        async fn get_expected_participant_count(&self, checkpoint_id: u64) -> CylonResult<usize> {
            let participants = self.get_participants(checkpoint_id).await?;
            if participants.is_empty() {
                self.get_expected_worker_count().await
            } else {
                Ok(participants.len())
            }
        }

        /// Check votes from participants
        async fn check_votes(&self, checkpoint_id: u64) -> CylonResult<(usize, usize, bool)> {
            let mut conn = self.get_connection().await?;
            let prefix = self.key_prefix();
            let vote_key = format!("{}:checkpoint:{}:votes", prefix, checkpoint_id);

            let votes: Vec<String> = conn.smembers(&vote_key).await.map_err(|e| {
                CylonError::new(Code::IoError, format!("Failed to get votes: {}", e))
            })?;

            let mut yes_count = 0;
            let mut no_count = 0;
            for vote in &votes {
                if vote.ends_with(":yes") {
                    yes_count += 1;
                } else if vote.ends_with(":no") {
                    no_count += 1;
                }
            }

            let total_votes = yes_count + no_count;
            let expected = self.get_expected_participant_count(checkpoint_id).await?;
            let all_yes = no_count == 0 && yes_count >= expected;

            Ok((yes_count, total_votes, all_yes))
        }

        /// Wait for all participants to vote (with failure detection)
        async fn wait_for_votes(&self, checkpoint_id: u64) -> CylonResult<bool> {
            let start = std::time::Instant::now();
            let timeout = self.config.coordination_timeout;
            let failure_check_interval = self.config.failure_check_interval;
            let mut last_failure_check = std::time::Instant::now();

            loop {
                // Check for failures periodically
                if last_failure_check.elapsed() >= failure_check_interval {
                    self.check_for_failures_and_abort(checkpoint_id).await?;
                    last_failure_check = std::time::Instant::now();
                }

                let expected = self.get_expected_participant_count(checkpoint_id).await?;
                let (_yes_count, total_votes, all_yes) = self.check_votes(checkpoint_id).await?;

                if total_votes >= expected {
                    return Ok(all_yes);
                }

                if start.elapsed() > timeout {
                    // Before timing out, do one final failure check
                    let failed = self.check_participant_failures(checkpoint_id).await?;
                    let reason = if !failed.is_empty() {
                        format!("Workers failed: {}", failed.join(", "))
                    } else {
                        format!("Timeout: {}/{} votes received", total_votes, expected)
                    };
                    self.signal_abort(checkpoint_id, &reason).await?;
                    return Err(CylonError::new(
                        Code::ExecutionError,
                        format!("Checkpoint aborted: {}", reason),
                    ));
                }

                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        }

        // ==================== Completion Tracking ====================

        /// Mark this worker as having completed a checkpoint
        async fn mark_checkpoint_complete(&self, checkpoint_id: u64) -> CylonResult<()> {
            // Check if aborted before marking complete
            self.check_for_failures_and_abort(checkpoint_id).await?;

            let mut conn = self.get_connection().await?;
            let prefix = self.key_prefix();
            let completed_key = format!("{}:checkpoint:{}:completed", prefix, checkpoint_id);

            conn.sadd::<_, _, ()>(&completed_key, &self.config.worker_id)
                .await
                .map_err(|e| {
                    CylonError::new(
                        Code::IoError,
                        format!("Failed to mark checkpoint complete: {}", e),
                    )
                })?;

            // Set expiry
            conn.expire::<_, ()>(&completed_key, self.config.coordination_timeout.as_secs() as i64)
                .await
                .ok();

            Ok(())
        }

        /// Wait for all participants to complete (barrier with failure detection)
        async fn wait_for_all_complete(&self, checkpoint_id: u64) -> CylonResult<()> {
            let mut conn = self.get_connection().await?;
            let prefix = self.key_prefix();
            let completed_key = format!("{}:checkpoint:{}:completed", prefix, checkpoint_id);
            let start = std::time::Instant::now();
            let timeout = self.config.coordination_timeout;
            let failure_check_interval = self.config.failure_check_interval;
            let mut last_failure_check = std::time::Instant::now();

            loop {
                // Check for failures periodically
                if last_failure_check.elapsed() >= failure_check_interval {
                    self.check_for_failures_and_abort(checkpoint_id).await?;
                    last_failure_check = std::time::Instant::now();
                }

                let count: usize = conn.scard(&completed_key).await.unwrap_or(0);
                let expected = self.get_expected_participant_count(checkpoint_id).await?;

                if count >= expected {
                    return Ok(());
                }

                if start.elapsed() > timeout {
                    // Before timing out, do one final failure check
                    let failed = self.check_participant_failures(checkpoint_id).await?;
                    let reason = if !failed.is_empty() {
                        format!("Workers failed: {}", failed.join(", "))
                    } else {
                        format!("Timeout: {}/{} workers completed", count, expected)
                    };
                    self.signal_abort(checkpoint_id, &reason).await?;
                    return Err(CylonError::new(
                        Code::ExecutionError,
                        format!("Checkpoint aborted: {}", reason),
                    ));
                }

                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        }

        /// Update the latest committed checkpoint
        async fn update_latest_checkpoint(&self, checkpoint_id: u64) -> CylonResult<()> {
            let mut conn = self.get_connection().await?;
            let prefix = self.key_prefix();
            let latest_key = format!("{}:checkpoint:latest", prefix);

            // Only update if this is newer (using Lua script for atomicity)
            let script = redis::Script::new(
                r#"
                local current = tonumber(redis.call("get", KEYS[1])) or 0
                if tonumber(ARGV[1]) > current then
                    redis.call("set", KEYS[1], ARGV[1])
                    return 1
                end
                return 0
                "#,
            );

            script
                .key(&latest_key)
                .arg(checkpoint_id)
                .invoke_async::<_, i32>(&mut conn)
                .await
                .map_err(|e| {
                    CylonError::new(
                        Code::IoError,
                        format!("Failed to update latest checkpoint: {}", e),
                    )
                })?;

            self.latest_committed.store(checkpoint_id, Ordering::SeqCst);

            Ok(())
        }

        /// Get the latest committed checkpoint from Redis
        async fn get_latest_checkpoint(&self) -> CylonResult<Option<u64>> {
            let mut conn = self.get_connection().await?;
            let prefix = self.key_prefix();
            let latest_key = format!("{}:checkpoint:latest", prefix);

            let latest: Option<u64> = conn.get(&latest_key).await.unwrap_or(None);
            Ok(latest)
        }

        /// Cleanup old checkpoint coordination data
        pub async fn cleanup_checkpoint(&self, checkpoint_id: u64) -> CylonResult<()> {
            let mut conn = self.get_connection().await?;
            let prefix = self.key_prefix();

            let keys_to_delete = vec![
                format!("{}:checkpoint:{}:participants", prefix, checkpoint_id),
                format!("{}:checkpoint:{}:votes", prefix, checkpoint_id),
                format!("{}:checkpoint:{}:completed", prefix, checkpoint_id),
                format!("{}:checkpoint:{}:status", prefix, checkpoint_id),
                format!("{}:checkpoint:{}:abort_reason", prefix, checkpoint_id),
            ];

            for key in keys_to_delete {
                conn.del::<_, ()>(&key).await.ok();
            }

            Ok(())
        }

        /// Get the abort reason for a checkpoint (if aborted)
        pub async fn get_abort_reason(&self, checkpoint_id: u64) -> CylonResult<Option<String>> {
            let mut conn = self.get_connection().await?;
            let prefix = self.key_prefix();
            let reason_key = format!("{}:checkpoint:{}:abort_reason", prefix, checkpoint_id);

            let reason: Option<String> = conn.get(&reason_key).await.unwrap_or(None);
            Ok(reason)
        }

        /// Unregister this worker (call on shutdown)
        pub async fn unregister(&self) -> CylonResult<()> {
            let mut conn = self.get_connection().await?;
            let prefix = self.key_prefix();
            let worker_key = format!("{}:workers", prefix);
            let heartbeat_key = format!("{}:worker:{}:heartbeat", prefix, self.config.worker_id);

            conn.srem::<_, _, ()>(&worker_key, &self.config.worker_id)
                .await
                .ok();
            conn.del::<_, ()>(&heartbeat_key).await.ok();

            Ok(())
        }

        /// Get the configuration
        pub fn config(&self) -> &RedisCoordinatorConfig {
            &self.config
        }

        /// Update the remaining time budget (for serverless)
        pub fn set_remaining_time_budget(&mut self, remaining: Duration) {
            self.config.remaining_time_budget = Some(remaining);
        }

        /// Try to become leader (using distributed lock)
        pub async fn try_become_leader(&self) -> CylonResult<bool> {
            self.acquire_lock("leader", self.config.lock_ttl).await
        }

        /// Release leader role
        pub async fn release_leader(&self) -> CylonResult<()> {
            self.release_lock("leader").await
        }
    }

    #[async_trait]
    impl CheckpointCoordinator for RedisCoordinator {
        fn worker_id(&self) -> WorkerId {
            self.worker_id.clone()
        }

        fn world_size(&self) -> usize {
            // Return expected_workers if set, otherwise return 1 as a default
            // For dynamic worker counts, use get_active_worker_count() instead
            self.config.expected_workers.unwrap_or(1)
        }

        fn should_checkpoint(&self, context: &CheckpointContext) -> bool {
            // Check time budget (for serverless)
            if let Some(remaining) = context.remaining_time_budget {
                if remaining < Duration::from_secs(30) {
                    return true;
                }
            }

            // Also check config's time budget
            if let Some(remaining) = self.config.remaining_time_budget {
                if remaining < Duration::from_secs(30) {
                    return true;
                }
            }

            context.operations_since_checkpoint >= 100
                || context.bytes_since_checkpoint >= 100 * 1024 * 1024
        }

        async fn begin_checkpoint(&self, checkpoint_id: u64) -> CylonResult<CheckpointDecision> {
            // Vote to proceed
            self.vote_for_checkpoint(checkpoint_id, true).await?;

            // Wait for all workers to vote
            let all_yes = self.wait_for_votes(checkpoint_id).await?;

            if all_yes {
                Ok(CheckpointDecision::Proceed(CheckpointPriority::Medium))
            } else {
                Ok(CheckpointDecision::Skip)
            }
        }

        async fn commit_checkpoint(&self, checkpoint_id: u64) -> CylonResult<()> {
            // Mark this worker as complete
            self.mark_checkpoint_complete(checkpoint_id).await?;

            // Wait for all workers to complete (barrier)
            self.wait_for_all_complete(checkpoint_id).await?;

            // Set status to committed
            self.set_checkpoint_status(checkpoint_id, CheckpointCoordinationStatus::Committed)
                .await?;

            // Update latest checkpoint
            self.update_latest_checkpoint(checkpoint_id).await?;

            Ok(())
        }

        async fn abort_checkpoint(&self, checkpoint_id: u64) -> CylonResult<()> {
            // Signal abort to other workers
            self.signal_abort(checkpoint_id, "Explicitly aborted").await?;
            // Cleanup coordination data
            self.cleanup_checkpoint(checkpoint_id).await?;
            Ok(())
        }

        async fn find_latest_checkpoint(&self) -> CylonResult<Option<u64>> {
            self.get_latest_checkpoint().await
        }

        async fn heartbeat(&self) -> CylonResult<()> {
            self.send_heartbeat().await
        }

        async fn claim_work(&self, work_unit_id: &str) -> CylonResult<bool> {
            // Try to acquire a lock on the work unit
            self.acquire_lock(&format!("work:{}", work_unit_id), self.config.lock_ttl)
                .await
        }

        fn is_leader(&self) -> bool {
            // With Redis coordination, leadership is dynamic via distributed lock
            // This returns false by default; use try_become_leader() for dynamic election
            false
        }
    }
}

#[cfg(feature = "redis")]
pub use redis_coordinator::{CheckpointCoordinationStatus, RedisCoordinator, RedisCoordinatorConfig};

// Re-export DistributedCoordinator as the main coordinator for backwards compatibility
// with any code that might have referenced MpiCoordinator
pub use DistributedCoordinator as MpiCoordinator;
