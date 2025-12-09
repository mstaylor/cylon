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
    //! This module provides coordination for both HPC and serverless environments
    //! using Redis as a central coordination point. It's particularly useful when:
    //!
    //! - MPI is not available or desired
    //! - Workers are ephemeral (serverless functions)
    //! - You want a simpler deployment than MPI
    //! - You need coordination across heterogeneous systems
    //!
    //! # Key Structure
    //!
    //! ```text
    //! cylon:{job_id}:workers                    - Set of active worker IDs
    //! cylon:{job_id}:worker:{worker_id}:heartbeat - Worker heartbeat (with TTL)
    //! cylon:{job_id}:checkpoint:current         - Current checkpoint being coordinated
    //! cylon:{job_id}:checkpoint:{id}:votes      - Set of worker votes for checkpoint
    //! cylon:{job_id}:checkpoint:{id}:status     - Checkpoint status (pending/committed/aborted)
    //! cylon:{job_id}:checkpoint:{id}:workers    - Set of workers that completed checkpoint
    //! cylon:{job_id}:checkpoint:latest          - Latest committed checkpoint ID
    //! cylon:{job_id}:lock:{resource}            - Distributed lock
    //! ```

    use super::*;
    use redis::aio::MultiplexedConnection;
    use redis::{AsyncCommands, Client};
    use std::time::Duration;
    use tokio::sync::RwLock;

    /// Configuration for Redis coordinator
    #[derive(Clone, Debug)]
    pub struct RedisCoordinatorConfig {
        /// Redis connection URL (e.g., "redis://localhost:6379")
        pub redis_url: String,
        /// Job identifier (used as key prefix)
        pub job_id: String,
        /// This worker's unique ID
        pub worker_id: String,
        /// Worker rank (0-based, for HPC compatibility)
        pub rank: i32,
        /// Expected number of workers (for coordination)
        pub expected_workers: usize,
        /// Heartbeat interval
        pub heartbeat_interval: Duration,
        /// Heartbeat TTL (worker considered dead if no heartbeat within this time)
        pub heartbeat_ttl: Duration,
        /// Lock TTL for distributed locks
        pub lock_ttl: Duration,
        /// Timeout for waiting on coordination
        pub coordination_timeout: Duration,
        /// Whether this is a serverless environment
        pub serverless: bool,
    }

    impl RedisCoordinatorConfig {
        /// Create a new config for HPC-style environment (fixed workers with ranks)
        pub fn hpc(
            redis_url: impl Into<String>,
            job_id: impl Into<String>,
            rank: i32,
            world_size: usize,
        ) -> Self {
            let job_id = job_id.into();
            Self {
                redis_url: redis_url.into(),
                worker_id: format!("rank_{}", rank),
                job_id,
                rank,
                expected_workers: world_size,
                heartbeat_interval: Duration::from_secs(5),
                heartbeat_ttl: Duration::from_secs(30),
                lock_ttl: Duration::from_secs(60),
                coordination_timeout: Duration::from_secs(300),
                serverless: false,
            }
        }

        /// Create a new config for serverless environment (dynamic workers)
        pub fn serverless(
            redis_url: impl Into<String>,
            job_id: impl Into<String>,
            worker_id: impl Into<String>,
            expected_workers: usize,
        ) -> Self {
            Self {
                redis_url: redis_url.into(),
                job_id: job_id.into(),
                worker_id: worker_id.into(),
                rank: -1, // No fixed rank in serverless
                expected_workers,
                heartbeat_interval: Duration::from_secs(5),
                heartbeat_ttl: Duration::from_secs(30),
                lock_ttl: Duration::from_secs(60),
                coordination_timeout: Duration::from_secs(300),
                serverless: true,
            }
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
    }

    /// Redis-based checkpoint coordinator.
    ///
    /// Uses Redis for distributed coordination in both HPC and serverless
    /// environments. This coordinator is a good choice when:
    ///
    /// - You don't want to use MPI for coordination
    /// - Workers may be ephemeral (Lambda, Cloud Functions, Kubernetes Jobs)
    /// - You need coordination across different machine types
    /// - You want Redis as a unified coordination/storage backend
    ///
    /// # Features
    ///
    /// - Distributed voting for checkpoint decisions
    /// - Worker heartbeats with automatic cleanup
    /// - Distributed locking for safe coordination
    /// - Support for both fixed (HPC) and dynamic (serverless) worker counts
    /// - Barrier-like synchronization via Redis
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

            let worker_id = if config.serverless {
                WorkerId::Serverless {
                    worker_id: config.worker_id.clone(),
                }
            } else {
                WorkerId::Rank(config.rank)
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

        /// Create a coordinator for HPC environment
        pub async fn for_hpc(
            redis_url: impl Into<String>,
            job_id: impl Into<String>,
            rank: i32,
            world_size: usize,
        ) -> CylonResult<Self> {
            let config = RedisCoordinatorConfig::hpc(redis_url, job_id, rank, world_size);
            Self::new(config).await
        }

        /// Create a coordinator for serverless environment
        pub async fn for_serverless(
            redis_url: impl Into<String>,
            job_id: impl Into<String>,
            worker_id: impl Into<String>,
            expected_workers: usize,
        ) -> CylonResult<Self> {
            let config = RedisCoordinatorConfig::serverless(redis_url, job_id, worker_id, expected_workers);
            Self::new(config).await
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

        /// Vote for a checkpoint
        async fn vote_for_checkpoint(&self, checkpoint_id: u64, vote: bool) -> CylonResult<()> {
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

        /// Check if all expected workers have voted yes
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
            let all_yes = no_count == 0 && yes_count >= self.config.expected_workers;

            Ok((yes_count, total_votes, all_yes))
        }

        /// Wait for all workers to vote
        async fn wait_for_votes(&self, checkpoint_id: u64) -> CylonResult<bool> {
            let start = std::time::Instant::now();
            let timeout = self.config.coordination_timeout;

            loop {
                let (yes_count, total_votes, all_yes) = self.check_votes(checkpoint_id).await?;

                if total_votes >= self.config.expected_workers {
                    return Ok(all_yes);
                }

                if start.elapsed() > timeout {
                    return Err(CylonError::new(
                        Code::ExecutionError,
                        format!(
                            "Timeout waiting for checkpoint votes: {}/{} received",
                            total_votes, self.config.expected_workers
                        ),
                    ));
                }

                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        }

        /// Mark this worker as having completed a checkpoint
        async fn mark_checkpoint_complete(&self, checkpoint_id: u64) -> CylonResult<()> {
            let mut conn = self.get_connection().await?;
            let prefix = self.key_prefix();
            let workers_key = format!("{}:checkpoint:{}:workers", prefix, checkpoint_id);

            conn.sadd::<_, _, ()>(&workers_key, &self.config.worker_id)
                .await
                .map_err(|e| {
                    CylonError::new(
                        Code::IoError,
                        format!("Failed to mark checkpoint complete: {}", e),
                    )
                })?;

            // Set expiry
            conn.expire::<_, ()>(&workers_key, self.config.coordination_timeout.as_secs() as i64)
                .await
                .ok();

            Ok(())
        }

        /// Wait for all workers to complete a checkpoint (barrier)
        async fn wait_for_all_complete(&self, checkpoint_id: u64) -> CylonResult<()> {
            let mut conn = self.get_connection().await?;
            let prefix = self.key_prefix();
            let workers_key = format!("{}:checkpoint:{}:workers", prefix, checkpoint_id);
            let start = std::time::Instant::now();
            let timeout = self.config.coordination_timeout;

            loop {
                let count: usize = conn.scard(&workers_key).await.unwrap_or(0);

                if count >= self.config.expected_workers {
                    return Ok(());
                }

                if start.elapsed() > timeout {
                    return Err(CylonError::new(
                        Code::ExecutionError,
                        format!(
                            "Timeout waiting for checkpoint completion: {}/{} workers",
                            count, self.config.expected_workers
                        ),
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
                format!("{}:checkpoint:{}:votes", prefix, checkpoint_id),
                format!("{}:checkpoint:{}:status", prefix, checkpoint_id),
                format!("{}:checkpoint:{}:workers", prefix, checkpoint_id),
            ];

            for key in keys_to_delete {
                conn.del::<_, ()>(&key).await.ok();
            }

            Ok(())
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
    }

    #[async_trait]
    impl CheckpointCoordinator for RedisCoordinator {
        fn worker_id(&self) -> WorkerId {
            self.worker_id.clone()
        }

        fn world_size(&self) -> usize {
            self.config.expected_workers
        }

        fn should_checkpoint(&self, context: &CheckpointContext) -> bool {
            // For serverless, also consider remaining time budget
            if self.config.serverless {
                if let Some(remaining) = context.remaining_time_budget {
                    if remaining < Duration::from_secs(30) {
                        return true;
                    }
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

            // Update latest checkpoint
            self.update_latest_checkpoint(checkpoint_id).await?;

            Ok(())
        }

        async fn abort_checkpoint(&self, checkpoint_id: u64) -> CylonResult<()> {
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
            // In HPC mode, rank 0 is leader
            // In serverless mode, there's no fixed leader
            if self.config.serverless {
                false
            } else {
                self.config.rank == 0
            }
        }
    }
}

#[cfg(feature = "redis")]
pub use redis_coordinator::{RedisCoordinator, RedisCoordinatorConfig};

// Re-export DistributedCoordinator as the main coordinator for backwards compatibility
// with any code that might have referenced MpiCoordinator
pub use DistributedCoordinator as MpiCoordinator;
