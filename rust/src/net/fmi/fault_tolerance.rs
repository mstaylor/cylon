// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Fault tolerance for distributed operations
//!
//! This module provides automatic fault detection and recovery for serverless
//! environments where workers can die unexpectedly (e.g., Lambda 15-minute timeout).
//!
//! # Components
//!
//! - [`HeartbeatWatcher`]: Background thread that monitors peer heartbeats
//! - [`WorkerPool`]: Manages dynamic worker membership and partition assignment
//! - [`ResilientExecutor`]: Wraps operations with automatic retry and recovery
//!
//! # Usage
//!
//! ```ignore
//! let executor = ResilientExecutor::new(coordinator, checkpoint_mgr, config);
//! executor.worker_pool().join()?;
//!
//! let result = executor.execute("shuffle", || {
//!     shuffle_data(&channel, &partitions)
//! })?;
//! // If peer dies, executor automatically restores and retries
//! ```

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::error::{CylonError, CylonResult, Code};

#[cfg(feature = "redis")]
use crate::checkpoint::RedisCoordinator;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for fault tolerance
#[derive(Clone, Debug)]
pub struct FaultToleranceConfig {
    /// How often to check heartbeats in background thread
    pub heartbeat_check_interval: Duration,
    /// Timeout for operations
    pub operation_timeout: Duration,
    /// Maximum retry attempts for transient failures
    pub max_retries: u32,
    /// Initial backoff duration for retries
    pub initial_backoff: Duration,
    /// Maximum backoff duration
    pub max_backoff: Duration,
    /// Time reserved for checkpoint before Lambda timeout
    pub checkpoint_reserve: Duration,
    /// Estimated time for a single operation
    pub estimated_op_time: Duration,
}

impl Default for FaultToleranceConfig {
    fn default() -> Self {
        Self {
            heartbeat_check_interval: Duration::from_millis(200),
            operation_timeout: Duration::from_secs(30),
            max_retries: 3,
            initial_backoff: Duration::from_millis(100),
            max_backoff: Duration::from_secs(5),
            checkpoint_reserve: Duration::from_secs(20),
            estimated_op_time: Duration::from_secs(30),
        }
    }
}

impl FaultToleranceConfig {
    /// Configuration optimized for serverless environments
    pub fn for_serverless() -> Self {
        Self::default()
    }

    /// Set heartbeat check interval
    pub fn with_heartbeat_check_interval(mut self, interval: Duration) -> Self {
        self.heartbeat_check_interval = interval;
        self
    }

    /// Set operation timeout
    pub fn with_operation_timeout(mut self, timeout: Duration) -> Self {
        self.operation_timeout = timeout;
        self
    }

    /// Set max retries
    pub fn with_max_retries(mut self, retries: u32) -> Self {
        self.max_retries = retries;
        self
    }
}

// ============================================================================
// HeartbeatWatcher
// ============================================================================

/// Watches peer heartbeats in a background thread
///
/// The watcher periodically checks Redis for peer heartbeats and sets
/// an atomic flag when any peer is detected as dead. The progress loop
/// can check this flag instantly without any blocking I/O.
pub struct HeartbeatWatcher {
    /// Flag set when any expected peer is detected dead
    peer_failed: Arc<AtomicBool>,
    /// Flag to signal abort to all workers
    abort_signaled: Arc<AtomicBool>,
    /// Abort reason (if any)
    abort_reason: Arc<RwLock<Option<String>>>,
    /// List of dead peers
    dead_peers: Arc<RwLock<Vec<String>>>,
    /// Expected peers to monitor
    expected_peers: Arc<RwLock<Vec<String>>>,
    /// This worker's ID
    worker_id: String,
    /// Stop signal for background thread
    stop: Arc<AtomicBool>,
    /// Background thread handle
    thread_handle: Mutex<Option<JoinHandle<()>>>,
    /// Configuration
    config: FaultToleranceConfig,
}

impl HeartbeatWatcher {
    /// Create a new heartbeat watcher
    pub fn new(worker_id: String, config: FaultToleranceConfig) -> Self {
        Self {
            peer_failed: Arc::new(AtomicBool::new(false)),
            abort_signaled: Arc::new(AtomicBool::new(false)),
            abort_reason: Arc::new(RwLock::new(None)),
            dead_peers: Arc::new(RwLock::new(Vec::new())),
            expected_peers: Arc::new(RwLock::new(Vec::new())),
            worker_id,
            stop: Arc::new(AtomicBool::new(false)),
            thread_handle: Mutex::new(None),
            config,
        }
    }

    /// Set the expected peers to monitor
    pub fn set_expected_peers(&self, peers: Vec<String>) {
        let filtered: Vec<String> = peers
            .into_iter()
            .filter(|p| p != &self.worker_id)
            .collect();
        *self.expected_peers.write().unwrap() = filtered;
    }

    /// Check if any peer has failed (instant, no I/O)
    #[inline]
    pub fn has_peer_failed(&self) -> bool {
        self.peer_failed.load(Ordering::Relaxed)
    }

    /// Check if abort has been signaled (instant, no I/O)
    #[inline]
    pub fn is_abort_signaled(&self) -> bool {
        self.abort_signaled.load(Ordering::Relaxed)
    }

    /// Get list of dead peers
    pub fn get_dead_peers(&self) -> Vec<String> {
        self.dead_peers.read().unwrap().clone()
    }

    /// Get abort reason if any
    pub fn get_abort_reason(&self) -> Option<String> {
        self.abort_reason.read().unwrap().clone()
    }

    /// Signal abort (can be called by any worker)
    pub fn signal_abort(&self, reason: &str) {
        *self.abort_reason.write().unwrap() = Some(reason.to_string());
        self.abort_signaled.store(true, Ordering::SeqCst);
    }

    /// Reset the failure state (for retry after recovery)
    pub fn reset(&self) {
        self.peer_failed.store(false, Ordering::SeqCst);
        self.abort_signaled.store(false, Ordering::SeqCst);
        *self.abort_reason.write().unwrap() = None;
        self.dead_peers.write().unwrap().clear();
    }

    /// Mark specific peers as dead (called by background thread)
    fn mark_peers_dead(&self, peers: Vec<String>) {
        if !peers.is_empty() {
            let mut dead = self.dead_peers.write().unwrap();
            for peer in peers {
                if !dead.contains(&peer) {
                    dead.push(peer);
                }
            }
            self.peer_failed.store(true, Ordering::SeqCst);
        }
    }

    /// Start the background heartbeat watching thread
    #[cfg(feature = "redis")]
    pub fn start(&self, coordinator: Arc<RedisCoordinator>) {
        let peer_failed = self.peer_failed.clone();
        let dead_peers = self.dead_peers.clone();
        let expected_peers = self.expected_peers.clone();
        let stop = self.stop.clone();
        let check_interval = self.config.heartbeat_check_interval;

        let handle = thread::spawn(move || {
            // Create a tokio runtime for async Redis calls
            let rt = match tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                Ok(rt) => rt,
                Err(e) => {
                    log::error!("Failed to create tokio runtime for heartbeat watcher: {}", e);
                    return;
                }
            };

            log::debug!("Heartbeat watcher thread started");

            while !stop.load(Ordering::Relaxed) {
                let peers_to_check: Vec<String> = expected_peers.read().unwrap().clone();

                if !peers_to_check.is_empty() {
                    let check_result = rt.block_on(async {
                        let mut dead = Vec::new();
                        for peer in &peers_to_check {
                            match coordinator.is_worker_alive(peer).await {
                                Ok(true) => {}
                                Ok(false) => dead.push(peer.clone()),
                                Err(e) => {
                                    log::warn!("Failed to check heartbeat for {}: {}", peer, e);
                                }
                            }
                        }
                        dead
                    });

                    if !check_result.is_empty() {
                        log::warn!("Detected dead peers: {:?}", check_result);
                        let mut dead = dead_peers.write().unwrap();
                        for peer in check_result {
                            if !dead.contains(&peer) {
                                dead.push(peer);
                            }
                        }
                        peer_failed.store(true, Ordering::SeqCst);
                    }
                }

                thread::sleep(check_interval);
            }

            log::debug!("Heartbeat watcher thread stopping");
        });

        *self.thread_handle.lock().unwrap() = Some(handle);
    }

    /// Stop the background thread
    pub fn stop(&self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(handle) = self.thread_handle.lock().unwrap().take() {
            let _ = handle.join();
        }
    }

    /// Check for failure in progress loop (instant, no I/O)
    pub fn check_for_failure(&self) -> CylonResult<()> {
        if self.is_abort_signaled() {
            let reason = self.get_abort_reason().unwrap_or_else(|| "Unknown".to_string());
            return Err(CylonError::new(
                Code::ExecutionError,
                format!("Operation aborted: {}", reason),
            ));
        }

        if self.has_peer_failed() {
            let dead = self.get_dead_peers();
            return Err(CylonError::new(
                Code::ExecutionError,
                format!("Peer(s) failed: {}", dead.join(", ")),
            ));
        }

        Ok(())
    }
}

impl Drop for HeartbeatWatcher {
    fn drop(&mut self) {
        self.stop();
    }
}

// ============================================================================
// WorkerPool
// ============================================================================

/// Manages dynamic worker membership and partition assignment
pub struct WorkerPool {
    /// Current epoch (increments on membership change)
    epoch: AtomicU64,
    /// Active workers in current epoch
    workers: RwLock<Vec<String>>,
    /// Partition assignments: worker_id -> partition_ids
    partitions: RwLock<HashMap<String, Vec<u32>>>,
    /// Total number of partitions
    total_partitions: RwLock<u32>,
    /// This worker's ID
    worker_id: String,
}

impl WorkerPool {
    /// Create a new worker pool
    pub fn new(worker_id: String) -> Self {
        Self {
            epoch: AtomicU64::new(0),
            workers: RwLock::new(Vec::new()),
            partitions: RwLock::new(HashMap::new()),
            total_partitions: RwLock::new(0),
            worker_id,
        }
    }

    /// Get current epoch
    pub fn epoch(&self) -> u64 {
        self.epoch.load(Ordering::SeqCst)
    }

    /// Get active workers
    pub fn workers(&self) -> Vec<String> {
        self.workers.read().unwrap().clone()
    }

    /// Get this worker's partitions
    pub fn my_partitions(&self) -> Vec<u32> {
        self.partitions
            .read()
            .unwrap()
            .get(&self.worker_id)
            .cloned()
            .unwrap_or_default()
    }

    /// Get all partition assignments
    pub fn all_partitions(&self) -> HashMap<String, Vec<u32>> {
        self.partitions.read().unwrap().clone()
    }

    /// Initialize with workers and partition count
    pub fn initialize(&self, workers: Vec<String>, num_partitions: u32) {
        *self.total_partitions.write().unwrap() = num_partitions;
        self.set_workers(workers);
    }

    /// Set active workers and rebalance partitions
    pub fn set_workers(&self, workers: Vec<String>) {
        let num_partitions = *self.total_partitions.read().unwrap();

        // Calculate partition assignment
        let mut assignments: HashMap<String, Vec<u32>> = HashMap::new();
        if !workers.is_empty() {
            for (i, partition) in (0..num_partitions).enumerate() {
                let worker_idx = i % workers.len();
                let worker = &workers[worker_idx];
                assignments
                    .entry(worker.clone())
                    .or_insert_with(Vec::new)
                    .push(partition);
            }
        }

        *self.workers.write().unwrap() = workers;
        *self.partitions.write().unwrap() = assignments;
        self.epoch.fetch_add(1, Ordering::SeqCst);
    }

    /// Rebalance after worker failure
    pub fn rebalance_after_failure(&self, dead_workers: &[String]) -> CylonResult<()> {
        let mut workers = self.workers.write().unwrap();
        let mut partitions = self.partitions.write().unwrap();

        // Collect orphaned partitions from dead workers
        let mut orphaned: Vec<u32> = Vec::new();
        for dead in dead_workers {
            workers.retain(|w| w != dead);
            if let Some(parts) = partitions.remove(dead) {
                orphaned.extend(parts);
            }
        }

        // Redistribute orphaned partitions round-robin
        if !workers.is_empty() {
            for (i, partition) in orphaned.into_iter().enumerate() {
                let worker_idx = i % workers.len();
                let worker = &workers[worker_idx];
                partitions
                    .entry(worker.clone())
                    .or_insert_with(Vec::new)
                    .push(partition);
            }
        }

        // Increment epoch
        self.epoch.fetch_add(1, Ordering::SeqCst);

        log::info!(
            "Rebalanced partitions after failure. Epoch: {}, Workers: {:?}",
            self.epoch.load(Ordering::SeqCst),
            *workers
        );

        Ok(())
    }

    /// Join the worker pool via Redis
    #[cfg(feature = "redis")]
    pub async fn join(&self, coordinator: &RedisCoordinator) -> CylonResult<()> {
        // Get active workers from Redis
        let active = coordinator.get_active_worker_count().await?;
        log::info!("Joining worker pool with {} active workers", active);
        Ok(())
    }
}

// ============================================================================
// Recovery Callback
// ============================================================================

/// Trait for recovery actions after peer failure
///
/// Implement this trait to provide custom recovery logic for your application.
/// The ResilientExecutor will call these methods during recovery.
#[cfg(feature = "redis")]
#[async_trait::async_trait]
pub trait RecoveryHandler: Send + Sync {
    /// Restore state from the latest checkpoint
    ///
    /// Called during recovery after peer failure. Should restore application
    /// state from the most recent valid checkpoint.
    ///
    /// Returns the checkpoint ID that was restored, or None if no checkpoint available.
    async fn restore_checkpoint(&self) -> CylonResult<Option<u64>>;

    /// Force a checkpoint (e.g., before Lambda timeout)
    async fn force_checkpoint(&self) -> CylonResult<u64>;
}

// ============================================================================
// ResilientExecutor
// ============================================================================

/// Executes operations with automatic fault tolerance
///
/// Wraps distributed operations with:
/// - Pre-flight checks (heartbeats, time budget)
/// - Automatic retry for transient failures
/// - Coordinated abort on peer failure
/// - Checkpoint restore and partition rebalancing
///
/// # Example
///
/// ```ignore
/// let executor = ResilientExecutor::new(
///     coordinator,
///     recovery_handler,
///     "worker-0".to_string(),
///     FaultToleranceConfig::for_serverless(),
/// );
///
/// executor.start_watcher();
/// let result = executor.execute("shuffle", || shuffle_data(&channel)).await?;
/// executor.stop_watcher();
/// ```
#[cfg(feature = "redis")]
pub struct ResilientExecutor<R: RecoveryHandler> {
    /// Redis coordinator for heartbeats and coordination
    coordinator: Arc<RedisCoordinator>,
    /// Recovery handler for checkpoint restore
    recovery_handler: Arc<R>,
    /// Heartbeat watcher
    watcher: Arc<HeartbeatWatcher>,
    /// Worker pool
    worker_pool: Arc<WorkerPool>,
    /// Configuration
    config: FaultToleranceConfig,
    /// Remaining time budget (for serverless)
    remaining_time: RwLock<Option<Duration>>,
}

#[cfg(feature = "redis")]
impl<R: RecoveryHandler> ResilientExecutor<R> {
    /// Create a new resilient executor
    pub fn new(
        coordinator: Arc<RedisCoordinator>,
        recovery_handler: Arc<R>,
        worker_id: String,
        config: FaultToleranceConfig,
    ) -> Self {
        let watcher = Arc::new(HeartbeatWatcher::new(worker_id.clone(), config.clone()));
        let worker_pool = Arc::new(WorkerPool::new(worker_id));

        Self {
            coordinator,
            recovery_handler,
            watcher,
            worker_pool,
            config,
            remaining_time: RwLock::new(None),
        }
    }

    /// Get the heartbeat watcher
    pub fn watcher(&self) -> &Arc<HeartbeatWatcher> {
        &self.watcher
    }

    /// Get the worker pool
    pub fn worker_pool(&self) -> &Arc<WorkerPool> {
        &self.worker_pool
    }

    /// Get the coordinator
    pub fn coordinator(&self) -> &Arc<RedisCoordinator> {
        &self.coordinator
    }

    /// Set remaining time budget (call periodically in Lambda)
    pub fn set_remaining_time(&self, remaining: Duration) {
        *self.remaining_time.write().unwrap() = Some(remaining);
    }

    /// Start the heartbeat watcher
    pub fn start_watcher(&self) {
        self.watcher.start(self.coordinator.clone());
    }

    /// Stop the heartbeat watcher
    pub fn stop_watcher(&self) {
        self.watcher.stop();
    }

    /// Pre-flight check before operation
    pub async fn preflight_check(&self) -> CylonResult<()> {
        // Check time budget
        if let Some(remaining) = *self.remaining_time.read().unwrap() {
            let required = self.config.estimated_op_time + self.config.checkpoint_reserve;
            if remaining < required {
                return Err(CylonError::new(
                    Code::ExecutionError,
                    format!(
                        "Insufficient time: {:?} remaining, {:?} required",
                        remaining, required
                    ),
                ));
            }
        }

        // Check for any pending abort
        if self.watcher.is_abort_signaled() {
            return Err(CylonError::new(
                Code::ExecutionError,
                "Abort signal pending".to_string(),
            ));
        }

        // Check all expected peers are alive
        self.watcher.check_for_failure()?;

        Ok(())
    }

    /// Execute an operation with fault tolerance
    ///
    /// Automatically handles:
    /// - Retry with exponential backoff for transient failures
    /// - Coordinated abort on peer failure
    /// - Checkpoint restore
    /// - Partition rebalancing
    pub async fn execute<F, T>(&self, op_name: &str, mut operation: F) -> CylonResult<T>
    where
        F: FnMut() -> CylonResult<T>,
    {
        let mut attempts = 0;
        let mut backoff = self.config.initial_backoff;

        loop {
            // Pre-flight check
            self.preflight_check().await?;

            // Reset watcher for new attempt
            self.watcher.reset();
            attempts += 1;

            log::debug!("Executing operation '{}', attempt {}", op_name, attempts);

            match operation() {
                Ok(result) => {
                    log::debug!("Operation '{}' succeeded", op_name);
                    return Ok(result);
                }

                Err(error) => {
                    // Check if peer failure
                    if self.watcher.has_peer_failed() {
                        let dead_peers = self.watcher.get_dead_peers();
                        log::warn!(
                            "Operation '{}' failed due to peer failure: {:?}",
                            op_name,
                            dead_peers
                        );

                        // Signal abort to other workers
                        let reason = format!("Peer(s) failed: {}", dead_peers.join(", "));
                        self.coordinator
                            .signal_abort(0, &reason)
                            .await
                            .ok();

                        // Recover
                        self.recover(&dead_peers).await?;

                        // Reset attempts after recovery
                        attempts = 0;
                        backoff = self.config.initial_backoff;
                        continue;
                    }

                    // Transient error - retry with backoff
                    if attempts < self.config.max_retries {
                        log::warn!(
                            "Operation '{}' failed (attempt {}), retrying in {:?}: {}",
                            op_name,
                            attempts,
                            backoff,
                            error
                        );

                        tokio::time::sleep(backoff).await;
                        backoff = std::cmp::min(backoff * 2, self.config.max_backoff);
                        continue;
                    }

                    // Retries exhausted
                    log::error!(
                        "Operation '{}' failed after {} attempts: {}",
                        op_name,
                        attempts,
                        error
                    );
                    return Err(error);
                }
            }
        }
    }

    /// Recover from peer failure
    async fn recover(&self, dead_peers: &[String]) -> CylonResult<()> {
        log::info!("Starting recovery after peer failure: {:?}", dead_peers);

        // 1. Restore from last checkpoint
        match self.recovery_handler.restore_checkpoint().await? {
            Some(checkpoint_id) => {
                log::info!("Restored from checkpoint {}", checkpoint_id);
            }
            None => {
                log::warn!("No checkpoint available for restore");
            }
        }

        // 2. Update worker pool
        self.worker_pool
            .rebalance_after_failure(&dead_peers.to_vec())?;

        // 3. Update watcher with new peer list
        let active_workers = self.worker_pool.workers();
        self.watcher.set_expected_peers(active_workers);
        self.watcher.reset();

        log::info!("Recovery complete. New epoch: {}", self.worker_pool.epoch());

        Ok(())
    }

    /// Force a checkpoint (e.g., before Lambda timeout)
    pub async fn force_checkpoint(&self) -> CylonResult<u64> {
        self.recovery_handler.force_checkpoint().await
    }
}

/// A no-op recovery handler for testing or when checkpointing is disabled
#[cfg(feature = "redis")]
pub struct NoOpRecoveryHandler;

#[cfg(feature = "redis")]
#[async_trait::async_trait]
impl RecoveryHandler for NoOpRecoveryHandler {
    async fn restore_checkpoint(&self) -> CylonResult<Option<u64>> {
        Ok(None)
    }

    async fn force_checkpoint(&self) -> CylonResult<u64> {
        Err(CylonError::new(
            Code::ExecutionError,
            "Checkpointing not configured".to_string(),
        ))
    }
}

// ============================================================================
// CheckpointRecoveryHandler - Bridge between fault tolerance and checkpointing
// ============================================================================

use crate::checkpoint::{
    CheckpointCoordinator, CheckpointManager, CheckpointSerializer, CheckpointStorage,
    CheckpointTrigger,
};

/// Recovery handler that uses CheckpointManager for actual checkpoint/restore
///
/// This bridges the fault tolerance layer (ResilientExecutor) with the
/// checkpointing system (CheckpointManager).
///
/// # Example
///
/// ```ignore
/// use cylon::checkpoint::{
///     CheckpointManager, CheckpointConfig, RedisCoordinator, S3Storage,
///     ArrowIpcSerializer, CompositeTrigger,
/// };
/// use cylon::net::fmi::{CheckpointRecoveryHandler, ResilientExecutor, FaultToleranceConfig};
///
/// // Create checkpoint manager
/// let manager = CheckpointManager::new(
///     ctx.clone(),
///     coordinator.clone(),
///     storage,
///     serializer,
///     trigger,
///     config,
/// );
///
/// // Create recovery handler wrapping the manager
/// let recovery = Arc::new(CheckpointRecoveryHandler::new(Arc::new(manager)));
///
/// // Create resilient executor with checkpoint support
/// let executor = ResilientExecutor::new(
///     redis_coordinator,
///     recovery,
///     worker_id,
///     FaultToleranceConfig::for_serverless(),
/// );
/// ```
#[cfg(feature = "redis")]
pub struct CheckpointRecoveryHandler<C, S, Z, T>
where
    C: CheckpointCoordinator + Send + Sync,
    S: CheckpointStorage + Send + Sync + 'static,
    Z: CheckpointSerializer + Send + Sync,
    T: CheckpointTrigger + Send + Sync,
{
    manager: Arc<CheckpointManager<C, S, Z, T>>,
}

#[cfg(feature = "redis")]
impl<C, S, Z, T> CheckpointRecoveryHandler<C, S, Z, T>
where
    C: CheckpointCoordinator + Send + Sync,
    S: CheckpointStorage + Send + Sync + 'static,
    Z: CheckpointSerializer + Send + Sync,
    T: CheckpointTrigger + Send + Sync,
{
    /// Create a new checkpoint recovery handler
    pub fn new(manager: Arc<CheckpointManager<C, S, Z, T>>) -> Self {
        Self { manager }
    }

    /// Get access to the underlying checkpoint manager
    ///
    /// Use this to register tables, set custom state, etc.
    pub fn manager(&self) -> &Arc<CheckpointManager<C, S, Z, T>> {
        &self.manager
    }

    /// Register a table for checkpointing
    ///
    /// Tables must be registered before they can be checkpointed.
    /// Convenience method that delegates to the underlying manager.
    pub async fn register_table(&self, name: &str, table: Arc<tokio::sync::RwLock<crate::table::Table>>) {
        self.manager.register_table(name, table).await;
    }

    /// Register custom state for checkpointing
    ///
    /// Convenience method that delegates to the underlying manager.
    pub async fn register_state(&self, key: &str, data: Vec<u8>) {
        self.manager.register_state(key, data).await;
    }

    /// Get the last checkpoint ID
    pub async fn last_checkpoint_id(&self) -> Option<u64> {
        self.manager.last_checkpoint_id().await
    }
}

#[cfg(feature = "redis")]
#[async_trait::async_trait]
impl<C, S, Z, T> RecoveryHandler for CheckpointRecoveryHandler<C, S, Z, T>
where
    C: CheckpointCoordinator + Send + Sync,
    S: CheckpointStorage + Send + Sync + 'static,
    Z: CheckpointSerializer + Send + Sync,
    T: CheckpointTrigger + Send + Sync,
{
    /// Restore state from the latest checkpoint
    ///
    /// Finds the latest valid checkpoint and restores all registered tables
    /// and custom state from it.
    async fn restore_checkpoint(&self) -> CylonResult<Option<u64>> {
        log::info!("Restoring from latest checkpoint...");

        match self.manager.restore().await {
            Ok(Some(checkpoint_id)) => {
                log::info!("Successfully restored from checkpoint {}", checkpoint_id);
                Ok(Some(checkpoint_id))
            }
            Ok(None) => {
                log::warn!("No checkpoint available for restore");
                Ok(None)
            }
            Err(e) => {
                log::error!("Failed to restore checkpoint: {}", e);
                Err(e)
            }
        }
    }

    /// Force a checkpoint immediately
    ///
    /// Creates a checkpoint of all registered tables and custom state.
    /// This is typically called before Lambda timeout or when peer failure
    /// is detected.
    async fn force_checkpoint(&self) -> CylonResult<u64> {
        log::info!("Forcing checkpoint...");

        match self.manager.checkpoint().await {
            Ok(checkpoint_id) => {
                log::info!("Successfully created checkpoint {}", checkpoint_id);
                Ok(checkpoint_id)
            }
            Err(e) => {
                log::error!("Failed to create checkpoint: {}", e);
                Err(e)
            }
        }
    }
}

// ============================================================================
// Error Types
// ============================================================================

/// Error types for fault-tolerant operations
#[derive(Debug)]
pub enum FaultError {
    /// Peer(s) failed during operation
    PeerFailure {
        dead_peers: Vec<String>,
        error: CylonError,
    },
    /// Operation was aborted
    Aborted { reason: String },
    /// Retries exhausted
    RetriesExhausted { attempts: u32, error: CylonError },
    /// Insufficient time budget
    InsufficientTime { remaining: Duration, required: Duration },
    /// Other error
    Other(CylonError),
}

impl std::fmt::Display for FaultError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FaultError::PeerFailure { dead_peers, error } => {
                write!(f, "Peer failure ({:?}): {}", dead_peers, error)
            }
            FaultError::Aborted { reason } => write!(f, "Aborted: {}", reason),
            FaultError::RetriesExhausted { attempts, error } => {
                write!(f, "Retries exhausted ({} attempts): {}", attempts, error)
            }
            FaultError::InsufficientTime { remaining, required } => {
                write!(f, "Insufficient time: {:?} < {:?}", remaining, required)
            }
            FaultError::Other(e) => write!(f, "{}", e),
        }
    }
}

impl std::error::Error for FaultError {}

impl From<CylonError> for FaultError {
    fn from(error: CylonError) -> Self {
        FaultError::Other(error)
    }
}
