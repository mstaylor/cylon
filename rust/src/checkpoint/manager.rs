//! Checkpoint manager implementation.
//!
//! The manager orchestrates the checkpoint process using coordinator, storage,
//! serializer, and trigger components.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::ctx::CylonContext;
use crate::error::{Code, CylonError, CylonResult};
use crate::table::Table;

use super::async_io::{AsyncCheckpointHandle, AsyncCheckpointWriter, AsyncIoConfig};
use super::config::{CheckpointConfig, CompressionAlgorithm, StorageConfig};
use super::coordinator::{DistributedCoordinator, LocalCoordinator};
use super::incremental::{ChangeTracker, DeltaTableInfo, DeltaType, IncrementalCheckpointInfo};
use super::serializer::ArrowIpcSerializer;
use super::storage::FileSystemStorage;
use super::traits::{
    CheckpointCoordinator, CheckpointSerializer, CheckpointStorage, CheckpointTrigger,
};
use super::trigger::CompositeTrigger;
use super::types::{
    CheckpointAction, CheckpointDecision, CheckpointEvent, CheckpointMetadata,
    CheckpointStatus, CheckpointUrgency, OperationType, WorkerId,
};

/// Checkpoint manager that orchestrates the checkpoint process.
///
/// The manager is responsible for:
/// - Tracking when checkpoints should occur (via triggers)
/// - Coordinating distributed checkpoints (via coordinator)
/// - Persisting checkpoint data (via storage)
/// - Serializing table data (via serializer)
/// - Managing checkpoint lifecycle and pruning
pub struct CheckpointManager<C, S, Z, T>
where
    C: CheckpointCoordinator,
    S: CheckpointStorage + 'static,
    Z: CheckpointSerializer,
    T: CheckpointTrigger,
{
    /// Coordinator for distributed checkpoint agreement
    coordinator: Arc<C>,
    /// Storage backend for persisting checkpoints
    storage: Arc<S>,
    /// Serializer for table data
    serializer: Arc<Z>,
    /// Trigger for checkpoint timing
    trigger: Arc<T>,
    /// Configuration
    config: CheckpointConfig,
    /// Cylon context
    ctx: Arc<CylonContext>,
    /// Current checkpoint ID being created (if any)
    current_checkpoint: RwLock<Option<u64>>,
    /// Next checkpoint ID to use
    next_checkpoint_id: AtomicU64,
    /// Last committed checkpoint ID (for incremental checkpoints)
    last_checkpoint_id: RwLock<Option<u64>>,
    /// Change tracker for incremental checkpoints
    change_tracker: Arc<ChangeTracker>,
    /// Async checkpoint writer (if async I/O is enabled)
    async_writer: Option<Arc<AsyncCheckpointWriter<S>>>,
    /// Pending async checkpoint handles
    pending_async_checkpoints: RwLock<HashMap<u64, AsyncCheckpointHandle>>,
    /// Registered tables to checkpoint
    tables: RwLock<HashMap<String, Arc<RwLock<Table>>>>,
    /// Custom state to checkpoint
    custom_state: RwLock<HashMap<String, Vec<u8>>>,
    /// Event listeners
    listeners: RwLock<Vec<Box<dyn Fn(CheckpointEvent) + Send + Sync>>>,
}

impl<C, S, Z, T> CheckpointManager<C, S, Z, T>
where
    C: CheckpointCoordinator,
    S: CheckpointStorage + Send + Sync + 'static,
    Z: CheckpointSerializer,
    T: CheckpointTrigger,
{
    /// Create a new checkpoint manager
    pub fn new(
        ctx: Arc<CylonContext>,
        coordinator: Arc<C>,
        storage: Arc<S>,
        serializer: Arc<Z>,
        trigger: Arc<T>,
        config: CheckpointConfig,
    ) -> Self {
        let change_tracker = if config.incremental_config.track_rows {
            Arc::new(ChangeTracker::with_row_tracking())
        } else {
            Arc::new(ChangeTracker::new())
        };

        // Create async writer if async I/O is enabled
        let async_writer = if config.async_io {
            Some(Arc::new(AsyncCheckpointWriter::new(
                storage.clone(),
                AsyncIoConfig::default(),
            )))
        } else {
            None
        };

        Self {
            coordinator,
            storage,
            serializer,
            trigger,
            config,
            ctx,
            current_checkpoint: RwLock::new(None),
            next_checkpoint_id: AtomicU64::new(1),
            last_checkpoint_id: RwLock::new(None),
            change_tracker,
            async_writer,
            pending_async_checkpoints: RwLock::new(HashMap::new()),
            tables: RwLock::new(HashMap::new()),
            custom_state: RwLock::new(HashMap::new()),
            listeners: RwLock::new(Vec::new()),
        }
    }

    /// Get the change tracker for marking table modifications
    pub fn change_tracker(&self) -> &Arc<ChangeTracker> {
        &self.change_tracker
    }

    /// Mark a table as modified (for incremental checkpoints)
    pub fn mark_table_modified(&self, table_name: &str) {
        self.change_tracker.mark_table_modified(table_name);
    }

    /// Check if incremental checkpoints are enabled
    pub fn is_incremental_enabled(&self) -> bool {
        self.config.incremental_config.enabled
    }

    /// Check if compression is enabled
    ///
    /// Note: Compression is now handled natively by the Arrow IPC serializer.
    pub fn is_compression_enabled(&self) -> bool {
        self.config
            .compression
            .as_ref()
            .map(|c| c.algorithm != CompressionAlgorithm::None)
            .unwrap_or(false)
    }

    /// Check if async I/O is enabled
    pub fn is_async_io_enabled(&self) -> bool {
        self.async_writer.is_some()
    }

    /// Get a reference to the async writer (if enabled)
    pub fn async_writer(&self) -> Option<&Arc<AsyncCheckpointWriter<S>>> {
        self.async_writer.as_ref()
    }

    /// Get the compression algorithm being used
    pub fn compression_algorithm(&self) -> CompressionAlgorithm {
        self.config
            .compression
            .as_ref()
            .map(|c| c.algorithm)
            .unwrap_or(CompressionAlgorithm::None)
    }

    /// Get the last committed checkpoint ID
    pub async fn last_checkpoint_id(&self) -> Option<u64> {
        *self.last_checkpoint_id.read().await
    }

    /// Get the worker ID
    pub fn worker_id(&self) -> WorkerId {
        self.coordinator.worker_id()
    }

    /// Get the world size
    pub fn world_size(&self) -> usize {
        self.coordinator.world_size()
    }

    /// Check if this worker is the leader
    pub fn is_leader(&self) -> bool {
        self.coordinator.is_leader()
    }

    /// Register a table for checkpointing
    pub async fn register_table(&self, name: impl Into<String>, table: Arc<RwLock<Table>>) {
        let mut tables = self.tables.write().await;
        tables.insert(name.into(), table);
    }

    /// Unregister a table from checkpointing
    pub async fn unregister_table(&self, name: &str) {
        let mut tables = self.tables.write().await;
        tables.remove(name);
    }

    /// Register custom state for checkpointing
    pub async fn register_state(&self, key: impl Into<String>, data: Vec<u8>) {
        let mut state = self.custom_state.write().await;
        state.insert(key.into(), data);
    }

    /// Add an event listener
    pub async fn add_listener<F>(&self, listener: F)
    where
        F: Fn(CheckpointEvent) + Send + Sync + 'static,
    {
        let mut listeners = self.listeners.write().await;
        listeners.push(Box::new(listener));
    }

    /// Notify all listeners of an event
    async fn notify(&self, event: CheckpointEvent) {
        let listeners = self.listeners.read().await;
        for listener in listeners.iter() {
            listener(event.clone());
        }
    }

    /// Record an operation for trigger tracking
    pub fn record_operation(&self, op_type: OperationType, bytes_processed: u64) {
        self.trigger.record_operation(op_type, bytes_processed);
    }

    /// Check if a checkpoint should be taken
    pub fn should_checkpoint(&self) -> bool {
        self.trigger.should_checkpoint()
    }

    /// Get the current urgency level
    pub fn urgency(&self) -> CheckpointUrgency {
        self.trigger.urgency()
    }

    /// Force a checkpoint on the next opportunity
    pub fn force_checkpoint(&self) {
        self.trigger.force_checkpoint();
    }

    /// Get the recommended action based on current state
    pub fn get_recommended_action(&self) -> CheckpointAction {
        let urgency = self.trigger.urgency();
        let context = self.trigger.get_context();

        match urgency {
            CheckpointUrgency::Critical => CheckpointAction::CheckpointNow,
            CheckpointUrgency::High => {
                if context.remaining_time_budget.map(|t| t.as_secs() < 30).unwrap_or(false) {
                    CheckpointAction::CheckpointNow
                } else {
                    CheckpointAction::CheckpointSoon
                }
            }
            CheckpointUrgency::Medium => CheckpointAction::CheckpointSoon,
            CheckpointUrgency::Low => CheckpointAction::Continue,
            CheckpointUrgency::None => CheckpointAction::Continue,
        }
    }

    /// Create a checkpoint
    ///
    /// This performs the full checkpoint protocol:
    /// 1. Get consensus on checkpoint ID
    /// 2. Begin checkpoint (distributed vote)
    /// 3. Write local data to staging (full or incremental)
    /// 4. Commit checkpoint (distributed barrier)
    /// 5. Move data from staging to committed
    /// 6. Update metadata
    ///
    /// If incremental checkpoints are enabled and a parent checkpoint exists,
    /// only modified tables will be written.
    pub async fn checkpoint(&self) -> CylonResult<u64> {
        // Check if already checkpointing
        {
            let current = self.current_checkpoint.read().await;
            if current.is_some() {
                return Err(CylonError::new(
                    Code::InvalidState,
                    "Checkpoint already in progress".to_string(),
                ));
            }
        }

        // Get next checkpoint ID
        let checkpoint_id = self.next_checkpoint_id.fetch_add(1, Ordering::SeqCst);

        // Determine if this should be incremental
        let parent_id = self.last_checkpoint_id.read().await.clone();
        let use_incremental = self.should_use_incremental(parent_id).await;

        // Set current checkpoint
        {
            let mut current = self.current_checkpoint.write().await;
            *current = Some(checkpoint_id);
        }

        // Notify start
        self.notify(CheckpointEvent::Started { checkpoint_id }).await;

        // Begin checkpoint (distributed agreement)
        let decision = self.coordinator.begin_checkpoint(checkpoint_id).await?;

        match decision {
            CheckpointDecision::Skip => {
                // Checkpoint was vetoed
                self.cleanup_checkpoint(checkpoint_id).await;
                return Err(CylonError::new(
                    Code::Cancelled,
                    "Checkpoint skipped by coordinator".to_string(),
                ));
            }
            CheckpointDecision::Defer(reason) => {
                // Checkpoint deferred
                self.cleanup_checkpoint(checkpoint_id).await;
                return Err(CylonError::new(
                    Code::Cancelled,
                    format!("Checkpoint deferred: {}", reason),
                ));
            }
            CheckpointDecision::Proceed(_priority) => {
                // Continue with checkpoint
            }
        }

        // Write checkpoint data (full or incremental)
        let incremental_info = if use_incremental {
            match self
                .write_incremental_checkpoint_data(checkpoint_id, parent_id.unwrap())
                .await
            {
                Ok(info) => Some(info),
                Err(e) => {
                    // Abort checkpoint
                    self.coordinator.abort_checkpoint(checkpoint_id).await?;
                    self.cleanup_checkpoint(checkpoint_id).await;
                    self.notify(CheckpointEvent::Failed {
                        checkpoint_id,
                        error: e.to_string(),
                    })
                    .await;
                    return Err(e);
                }
            }
        } else {
            match self.write_checkpoint_data(checkpoint_id).await {
                Ok(()) => None,
                Err(e) => {
                    // Abort checkpoint
                    self.coordinator.abort_checkpoint(checkpoint_id).await?;
                    self.cleanup_checkpoint(checkpoint_id).await;
                    self.notify(CheckpointEvent::Failed {
                        checkpoint_id,
                        error: e.to_string(),
                    })
                    .await;
                    return Err(e);
                }
            }
        };

        // Commit checkpoint (distributed barrier)
        self.coordinator.commit_checkpoint(checkpoint_id).await?;

        // Move from staging to committed
        self.storage
            .commit_write(checkpoint_id, &self.worker_id())
            .await?;

        // Write metadata (leader only)
        if self.is_leader() {
            let metadata = self.create_metadata(checkpoint_id, incremental_info).await;
            self.storage.write_metadata(checkpoint_id, &metadata).await?;
        }

        // Reset trigger and change tracker
        self.trigger.reset();
        self.change_tracker.reset();
        self.change_tracker.set_parent_checkpoint(checkpoint_id);

        // Update last checkpoint ID
        {
            let mut last = self.last_checkpoint_id.write().await;
            *last = Some(checkpoint_id);
        }

        // Clear current checkpoint
        {
            let mut current = self.current_checkpoint.write().await;
            *current = None;
        }

        // Notify completion
        self.notify(CheckpointEvent::Completed { checkpoint_id }).await;

        // Prune old checkpoints if needed
        self.prune_old_checkpoints().await?;

        Ok(checkpoint_id)
    }

    /// Create an asynchronous checkpoint that writes in the background.
    ///
    /// This method returns immediately with a handle that can be used to
    /// wait for the checkpoint to complete. The checkpoint data is written
    /// asynchronously in the background.
    ///
    /// Returns a handle to track and wait for the async checkpoint.
    pub async fn checkpoint_async(&self) -> CylonResult<AsyncCheckpointHandle> {
        let async_writer = self.async_writer.as_ref().ok_or_else(|| {
            CylonError::new(
                Code::InvalidState,
                "Async I/O is not enabled. Enable it via CheckpointConfig::with_async_io(true)"
                    .to_string(),
            )
        })?;

        // Check if already checkpointing
        {
            let current = self.current_checkpoint.read().await;
            if current.is_some() {
                return Err(CylonError::new(
                    Code::InvalidState,
                    "Checkpoint already in progress".to_string(),
                ));
            }
        }

        // Get next checkpoint ID
        let checkpoint_id = self.next_checkpoint_id.fetch_add(1, Ordering::SeqCst);

        // Determine if this should be incremental
        let parent_id = self.last_checkpoint_id.read().await.clone();
        let use_incremental = self.should_use_incremental(parent_id).await;

        // Set current checkpoint
        {
            let mut current = self.current_checkpoint.write().await;
            *current = Some(checkpoint_id);
        }

        // Notify start
        self.notify(CheckpointEvent::Started { checkpoint_id }).await;

        // Begin checkpoint (distributed agreement)
        let decision = self.coordinator.begin_checkpoint(checkpoint_id).await?;

        match decision {
            CheckpointDecision::Skip => {
                self.cleanup_checkpoint(checkpoint_id).await;
                return Err(CylonError::new(
                    Code::Cancelled,
                    "Checkpoint skipped by coordinator".to_string(),
                ));
            }
            CheckpointDecision::Defer(reason) => {
                self.cleanup_checkpoint(checkpoint_id).await;
                return Err(CylonError::new(
                    Code::Cancelled,
                    format!("Checkpoint deferred: {}", reason),
                ));
            }
            CheckpointDecision::Proceed(_priority) => {
                // Continue with checkpoint
            }
        }

        // Begin async checkpoint tracking
        let state = async_writer.begin_checkpoint(checkpoint_id).await;

        // Write checkpoint data asynchronously
        let write_result = if use_incremental {
            self.write_checkpoint_data_async(checkpoint_id, async_writer, parent_id)
                .await
        } else {
            self.write_checkpoint_data_async_full(checkpoint_id, async_writer)
                .await
        };

        if let Err(e) = write_result {
            self.coordinator.abort_checkpoint(checkpoint_id).await?;
            self.cleanup_checkpoint(checkpoint_id).await;
            async_writer.cleanup_checkpoint(checkpoint_id).await;
            self.notify(CheckpointEvent::Failed {
                checkpoint_id,
                error: e.to_string(),
            })
            .await;
            return Err(e);
        }

        // Finish checkpoint and get handle
        let handle = async_writer.finish_checkpoint(checkpoint_id).await?;

        // Store handle for later retrieval
        {
            let mut pending = self.pending_async_checkpoints.write().await;
            pending.insert(
                checkpoint_id,
                AsyncCheckpointHandle::new(checkpoint_id, state),
            );
        }

        Ok(handle)
    }

    /// Wait for an async checkpoint to complete and finalize it.
    ///
    /// This should be called after checkpoint_async() to wait for the background
    /// writes to complete and then commit the checkpoint.
    pub async fn wait_for_async_checkpoint(
        &self,
        mut handle: AsyncCheckpointHandle,
    ) -> CylonResult<u64> {
        let checkpoint_id = handle.checkpoint_id();

        // Wait for all writes to complete
        handle.wait().await?;

        // Commit checkpoint (distributed barrier)
        self.coordinator.commit_checkpoint(checkpoint_id).await?;

        // Move from staging to committed
        self.storage
            .commit_write(checkpoint_id, &self.worker_id())
            .await?;

        // Write metadata (leader only)
        if self.is_leader() {
            // For async checkpoints, we don't track incremental info yet
            let metadata = self.create_metadata(checkpoint_id, None).await;
            self.storage.write_metadata(checkpoint_id, &metadata).await?;
        }

        // Reset trigger and change tracker
        self.trigger.reset();
        self.change_tracker.reset();
        self.change_tracker.set_parent_checkpoint(checkpoint_id);

        // Update last checkpoint ID
        {
            let mut last = self.last_checkpoint_id.write().await;
            *last = Some(checkpoint_id);
        }

        // Clear current checkpoint
        {
            let mut current = self.current_checkpoint.write().await;
            *current = None;
        }

        // Remove from pending
        {
            let mut pending = self.pending_async_checkpoints.write().await;
            pending.remove(&checkpoint_id);
        }

        // Clean up async writer state
        if let Some(writer) = &self.async_writer {
            writer.cleanup_checkpoint(checkpoint_id).await;
        }

        // Notify completion
        self.notify(CheckpointEvent::Completed { checkpoint_id }).await;

        // Prune old checkpoints if needed
        self.prune_old_checkpoints().await?;

        Ok(checkpoint_id)
    }

    /// Check the progress of a pending async checkpoint.
    ///
    /// Returns (completed, failed, total) write counts.
    pub async fn async_checkpoint_progress(&self, checkpoint_id: u64) -> Option<(usize, usize, usize)> {
        if let Some(writer) = &self.async_writer {
            if let Some(state) = writer.get_state(checkpoint_id).await {
                return Some(state.progress());
            }
        }
        None
    }

    /// Check if an async checkpoint is complete.
    pub async fn is_async_checkpoint_complete(&self, checkpoint_id: u64) -> bool {
        if let Some(writer) = &self.async_writer {
            if let Some(state) = writer.get_state(checkpoint_id).await {
                return state.is_complete().await;
            }
        }
        false
    }

    /// Write checkpoint data asynchronously (full checkpoint)
    ///
    /// Note: Compression is handled by the serializer using Arrow IPC native compression.
    async fn write_checkpoint_data_async_full(
        &self,
        checkpoint_id: u64,
        async_writer: &Arc<AsyncCheckpointWriter<S>>,
    ) -> CylonResult<()> {
        let worker_id = self.worker_id();

        // Write tables (serializer handles compression)
        let tables = self.tables.read().await;
        for (name, table_lock) in tables.iter() {
            let table = table_lock.read().await;
            let data = self.serializer.serialize_table(&table)?;
            let key = format!("{}.arrow", name);

            // Queue for async writing
            async_writer
                .queue_write(checkpoint_id, worker_id.clone(), key, data)
                .await?;
        }

        // Write custom state (raw bytes)
        let state = self.custom_state.read().await;
        for (key, data) in state.iter() {
            let state_key = format!("state_{}.bin", key);

            // Queue for async writing
            async_writer
                .queue_write(checkpoint_id, worker_id.clone(), state_key, data.clone())
                .await?;
        }

        Ok(())
    }

    /// Write checkpoint data asynchronously (incremental or full based on parent)
    ///
    /// Note: Compression is handled by the serializer using Arrow IPC native compression.
    async fn write_checkpoint_data_async(
        &self,
        checkpoint_id: u64,
        async_writer: &Arc<AsyncCheckpointWriter<S>>,
        parent_id: Option<u64>,
    ) -> CylonResult<()> {
        let worker_id = self.worker_id();

        // If no parent, write full checkpoint
        if parent_id.is_none() {
            return self
                .write_checkpoint_data_async_full(checkpoint_id, async_writer)
                .await;
        }

        // Get all table names
        let tables = self.tables.read().await;

        // Write modified tables (serializer handles compression)
        for (name, table_lock) in tables.iter() {
            if self.change_tracker.needs_checkpoint(name) {
                let table = table_lock.read().await;
                let data = self.serializer.serialize_table(&table)?;
                let key = format!("{}.arrow", name);

                // Queue for async writing
                async_writer
                    .queue_write(checkpoint_id, worker_id.clone(), key, data)
                    .await?;
            }
        }

        // Write custom state (raw bytes)
        let state = self.custom_state.read().await;
        for (key, data) in state.iter() {
            let state_key = format!("state_{}.bin", key);

            // Queue for async writing
            async_writer
                .queue_write(checkpoint_id, worker_id.clone(), state_key, data.clone())
                .await?;
        }

        Ok(())
    }

    /// Determine if we should use incremental checkpointing
    async fn should_use_incremental(&self, parent_id: Option<u64>) -> bool {
        // Not enabled
        if !self.config.incremental_config.enabled {
            return false;
        }

        // No parent checkpoint
        let parent_id = match parent_id {
            Some(id) => id,
            None => return false,
        };

        // Check chain depth
        if let Ok(parent_metadata) = self.storage.read_metadata(parent_id).await {
            let chain_depth = parent_metadata.chain_depth() + 1;
            if chain_depth > self.config.incremental_config.max_chain_depth {
                return false; // Force full checkpoint to limit chain depth
            }
        }

        // Check savings ratio
        let tables = self.tables.read().await;
        let all_table_names: Vec<String> = tables.keys().cloned().collect();
        let unchanged = self.change_tracker.get_unchanged_tables(&all_table_names);
        let savings_ratio = if all_table_names.is_empty() {
            0.0
        } else {
            unchanged.len() as f64 / all_table_names.len() as f64
        };

        if savings_ratio < self.config.incremental_config.min_savings_ratio {
            return false; // Not enough savings to justify incremental
        }

        true
    }

    /// Write checkpoint data to staging (full checkpoint)
    ///
    /// Note: Compression is handled by the serializer using Arrow IPC native compression.
    async fn write_checkpoint_data(&self, checkpoint_id: u64) -> CylonResult<()> {
        let worker_id = self.worker_id();

        // Write tables (serializer handles compression)
        let tables = self.tables.read().await;
        for (name, table_lock) in tables.iter() {
            let table = table_lock.read().await;
            let data = self.serializer.serialize_table(&table)?;
            let key = format!("{}.arrow", name);

            self.storage
                .write(checkpoint_id, &worker_id, &key, &data)
                .await?;
        }

        // Write custom state (raw bytes, no compression for state)
        let state = self.custom_state.read().await;
        for (key, data) in state.iter() {
            let state_key = format!("state_{}.bin", key);

            self.storage
                .write(checkpoint_id, &worker_id, &state_key, data)
                .await?;
        }

        Ok(())
    }

    /// Write incremental checkpoint data to staging
    ///
    /// Only writes tables that have been modified since the parent checkpoint.
    /// Returns the incremental checkpoint info for metadata.
    ///
    /// Note: Compression is handled by the serializer using Arrow IPC native compression.
    async fn write_incremental_checkpoint_data(
        &self,
        checkpoint_id: u64,
        parent_checkpoint_id: u64,
    ) -> CylonResult<IncrementalCheckpointInfo> {
        let worker_id = self.worker_id();

        // Get parent metadata to determine chain depth
        let parent_metadata = self.storage.read_metadata(parent_checkpoint_id).await?;
        let chain_depth = parent_metadata.chain_depth() + 1;

        let mut info = IncrementalCheckpointInfo::new(parent_checkpoint_id);
        info.chain_depth = chain_depth;

        // Get all table names
        let tables = self.tables.read().await;
        let all_table_names: Vec<String> = tables.keys().cloned().collect();

        // Categorize tables
        for name in &all_table_names {
            if self.change_tracker.needs_checkpoint(name) {
                // Table was modified - write it (serializer handles compression)
                let table_lock = tables.get(name).unwrap();
                let table = table_lock.read().await;
                let data = self.serializer.serialize_table(&table)?;
                let key = format!("{}.arrow", name);

                self.storage
                    .write(checkpoint_id, &worker_id, &key, &data)
                    .await?;

                // Determine delta type
                let delta_type = self.change_tracker.get_delta_type(name);
                let rows = table.rows() as u64;

                match delta_type {
                    DeltaType::Full => {
                        info.add_full(name.clone());
                    }
                    _ => {
                        // For now, treat all modifications as full table writes
                        // Row-level deltas would require more complex logic
                        let delta_info = DeltaTableInfo::full(name.clone(), rows);
                        info.add_delta(delta_info);
                    }
                }
            } else {
                // Table unchanged - reference parent
                info.add_unchanged(name.clone());
            }
        }

        // Write custom state (raw bytes, no compression for state)
        let state = self.custom_state.read().await;
        for (key, data) in state.iter() {
            let state_key = format!("state_{}.bin", key);

            self.storage
                .write(checkpoint_id, &worker_id, &state_key, data)
                .await?;
        }

        Ok(info)
    }

    /// Create checkpoint metadata
    async fn create_metadata(
        &self,
        checkpoint_id: u64,
        incremental_info: Option<IncrementalCheckpointInfo>,
    ) -> CheckpointMetadata {
        let tables = self.tables.read().await;
        let table_names: Vec<String> = tables.keys().cloned().collect();

        let mut workers = Vec::new();
        for i in 0..self.world_size() {
            workers.push(WorkerId::Rank(i as i32));
        }

        let parent_checkpoint_id = incremental_info
            .as_ref()
            .map(|i| i.parent_checkpoint_id);

        CheckpointMetadata {
            checkpoint_id,
            job_id: self.config.job_id.clone(),
            timestamp: std::time::SystemTime::now(),
            status: CheckpointStatus::Committed,
            workers,
            tables: table_names,
            total_bytes: 0, // Could be tracked during write
            format_version: "1.0".to_string(),
            metadata: HashMap::new(),
            parent_checkpoint_id,
            incremental_info,
        }
    }

    /// Clean up after a failed or skipped checkpoint
    async fn cleanup_checkpoint(&self, checkpoint_id: u64) {
        // Clear current checkpoint marker
        let mut current = self.current_checkpoint.write().await;
        if *current == Some(checkpoint_id) {
            *current = None;
        }
    }

    /// Restore from the latest checkpoint
    pub async fn restore(&self) -> CylonResult<Option<u64>> {
        // Find latest checkpoint across all workers
        let latest = self.coordinator.find_latest_checkpoint().await?;

        match latest {
            Some(checkpoint_id) => {
                self.restore_from(checkpoint_id).await?;
                Ok(Some(checkpoint_id))
            }
            None => Ok(None),
        }
    }

    /// Restore from a specific checkpoint
    ///
    /// If the checkpoint is incremental, this will automatically build and apply
    /// the checkpoint chain from the base checkpoint.
    pub async fn restore_from(&self, checkpoint_id: u64) -> CylonResult<()> {
        let metadata = self.storage.read_metadata(checkpoint_id).await?;

        // Check if this is an incremental checkpoint
        if metadata.is_incremental() {
            self.restore_incremental(checkpoint_id).await
        } else {
            self.restore_full(checkpoint_id).await
        }
    }

    /// Restore from a full (non-incremental) checkpoint
    ///
    /// Note: Arrow IPC reader handles decompression transparently.
    async fn restore_full(&self, checkpoint_id: u64) -> CylonResult<()> {
        let worker_id = self.worker_id();

        self.notify(CheckpointEvent::RestoreStarted { checkpoint_id })
            .await;

        // Restore tables (Arrow IPC reader handles decompression transparently)
        let mut tables = self.tables.write().await;
        for (name, table_lock) in tables.iter_mut() {
            let key = format!("{}.arrow", name);

            if self
                .storage
                .exists(checkpoint_id, &worker_id, &key)
                .await?
            {
                let data = self.storage.read(checkpoint_id, &worker_id, &key).await?;
                let restored_table =
                    self.serializer
                        .deserialize_table(&data, self.ctx.clone())?;

                let mut table = table_lock.write().await;
                *table = restored_table;
            }
        }

        // Restore custom state
        let keys = self.storage.list_keys(checkpoint_id, &worker_id).await?;
        let mut state = self.custom_state.write().await;

        for key in keys {
            if key.starts_with("state_") && key.ends_with(".bin") {
                // Extract state name by stripping prefix and suffix
                let state_name = key
                    .strip_prefix("state_")
                    .and_then(|s| s.strip_suffix(".bin"))
                    .unwrap_or(&key);

                let data = self.storage.read(checkpoint_id, &worker_id, &key).await?;
                state.insert(state_name.to_string(), data);
            }
        }

        // Update checkpoint ID counter and last checkpoint
        self.next_checkpoint_id
            .store(checkpoint_id + 1, Ordering::SeqCst);
        {
            let mut last = self.last_checkpoint_id.write().await;
            *last = Some(checkpoint_id);
        }
        self.change_tracker.set_parent_checkpoint(checkpoint_id);

        self.notify(CheckpointEvent::RestoreCompleted { checkpoint_id })
            .await;

        Ok(())
    }

    /// Restore from an incremental checkpoint by building the checkpoint chain
    ///
    /// Note: Arrow IPC reader handles decompression transparently.
    async fn restore_incremental(&self, checkpoint_id: u64) -> CylonResult<()> {
        self.notify(CheckpointEvent::RestoreStarted { checkpoint_id })
            .await;

        // Build the checkpoint chain (from oldest to newest)
        let chain = self.build_checkpoint_chain(checkpoint_id).await?;

        if chain.is_empty() {
            return Err(CylonError::new(
                Code::NotFound,
                "No checkpoints found in chain".to_string(),
            ));
        }

        let worker_id = self.worker_id();

        // Start with the base (full) checkpoint
        let base_id = chain[0];
        let mut tables = self.tables.write().await;

        // Restore from base checkpoint (Arrow IPC reader handles decompression)
        for (name, table_lock) in tables.iter_mut() {
            let key = format!("{}.arrow", name);

            if self.storage.exists(base_id, &worker_id, &key).await? {
                let data = self.storage.read(base_id, &worker_id, &key).await?;
                let restored_table = self
                    .serializer
                    .deserialize_table(&data, self.ctx.clone())?;
                let mut table = table_lock.write().await;
                *table = restored_table;
            }
        }

        // Apply each incremental checkpoint in order
        for &inc_checkpoint_id in &chain[1..] {
            let inc_metadata = self.storage.read_metadata(inc_checkpoint_id).await?;

            if let Some(ref inc_info) = inc_metadata.incremental_info {
                // Apply delta tables
                for delta in &inc_info.delta_tables {
                    if let Some(table_lock) = tables.get(&delta.name) {
                        let key = format!("{}.arrow", delta.name);

                        if self
                            .storage
                            .exists(inc_checkpoint_id, &worker_id, &key)
                            .await?
                        {
                            let data = self
                                .storage
                                .read(inc_checkpoint_id, &worker_id, &key)
                                .await?;
                            let delta_table = self
                                .serializer
                                .deserialize_table(&data, self.ctx.clone())?;

                            // For now, replace the entire table (full delta)
                            // Future: implement actual delta application (append, update, delete)
                            let mut table = table_lock.write().await;
                            *table = delta_table;
                        }
                    }
                }

                // Apply full tables
                for table_name in &inc_info.full_tables {
                    if let Some(table_lock) = tables.get(table_name) {
                        let key = format!("{}.arrow", table_name);

                        if self
                            .storage
                            .exists(inc_checkpoint_id, &worker_id, &key)
                            .await?
                        {
                            let data = self
                                .storage
                                .read(inc_checkpoint_id, &worker_id, &key)
                                .await?;
                            let restored_table = self
                                .serializer
                                .deserialize_table(&data, self.ctx.clone())?;
                            let mut table = table_lock.write().await;
                            *table = restored_table;
                        }
                    }
                }

                // Unchanged tables already have correct data from previous checkpoint
            }
        }

        // Restore custom state from the final checkpoint
        let keys = self.storage.list_keys(checkpoint_id, &worker_id).await?;
        let mut state = self.custom_state.write().await;

        for key in keys {
            if key.starts_with("state_") && key.ends_with(".bin") {
                // Extract state name by stripping prefix and suffix
                let state_name = key
                    .strip_prefix("state_")
                    .and_then(|s| s.strip_suffix(".bin"))
                    .unwrap_or(&key);

                let data = self.storage.read(checkpoint_id, &worker_id, &key).await?;
                state.insert(state_name.to_string(), data);
            }
        }

        // Update checkpoint ID counter and last checkpoint
        self.next_checkpoint_id
            .store(checkpoint_id + 1, Ordering::SeqCst);
        {
            let mut last = self.last_checkpoint_id.write().await;
            *last = Some(checkpoint_id);
        }
        self.change_tracker.set_parent_checkpoint(checkpoint_id);

        self.notify(CheckpointEvent::RestoreCompleted { checkpoint_id })
            .await;

        Ok(())
    }

    /// Build the checkpoint chain from a given checkpoint back to the base
    ///
    /// Returns a vector of checkpoint IDs ordered from oldest (base) to newest.
    async fn build_checkpoint_chain(&self, checkpoint_id: u64) -> CylonResult<Vec<u64>> {
        let mut chain = vec![checkpoint_id];
        let mut current_id = checkpoint_id;

        loop {
            let metadata = self.storage.read_metadata(current_id).await?;

            match metadata.parent_checkpoint_id {
                Some(parent_id) => {
                    chain.push(parent_id);
                    current_id = parent_id;
                }
                None => {
                    // Reached the base (full) checkpoint
                    break;
                }
            }
        }

        // Reverse so oldest is first
        chain.reverse();

        Ok(chain)
    }

    /// Prune old checkpoints based on retention policy.
    ///
    /// This method enforces the following pruning rules:
    /// 1. Always keep at least `min_retain` checkpoints
    /// 2. Remove checkpoints exceeding `max_checkpoints`
    /// 3. Remove checkpoints older than `max_age` (if set)
    /// 4. Only prune committed checkpoints if `only_prune_committed` is true
    /// 5. Never prune checkpoints that are parents of other checkpoints (incremental chain)
    async fn prune_old_checkpoints(&self) -> CylonResult<()> {
        // Only leader prunes to avoid race conditions
        if !self.is_leader() {
            return Ok(());
        }

        let policy = &self.config.retention;

        // Get all checkpoint metadata for analysis
        let checkpoint_ids = self.storage.list_checkpoints().await?;

        // Early exit if we have fewer checkpoints than min_retain
        if checkpoint_ids.len() <= policy.min_retain {
            return Ok(());
        }

        // Load metadata for all checkpoints
        let mut checkpoints_with_metadata: Vec<(u64, Option<CheckpointMetadata>)> = Vec::new();
        for id in checkpoint_ids {
            let metadata = self.storage.read_metadata(id).await.ok();
            checkpoints_with_metadata.push((id, metadata));
        }

        // Build set of parent checkpoint IDs (these cannot be pruned)
        let parent_ids: std::collections::HashSet<u64> = checkpoints_with_metadata
            .iter()
            .filter_map(|(_, meta)| meta.as_ref()?.parent_checkpoint_id)
            .collect();

        // Determine which checkpoints are eligible for pruning
        let now = std::time::SystemTime::now();
        let mut prunable: Vec<(u64, std::time::SystemTime)> = checkpoints_with_metadata
            .into_iter()
            .filter_map(|(id, meta)| {
                let meta = meta?;

                // Cannot prune checkpoints that are parents in incremental chains
                if parent_ids.contains(&id) {
                    return None;
                }

                // Check only_prune_committed policy
                if policy.only_prune_committed && meta.status != CheckpointStatus::Committed {
                    return None;
                }

                Some((id, meta.timestamp))
            })
            .collect();

        // Sort by timestamp (oldest first), use checkpoint ID as tiebreaker
        prunable.sort_by(|a, b| {
            a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0))
        });

        // Calculate how many to keep (at least min_retain)
        let total_checkpoints = prunable.len();
        let keep_count = policy.min_retain.max(
            total_checkpoints.saturating_sub(
                total_checkpoints.saturating_sub(policy.max_checkpoints),
            ),
        );

        // Determine checkpoints to prune
        let mut to_prune: Vec<u64> = Vec::new();

        for (i, (checkpoint_id, timestamp)) in prunable.iter().enumerate() {
            // Always keep the most recent min_retain checkpoints
            if total_checkpoints - i <= policy.min_retain {
                break;
            }

            // Check if exceeds max_checkpoints
            let exceeds_count = total_checkpoints - i > policy.max_checkpoints;

            // Check if exceeds max_age
            let exceeds_age = if let Some(max_age) = policy.max_age {
                now.duration_since(*timestamp)
                    .map(|age| age > max_age)
                    .unwrap_or(false)
            } else {
                false
            };

            // Prune if either limit is exceeded
            if exceeds_count || exceeds_age {
                to_prune.push(*checkpoint_id);
            }
        }

        // Prune the selected checkpoints
        for checkpoint_id in to_prune {
            if let Err(e) = self.storage.delete(checkpoint_id).await {
                // Log error but continue with other pruning
                eprintln!("Warning: Failed to prune checkpoint {}: {}", checkpoint_id, e);
                continue;
            }
            self.notify(CheckpointEvent::Pruned { checkpoint_id }).await;
        }

        Ok(())
    }

    /// Manually trigger pruning of old checkpoints.
    ///
    /// This can be called explicitly to clean up old checkpoints without
    /// waiting for the next checkpoint operation.
    pub async fn prune(&self) -> CylonResult<usize> {
        let before_count = self.storage.list_checkpoints().await?.len();
        self.prune_old_checkpoints().await?;
        let after_count = self.storage.list_checkpoints().await?.len();
        Ok(before_count.saturating_sub(after_count))
    }

    /// Get pruning statistics.
    ///
    /// Returns information about which checkpoints would be pruned if prune() is called.
    pub async fn get_prune_candidates(&self) -> CylonResult<Vec<u64>> {
        let policy = &self.config.retention;
        let checkpoint_ids = self.storage.list_checkpoints().await?;

        if checkpoint_ids.len() <= policy.min_retain {
            return Ok(Vec::new());
        }

        // Load metadata for all checkpoints
        let mut checkpoints_with_metadata: Vec<(u64, Option<CheckpointMetadata>)> = Vec::new();
        for id in &checkpoint_ids {
            let metadata = self.storage.read_metadata(*id).await.ok();
            checkpoints_with_metadata.push((*id, metadata));
        }

        // Build set of parent checkpoint IDs
        let parent_ids: std::collections::HashSet<u64> = checkpoints_with_metadata
            .iter()
            .filter_map(|(_, meta)| meta.as_ref()?.parent_checkpoint_id)
            .collect();

        let now = std::time::SystemTime::now();
        let mut prunable: Vec<(u64, std::time::SystemTime)> = checkpoints_with_metadata
            .into_iter()
            .filter_map(|(id, meta)| {
                let meta = meta?;
                if parent_ids.contains(&id) {
                    return None;
                }
                if policy.only_prune_committed && meta.status != CheckpointStatus::Committed {
                    return None;
                }
                Some((id, meta.timestamp))
            })
            .collect();

        // Sort by timestamp (oldest first), use checkpoint ID as tiebreaker
        prunable.sort_by(|a, b| {
            a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0))
        });

        let total_checkpoints = prunable.len();
        let mut candidates: Vec<u64> = Vec::new();

        for (i, (checkpoint_id, timestamp)) in prunable.iter().enumerate() {
            if total_checkpoints - i <= policy.min_retain {
                break;
            }

            let exceeds_count = total_checkpoints - i > policy.max_checkpoints;
            let exceeds_age = if let Some(max_age) = policy.max_age {
                now.duration_since(*timestamp)
                    .map(|age| age > max_age)
                    .unwrap_or(false)
            } else {
                false
            };

            if exceeds_count || exceeds_age {
                candidates.push(*checkpoint_id);
            }
        }

        Ok(candidates)
    }

    /// List available checkpoints
    pub async fn list_checkpoints(&self) -> CylonResult<Vec<u64>> {
        self.storage.list_checkpoints().await
    }

    /// Get metadata for a checkpoint
    pub async fn get_metadata(&self, checkpoint_id: u64) -> CylonResult<CheckpointMetadata> {
        self.storage.read_metadata(checkpoint_id).await
    }

    /// Delete a specific checkpoint
    pub async fn delete_checkpoint(&self, checkpoint_id: u64) -> CylonResult<()> {
        self.storage.delete(checkpoint_id).await
    }
}

/// Builder for creating CheckpointManager instances.
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

    /// Build a local (single-worker) checkpoint manager
    pub async fn build_local(
        self,
    ) -> CylonResult<
        CheckpointManager<LocalCoordinator, FileSystemStorage, ArrowIpcSerializer, CompositeTrigger>,
    > {
        let ctx = self.ctx.ok_or_else(|| {
            CylonError::new(Code::InvalidArgument, "Context is required".to_string())
        })?;

        let base_path = match &self.config.storage {
            StorageConfig::FileSystem { base_path } => base_path.clone(),
            #[cfg(feature = "s3")]
            _ => {
                return Err(CylonError::new(
                    Code::InvalidArgument,
                    "Local manager requires filesystem storage".to_string(),
                ))
            }
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

    /// Build a local (single-worker) checkpoint manager with S3 storage
    #[cfg(feature = "s3")]
    pub async fn build_local_s3(
        self,
    ) -> CylonResult<
        CheckpointManager<LocalCoordinator, super::storage::S3Storage, ArrowIpcSerializer, CompositeTrigger>,
    > {
        use super::storage::{S3Storage, S3StorageConfig};

        let ctx = self.ctx.ok_or_else(|| {
            CylonError::new(Code::InvalidArgument, "Context is required".to_string())
        })?;

        let storage = match &self.config.storage {
            StorageConfig::S3 {
                bucket,
                prefix,
                region,
                endpoint,
                force_path_style,
            } => {
                let mut config = S3StorageConfig::new(bucket, format!("{}/{}", prefix, self.config.job_id));
                if let Some(r) = region {
                    config = config.with_region(r);
                }
                if let Some(e) = endpoint {
                    config = config.with_endpoint(e);
                }
                config = config.with_path_style(*force_path_style);
                S3Storage::new(config).await?
            }
            _ => {
                return Err(CylonError::new(
                    Code::InvalidArgument,
                    "S3 manager requires S3 storage configuration".to_string(),
                ))
            }
        };

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

    /// Build a distributed checkpoint manager using the provided communicator
    pub async fn build_distributed(
        self,
        communicator: Arc<dyn crate::net::communicator::Communicator>,
    ) -> CylonResult<
        CheckpointManager<
            DistributedCoordinator,
            FileSystemStorage,
            ArrowIpcSerializer,
            CompositeTrigger,
        >,
    > {
        let ctx = self.ctx.ok_or_else(|| {
            CylonError::new(Code::InvalidArgument, "Context is required".to_string())
        })?;

        let base_path = match &self.config.storage {
            StorageConfig::FileSystem { base_path } => base_path.clone(),
            #[cfg(feature = "s3")]
            _ => {
                return Err(CylonError::new(
                    Code::InvalidArgument,
                    "Distributed manager requires filesystem storage".to_string(),
                ))
            }
        };

        let storage = FileSystemStorage::new(base_path, &self.config.job_id);
        storage.initialize().await?;

        let coordinator = DistributedCoordinator::new(communicator);
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

    /// Build a distributed checkpoint manager with S3 storage
    #[cfg(feature = "s3")]
    pub async fn build_distributed_s3(
        self,
        communicator: Arc<dyn crate::net::communicator::Communicator>,
    ) -> CylonResult<
        CheckpointManager<
            DistributedCoordinator,
            super::storage::S3Storage,
            ArrowIpcSerializer,
            CompositeTrigger,
        >,
    > {
        use super::storage::{S3Storage, S3StorageConfig};

        let ctx = self.ctx.ok_or_else(|| {
            CylonError::new(Code::InvalidArgument, "Context is required".to_string())
        })?;

        let storage = match &self.config.storage {
            StorageConfig::S3 {
                bucket,
                prefix,
                region,
                endpoint,
                force_path_style,
            } => {
                let mut config = S3StorageConfig::new(bucket, format!("{}/{}", prefix, self.config.job_id));
                if let Some(r) = region {
                    config = config.with_region(r);
                }
                if let Some(e) = endpoint {
                    config = config.with_endpoint(e);
                }
                config = config.with_path_style(*force_path_style);
                S3Storage::new(config).await?
            }
            _ => {
                return Err(CylonError::new(
                    Code::InvalidArgument,
                    "S3 distributed manager requires S3 storage configuration".to_string(),
                ))
            }
        };

        let coordinator = DistributedCoordinator::new(communicator);
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
}

impl Default for CheckpointManagerBuilder {
    fn default() -> Self {
        Self::new()
    }
}
