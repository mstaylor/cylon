//! Asynchronous I/O support for checkpoint operations.
//!
//! This module provides background checkpoint writing capabilities that allow
//! the main processing to continue while checkpoint data is being persisted.
//!
//! # Architecture
//!
//! When async I/O is enabled:
//! 1. Checkpoint data is serialized and queued for writing
//! 2. A background task handles the actual I/O operations
//! 3. The main thread can continue processing immediately
//! 4. Commit only happens after all writes complete
//!
//! This is particularly useful for:
//! - Large checkpoints that would otherwise block processing
//! - Systems with slow storage backends
//! - Overlapping computation with I/O

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot, RwLock, Semaphore};

use crate::error::{Code, CylonError, CylonResult};

use super::traits::CheckpointStorage;
use super::types::WorkerId;

/// A write operation to be performed asynchronously.
#[derive(Debug)]
pub struct WriteOperation {
    /// Checkpoint ID
    pub checkpoint_id: u64,
    /// Worker ID
    pub worker_id: WorkerId,
    /// Storage key
    pub key: String,
    /// Data to write (already compressed if compression is enabled)
    pub data: Vec<u8>,
    /// Channel to signal completion
    pub completion: Option<oneshot::Sender<CylonResult<()>>>,
}

/// Status of an async checkpoint write.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AsyncWriteStatus {
    /// Write is pending/in progress
    Pending,
    /// Write completed successfully
    Completed,
    /// Write failed
    Failed(String),
}

/// Tracks the status of async writes for a checkpoint.
#[derive(Debug)]
pub struct CheckpointWriteState {
    /// Checkpoint ID
    pub checkpoint_id: u64,
    /// Total number of writes expected
    pub total_writes: AtomicUsize,
    /// Number of completed writes
    pub completed_writes: AtomicUsize,
    /// Number of failed writes
    pub failed_writes: AtomicUsize,
    /// Error messages from failed writes
    pub errors: RwLock<Vec<String>>,
    /// Whether all writes have been submitted
    pub all_submitted: RwLock<bool>,
    /// Channel to notify when all writes complete
    pub completion_notify: RwLock<Option<oneshot::Sender<CylonResult<()>>>>,
}

impl CheckpointWriteState {
    /// Create a new checkpoint write state
    pub fn new(checkpoint_id: u64) -> Self {
        Self {
            checkpoint_id,
            total_writes: AtomicUsize::new(0),
            completed_writes: AtomicUsize::new(0),
            failed_writes: AtomicUsize::new(0),
            errors: RwLock::new(Vec::new()),
            all_submitted: RwLock::new(false),
            completion_notify: RwLock::new(None),
        }
    }

    /// Increment the total expected writes
    pub fn expect_write(&self) {
        self.total_writes.fetch_add(1, Ordering::SeqCst);
    }

    /// Record a completed write
    pub async fn write_completed(&self) {
        self.completed_writes.fetch_add(1, Ordering::SeqCst);
        self.check_completion().await;
    }

    /// Record a failed write
    pub async fn write_failed(&self, error: String) {
        self.failed_writes.fetch_add(1, Ordering::SeqCst);
        {
            let mut errors = self.errors.write().await;
            errors.push(error);
        }
        self.check_completion().await;
    }

    /// Mark that all writes have been submitted
    pub async fn all_writes_submitted(&self) {
        {
            let mut submitted = self.all_submitted.write().await;
            *submitted = true;
        }
        self.check_completion().await;
    }

    /// Check if all writes are complete and notify if so
    async fn check_completion(&self) {
        let all_submitted = *self.all_submitted.read().await;
        if !all_submitted {
            return;
        }

        let total = self.total_writes.load(Ordering::SeqCst);
        let completed = self.completed_writes.load(Ordering::SeqCst);
        let failed = self.failed_writes.load(Ordering::SeqCst);

        if completed + failed >= total {
            // All writes done, notify
            let mut notify = self.completion_notify.write().await;
            if let Some(sender) = notify.take() {
                let result = if failed > 0 {
                    let errors = self.errors.read().await;
                    Err(CylonError::new(
                        Code::IoError,
                        format!("Async checkpoint failed: {} errors: {:?}", failed, *errors),
                    ))
                } else {
                    Ok(())
                };
                let _ = sender.send(result);
            }
        }
    }

    /// Get the current progress
    pub fn progress(&self) -> (usize, usize, usize) {
        (
            self.completed_writes.load(Ordering::SeqCst),
            self.failed_writes.load(Ordering::SeqCst),
            self.total_writes.load(Ordering::SeqCst),
        )
    }

    /// Check if all writes are complete
    pub async fn is_complete(&self) -> bool {
        let all_submitted = *self.all_submitted.read().await;
        if !all_submitted {
            return false;
        }

        let total = self.total_writes.load(Ordering::SeqCst);
        let completed = self.completed_writes.load(Ordering::SeqCst);
        let failed = self.failed_writes.load(Ordering::SeqCst);

        completed + failed >= total
    }

    /// Check if all writes succeeded
    pub fn is_success(&self) -> bool {
        self.failed_writes.load(Ordering::SeqCst) == 0
    }
}

/// Configuration for async checkpoint I/O.
#[derive(Clone, Debug)]
pub struct AsyncIoConfig {
    /// Maximum number of concurrent write operations
    pub max_concurrent_writes: usize,
    /// Size of the write queue
    pub queue_size: usize,
    /// Whether to use memory-mapped I/O for large files (future)
    pub use_mmap: bool,
}

impl Default for AsyncIoConfig {
    fn default() -> Self {
        Self {
            max_concurrent_writes: 4,
            queue_size: 64,
            use_mmap: false,
        }
    }
}

impl AsyncIoConfig {
    /// Create config optimized for SSDs
    pub fn for_ssd() -> Self {
        Self {
            max_concurrent_writes: 8,
            queue_size: 128,
            use_mmap: false,
        }
    }

    /// Create config optimized for HDDs
    pub fn for_hdd() -> Self {
        Self {
            max_concurrent_writes: 2,
            queue_size: 32,
            use_mmap: false,
        }
    }

    /// Create config optimized for network storage
    pub fn for_network() -> Self {
        Self {
            max_concurrent_writes: 16,
            queue_size: 256,
            use_mmap: false,
        }
    }
}

/// Handle for controlling and monitoring async checkpoint writes.
pub struct AsyncCheckpointHandle {
    /// Checkpoint ID
    checkpoint_id: u64,
    /// Write state
    state: Arc<CheckpointWriteState>,
    /// Receiver for completion notification
    completion_rx: Option<oneshot::Receiver<CylonResult<()>>>,
}

impl AsyncCheckpointHandle {
    /// Create a new handle
    pub fn new(checkpoint_id: u64, state: Arc<CheckpointWriteState>) -> Self {
        Self {
            checkpoint_id,
            state,
            completion_rx: None,
        }
    }

    /// Get the checkpoint ID
    pub fn checkpoint_id(&self) -> u64 {
        self.checkpoint_id
    }

    /// Get the current progress (completed, failed, total)
    pub fn progress(&self) -> (usize, usize, usize) {
        self.state.progress()
    }

    /// Check if all writes are complete
    pub async fn is_complete(&self) -> bool {
        self.state.is_complete().await
    }

    /// Check if all writes succeeded
    pub fn is_success(&self) -> bool {
        self.state.is_success()
    }

    /// Wait for all writes to complete
    pub async fn wait(mut self) -> CylonResult<()> {
        if let Some(rx) = self.completion_rx.take() {
            rx.await.map_err(|_| {
                CylonError::new(Code::IoError, "Async checkpoint completion channel closed")
            })?
        } else {
            // Poll until complete
            while !self.state.is_complete().await {
                tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
            }
            if !self.state.is_success() {
                let errors = self.state.errors.read().await;
                return Err(CylonError::new(
                    Code::IoError,
                    format!("Async checkpoint failed: {:?}", *errors),
                ));
            }
            Ok(())
        }
    }

    /// Set the completion receiver
    pub fn set_completion_receiver(&mut self, rx: oneshot::Receiver<CylonResult<()>>) {
        self.completion_rx = Some(rx);
    }
}

/// Async checkpoint writer that processes writes in the background.
pub struct AsyncCheckpointWriter<S: CheckpointStorage + 'static> {
    /// Storage backend
    storage: Arc<S>,
    /// Write operation sender
    write_tx: mpsc::Sender<WriteOperation>,
    /// Semaphore for limiting concurrent writes
    semaphore: Arc<Semaphore>,
    /// Active checkpoint states
    states: Arc<RwLock<HashMap<u64, Arc<CheckpointWriteState>>>>,
    /// Configuration
    config: AsyncIoConfig,
    /// Whether the writer is running
    running: Arc<RwLock<bool>>,
}

impl<S: CheckpointStorage + Send + Sync + 'static> AsyncCheckpointWriter<S> {
    /// Create a new async checkpoint writer
    pub fn new(storage: Arc<S>, config: AsyncIoConfig) -> Self {
        let (write_tx, write_rx) = mpsc::channel(config.queue_size);
        let semaphore = Arc::new(Semaphore::new(config.max_concurrent_writes));
        let states: Arc<RwLock<HashMap<u64, Arc<CheckpointWriteState>>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let running = Arc::new(RwLock::new(true));

        // Spawn background writer task
        let writer = BackgroundWriter {
            storage: storage.clone(),
            write_rx,
            semaphore: semaphore.clone(),
            states: states.clone(),
            running: running.clone(),
        };
        tokio::spawn(writer.run());

        Self {
            storage,
            write_tx,
            semaphore,
            states,
            config,
            running,
        }
    }

    /// Begin tracking writes for a new checkpoint
    pub async fn begin_checkpoint(&self, checkpoint_id: u64) -> Arc<CheckpointWriteState> {
        let state = Arc::new(CheckpointWriteState::new(checkpoint_id));
        {
            let mut states = self.states.write().await;
            states.insert(checkpoint_id, state.clone());
        }
        state
    }

    /// Queue a write operation
    pub async fn queue_write(
        &self,
        checkpoint_id: u64,
        worker_id: WorkerId,
        key: String,
        data: Vec<u8>,
    ) -> CylonResult<()> {
        // Get or create state
        let state = {
            let states = self.states.read().await;
            states.get(&checkpoint_id).cloned()
        };

        if let Some(state) = state {
            state.expect_write();
        }

        let op = WriteOperation {
            checkpoint_id,
            worker_id,
            key,
            data,
            completion: None,
        };

        self.write_tx.send(op).await.map_err(|_| {
            CylonError::new(Code::IoError, "Failed to queue async write operation")
        })?;

        Ok(())
    }

    /// Queue a write operation and wait for completion
    pub async fn queue_write_and_wait(
        &self,
        checkpoint_id: u64,
        worker_id: WorkerId,
        key: String,
        data: Vec<u8>,
    ) -> CylonResult<()> {
        let (tx, rx) = oneshot::channel();

        // Get or create state
        let state = {
            let states = self.states.read().await;
            states.get(&checkpoint_id).cloned()
        };

        if let Some(state) = state {
            state.expect_write();
        }

        let op = WriteOperation {
            checkpoint_id,
            worker_id,
            key,
            data,
            completion: Some(tx),
        };

        self.write_tx.send(op).await.map_err(|_| {
            CylonError::new(Code::IoError, "Failed to queue async write operation")
        })?;

        rx.await.map_err(|_| {
            CylonError::new(Code::IoError, "Async write completion channel closed")
        })?
    }

    /// Signal that all writes for a checkpoint have been submitted
    pub async fn finish_checkpoint(&self, checkpoint_id: u64) -> CylonResult<AsyncCheckpointHandle> {
        let state = {
            let states = self.states.read().await;
            states.get(&checkpoint_id).cloned()
        };

        let state = state.ok_or_else(|| {
            CylonError::new(
                Code::NotFound,
                format!("No async checkpoint state for {}", checkpoint_id),
            )
        })?;

        // Create completion channel
        let (tx, rx) = oneshot::channel();
        {
            let mut notify = state.completion_notify.write().await;
            *notify = Some(tx);
        }

        // Mark all writes submitted
        state.all_writes_submitted().await;

        let mut handle = AsyncCheckpointHandle::new(checkpoint_id, state);
        handle.set_completion_receiver(rx);

        Ok(handle)
    }

    /// Get the state for a checkpoint
    pub async fn get_state(&self, checkpoint_id: u64) -> Option<Arc<CheckpointWriteState>> {
        let states = self.states.read().await;
        states.get(&checkpoint_id).cloned()
    }

    /// Clean up state for a completed checkpoint
    pub async fn cleanup_checkpoint(&self, checkpoint_id: u64) {
        let mut states = self.states.write().await;
        states.remove(&checkpoint_id);
    }

    /// Shutdown the async writer
    pub async fn shutdown(&self) {
        let mut running = self.running.write().await;
        *running = false;
    }
}

/// Background task that processes write operations.
struct BackgroundWriter<S: CheckpointStorage + 'static> {
    storage: Arc<S>,
    write_rx: mpsc::Receiver<WriteOperation>,
    semaphore: Arc<Semaphore>,
    states: Arc<RwLock<HashMap<u64, Arc<CheckpointWriteState>>>>,
    running: Arc<RwLock<bool>>,
}

impl<S: CheckpointStorage + Send + Sync + 'static> BackgroundWriter<S> {
    async fn run(mut self) {
        while let Some(op) = self.write_rx.recv().await {
            // Check if we should stop
            {
                let running = self.running.read().await;
                if !*running {
                    break;
                }
            }

            // Acquire semaphore permit
            let permit = self.semaphore.clone().acquire_owned().await;

            let storage = self.storage.clone();
            let states = self.states.clone();

            // Spawn task to handle this write
            tokio::spawn(async move {
                let _permit = permit; // Hold permit until done

                let result = storage
                    .write(op.checkpoint_id, &op.worker_id, &op.key, &op.data)
                    .await;

                // Update state
                if let Some(state) = {
                    let states = states.read().await;
                    states.get(&op.checkpoint_id).cloned()
                } {
                    match &result {
                        Ok(_) => state.write_completed().await,
                        Err(e) => state.write_failed(e.to_string()).await,
                    }
                }

                // Notify completion if requested
                if let Some(completion) = op.completion {
                    let _ = completion.send(result.map(|_| ()));
                }
            });
        }
    }
}
