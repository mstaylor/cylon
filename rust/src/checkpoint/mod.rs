//! Unified Checkpointing Infrastructure for Cylon
//!
//! This module provides checkpointing support for both HPC/MPI and serverless environments
//! through a common API with pluggable backends.
//!
//! # Architecture
//!
//! The checkpointing system is built on four core traits:
//! - [`CheckpointCoordinator`] - Distributed coordination (when/how to sync)
//! - [`CheckpointStorage`] - Reading/writing checkpoint data
//! - [`CheckpointSerializer`] - Serialization of tables and state
//! - [`CheckpointTrigger`] - Determining when to checkpoint
//!
//! # Example
//!
//! ```rust,ignore
//! use cylon::checkpoint::{CheckpointManager, CheckpointConfig};
//!
//! // Create checkpoint manager with default config
//! let manager = CheckpointManager::new(ctx.clone(), config)?;
//!
//! // Register tables for checkpointing
//! manager.register_table("orders", orders_table);
//! manager.register_table("customers", customers_table);
//!
//! // Checkpoint
//! let checkpoint_id = manager.checkpoint().await?;
//!
//! // Restore
//! let tables = manager.restore(checkpoint_id).await?;
//! ```

mod types;
mod traits;
mod serializer;
mod storage;
mod coordinator;
mod trigger;
mod manager;
mod config;
pub mod async_io;
pub mod incremental;

// Re-export core types
pub use types::{
    CheckpointAction,
    CheckpointContext,
    CheckpointDecision,
    CheckpointEvent,
    CheckpointMetadata,
    CheckpointPriority,
    CheckpointStatus,
    CheckpointUrgency,
    OperationType,
    TriggerReason,
    WorkerCheckpointInfo,
    WorkerCheckpointStatus,
    WorkerId,
};

// Re-export traits
pub use traits::{
    CheckpointCoordinator,
    CheckpointStorage,
    CheckpointSerializer,
    CheckpointTrigger,
};

// Re-export implementations
pub use serializer::{ArrowIpcSerializer, JsonSerializer};
pub use storage::FileSystemStorage;
pub use coordinator::{DistributedCoordinator, LocalCoordinator, MpiCoordinator};
pub use trigger::{OperationCountTrigger, TimeBudgetTrigger, IntervalTrigger, CompositeTrigger};
pub use manager::{CheckpointManager, CheckpointManagerBuilder};
pub use config::{
    CheckpointConfig, CompressionAlgorithm, CompressionConfig, PrunePolicy, StorageConfig,
    TriggerConfig,
};

// Re-export incremental checkpoint types
pub use incremental::{
    ChangeTracker, DeltaTableInfo, DeltaType, IncrementalCheckpointInfo, IncrementalConfig,
    IncrementalRestoreResult, RowChangeType, RowRange,
};

// Re-export async I/O types
pub use async_io::{
    AsyncCheckpointHandle, AsyncCheckpointWriter, AsyncIoConfig, AsyncWriteStatus,
    CheckpointWriteState,
};

#[cfg(feature = "redis")]
pub use coordinator::RedisCoordinator;

#[cfg(feature = "redis")]
pub use storage::S3Storage;
