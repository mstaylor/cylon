//! Checkpoint serializer implementations.
//!
//! Provides serializers for converting tables and state to bytes for checkpointing.

use std::sync::Arc;
use serde::{de::DeserializeOwned, Serialize};

use crate::ctx::CylonContext;
use crate::error::{Code, CylonError, CylonResult};
use crate::net::serialize::{deserialize_table, serialize_table};
use crate::table::Table;

use super::traits::CheckpointSerializer;

/// Arrow IPC format serializer.
///
/// Uses Arrow's Inter-Process Communication format for efficient serialization
/// of table data. This is the same format used for network communication in
/// distributed operations.
///
/// Features:
/// - Zero-copy serialization where possible
/// - Maintains columnar format
/// - Supports multi-batch tables
/// - Schema preservation
pub struct ArrowIpcSerializer;

impl ArrowIpcSerializer {
    /// Create a new Arrow IPC serializer
    pub fn new() -> Self {
        Self
    }
}

impl Default for ArrowIpcSerializer {
    fn default() -> Self {
        Self::new()
    }
}

impl CheckpointSerializer for ArrowIpcSerializer {
    fn serialize_table(&self, table: &Table) -> CylonResult<Vec<u8>> {
        serialize_table(table)
    }

    fn deserialize_table(&self, data: &[u8], ctx: Arc<CylonContext>) -> CylonResult<Table> {
        deserialize_table(ctx, data)
    }

    fn serialize_state<T: Serialize>(&self, state: &T) -> CylonResult<Vec<u8>> {
        bincode::serialize(state).map_err(|e| {
            CylonError::new(
                Code::SerializationError,
                format!("Failed to serialize state: {}", e),
            )
        })
    }

    fn deserialize_state<T: DeserializeOwned>(&self, data: &[u8]) -> CylonResult<T> {
        bincode::deserialize(data).map_err(|e| {
            CylonError::new(
                Code::SerializationError,
                format!("Failed to deserialize state: {}", e),
            )
        })
    }

    fn format_id(&self) -> &str {
        "arrow_ipc"
    }
}

/// JSON serializer for checkpoint metadata.
///
/// Uses JSON format which is human-readable and useful for debugging.
/// Not recommended for large tables due to overhead.
pub struct JsonSerializer;

impl JsonSerializer {
    /// Create a new JSON serializer
    pub fn new() -> Self {
        Self
    }
}

impl Default for JsonSerializer {
    fn default() -> Self {
        Self::new()
    }
}

impl JsonSerializer {
    /// Serialize state to JSON
    pub fn serialize_json<T: Serialize>(&self, state: &T) -> CylonResult<Vec<u8>> {
        serde_json::to_vec_pretty(state).map_err(|e| {
            CylonError::new(
                Code::SerializationError,
                format!("Failed to serialize to JSON: {}", e),
            )
        })
    }

    /// Deserialize state from JSON
    pub fn deserialize_json<T: DeserializeOwned>(&self, data: &[u8]) -> CylonResult<T> {
        serde_json::from_slice(data).map_err(|e| {
            CylonError::new(
                Code::SerializationError,
                format!("Failed to deserialize from JSON: {}", e),
            )
        })
    }
}
