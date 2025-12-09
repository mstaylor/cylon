//! Checkpoint serializer implementations.
//!
//! Provides serializers for converting tables and state to bytes for checkpointing.

use std::sync::Arc;
use serde::{de::DeserializeOwned, Serialize};

use crate::ctx::CylonContext;
use crate::error::{Code, CylonError, CylonResult};
use crate::net::serialize::{deserialize_table, serialize_table_with_compression, IpcCompression};
use crate::table::Table;

use super::config::CompressionAlgorithm;
use super::traits::CheckpointSerializer;

/// Arrow IPC format serializer with optional compression.
///
/// Uses Arrow's Inter-Process Communication format for efficient serialization
/// of table data. Supports native Arrow IPC compression (LZ4, ZSTD) at the
/// buffer level for efficient storage.
///
/// Features:
/// - Zero-copy serialization where possible
/// - Maintains columnar format
/// - Supports multi-batch tables
/// - Schema preservation
/// - Native LZ4/ZSTD compression
pub struct ArrowIpcSerializer {
    /// Compression algorithm to use
    compression: IpcCompression,
}

impl ArrowIpcSerializer {
    /// Create a new Arrow IPC serializer without compression
    pub fn new() -> Self {
        Self {
            compression: IpcCompression::None,
        }
    }

    /// Create a new Arrow IPC serializer with specified compression
    pub fn with_compression(algorithm: CompressionAlgorithm) -> Self {
        let compression = match algorithm {
            CompressionAlgorithm::None => IpcCompression::None,
            CompressionAlgorithm::Lz4 => IpcCompression::Lz4,
            CompressionAlgorithm::Zstd => IpcCompression::Zstd,
        };
        Self { compression }
    }

    /// Get the compression algorithm being used
    pub fn compression(&self) -> IpcCompression {
        self.compression
    }
}

impl Default for ArrowIpcSerializer {
    fn default() -> Self {
        Self::new()
    }
}

impl CheckpointSerializer for ArrowIpcSerializer {
    fn serialize_table(&self, table: &Table) -> CylonResult<Vec<u8>> {
        serialize_table_with_compression(table, self.compression)
    }

    fn deserialize_table(&self, data: &[u8], ctx: Arc<CylonContext>) -> CylonResult<Table> {
        // Arrow IPC reader automatically handles compressed data
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
        match self.compression {
            IpcCompression::None => "arrow_ipc",
            IpcCompression::Lz4 => "arrow_ipc_lz4",
            IpcCompression::Zstd => "arrow_ipc_zstd",
        }
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
