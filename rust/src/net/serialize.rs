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

//! Arrow table serialization for distributed operations
//!
//! This module provides serialization/deserialization of Arrow tables
//! using the Arrow IPC (Inter-Process Communication) format for efficient
//! network transmission in distributed operations.
//!
//! Key Features:
//! - Zero-copy serialization using Arrow IPC
//! - Maintains columnar format during transmission
//! - Supports multi-batch tables
//! - Schema preservation

use std::sync::Arc;
use arrow::ipc::reader::StreamReader;
use arrow::ipc::writer::{IpcWriteOptions, StreamWriter};
use arrow::ipc::CompressionType;
use arrow::record_batch::RecordBatch;

use crate::error::{Code, CylonError, CylonResult};
use crate::ctx::CylonContext;
use crate::table::Table;

/// Compression algorithm for Arrow IPC serialization.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum IpcCompression {
    /// No compression
    #[default]
    None,
    /// LZ4 frame compression (fast)
    Lz4,
    /// Zstandard compression (better ratio)
    Zstd,
}

impl IpcCompression {
    /// Convert to Arrow IPC CompressionType
    fn to_arrow_compression(self) -> Option<CompressionType> {
        match self {
            IpcCompression::None => None,
            IpcCompression::Lz4 => Some(CompressionType::LZ4_FRAME),
            IpcCompression::Zstd => Some(CompressionType::ZSTD),
        }
    }
}

/// Serialize an Arrow RecordBatch to bytes using Arrow IPC format
///
/// # Arguments
/// * `batch` - The RecordBatch to serialize
///
/// # Returns
/// A vector of bytes containing the serialized data in Arrow IPC stream format
///
/// # Example
/// ```ignore
/// let batch = // ... create RecordBatch
/// let bytes = serialize_record_batch(&batch)?;
/// // Send bytes over network
/// ```
pub fn serialize_record_batch(batch: &RecordBatch) -> CylonResult<Vec<u8>> {
    let mut buffer = Vec::new();

    {
        // Create a StreamWriter that writes to our buffer
        let mut writer = StreamWriter::try_new(&mut buffer, &batch.schema())
            .map_err(|e| CylonError::new(
                Code::Invalid,
                format!("Failed to create IPC writer: {}", e)
            ))?;

        // Write the batch
        writer.write(batch)
            .map_err(|e| CylonError::new(
                Code::IoError,
                format!("Failed to write batch: {}", e)
            ))?;

        // Finish writing (flushes footer)
        writer.finish()
            .map_err(|e| CylonError::new(
                Code::IoError,
                format!("Failed to finish writing: {}", e)
            ))?;
    }

    Ok(buffer)
}

/// Deserialize bytes to an Arrow RecordBatch using Arrow IPC format
///
/// # Arguments
/// * `data` - Bytes containing serialized RecordBatch in Arrow IPC stream format
///
/// # Returns
/// The deserialized RecordBatch
///
/// # Example
/// ```ignore
/// let bytes = // ... receive from network
/// let batch = deserialize_record_batch(&bytes)?;
/// ```
pub fn deserialize_record_batch(data: &[u8]) -> CylonResult<RecordBatch> {
    // Create a StreamReader from the bytes
    let cursor = std::io::Cursor::new(data);
    let mut reader = StreamReader::try_new(cursor, None)
        .map_err(|e| CylonError::new(
            Code::Invalid,
            format!("Failed to create IPC reader: {}", e)
        ))?;

    // Read the first (and should be only) batch
    reader.next()
        .ok_or_else(|| CylonError::new(
            Code::Invalid,
            "No batch found in serialized data".to_string()
        ))?
        .map_err(|e| CylonError::new(
            Code::IoError,
            format!("Failed to read batch: {}", e)
        ))
}

/// Serialize a Cylon Table to bytes using Arrow IPC format
///
/// This function serializes all batches in the table along with the schema.
/// The format is compatible with Arrow IPC stream format.
///
/// # Arguments
/// * `table` - The Table to serialize
///
/// # Returns
/// A vector of bytes containing the serialized table
///
/// # Example
/// ```ignore
/// let table = // ... create Table
/// let bytes = serialize_table(&table)?;
/// // Send bytes over network
/// ```
pub fn serialize_table(table: &Table) -> CylonResult<Vec<u8>> {
    serialize_table_with_compression(table, IpcCompression::None)
}

/// Serialize a Cylon Table to bytes using Arrow IPC format with compression.
///
/// This function serializes all batches in the table with optional compression.
/// Compression is applied at the Arrow buffer level for efficient storage.
///
/// # Arguments
/// * `table` - The Table to serialize
/// * `compression` - Compression algorithm to use
///
/// # Returns
/// A vector of bytes containing the serialized (and optionally compressed) table
///
/// # Example
/// ```ignore
/// let table = // ... create Table
/// let bytes = serialize_table_with_compression(&table, IpcCompression::Lz4)?;
/// ```
pub fn serialize_table_with_compression(
    table: &Table,
    compression: IpcCompression,
) -> CylonResult<Vec<u8>> {
    let mut buffer = Vec::new();

    // Get the table's schema
    let schema = table.schema().ok_or_else(|| CylonError::new(
        Code::Invalid,
        "Table has no schema".to_string()
    ))?;

    {
        // Create IPC write options with compression if specified
        let write_options = IpcWriteOptions::default()
            .try_with_compression(compression.to_arrow_compression())
            .map_err(|e| CylonError::new(
                Code::Invalid,
                format!("Failed to set compression options: {}", e)
            ))?;

        // Create a StreamWriter with the table's schema and options
        let mut writer = StreamWriter::try_new_with_options(&mut buffer, &schema, write_options)
            .map_err(|e| CylonError::new(
                Code::Invalid,
                format!("Failed to create IPC writer: {}", e)
            ))?;

        // Write all batches
        for i in 0..table.num_batches() {
            if let Some(batch) = table.batch(i) {
                writer.write(&batch)
                    .map_err(|e| CylonError::new(
                        Code::IoError,
                        format!("Failed to write batch {}: {}", i, e)
                    ))?;
            }
        }

        // Finish writing
        writer.finish()
            .map_err(|e| CylonError::new(
                Code::IoError,
                format!("Failed to finish writing: {}", e)
            ))?;
    }

    Ok(buffer)
}

/// Deserialize bytes to a Cylon Table using Arrow IPC format
///
/// This function deserializes all batches from the Arrow IPC stream format
/// and reconstructs the Table.
///
/// # Arguments
/// * `ctx` - CylonContext for the table
/// * `data` - Bytes containing serialized Table in Arrow IPC stream format
///
/// # Returns
/// The deserialized Table
///
/// # Example
/// ```ignore
/// let ctx = CylonContext::new()?;
/// let bytes = // ... receive from network
/// let table = deserialize_table(ctx, &bytes)?;
/// ```
pub fn deserialize_table(ctx: Arc<CylonContext>, data: &[u8]) -> CylonResult<Table> {
    // Create a StreamReader from the bytes
    let cursor = std::io::Cursor::new(data);
    let mut reader = StreamReader::try_new(cursor, None)
        .map_err(|e| CylonError::new(
            Code::Invalid,
            format!("Failed to create IPC reader: {}", e)
        ))?;

    // Read all batches
    let mut batches = Vec::new();

    for result in reader {
        let batch = result.map_err(|e| CylonError::new(
            Code::IoError,
            format!("Failed to read batch: {}", e)
        ))?;
        batches.push(batch);
    }

    // Create Table from batches (handles empty case automatically)
    Table::from_record_batches(ctx, batches)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, Int32Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn create_test_batch() -> RecordBatch {
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Utf8, false),
        ]);

        let a = Int32Array::from(vec![1, 2, 3, 4, 5]);
        let b = StringArray::from(vec!["a", "b", "c", "d", "e"]);

        RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(a), Arc::new(b)],
        ).unwrap()
    }

    #[test]
    fn test_serialize_deserialize_record_batch() {
        let batch = create_test_batch();

        // Serialize
        let bytes = serialize_record_batch(&batch).unwrap();
        assert!(!bytes.is_empty());

        // Deserialize
        let result = deserialize_record_batch(&bytes).unwrap();

        // Verify schema
        assert_eq!(result.schema(), batch.schema());
        assert_eq!(result.num_rows(), batch.num_rows());
        assert_eq!(result.num_columns(), batch.num_columns());

        // Verify data
        let a_orig = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        let a_result = result.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(a_orig.len(), a_result.len());
        for i in 0..a_orig.len() {
            assert_eq!(a_orig.value(i), a_result.value(i));
        }

        let b_orig = batch.column(1).as_any().downcast_ref::<StringArray>().unwrap();
        let b_result = result.column(1).as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(b_orig.len(), b_result.len());
        for i in 0..b_orig.len() {
            assert_eq!(b_orig.value(i), b_result.value(i));
        }
    }

    #[test]
    fn test_serialize_deserialize_table() {
        let ctx = Arc::new(CylonContext::new(false));

        // Create a table with multiple batches
        let batch1 = create_test_batch();
        let batch2 = create_test_batch();

        let table = Table::from_record_batches(
            ctx.clone(),
            vec![batch1, batch2],
        ).unwrap();

        // Serialize
        let bytes = serialize_table(&table).unwrap();
        assert!(!bytes.is_empty());

        // Deserialize
        let result = deserialize_table(ctx, &bytes).unwrap();

        // Verify
        assert_eq!(result.num_batches(), table.num_batches());
        assert_eq!(result.rows(), table.rows());
        assert_eq!(result.columns(), table.columns());
        assert_eq!(result.schema(), table.schema());
    }

    #[test]
    fn test_empty_table() {
        let ctx = Arc::new(CylonContext::new(false));

        // Create an empty batch with schema
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Utf8, false),
        ]);

        let a = Int32Array::from(vec![] as Vec<i32>);
        let b = StringArray::from(vec![] as Vec<&str>);

        let empty_batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(a), Arc::new(b)],
        ).unwrap();

        let table = Table::from_record_batches(
            ctx.clone(),
            vec![empty_batch],
        ).unwrap();

        // Serialize
        let bytes = serialize_table(&table).unwrap();

        // Deserialize
        let result = deserialize_table(ctx, &bytes).unwrap();

        // Verify
        assert_eq!(result.num_batches(), 1);
        assert_eq!(result.rows(), 0); // No rows but has schema
    }
}
