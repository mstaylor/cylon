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

//! Tests for base operations (TableBcastImpl, TableGatherImpl, TableAllgatherImpl)
//!
//! These tests verify the serialization/deserialization logic in the Execute methods.
//! For actual distributed tests, see ucx_communicator_test.rs and ucc_operations_test.rs

use std::sync::Arc;
use arrow::array::{Int64Array, Float64Array};
use arrow::datatypes::{Schema, Field, DataType};
use arrow::record_batch::RecordBatch;

use cylon::ctx::CylonContext;
use cylon::table::Table;
use cylon::net::serialize::{serialize_table, deserialize_table};

/// Create a test table with sample data
fn create_test_table(ctx: Arc<CylonContext>, id: i64) -> Table {
    let col0 = Int64Array::from(vec![id * 10, id * 10 + 1, id * 10 + 2]);
    let col1 = Float64Array::from(vec![id as f64 * 1.1, id as f64 * 2.2, id as f64 * 3.3]);

    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int64, false),
        Field::new("b", DataType::Float64, false),
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(col0), Arc::new(col1)],
    ).unwrap();

    Table::from_record_batch(ctx, batch).unwrap()
}

// =============================================================================
// Serialization Tests (Foundation for Execute methods)
// =============================================================================

#[test]
fn test_table_serialization_roundtrip() {
    let ctx = Arc::new(CylonContext::new(false));
    let table = create_test_table(ctx.clone(), 1);

    // Serialize
    let bytes = serialize_table(&table).unwrap();
    assert!(!bytes.is_empty(), "Serialized bytes should not be empty");

    // Deserialize
    let restored = deserialize_table(ctx.clone(), &bytes).unwrap();

    // Verify
    assert_eq!(restored.rows(), table.rows());
    assert_eq!(restored.columns(), table.columns());
    assert_eq!(restored.schema(), table.schema());
}

#[test]
fn test_empty_table_serialization() {
    let ctx = Arc::new(CylonContext::new(false));

    // Create empty table with schema
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int64, false),
        Field::new("b", DataType::Float64, false),
    ]));

    let col0 = Int64Array::from(vec![] as Vec<i64>);
    let col1 = Float64Array::from(vec![] as Vec<f64>);

    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(col0), Arc::new(col1)],
    ).unwrap();

    let table = Table::from_record_batch(ctx.clone(), batch).unwrap();

    // Serialize
    let bytes = serialize_table(&table).unwrap();

    // Deserialize
    let restored = deserialize_table(ctx.clone(), &bytes).unwrap();

    // Verify
    assert_eq!(restored.rows(), 0);
    assert_eq!(restored.columns(), table.columns());
}

#[test]
fn test_multi_batch_table_serialization() {
    let ctx = Arc::new(CylonContext::new(false));

    // Create table with multiple batches
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int64, false),
    ]));

    let batch1 = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int64Array::from(vec![1, 2, 3]))],
    ).unwrap();

    let batch2 = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int64Array::from(vec![4, 5, 6]))],
    ).unwrap();

    let table = Table::from_record_batches(ctx.clone(), vec![batch1, batch2]).unwrap();
    assert_eq!(table.rows(), 6);

    // Serialize
    let bytes = serialize_table(&table).unwrap();

    // Deserialize
    let restored = deserialize_table(ctx.clone(), &bytes).unwrap();

    // Verify total rows preserved
    assert_eq!(restored.rows(), 6);
}

// =============================================================================
// Mock TableBcastImpl for testing the default execute() implementation
// =============================================================================

use cylon::error::CylonResult;
use cylon::net::ops::base_ops::TableBcastImpl;

/// Mock broadcast implementation for single-process testing
/// Simulates broadcast by copying data
struct MockTableBcastImpl {
    /// Stored data buffers for verification
    stored_data: Vec<Vec<u8>>,
    /// Buffer sizes
    sizes: Vec<i32>,
    /// Simulated world size
    world_size: i32,
    /// Simulated rank
    rank: i32,
    /// Root process rank
    root: i32,
}

impl MockTableBcastImpl {
    fn new(rank: i32, world_size: i32, root: i32) -> Self {
        Self {
            stored_data: Vec::new(),
            sizes: Vec::new(),
            world_size,
            rank,
            root,
        }
    }
}

impl TableBcastImpl for MockTableBcastImpl {
    fn init(&mut self, num_buffers: i32) {
        self.stored_data.resize(num_buffers as usize, Vec::new());
    }

    fn bcast_buffer_sizes(&self, buffer: &mut [i32], _count: i32, _bcast_root: i32) -> CylonResult<()> {
        // In single-process mock, sizes are already correct on root
        // Non-root would receive the size
        Ok(())
    }

    fn bcast_buffer_data(&self, _buf_data: &mut [u8], _send_count: i32, _bcast_root: i32) -> CylonResult<()> {
        // In single-process mock, data is already in place
        Ok(())
    }

    fn ibcast_buffer_data(&mut self, _buf_idx: i32, _buf_data: &mut [u8], _send_count: i32, _bcast_root: i32) -> CylonResult<()> {
        // Non-blocking version - data already in place in single process
        Ok(())
    }

    fn wait_all(&mut self, _num_buffers: i32) -> CylonResult<()> {
        Ok(())
    }
}

#[test]
fn test_bcast_execute_on_root() {
    let ctx = Arc::new(CylonContext::new(false));
    let table = create_test_table(ctx.clone(), 42);

    // Single-process mock where rank == root
    let mut impl_ = MockTableBcastImpl::new(0, 1, 0);
    let mut table_opt = Some(table.clone());

    let result = impl_.execute(&mut table_opt, 0, ctx.clone());
    assert!(result.is_ok(), "Execute should succeed on root");

    // Table should still be present on root
    assert!(table_opt.is_some());
    let restored = table_opt.unwrap();
    assert_eq!(restored.rows(), table.rows());
}

// =============================================================================
// Mock TableAllgatherImpl for testing
// =============================================================================

use cylon::net::ops::base_ops::TableAllgatherImpl;

/// Mock allgather implementation - simulates single-process allgather
struct MockTableAllgatherImpl {
    world_size: i32,
}

impl MockTableAllgatherImpl {
    fn new(world_size: i32) -> Self {
        Self { world_size }
    }
}

impl TableAllgatherImpl for MockTableAllgatherImpl {
    fn init(&mut self, _num_buffers: i32) {}

    fn allgather_buffer_sizes(&self, send_data: &[i32], _num_buffers: i32, rcv_data: &mut [i32]) -> CylonResult<()> {
        // In single-process, just copy send to all receive positions
        for i in 0..self.world_size as usize {
            for (j, &val) in send_data.iter().enumerate() {
                if i * send_data.len() + j < rcv_data.len() {
                    rcv_data[i * send_data.len() + j] = val;
                }
            }
        }
        Ok(())
    }

    fn iallgather_buffer_data(
        &mut self,
        _buf_idx: i32,
        send_data: &[u8],
        send_count: i32,
        recv_data: &mut [u8],
        _recv_count: &[i32],
        displacements: &[i32],
    ) -> CylonResult<()> {
        // In single-process, copy send data to all positions
        for i in 0..self.world_size as usize {
            let offset = displacements[i] as usize;
            let end = offset + send_count as usize;
            if end <= recv_data.len() && send_count as usize <= send_data.len() {
                recv_data[offset..end].copy_from_slice(&send_data[..send_count as usize]);
            }
        }
        Ok(())
    }

    fn wait_all(&mut self, _num_buffers: i32) -> CylonResult<()> {
        Ok(())
    }
}

#[test]
fn test_allgather_execute_single_process() {
    let ctx = Arc::new(CylonContext::new(false));
    let table = create_test_table(ctx.clone(), 1);

    let mut impl_ = MockTableAllgatherImpl::new(1);
    let result = impl_.execute(&table, ctx.clone());

    assert!(result.is_ok(), "Execute should succeed");
    let tables = result.unwrap();

    // With world_size=1, should get 1 table back
    assert_eq!(tables.len(), 1);
    assert_eq!(tables[0].rows(), table.rows());
}

// =============================================================================
// Mock TableGatherImpl for testing
// =============================================================================

use cylon::net::ops::base_ops::TableGatherImpl;

/// Mock gather implementation - simulates single-process gather
struct MockTableGatherImpl {
    world_size: i32,
    rank: i32,
}

impl MockTableGatherImpl {
    fn new(rank: i32, world_size: i32) -> Self {
        Self { world_size, rank }
    }
}

impl TableGatherImpl for MockTableGatherImpl {
    fn init(&mut self, _num_buffers: i32) {}

    fn gather_buffer_sizes(&self, send_data: &[i32], _num_buffers: i32, rcv_data: &mut [i32], gather_root: i32) -> CylonResult<()> {
        // Only root receives
        if self.rank == gather_root {
            for (i, &val) in send_data.iter().enumerate() {
                if i < rcv_data.len() {
                    rcv_data[i] = val;
                }
            }
        }
        Ok(())
    }

    fn igather_buffer_data(
        &mut self,
        _buf_idx: i32,
        send_data: &[u8],
        send_count: i32,
        recv_data: &mut [u8],
        _recv_count: &[i32],
        _displacements: &[i32],
        gather_root: i32,
    ) -> CylonResult<()> {
        // Only root receives
        if self.rank == gather_root && send_count as usize <= recv_data.len() {
            recv_data[..send_count as usize].copy_from_slice(&send_data[..send_count as usize]);
        }
        Ok(())
    }

    fn wait_all(&mut self, _num_buffers: i32) -> CylonResult<()> {
        Ok(())
    }
}

#[test]
fn test_gather_execute_on_root() {
    let ctx = Arc::new(CylonContext::new(false));
    let table = create_test_table(ctx.clone(), 1);

    // Single process acting as root
    let mut impl_ = MockTableGatherImpl::new(0, 1);
    let result = impl_.execute(&table, 0, true, ctx.clone());

    assert!(result.is_ok(), "Execute should succeed");
    let tables = result.unwrap();

    // Root gathers from itself in single-process case
    assert_eq!(tables.len(), 1);
    assert_eq!(tables[0].rows(), table.rows());
}

// =============================================================================
// UCC Operations Tests (require feature flag and actual UCC library)
// =============================================================================

#[cfg(feature = "ucc")]
mod ucc_tests {
    // UCC tests would go here, but they require actual UCC infrastructure
    // and multiple processes, so they would be marked as #[ignore]

    #[test]
    #[ignore]
    fn test_ucc_allgather_with_infrastructure() {
        // This test requires:
        // - UCC library installed
        // - Multiple processes (MPI or Redis OOB)
        // - Proper environment setup
        println!("UCC allgather test - requires infrastructure");
    }
}
