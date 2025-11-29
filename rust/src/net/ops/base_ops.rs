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

//! Base operation implementations for distributed table operations
//!
//! Ported from cpp/src/cylon/net/ops/base_ops.hpp
//!
//! This module defines traits for collective operations on tables.
//! The architecture follows a two-level design:
//! 1. Base traits (like TableBcastImpl) define the interface and orchestration logic
//! 2. Communication-specific implementations (like MpiTableBcastImpl) provide primitives
//!
//! Each operation has:
//! - Virtual methods for primitive operations (broadcast buffer sizes, broadcast data, etc.)
//! - Execute methods that orchestrate the full operation

use std::sync::Arc;
use crate::error::CylonResult;
use crate::ctx::CylonContext;
use crate::table::Table;

/// Buffer abstraction for network transmission
/// Corresponds to C++ Buffer class
pub struct Buffer {
    data: Vec<u8>,
}

impl Buffer {
    pub fn new(size: usize) -> Self {
        Self {
            data: vec![0; size],
        }
    }

    pub fn from_vec(data: Vec<u8>) -> Self {
        Self { data }
    }

    pub fn data(&self) -> &[u8] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// Trait for broadcasting tables across all processes
/// Corresponds to C++ TableBcastImpl class from cpp/src/cylon/net/ops/base_ops.hpp
pub trait TableBcastImpl {
    /// Initialize asynchronous operations for the given number of buffers
    ///
    /// # Arguments
    /// * `num_buffers` - Number of buffers that will be broadcast
    fn init(&mut self, num_buffers: i32);

    /// Broadcast buffer size information
    ///
    /// # Arguments
    /// * `buffer` - Buffer containing sizes to broadcast
    /// * `count` - Number of integers to broadcast
    /// * `bcast_root` - Root process rank
    fn bcast_buffer_sizes(&self, buffer: &mut [i32], count: i32, bcast_root: i32) -> CylonResult<()>;

    /// Synchronous broadcast of buffer data
    ///
    /// # Arguments
    /// * `buf_data` - Buffer data to broadcast
    /// * `send_count` - Number of bytes to broadcast
    /// * `bcast_root` - Root process rank
    fn bcast_buffer_data(&self, buf_data: &mut [u8], send_count: i32, bcast_root: i32) -> CylonResult<()>;

    /// Non-blocking broadcast of buffer data
    ///
    /// # Arguments
    /// * `buf_idx` - Index of the buffer (for tracking async operations)
    /// * `buf_data` - Buffer data to broadcast
    /// * `send_count` - Number of bytes to broadcast
    /// * `bcast_root` - Root process rank
    fn ibcast_buffer_data(&mut self, buf_idx: i32, buf_data: &mut [u8], send_count: i32, bcast_root: i32) -> CylonResult<()>;

    /// Wait for all asynchronous operations to complete
    ///
    /// # Arguments
    /// * `num_buffers` - Number of buffers to wait for
    fn wait_all(&mut self, num_buffers: i32) -> CylonResult<()>;

    /// Execute table broadcast operation
    ///
    /// This is the main entry point that orchestrates the entire broadcast:
    /// 1. Broadcast the Arrow schema
    /// 2. Serialize table on root process
    /// 3. Broadcast number of buffers
    /// 4. Broadcast buffer sizes
    /// 5. Broadcast actual data buffers (using non-blocking operations)
    /// 6. Deserialize on non-root processes
    ///
    /// # Arguments
    /// * `table` - Table to broadcast (Some on root, None on non-root initially)
    /// * `bcast_root` - Root process rank
    /// * `ctx` - Cylon context
    ///
    /// # Returns
    /// After execution, all processes will have the same table
    fn execute(&mut self, table: &mut Option<Table>, bcast_root: i32, ctx: Arc<CylonContext>) -> CylonResult<()>;
}

/// Trait for gathering tables from all processes
/// Corresponds to C++ TableGatherImpl class
pub trait TableGatherImpl {
    /// Initialize asynchronous operations
    fn init(&mut self, num_buffers: i32);

    /// Gather buffer sizes from all processes
    fn gather_buffer_sizes(&self, send_data: &[i32], num_buffers: i32, rcv_data: &mut [i32], gather_root: i32) -> CylonResult<()>;

    /// Non-blocking gather of buffer data
    fn igather_buffer_data(
        &mut self,
        buf_idx: i32,
        send_data: &[u8],
        send_count: i32,
        recv_data: &mut [u8],
        recv_count: &[i32],
        displacements: &[i32],
        gather_root: i32,
    ) -> CylonResult<()>;

    /// Wait for all asynchronous operations
    fn wait_all(&mut self, num_buffers: i32) -> CylonResult<()>;

    /// Execute table gather operation
    fn execute(
        &mut self,
        table: &Table,
        gather_root: i32,
        gather_from_root: bool,
        ctx: Arc<CylonContext>,
    ) -> CylonResult<Vec<Table>>;
}

/// Trait for all-gathering tables (gather to all processes)
/// Corresponds to C++ TableAllgatherImpl class
pub trait TableAllgatherImpl {
    /// Initialize asynchronous operations
    fn init(&mut self, num_buffers: i32);

    /// All-gather buffer sizes
    fn allgather_buffer_sizes(&self, send_data: &[i32], num_buffers: i32, rcv_data: &mut [i32]) -> CylonResult<()>;

    /// Non-blocking all-gather of buffer data
    fn iallgather_buffer_data(
        &mut self,
        buf_idx: i32,
        send_data: &[u8],
        send_count: i32,
        recv_data: &mut [u8],
        recv_count: &[i32],
        displacements: &[i32],
    ) -> CylonResult<()>;

    /// Wait for all asynchronous operations
    fn wait_all(&mut self, num_buffers: i32) -> CylonResult<()>;

    /// Execute table all-gather operation
    fn execute(&mut self, table: &Table, ctx: Arc<CylonContext>) -> CylonResult<Vec<Table>>;
}

use super::super::comm_operations::ReduceOp;
use crate::DataType;

/// Trait for AllReduce operations
/// Corresponds to C++ AllReduceImpl class from cpp/src/cylon/net/ops/base_ops.hpp
pub trait AllReduceImpl {
    /// Perform allreduce on a buffer
    ///
    /// # Arguments
    /// * `send_buf` - Send buffer
    /// * `rcv_buf` - Receive buffer
    /// * `count` - Number of elements
    /// * `data_type` - Data type of elements
    /// * `reduce_op` - Reduction operation
    fn allreduce_buffer(
        &self,
        send_buf: &[u8],
        rcv_buf: &mut [u8],
        count: i32,
        data_type: &DataType,
        reduce_op: ReduceOp,
    ) -> CylonResult<()>;
}

/// Trait for AllGather operations (Column/Scalar level)
/// Corresponds to C++ AllGatherImpl class from cpp/src/cylon/net/ops/base_ops.hpp
pub trait AllGatherImpl {
    /// Allgather buffer sizes
    ///
    /// # Arguments
    /// * `send_data` - Send buffer sizes
    /// * `num_buffers` - Number of buffers
    /// * `rcv_data` - Receive buffer for sizes from all processes
    fn allgather_buffer_size(
        &self,
        send_data: &[i32],
        num_buffers: i32,
        rcv_data: &mut [i32],
    ) -> CylonResult<()>;

    /// Non-blocking allgather of buffer data
    ///
    /// # Arguments
    /// * `buf_idx` - Buffer index
    /// * `send_data` - Send buffer
    /// * `send_count` - Send count
    /// * `recv_data` - Receive buffer
    /// * `recv_count` - Receive counts from each process
    /// * `displacements` - Displacements in receive buffer
    fn iallgather_buffer_data(
        &mut self,
        buf_idx: i32,
        send_data: &[u8],
        send_count: i32,
        recv_data: &mut [u8],
        recv_count: &[i32],
        displacements: &[i32],
    ) -> CylonResult<()>;

    /// Wait for all asynchronous operations
    fn wait_all(&self) -> CylonResult<()>;
}
