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
use crate::error::{CylonResult, CylonError, Code};
use crate::ctx::CylonContext;
use crate::table::Table;
use crate::net::serialize::{serialize_table, deserialize_table};

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
    /// 1. Serialize table on root process
    /// 2. Broadcast serialized data size
    /// 3. Broadcast actual data
    /// 4. Deserialize on non-root processes
    ///
    /// Corresponds to C++ TableBcastImpl::Execute (base_ops.cpp:299-339)
    ///
    /// # Arguments
    /// * `table` - Table to broadcast (Some on root, None on non-root initially)
    /// * `bcast_root` - Root process rank
    /// * `ctx` - Cylon context
    ///
    /// # Returns
    /// After execution, all processes will have the same table
    fn execute(&mut self, table: &mut Option<Table>, bcast_root: i32, ctx: Arc<CylonContext>) -> CylonResult<()> {
        let is_root = ctx.get_rank() == bcast_root;

        // Serialize on root
        let serialized = if is_root {
            let t = table.as_ref().ok_or_else(|| CylonError::new(
                Code::Invalid,
                "Root must provide a table for broadcast",
            ))?;
            serialize_table(t)?
        } else {
            Vec::new()
        };

        // Broadcast the size first
        let mut size = if is_root { serialized.len() as i32 } else { 0 };
        self.bcast_buffer_sizes(std::slice::from_mut(&mut size), 1, bcast_root)?;

        // If size is 0, nothing to broadcast (empty table case)
        if size == 0 {
            return Ok(());
        }

        // Initialize for 1 buffer
        self.init(1);

        // Prepare data buffer
        let mut data = if is_root {
            serialized
        } else {
            vec![0u8; size as usize]
        };

        // Use non-blocking broadcast
        self.ibcast_buffer_data(0, &mut data, size, bcast_root)?;
        self.wait_all(1)?;

        // Deserialize on non-root
        if !is_root {
            let received_table = deserialize_table(ctx, &data)?;
            *table = Some(received_table);
        }

        Ok(())
    }
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
    ///
    /// Corresponds to C++ TableGatherImpl::Execute (base_ops.cpp:183-215)
    fn execute(
        &mut self,
        table: &Table,
        gather_root: i32,
        gather_from_root: bool,
        ctx: Arc<CylonContext>,
    ) -> CylonResult<Vec<Table>> {
        let rank = ctx.get_rank();
        let world_size = ctx.get_world_size();
        let is_root = rank == gather_root;

        // Serialize local table (or empty if root and not gather_from_root)
        let serialized = if is_root && !gather_from_root {
            Vec::new()
        } else {
            serialize_table(table)?
        };

        let local_size = serialized.len() as i32;

        // Gather sizes from all processes
        let mut all_sizes = if is_root {
            vec![0i32; world_size as usize]
        } else {
            Vec::new()
        };

        self.gather_buffer_sizes(
            std::slice::from_ref(&local_size),
            1,
            &mut all_sizes,
            gather_root,
        )?;

        // Calculate total size and displacements on root
        let (total_size, displacements) = if is_root {
            let total: i32 = all_sizes.iter().sum();
            let mut disps = vec![0i32; world_size as usize];
            for i in 1..world_size as usize {
                disps[i] = disps[i - 1] + all_sizes[i - 1];
            }
            (total, disps)
        } else {
            (0, Vec::new())
        };

        // Initialize for 1 buffer
        self.init(1);

        // Allocate receive buffer on root
        let mut recv_buf = if is_root {
            vec![0u8; total_size as usize]
        } else {
            Vec::new()
        };

        // Gather data
        self.igather_buffer_data(
            0,
            &serialized,
            local_size,
            &mut recv_buf,
            &all_sizes,
            &displacements,
            gather_root,
        )?;

        self.wait_all(1)?;

        // Deserialize on root
        if is_root {
            let mut tables = Vec::with_capacity(world_size as usize);
            for i in 0..world_size as usize {
                let start = displacements[i] as usize;
                let end = start + all_sizes[i] as usize;
                if all_sizes[i] > 0 {
                    let t = deserialize_table(ctx.clone(), &recv_buf[start..end])?;
                    tables.push(t);
                }
            }
            Ok(tables)
        } else {
            Ok(Vec::new())
        }
    }
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
    ///
    /// Corresponds to C++ TableAllgatherImpl::Execute (base_ops.cpp:76-108)
    fn execute(&mut self, table: &Table, ctx: Arc<CylonContext>) -> CylonResult<Vec<Table>> {
        let world_size = ctx.get_world_size();

        // Serialize local table
        let serialized = serialize_table(table)?;
        let local_size = serialized.len() as i32;

        // Allgather sizes from all processes
        let mut all_sizes = vec![0i32; world_size as usize];
        self.allgather_buffer_sizes(
            std::slice::from_ref(&local_size),
            1,
            &mut all_sizes,
        )?;

        // Calculate total size and displacements
        let total_size: i32 = all_sizes.iter().sum();
        let mut displacements = vec![0i32; world_size as usize];
        for i in 1..world_size as usize {
            displacements[i] = displacements[i - 1] + all_sizes[i - 1];
        }

        // Initialize for 1 buffer
        self.init(1);

        // Allocate receive buffer
        let mut recv_buf = vec![0u8; total_size as usize];

        // Allgather data
        self.iallgather_buffer_data(
            0,
            &serialized,
            local_size,
            &mut recv_buf,
            &all_sizes,
            &displacements,
        )?;

        self.wait_all(1)?;

        // Deserialize all tables
        let mut tables = Vec::with_capacity(world_size as usize);
        for i in 0..world_size as usize {
            let start = displacements[i] as usize;
            let end = start + all_sizes[i] as usize;
            if all_sizes[i] > 0 {
                let t = deserialize_table(ctx.clone(), &recv_buf[start..end])?;
                tables.push(t);
            }
        }

        Ok(tables)
    }
}

use super::super::comm_operations::ReduceOp;
use crate::DataType;
use crate::data_types::Type;
use crate::table::Column;
use crate::scalar::Scalar;
use crate::net::serialize::ColumnSerializer;
use arrow::array::{Array, ArrayData};
use arrow::buffer::Buffer as ArrowBuffer;

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

    /// Execute AllReduce on a Column
    ///
    /// Corresponds to C++ AllReduceImpl::Execute(Column) (base_ops.cpp:341-382)
    ///
    /// # Arguments
    /// * `values` - Input column
    /// * `reduce_op` - Reduction operation
    ///
    /// # Returns
    /// Reduced column with same length
    fn execute_column(&self, values: &Column, reduce_op: ReduceOp) -> CylonResult<Arc<Column>> {
        let arr = values.data();
        let arrow_type = arr.data_type();

        // Get byte width - allreduce only supports fixed-width numeric types
        let byte_width = match arrow_type {
            arrow::datatypes::DataType::Int8 | arrow::datatypes::DataType::UInt8 => 1,
            arrow::datatypes::DataType::Int16 | arrow::datatypes::DataType::UInt16 => 2,
            arrow::datatypes::DataType::Int32 | arrow::datatypes::DataType::UInt32 |
            arrow::datatypes::DataType::Float32 => 4,
            arrow::datatypes::DataType::Int64 | arrow::datatypes::DataType::UInt64 |
            arrow::datatypes::DataType::Float64 => 8,
            _ => return Err(CylonError::new(
                Code::Invalid,
                format!("AllReduce does not support {}", arrow_type),
            )),
        };

        // Validate: all ranks should have 0 null count, and equal size.
        // Use the trick from C++: send [null_count, length, -length] and reduce with MAX
        // If max(null_count) > 0, there are nulls
        // If max(length) != max(-length) negated, lengths differ
        let null_count = arr.null_count() as i64;
        let length = arr.len() as i64;
        let metadata: [i64; 3] = [null_count, length, -length];
        let mut metadata_res: [i64; 3] = [0, 0, 0];

        let int64_type = DataType::new(Type::Int64);
        self.allreduce_buffer(
            unsafe { std::slice::from_raw_parts(metadata.as_ptr() as *const u8, 24) },
            unsafe { std::slice::from_raw_parts_mut(metadata_res.as_mut_ptr() as *mut u8, 24) },
            3,
            &int64_type,
            ReduceOp::Max,
        )?;

        // Check validation results
        if metadata_res[0] > 0 {
            return Err(CylonError::new(
                Code::Invalid,
                "AllReduce does not support null values".to_string(),
            ));
        }
        if metadata_res[1] != -metadata_res[2] {
            return Err(CylonError::new(
                Code::Invalid,
                "AllReduce values should be the same length in all ranks".to_string(),
            ));
        }

        let count = arr.len() as i32;
        let buf_size = byte_width * count as usize;

        // Allocate output buffer
        let mut output_buf = vec![0u8; buf_size];

        // Get input data buffer (buffer at index 1 for primitive arrays)
        let data = arr.to_data();
        let input_buf = data.buffers().get(0).ok_or_else(|| CylonError::new(
            Code::Invalid,
            "Column has no data buffer".to_string(),
        ))?;

        let offset_bytes = data.offset() * byte_width;
        let input_slice = &input_buf.as_slice()[offset_bytes..offset_bytes + buf_size];

        // Perform allreduce
        self.allreduce_buffer(
            input_slice,
            &mut output_buf,
            count,
            values.data_type(),
            reduce_op,
        )?;

        // Create output array
        let output_arrow_buf = ArrowBuffer::from(output_buf);
        let output_data = ArrayData::builder(arrow_type.clone())
            .len(count as usize)
            .add_buffer(output_arrow_buf)
            .build()
            .map_err(|e| CylonError::new(Code::ExecutionError, e.to_string()))?;

        Ok(Column::make(arrow::array::make_array(output_data)))
    }

    /// Execute AllReduce on a Scalar
    ///
    /// Corresponds to C++ AllReduceImpl::Execute(Scalar) (base_ops.cpp:384-396)
    ///
    /// # Arguments
    /// * `value` - Input scalar
    /// * `reduce_op` - Reduction operation
    ///
    /// # Returns
    /// Reduced scalar
    fn execute_scalar(&self, value: &Scalar, reduce_op: ReduceOp) -> CylonResult<Arc<Scalar>> {
        // Convert scalar to column (it's already stored as single-element array)
        let col = Column::new(value.data().clone());

        // Perform allreduce on column
        let out_col = self.execute_column(&col, reduce_op)?;

        // Extract result as scalar
        Ok(Scalar::make(out_col.data().clone()))
    }
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
    fn wait_all(&mut self) -> CylonResult<()>;

    /// Execute AllGather on a Column
    ///
    /// Corresponds to C++ AllGatherImpl::Execute(Column) (base_ops.cpp:402-460)
    ///
    /// # Arguments
    /// * `values` - Input column
    /// * `world_size` - Number of processes
    ///
    /// # Returns
    /// Vector of columns, one from each process
    fn execute_column(&mut self, values: &Column, world_size: i32) -> CylonResult<Vec<Arc<Column>>> {
        use crate::net::serialize::CylonColumnSerializer;

        if world_size == 1 {
            return Ok(vec![Column::make(values.data().clone())]);
        }

        let arrow_type = values.data().data_type().clone();

        // Create serializer for local column
        let serializer = CylonColumnSerializer::make(values.data())?;
        let buf_sizes = serializer.buffer_sizes();
        let buffers = serializer.data_buffers();

        // Allgather buffer sizes from all processes
        // Layout: |b_0, b_1, b_2|...|b_0, b_1, b_2| for each column
        let mut all_buf_sizes = vec![0i32; world_size as usize * 3];
        self.allgather_buffer_size(&buf_sizes, 3, &mut all_buf_sizes)?;

        // Calculate total buffer sizes
        let mut total_buf_sizes = [0i32; 3];
        for i in 0..world_size as usize {
            total_buf_sizes[0] += all_buf_sizes[3 * i];
            total_buf_sizes[1] += all_buf_sizes[3 * i + 1];
            total_buf_sizes[2] += all_buf_sizes[3 * i + 2];
        }

        // Calculate displacements and receive counts for each buffer
        let mut displacements: [Vec<i32>; 3] = [
            vec![0i32; world_size as usize],
            vec![0i32; world_size as usize],
            vec![0i32; world_size as usize],
        ];
        let mut recv_counts: [Vec<i32>; 3] = [
            vec![0i32; world_size as usize],
            vec![0i32; world_size as usize],
            vec![0i32; world_size as usize],
        ];

        for buf_idx in 0..3 {
            for i in 0..world_size as usize {
                recv_counts[buf_idx][i] = all_buf_sizes[3 * i + buf_idx];
            }
            // Compute displacements as prefix sum
            for i in 1..world_size as usize {
                displacements[buf_idx][i] = displacements[buf_idx][i - 1] + recv_counts[buf_idx][i - 1];
            }
        }

        // Allocate receive buffers
        let mut received_bufs: [Vec<u8>; 3] = [
            vec![0u8; total_buf_sizes[0] as usize],
            vec![0u8; total_buf_sizes[1] as usize],
            vec![0u8; total_buf_sizes[2] as usize],
        ];

        // Perform allgather for each buffer
        for buf_idx in 0..3 {
            let send_data = buffers[buf_idx].unwrap_or(&[]);
            self.iallgather_buffer_data(
                buf_idx as i32,
                send_data,
                buf_sizes[buf_idx],
                &mut received_bufs[buf_idx],
                &recv_counts[buf_idx],
                &displacements[buf_idx],
            )?;
        }
        self.wait_all()?;

        // Deserialize columns from received buffers
        let mut output = Vec::with_capacity(world_size as usize);
        for i in 0..world_size as usize {
            let sizes = [
                all_buf_sizes[3 * i],
                all_buf_sizes[3 * i + 1],
                all_buf_sizes[3 * i + 2],
            ];
            let offsets = [
                displacements[0][i],
                displacements[1][i],
                displacements[2][i],
            ];

            let col = deserialize_column(
                &arrow_type,
                &received_bufs,
                &sizes,
                &offsets,
            )?;
            output.push(col);
        }

        Ok(output)
    }

    /// Execute AllGather on a Scalar
    ///
    /// Corresponds to C++ AllGatherImpl::Execute(Scalar) (base_ops.cpp:462-480)
    ///
    /// # Arguments
    /// * `value` - Input scalar
    /// * `world_size` - Number of processes
    ///
    /// # Returns
    /// Column containing scalars from all processes
    fn execute_scalar(&mut self, value: &Scalar, world_size: i32) -> CylonResult<Arc<Column>> {
        // Convert scalar to column (it's already stored as single-element array)
        let col = Column::new(value.data().clone());

        // Perform allgather
        let columns = self.execute_column(&col, world_size)?;

        // Concatenate all columns into one
        let arrays: Vec<_> = columns.iter().map(|c| c.data().clone()).collect();
        let concatenated = arrow::compute::concat(&arrays.iter().map(|a| a.as_ref()).collect::<Vec<_>>())
            .map_err(|e| CylonError::new(Code::ExecutionError, e.to_string()))?;

        Ok(Column::make(concatenated))
    }
}

/// Deserialize a column from raw buffers
///
/// Corresponds to C++ DeserializeColumn (table_serialize.cpp:442-470)
fn deserialize_column(
    data_type: &arrow::datatypes::DataType,
    received_bufs: &[Vec<u8>; 3],
    sizes: &[i32; 3],
    offsets: &[i32; 3],
) -> CylonResult<Arc<Column>> {
    // Boolean type is not supported for deserialization (as per C++)
    if matches!(data_type, arrow::datatypes::DataType::Boolean) {
        return Err(CylonError::new(
            Code::Invalid,
            "Deserializing bool type column is not supported".to_string(),
        ));
    }

    // Calculate number of rows from buffer sizes
    let num_rows = calculate_num_rows(data_type, sizes)?;

    // Extract buffer slices
    let validity_buf = if sizes[0] > 0 {
        let start = offsets[0] as usize;
        let end = start + sizes[0] as usize;
        Some(ArrowBuffer::from(received_bufs[0][start..end].to_vec()))
    } else {
        None
    };

    let data_buf = {
        let start = offsets[2] as usize;
        let end = start + sizes[2] as usize;
        ArrowBuffer::from(received_bufs[2][start..end].to_vec())
    };

    // Build array data based on type
    let array_data = if is_fixed_width_type(data_type) {
        ArrayData::builder(data_type.clone())
            .len(num_rows as usize)
            .add_buffer(data_buf)
            .null_bit_buffer(validity_buf)
            .build()
            .map_err(|e| CylonError::new(Code::ExecutionError, e.to_string()))?
    } else if is_binary_like_type(data_type) || is_large_binary_like_type(data_type) {
        let offset_buf = {
            let start = offsets[1] as usize;
            let end = start + sizes[1] as usize;
            ArrowBuffer::from(received_bufs[1][start..end].to_vec())
        };

        ArrayData::builder(data_type.clone())
            .len(num_rows as usize)
            .add_buffer(offset_buf)
            .add_buffer(data_buf)
            .null_bit_buffer(validity_buf)
            .build()
            .map_err(|e| CylonError::new(Code::ExecutionError, e.to_string()))?
    } else {
        return Err(CylonError::new(
            Code::Invalid,
            format!("Unsupported data type for deserialization: {:?}", data_type),
        ));
    };

    Ok(Column::make(arrow::array::make_array(array_data)))
}

/// Calculate number of rows from buffer sizes
///
/// Corresponds to C++ CalculateNumRows (table_serialize.cpp:182-203)
fn calculate_num_rows(data_type: &arrow::datatypes::DataType, buffer_sizes: &[i32; 3]) -> CylonResult<i32> {
    if matches!(data_type, arrow::datatypes::DataType::Boolean) {
        return Err(CylonError::new(
            Code::Invalid,
            "Bool arrays cannot compute rows".to_string(),
        ));
    }

    if is_fixed_width_type(data_type) {
        let byte_width = get_byte_width_for_type(data_type).ok_or_else(|| CylonError::new(
            Code::Invalid,
            format!("Unknown byte width for type {:?}", data_type),
        ))?;
        return Ok(buffer_sizes[2] / byte_width as i32);
    }

    if is_binary_like_type(data_type) {
        return Ok(buffer_sizes[1] / std::mem::size_of::<i32>() as i32 - 1);
    }

    if is_large_binary_like_type(data_type) {
        return Ok(buffer_sizes[1] / std::mem::size_of::<i64>() as i32 - 1);
    }

    Err(CylonError::new(
        Code::Invalid,
        format!("Cannot calculate rows for type {:?}", data_type),
    ))
}

fn is_fixed_width_type(data_type: &arrow::datatypes::DataType) -> bool {
    matches!(
        data_type,
        arrow::datatypes::DataType::Int8 |
        arrow::datatypes::DataType::Int16 |
        arrow::datatypes::DataType::Int32 |
        arrow::datatypes::DataType::Int64 |
        arrow::datatypes::DataType::UInt8 |
        arrow::datatypes::DataType::UInt16 |
        arrow::datatypes::DataType::UInt32 |
        arrow::datatypes::DataType::UInt64 |
        arrow::datatypes::DataType::Float16 |
        arrow::datatypes::DataType::Float32 |
        arrow::datatypes::DataType::Float64 |
        arrow::datatypes::DataType::Date32 |
        arrow::datatypes::DataType::Date64 |
        arrow::datatypes::DataType::Time32(_) |
        arrow::datatypes::DataType::Time64(_) |
        arrow::datatypes::DataType::Timestamp(_, _) |
        arrow::datatypes::DataType::Duration(_)
    )
}

fn is_binary_like_type(data_type: &arrow::datatypes::DataType) -> bool {
    matches!(
        data_type,
        arrow::datatypes::DataType::Utf8 | arrow::datatypes::DataType::Binary
    )
}

fn is_large_binary_like_type(data_type: &arrow::datatypes::DataType) -> bool {
    matches!(
        data_type,
        arrow::datatypes::DataType::LargeUtf8 | arrow::datatypes::DataType::LargeBinary
    )
}

fn get_byte_width_for_type(data_type: &arrow::datatypes::DataType) -> Option<usize> {
    match data_type {
        arrow::datatypes::DataType::Int8 | arrow::datatypes::DataType::UInt8 => Some(1),
        arrow::datatypes::DataType::Int16 | arrow::datatypes::DataType::UInt16 |
        arrow::datatypes::DataType::Float16 => Some(2),
        arrow::datatypes::DataType::Int32 | arrow::datatypes::DataType::UInt32 |
        arrow::datatypes::DataType::Float32 | arrow::datatypes::DataType::Date32 |
        arrow::datatypes::DataType::Time32(_) => Some(4),
        arrow::datatypes::DataType::Int64 | arrow::datatypes::DataType::UInt64 |
        arrow::datatypes::DataType::Float64 | arrow::datatypes::DataType::Date64 |
        arrow::datatypes::DataType::Time64(_) | arrow::datatypes::DataType::Timestamp(_, _) |
        arrow::datatypes::DataType::Duration(_) => Some(8),
        _ => None,
    }
}
