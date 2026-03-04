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

//! Libfabric Operation Implementations
//!
//! This module bridges libfabric's CollectiveOps primitives with the
//! base_ops.rs orchestration traits for table-level operations.
//!
//! Provides operation implementations:
//! - LfTableAllgatherImpl
//! - LfTableGatherImpl
//! - LfTableBcastImpl
//! - LfAllReduceImpl
//! - LfAllgatherImpl

use crate::error::{CylonError, CylonResult, Code};
use crate::net::ops::{TableAllgatherImpl, TableGatherImpl, TableBcastImpl};
use crate::net::ops::base_ops::{AllReduceImpl, AllGatherImpl};
use crate::net::comm_operations::ReduceOp;
use crate::DataType;

use super::communicator::LibfabricCommunicator;
use super::libfabric_sys::*;

/// Libfabric Table Allgather Implementation
///
/// Uses libfabric's native fi_allgather for fixed-size buffer sizes,
/// and point-to-point send/recv for variable-length data (allgatherv).
pub struct LfTableAllgatherImpl<'a> {
    comm: &'a LibfabricCommunicator,
    rank: i32,
    world_size: i32,
}

impl<'a> LfTableAllgatherImpl<'a> {
    pub fn new(comm: &'a LibfabricCommunicator) -> Self {
        let rank = comm.rank();
        let world_size = comm.world_size();
        Self { comm, rank, world_size }
    }
}

impl<'a> TableAllgatherImpl for LfTableAllgatherImpl<'a> {
    fn init(&mut self, _num_buffers: i32) {
        // No-op
    }

    fn allgather_buffer_sizes(
        &self,
        send_data: &[i32],
        num_buffers: i32,
        rcv_data: &mut [i32],
    ) -> CylonResult<()> {
        // Use native fi_allgather for fixed-size i32 data
        let send_bytes = unsafe {
            std::slice::from_raw_parts(
                send_data.as_ptr() as *const u8,
                (num_buffers as usize) * std::mem::size_of::<i32>(),
            )
        };

        let recv_byte_size = (self.world_size as usize)
            * (num_buffers as usize)
            * std::mem::size_of::<i32>();

        let recv_bytes = unsafe {
            std::slice::from_raw_parts_mut(
                rcv_data.as_mut_ptr() as *mut u8,
                recv_byte_size,
            )
        };

        let op_id = self.comm.coll_ops().allgather(send_bytes, recv_bytes)?;
        self.comm.wait_for_coll_op(op_id)
    }

    fn iallgather_buffer_data(
        &mut self,
        _buf_idx: i32,
        send_data: &[u8],
        send_count: i32,
        recv_data: &mut [u8],
        recv_count: &[i32],
        displacements: &[i32],
    ) -> CylonResult<()> {
        // Libfabric fi_allgather only supports fixed-size data.
        // For variable-length allgatherv, use point-to-point send/recv.
        // Pattern: each rank sends to all, each rank recvs from all.
        let send_slice = &send_data[..send_count as usize];

        for peer in 0..self.world_size {
            if peer != self.rank {
                // Send our data to this peer
                self.comm.send_bytes(send_slice, peer, 0)?;
            }
        }

        for peer in 0..self.world_size {
            let offset = displacements[peer as usize] as usize;
            let count = recv_count[peer as usize] as usize;

            if peer == self.rank {
                // Copy local data directly
                recv_data[offset..offset + count]
                    .copy_from_slice(&send_slice[..count]);
            } else {
                // Receive from peer
                let mut buf = vec![0u8; count];
                self.comm.recv_bytes(&mut buf, peer, 0)?;
                recv_data[offset..offset + count].copy_from_slice(&buf);
            }
        }

        Ok(())
    }

    fn wait_all(&mut self, _num_buffers: i32) -> CylonResult<()> {
        // Operations are blocking, nothing to wait for
        Ok(())
    }
}

/// Libfabric Table Gather Implementation
///
/// Uses fi_gather for fixed-size buffer sizes and point-to-point for
/// variable-length gatherv.
pub struct LfTableGatherImpl<'a> {
    comm: &'a LibfabricCommunicator,
    rank: i32,
    world_size: i32,
}

impl<'a> LfTableGatherImpl<'a> {
    pub fn new(comm: &'a LibfabricCommunicator) -> Self {
        let rank = comm.rank();
        let world_size = comm.world_size();
        Self { comm, rank, world_size }
    }
}

impl<'a> TableGatherImpl for LfTableGatherImpl<'a> {
    fn init(&mut self, _num_buffers: i32) {
        // No-op
    }

    fn gather_buffer_sizes(
        &self,
        send_data: &[i32],
        num_buffers: i32,
        rcv_data: &mut [i32],
        gather_root: i32,
    ) -> CylonResult<()> {
        // Use native fi_gather for fixed-size i32 data
        let send_bytes = unsafe {
            std::slice::from_raw_parts(
                send_data.as_ptr() as *const u8,
                (num_buffers as usize) * std::mem::size_of::<i32>(),
            )
        };

        let recv_byte_size = (self.world_size as usize)
            * (num_buffers as usize)
            * std::mem::size_of::<i32>();

        let recv_bytes = unsafe {
            std::slice::from_raw_parts_mut(
                rcv_data.as_mut_ptr() as *mut u8,
                recv_byte_size,
            )
        };

        let op_id = self.comm.coll_ops().gather(send_bytes, recv_bytes, gather_root)?;
        self.comm.wait_for_coll_op(op_id)
    }

    fn igather_buffer_data(
        &mut self,
        _buf_idx: i32,
        send_data: &[u8],
        send_count: i32,
        recv_data: &mut [u8],
        recv_count: &[i32],
        displacements: &[i32],
        gather_root: i32,
    ) -> CylonResult<()> {
        // Variable-length gatherv via point-to-point
        let send_slice = &send_data[..send_count as usize];

        if self.rank == gather_root {
            // Root: copy own data and receive from all others
            let my_offset = displacements[self.rank as usize] as usize;
            let my_count = recv_count[self.rank as usize] as usize;
            if my_count > 0 {
                recv_data[my_offset..my_offset + my_count]
                    .copy_from_slice(&send_slice[..my_count]);
            }

            for peer in 0..self.world_size {
                if peer != self.rank {
                    let offset = displacements[peer as usize] as usize;
                    let count = recv_count[peer as usize] as usize;
                    if count > 0 {
                        let mut buf = vec![0u8; count];
                        self.comm.recv_bytes(&mut buf, peer, 0)?;
                        recv_data[offset..offset + count].copy_from_slice(&buf);
                    }
                }
            }
        } else {
            // Non-root: send data to root
            if send_count > 0 {
                self.comm.send_bytes(send_slice, gather_root, 0)?;
            }
        }

        Ok(())
    }

    fn wait_all(&mut self, _num_buffers: i32) -> CylonResult<()> {
        // Operations are blocking
        Ok(())
    }
}

/// Libfabric Table Broadcast Implementation
///
/// Uses fi_broadcast for both size and data broadcasting.
pub struct LfTableBcastImpl<'a> {
    comm: &'a LibfabricCommunicator,
}

impl<'a> LfTableBcastImpl<'a> {
    pub fn new(comm: &'a LibfabricCommunicator) -> Self {
        Self { comm }
    }
}

impl<'a> TableBcastImpl for LfTableBcastImpl<'a> {
    fn init(&mut self, _num_buffers: i32) {
        // No-op
    }

    fn bcast_buffer_sizes(
        &self,
        buffer: &mut [i32],
        _count: i32,
        bcast_root: i32,
    ) -> CylonResult<()> {
        // Use native fi_broadcast for i32 data
        let op_id = self.comm.coll_ops().broadcast(buffer, bcast_root)?;
        self.comm.wait_for_coll_op(op_id)
    }

    fn bcast_buffer_data(
        &self,
        buf_data: &mut [u8],
        send_count: i32,
        bcast_root: i32,
    ) -> CylonResult<()> {
        let op_id = self.comm.coll_ops().broadcast(
            &mut buf_data[..send_count as usize],
            bcast_root,
        )?;
        self.comm.wait_for_coll_op(op_id)
    }

    fn ibcast_buffer_data(
        &mut self,
        _buf_idx: i32,
        buf_data: &mut [u8],
        send_count: i32,
        bcast_root: i32,
    ) -> CylonResult<()> {
        // Use blocking broadcast (we wait immediately in wait_all)
        let op_id = self.comm.coll_ops().broadcast(
            &mut buf_data[..send_count as usize],
            bcast_root,
        )?;
        self.comm.wait_for_coll_op(op_id)
    }

    fn wait_all(&mut self, _num_buffers: i32) -> CylonResult<()> {
        // Operations are blocking
        Ok(())
    }
}

/// Libfabric AllReduce Implementation
///
/// Uses native fi_allreduce with libfabric's built-in reduction operations.
pub struct LfAllReduceImpl<'a> {
    comm: &'a LibfabricCommunicator,
}

impl<'a> LfAllReduceImpl<'a> {
    pub fn new(comm: &'a LibfabricCommunicator) -> Self {
        Self { comm }
    }
}

/// Get the byte size of a DataType for allreduce buffer sizing
fn type_size_bytes(data_type: &DataType) -> CylonResult<usize> {
    use crate::Type;
    match data_type.get_type() {
        Type::Bool => Ok(1),
        Type::UInt8 | Type::Int8 => Ok(1),
        Type::UInt16 | Type::Int16 | Type::HalfFloat => Ok(2),
        Type::UInt32 | Type::Int32 | Type::Float | Type::Date32 | Type::Time32 => Ok(4),
        Type::UInt64 | Type::Int64 | Type::Double | Type::Date64 | Type::Time64 | Type::Timestamp | Type::Duration => Ok(8),
        _ => Err(CylonError::new(
            Code::NotImplemented,
            format!("allreduce not implemented for type {:?}", data_type.get_type()),
        )),
    }
}

/// Map Cylon DataType to libfabric fi_datatype for allreduce
fn cylon_type_to_fi_datatype(data_type: &DataType) -> CylonResult<fi_datatype> {
    use crate::Type;
    match data_type.get_type() {
        Type::Int8 => Ok(FI_INT8),
        Type::UInt8 | Type::Bool => Ok(FI_UINT8),
        Type::Int16 => Ok(FI_INT16),
        Type::UInt16 => Ok(FI_UINT16),
        Type::Int32 => Ok(FI_INT32),
        Type::UInt32 | Type::Date32 | Type::Time32 => Ok(FI_UINT32),
        Type::Int64 => Ok(FI_INT64),
        Type::UInt64 | Type::Date64 | Type::Time64 | Type::Timestamp | Type::Duration => Ok(FI_UINT64),
        Type::Float => Ok(FI_FLOAT),
        Type::Double => Ok(FI_DOUBLE),
        _ => Err(CylonError::new(
            Code::NotImplemented,
            format!("No fi_datatype mapping for {:?}", data_type.get_type()),
        )),
    }
}

impl<'a> AllReduceImpl for LfAllReduceImpl<'a> {
    fn allreduce_buffer(
        &self,
        send_buf: &[u8],
        rcv_buf: &mut [u8],
        count: i32,
        data_type: &DataType,
        reduce_op: ReduceOp,
    ) -> CylonResult<()> {
        let elem_size = type_size_bytes(data_type)?;
        let data_byte_size = (count as usize) * elem_size;

        // We need to dispatch based on the element size to get the right
        // fi_datatype for the native allreduce call
        // Validate that the type is supported for allreduce
        cylon_type_to_fi_datatype(data_type)?;

        // Use CollectiveOps::allreduce which handles fi_op/fi_datatype mapping.
        // Dispatch by element size
        // We need to call allreduce with typed buffers, but CollectiveOps::allreduce
        // is generic. We dispatch based on element size.
        match elem_size {
            1 => {
                let send_slice = &send_buf[..data_byte_size];
                let recv_slice = &mut rcv_buf[..data_byte_size];
                let op_id = self.comm.coll_ops().allreduce(send_slice, recv_slice, reduce_op)?;
                self.comm.wait_for_coll_op(op_id)
            }
            2 => {
                let send_slice = unsafe {
                    std::slice::from_raw_parts(send_buf.as_ptr() as *const u16, count as usize)
                };
                let recv_slice = unsafe {
                    std::slice::from_raw_parts_mut(rcv_buf.as_mut_ptr() as *mut u16, count as usize)
                };
                let op_id = self.comm.coll_ops().allreduce(send_slice, recv_slice, reduce_op)?;
                self.comm.wait_for_coll_op(op_id)
            }
            4 => {
                let send_slice = unsafe {
                    std::slice::from_raw_parts(send_buf.as_ptr() as *const u32, count as usize)
                };
                let recv_slice = unsafe {
                    std::slice::from_raw_parts_mut(rcv_buf.as_mut_ptr() as *mut u32, count as usize)
                };
                let op_id = self.comm.coll_ops().allreduce(send_slice, recv_slice, reduce_op)?;
                self.comm.wait_for_coll_op(op_id)
            }
            8 => {
                let send_slice = unsafe {
                    std::slice::from_raw_parts(send_buf.as_ptr() as *const u64, count as usize)
                };
                let recv_slice = unsafe {
                    std::slice::from_raw_parts_mut(rcv_buf.as_mut_ptr() as *mut u64, count as usize)
                };
                let op_id = self.comm.coll_ops().allreduce(send_slice, recv_slice, reduce_op)?;
                self.comm.wait_for_coll_op(op_id)
            }
            _ => Err(CylonError::new(
                Code::NotImplemented,
                format!("Unsupported element size {} for allreduce", elem_size),
            )),
        }
    }
}

/// Libfabric Allgather Implementation (for Column/Scalar)
///
/// Uses fi_allgather for fixed-size buffer sizes and point-to-point
/// for variable-length allgatherv.
pub struct LfAllgatherImpl<'a> {
    comm: &'a LibfabricCommunicator,
    world_size: i32,
    rank: i32,
}

impl<'a> LfAllgatherImpl<'a> {
    pub fn new(comm: &'a LibfabricCommunicator) -> Self {
        let world_size = comm.world_size();
        let rank = comm.rank();
        Self { comm, world_size, rank }
    }
}

impl<'a> AllGatherImpl for LfAllgatherImpl<'a> {
    fn allgather_buffer_size(
        &self,
        send_data: &[i32],
        num_buffers: i32,
        rcv_data: &mut [i32],
    ) -> CylonResult<()> {
        // Use native fi_allgather for fixed-size i32 data
        let send_bytes = unsafe {
            std::slice::from_raw_parts(
                send_data.as_ptr() as *const u8,
                (num_buffers as usize) * std::mem::size_of::<i32>(),
            )
        };

        let recv_byte_size = (self.world_size as usize)
            * (num_buffers as usize)
            * std::mem::size_of::<i32>();

        let recv_bytes = unsafe {
            std::slice::from_raw_parts_mut(
                rcv_data.as_mut_ptr() as *mut u8,
                recv_byte_size,
            )
        };

        let op_id = self.comm.coll_ops().allgather(send_bytes, recv_bytes)?;
        self.comm.wait_for_coll_op(op_id)
    }

    fn iallgather_buffer_data(
        &mut self,
        _buf_idx: i32,
        send_data: &[u8],
        send_count: i32,
        recv_data: &mut [u8],
        recv_count: &[i32],
        displacements: &[i32],
    ) -> CylonResult<()> {
        // Variable-length allgatherv via point-to-point
        let send_slice = &send_data[..send_count as usize];

        for peer in 0..self.world_size {
            if peer != self.rank {
                self.comm.send_bytes(send_slice, peer, 0)?;
            }
        }

        for peer in 0..self.world_size {
            let offset = displacements[peer as usize] as usize;
            let count = recv_count[peer as usize] as usize;

            if peer == self.rank {
                recv_data[offset..offset + count]
                    .copy_from_slice(&send_slice[..count]);
            } else {
                let mut buf = vec![0u8; count];
                self.comm.recv_bytes(&mut buf, peer, 0)?;
                recv_data[offset..offset + count].copy_from_slice(&buf);
            }
        }

        Ok(())
    }

    fn wait_all(&mut self) -> CylonResult<()> {
        // Operations are blocking
        Ok(())
    }
}