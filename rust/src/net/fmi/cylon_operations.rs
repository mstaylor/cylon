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

//! FMI Operation Implementations
//!
//! This module corresponds to cpp/src/cylon/net/fmi/fmi_operations.hpp/cpp
//!
//! Provides operation implementations for table-level collectives:
//! - FmiTableAllgatherImpl
//! - FmiTableGatherImpl
//! - FmiTableBcastImpl
//! - FmiAllReduceImpl
//! - FmiAllgatherImpl

use std::sync::Arc;

use crate::error::{CylonError, CylonResult, Code};
use crate::net::ops::{TableAllgatherImpl, TableGatherImpl, TableBcastImpl};
use crate::net::ops::base_ops::{AllReduceImpl, AllGatherImpl};
use crate::net::comm_operations::ReduceOp;
use crate::net::serialize::{serialize_table, deserialize_table};
use crate::ctx::CylonContext;
use crate::table::Table;
use crate::DataType;

use super::common::{Mode, Operation, EventProcessStatus, NbxStatus};
use super::communicator::Communicator as FmiCommunicator;

/// Convert NbxStatus to string for logging
fn nbx_status_to_string(status: NbxStatus) -> &'static str {
    match status {
        NbxStatus::Success => "SUCCESS",
        NbxStatus::SendFailed => "SEND_FAILED",
        NbxStatus::ReceiveFailed => "RECEIVE_FAILED",
        NbxStatus::DummySendFailed => "DUMMY_SEND_FAILED",
        NbxStatus::ConnectionClosedByPeer => "CONNECTION_CLOSED_BY_PEER",
        NbxStatus::SocketCreateFailed => "SOCKET_CREATE_FAILED",
        NbxStatus::TcpNoDelayFailed => "TCP_NODELAY_FAILED",
        NbxStatus::FcntlGetFailed => "FCNTL_GET_FAILED",
        NbxStatus::FcntlSetFailed => "FCNTL_SET_FAILED",
        NbxStatus::AddEventFailed => "ADD_EVENT_FAILED",
        NbxStatus::EpollWaitFailed => "EPOLL_WAIT_FAILED",
        NbxStatus::SocketPairFailed => "SOCKET_PAIR_FAILED",
        NbxStatus::SocketSetRcvTimeoFailed => "SOCKET_SET_SO_RCVTIMEO_FAILED",
        NbxStatus::SocketSetSndTimeoFailed => "SOCKET_SET_SO_SNDTIMEO_FAILED",
        NbxStatus::SocketSetTcpNoDelayFailed => "SOCKET_SET_TCP_NODELAY_FAILED",
        NbxStatus::SocketSetNonBlockingFailed => "SOCKET_SET_NONBLOCKING_FAILED",
        NbxStatus::NbxTimeout => "NBX_TIMEOUT",
    }
}

/// FMI Table Allgather Implementation
///
/// Matches C++ class: cylon::fmi::FmiTableAllgatherImpl
pub struct FmiTableAllgatherImpl {
    comm_ptr: Arc<FmiCommunicator>,
    mode: Mode,
    world_size: i32,
}

impl FmiTableAllgatherImpl {
    pub fn new(comm_ptr: Arc<FmiCommunicator>, mode: Mode) -> Self {
        let world_size = comm_ptr.get_num_peers();
        Self { comm_ptr, mode, world_size }
    }
}

impl TableAllgatherImpl for FmiTableAllgatherImpl {
    fn init(&mut self, _num_buffers: i32) {
        // No-op, matches C++ CYLON_UNUSED(num_buffers)
    }

    fn allgather_buffer_sizes(
        &self,
        send_data: &[i32],
        num_buffers: i32,
        rcv_data: &mut [i32],
    ) -> CylonResult<()> {
        // Matches C++: comm_ptr_->allgather(send_void_data, recv_void_data, 0)
        let send_bytes = unsafe {
            std::slice::from_raw_parts(
                send_data.as_ptr() as *const u8,
                (num_buffers as usize) * std::mem::size_of::<i32>(),
            )
        };

        let recv_byte_size = (self.comm_ptr.get_num_peers() as usize)
            * (num_buffers as usize)
            * std::mem::size_of::<i32>();

        let recv_bytes = unsafe {
            std::slice::from_raw_parts_mut(
                rcv_data.as_mut_ptr() as *mut u8,
                recv_byte_size,
            )
        };

        self.comm_ptr.allgather(send_bytes, recv_bytes, 0)
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
        // Matches C++: comm_ptr_->allgatherv(send_void_data, recv_void_data, 0, recv_count, displacements, mode_, callback)
        let callback = Arc::new(|status: NbxStatus, msg: &str, _ctx: &mut super::common::FmiContext| {
            if status != NbxStatus::Success {
                log::error!(
                    "FMI IallgatherBufferData status: {} msg: {}",
                    nbx_status_to_string(status),
                    msg
                );
            }
        });

        self.comm_ptr.allgatherv_async(
            &send_data[..send_count as usize],
            recv_data,
            0,
            recv_count,
            displacements,
            self.mode,
            Some(callback),
        )
    }

    fn wait_all(&mut self, _num_buffers: i32) -> CylonResult<()> {
        // Matches C++: while (comm_ptr_->communicator_event_progress(FMI::Utils::Operation::DEFAULT) == ...)
        if self.mode == Mode::NonBlocking {
            while self.comm_ptr.communicator_event_progress(Operation::Default)
                == EventProcessStatus::Processing
            {}
        }
        Ok(())
    }

    fn execute(&mut self, table: &Table, ctx: Arc<CylonContext>) -> CylonResult<Vec<Table>> {
        // 1. Serialize local table
        let serialized_data = serialize_table(table)?;
        let send_size = serialized_data.len() as i32;

        // 2. Allgather buffer sizes
        let mut all_sizes = vec![0i32; self.world_size as usize];
        self.allgather_buffer_sizes(&[send_size], 1, &mut all_sizes)?;

        // 3. Calculate displacements and total size
        let mut displacements = vec![0i32; self.world_size as usize];
        let mut cumulative = 0i32;
        for i in 0..self.world_size as usize {
            displacements[i] = cumulative;
            cumulative += all_sizes[i];
        }
        let total_size = cumulative as usize;

        // 4. Allgather actual data
        let mut recv_buffer = vec![0u8; total_size];
        self.iallgather_buffer_data(
            0,
            &serialized_data,
            send_size,
            &mut recv_buffer,
            &all_sizes,
            &displacements,
        )?;

        self.wait_all(1)?;

        // 5. Deserialize all tables
        let mut result = Vec::new();
        for i in 0..self.world_size as usize {
            let start = displacements[i] as usize;
            let end = start + all_sizes[i] as usize;
            let table_data = &recv_buffer[start..end];
            let deserialized_table = deserialize_table(ctx.clone(), table_data)?;
            result.push(deserialized_table);
        }

        Ok(result)
    }
}

/// FMI Table Gather Implementation
///
/// Matches C++ class: cylon::fmi::FmiTableGatherImpl
pub struct FmiTableGatherImpl {
    comm_ptr: Arc<FmiCommunicator>,
    mode: Mode,
    rank: i32,
    world_size: i32,
}

impl FmiTableGatherImpl {
    pub fn new(comm_ptr: Arc<FmiCommunicator>, mode: Mode) -> Self {
        let rank = comm_ptr.get_peer_id();
        let world_size = comm_ptr.get_num_peers();
        Self { comm_ptr, mode, rank, world_size }
    }
}

impl TableGatherImpl for FmiTableGatherImpl {
    fn init(&mut self, _num_buffers: i32) {
        // No-op
    }

    fn gather_buffer_sizes(
        &self,
        send_data: &[i32],
        num_buffers: i32,
        rcv_data: &mut [i32],
        _gather_root: i32,
    ) -> CylonResult<()> {
        // Matches C++: comm_ptr_->gather(send_void_data, recv_void_data, 0)
        let send_bytes = unsafe {
            std::slice::from_raw_parts(
                send_data.as_ptr() as *const u8,
                (num_buffers as usize) * std::mem::size_of::<i32>(),
            )
        };

        let recv_byte_size = (self.comm_ptr.get_num_peers() as usize)
            * (num_buffers as usize)
            * std::mem::size_of::<i32>();

        let recv_bytes = unsafe {
            std::slice::from_raw_parts_mut(
                rcv_data.as_mut_ptr() as *mut u8,
                recv_byte_size,
            )
        };

        self.comm_ptr.gather(send_bytes, recv_bytes, 0)
    }

    fn igather_buffer_data(
        &mut self,
        _buf_idx: i32,
        send_data: &[u8],
        send_count: i32,
        recv_data: &mut [u8],
        recv_count: &[i32],
        displacements: &[i32],
        _gather_root: i32,
    ) -> CylonResult<()> {
        // Matches C++: comm_ptr_->gatherv(send_void_data, recv_void_data, 0, recv_count, displacements, mode_, callback)
        let callback = Arc::new(|status: NbxStatus, msg: &str, _ctx: &mut super::common::FmiContext| {
            if status != NbxStatus::Success {
                log::error!(
                    "FMI IgatherBufferData status: {} msg: {}",
                    nbx_status_to_string(status),
                    msg
                );
            }
        });

        self.comm_ptr.gatherv_async(
            &send_data[..send_count as usize],
            recv_data,
            0,
            recv_count,
            displacements,
            self.mode,
            Some(callback),
        )
    }

    fn wait_all(&mut self, _num_buffers: i32) -> CylonResult<()> {
        if self.mode == Mode::NonBlocking {
            while self.comm_ptr.communicator_event_progress(Operation::Default)
                == EventProcessStatus::Processing
            {}
        }
        Ok(())
    }

    fn execute(
        &mut self,
        table: &Table,
        gather_root: i32,
        gather_from_root: bool,
        ctx: Arc<CylonContext>,
    ) -> CylonResult<Vec<Table>> {
        let is_root = self.rank == gather_root;

        // 1. Serialize local table
        let serialized_data = serialize_table(table)?;
        let send_size = serialized_data.len() as i32;

        // 2. Gather buffer sizes
        let mut all_sizes = if is_root {
            vec![0i32; self.world_size as usize]
        } else {
            vec![]
        };

        if is_root {
            self.gather_buffer_sizes(&[send_size], 1, &mut all_sizes, gather_root)?;
        } else {
            self.gather_buffer_sizes(&[send_size], 1, &mut [], gather_root)?;
        }

        // 3. Calculate displacements and total size (root only)
        let (total_size, displacements) = if is_root {
            let mut disps = vec![0i32; self.world_size as usize];
            let mut cumulative = 0i32;
            for i in 0..self.world_size as usize {
                disps[i] = cumulative;
                cumulative += all_sizes[i];
            }
            (cumulative as usize, disps)
        } else {
            (0, vec![])
        };

        // 4. Gather actual data
        let mut recv_buffer = if is_root {
            vec![0u8; total_size]
        } else {
            vec![]
        };

        if is_root {
            self.igather_buffer_data(
                0,
                &serialized_data,
                send_size,
                &mut recv_buffer,
                &all_sizes,
                &displacements,
                gather_root,
            )?;
        } else {
            self.igather_buffer_data(
                0,
                &serialized_data,
                send_size,
                &mut [],
                &[],
                &[],
                gather_root,
            )?;
        }

        self.wait_all(1)?;

        // 5. Deserialize tables (root only)
        let mut result = Vec::new();
        if is_root {
            let start_idx = if gather_from_root { 0 } else { 1 };
            for i in start_idx..self.world_size as usize {
                let start = displacements[i] as usize;
                let end = start + all_sizes[i] as usize;
                let table_data = &recv_buffer[start..end];
                let deserialized_table = deserialize_table(ctx.clone(), table_data)?;
                result.push(deserialized_table);
            }
        }

        Ok(result)
    }
}

/// FMI Table Broadcast Implementation
///
/// Matches C++ class: cylon::fmi::FmiTableBcastImpl
pub struct FmiTableBcastImpl {
    comm_ptr: Arc<FmiCommunicator>,
    mode: Mode,
    rank: i32,
}

impl FmiTableBcastImpl {
    pub fn new(comm_ptr: Arc<FmiCommunicator>, mode: Mode) -> Self {
        let rank = comm_ptr.get_peer_id();
        Self { comm_ptr, mode, rank }
    }
}

impl TableBcastImpl for FmiTableBcastImpl {
    fn init(&mut self, _num_buffers: i32) {
        // No-op
    }

    fn bcast_buffer_sizes(
        &self,
        buffer: &mut [i32],
        count: i32,
        bcast_root: i32,
    ) -> CylonResult<()> {
        // Matches C++: comm_ptr_->bcast(send_void_data, bcast_root)
        let data_bytes = unsafe {
            std::slice::from_raw_parts_mut(
                buffer.as_mut_ptr() as *mut u8,
                (count as usize) * std::mem::size_of::<i32>(),
            )
        };

        self.comm_ptr.bcast(data_bytes, bcast_root)
    }

    fn bcast_buffer_data(
        &self,
        buf_data: &mut [u8],
        send_count: i32,
        bcast_root: i32,
    ) -> CylonResult<()> {
        // Matches C++: comm_ptr_->bcast(send_void_data, bcast_root)
        self.comm_ptr.bcast(&mut buf_data[..send_count as usize], bcast_root)
    }

    fn ibcast_buffer_data(
        &mut self,
        _buf_idx: i32,
        buf_data: &mut [u8],
        send_count: i32,
        bcast_root: i32,
    ) -> CylonResult<()> {
        // Matches C++: comm_ptr_->bcast(send_void_data, bcast_root, mode_, callback)
        let callback = Arc::new(|status: NbxStatus, msg: &str, _ctx: &mut super::common::FmiContext| {
            if status != NbxStatus::Success {
                log::error!(
                    "FMI IbcastBufferData status: {} msg: {}",
                    nbx_status_to_string(status),
                    msg
                );
            }
        });

        self.comm_ptr.bcast_async(
            &mut buf_data[..send_count as usize],
            bcast_root,
            self.mode,
            Some(callback),
        )
    }

    fn wait_all(&mut self, _num_buffers: i32) -> CylonResult<()> {
        if self.mode == Mode::NonBlocking {
            while self.comm_ptr.communicator_event_progress(Operation::Default)
                == EventProcessStatus::Processing
            {}
        }
        Ok(())
    }

    fn execute(&mut self, table: &mut Option<Table>, bcast_root: i32, ctx: Arc<CylonContext>) -> CylonResult<()> {
        let is_root = self.rank == bcast_root;

        // 1. Serialize on root, get size
        let (serialized_data, data_size) = if is_root {
            let t = table.as_ref().ok_or_else(|| {
                CylonError::new(Code::Invalid, "Root must have a table for broadcast")
            })?;
            let data = serialize_table(t)?;
            let size = data.len() as i32;
            (data, size)
        } else {
            (Vec::new(), 0)
        };

        // 2. Broadcast size
        let mut size_buf = [data_size];
        self.bcast_buffer_sizes(&mut size_buf, 1, bcast_root)?;
        let recv_size = size_buf[0] as usize;

        // 3. Broadcast data
        let mut data_buf = if is_root {
            serialized_data
        } else {
            vec![0u8; recv_size]
        };

        self.bcast_buffer_data(&mut data_buf, recv_size as i32, bcast_root)?;

        // 4. Deserialize on non-root
        if !is_root {
            let deserialized = deserialize_table(ctx, &data_buf)?;
            *table = Some(deserialized);
        }

        Ok(())
    }
}

/// Get the byte size of a DataType
///
/// Matches C++ type size dispatch logic
fn type_size_bytes(data_type: &DataType) -> usize {
    use crate::Type;
    match data_type.get_type() {
        Type::Bool => 1, // Stored as byte in Arrow
        Type::UInt8 | Type::Int8 => 1,
        Type::UInt16 | Type::Int16 | Type::HalfFloat => 2,
        Type::UInt32 | Type::Int32 | Type::Float | Type::Date32 | Type::Time32 => 4,
        Type::UInt64 | Type::Int64 | Type::Double | Type::Date64 | Type::Time64 | Type::Timestamp | Type::Duration => 8,
        Type::Decimal => 16, // Decimal128
        Type::Interval => 8, // Year-month or day-time
        // Variable width types - return 0 as they don't have fixed size
        Type::String | Type::Binary | Type::LargeString | Type::LargeBinary | Type::List | Type::FixedSizeList | Type::FixedSizeBinary | Type::Extension | Type::MaxId => 0,
    }
}

/// FMI AllReduce Implementation
///
/// Matches C++ class: cylon::fmi::FmiAllReduceImpl
pub struct FmiAllReduceImpl {
    comm_ptr: Arc<FmiCommunicator>,
    _mode: Mode,
}

impl FmiAllReduceImpl {
    pub fn new(comm_ptr: Arc<FmiCommunicator>, mode: Mode) -> Self {
        Self { comm_ptr, _mode: mode }
    }
}

impl AllReduceImpl for FmiAllReduceImpl {
    fn allreduce_buffer(
        &self,
        send_buf: &[u8],
        rcv_buf: &mut [u8],
        count: i32,
        data_type: &DataType,
        reduce_op: ReduceOp,
    ) -> CylonResult<()> {
        // Get the reduction function based on data type and operation
        let data_type_size = type_size_bytes(data_type);
        let (func, commutative, associative) = get_reduce_function(data_type, reduce_op)?;

        // Matches C++: comm_ptr->allreduce(send_void_data, recv_void_data, commutative, associative, f)
        let data_byte_size = (count as usize) * data_type_size;
        self.comm_ptr.allreduce(
            &send_buf[..data_byte_size],
            &mut rcv_buf[..data_byte_size],
            func,
            associative,
            commutative,
        )
    }
}

/// FMI Allgather Implementation (for Column/Scalar)
///
/// Matches C++ class: cylon::fmi::FmiAllgatherImpl
pub struct FmiAllgatherImpl {
    comm_ptr: Arc<FmiCommunicator>,
    mode: Mode,
}

impl FmiAllgatherImpl {
    pub fn new(comm_ptr: Arc<FmiCommunicator>, mode: Mode) -> Self {
        Self { comm_ptr, mode }
    }
}

impl AllGatherImpl for FmiAllgatherImpl {
    fn allgather_buffer_size(
        &self,
        send_data: &[i32],
        num_buffers: i32,
        rcv_data: &mut [i32],
    ) -> CylonResult<()> {
        // Matches C++: comm_ptr_->allgather(send_void_data, recv_void_data, 0)
        let send_bytes = unsafe {
            std::slice::from_raw_parts(
                send_data.as_ptr() as *const u8,
                (num_buffers as usize) * std::mem::size_of::<i32>(),
            )
        };

        let recv_byte_size = (self.comm_ptr.get_num_peers() as usize)
            * (num_buffers as usize)
            * std::mem::size_of::<i32>();

        let recv_bytes = unsafe {
            std::slice::from_raw_parts_mut(
                rcv_data.as_mut_ptr() as *mut u8,
                recv_byte_size,
            )
        };

        self.comm_ptr.allgather(send_bytes, recv_bytes, 0)
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
        // Matches C++: comm_ptr_->allgatherv(send_void_data, recv_void_data, 0, recv_count, displacements, mode_, callback)
        let callback = Arc::new(|status: NbxStatus, msg: &str, _ctx: &mut super::common::FmiContext| {
            if status != NbxStatus::Success {
                log::error!(
                    "FMI IallgatherBufferData status: {} msg: {}",
                    nbx_status_to_string(status),
                    msg
                );
            }
        });

        self.comm_ptr.allgatherv_async(
            &send_data[..send_count as usize],
            recv_data,
            0,
            recv_count,
            displacements,
            self.mode,
            Some(callback),
        )
    }

    fn wait_all(&self) -> CylonResult<()> {
        if self.mode == Mode::NonBlocking {
            while self.comm_ptr.communicator_event_progress(Operation::Default)
                == EventProcessStatus::Processing
            {}
        }
        Ok(())
    }
}

/// Get reduction function for the given data type and reduce operation
/// Matches C++ all_reduce_buffer<T> template dispatch
fn get_reduce_function(
    data_type: &DataType,
    reduce_op: ReduceOp,
) -> CylonResult<(Box<dyn Fn(&mut [u8], &[u8]) + Send + Sync>, bool, bool)> {
    use crate::Type;

    match data_type.get_type() {
        Type::UInt8 => get_typed_reduce_function::<u8>(reduce_op),
        Type::Int8 => get_typed_reduce_function::<i8>(reduce_op),
        Type::UInt16 => get_typed_reduce_function::<u16>(reduce_op),
        Type::Int16 => get_typed_reduce_function::<i16>(reduce_op),
        Type::UInt32 | Type::Date32 | Type::Time32 => get_typed_reduce_function::<u32>(reduce_op),
        Type::Int32 => get_typed_reduce_function::<i32>(reduce_op),
        Type::UInt64 | Type::Date64 | Type::Timestamp | Type::Time64 => get_typed_reduce_function::<u64>(reduce_op),
        Type::Int64 => get_typed_reduce_function::<i64>(reduce_op),
        Type::Float => get_typed_reduce_function_float::<f32>(reduce_op),
        Type::Double => get_typed_reduce_function_float::<f64>(reduce_op),
        _ => Err(CylonError::new(
            Code::NotImplemented,
            format!("allreduce not implemented for type {:?}", data_type.get_type()),
        )),
    }
}

fn get_typed_reduce_function<T>(
    reduce_op: ReduceOp,
) -> CylonResult<(Box<dyn Fn(&mut [u8], &[u8]) + Send + Sync>, bool, bool)>
where
    T: Copy + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + Ord + 'static,
{
    let (func, associative, commutative): (Box<dyn Fn(&mut [u8], &[u8]) + Send + Sync>, bool, bool) = match reduce_op {
        ReduceOp::Sum => {
            let f = Box::new(move |acc: &mut [u8], val: &[u8]| {
                let acc_ptr = acc.as_mut_ptr() as *mut T;
                let val_ptr = val.as_ptr() as *const T;
                let count = acc.len() / std::mem::size_of::<T>();
                for i in 0..count {
                    unsafe {
                        *acc_ptr.add(i) = *acc_ptr.add(i) + *val_ptr.add(i);
                    }
                }
            });
            (f, true, true)
        }
        ReduceOp::Min => {
            let f = Box::new(move |acc: &mut [u8], val: &[u8]| {
                let acc_ptr = acc.as_mut_ptr() as *mut T;
                let val_ptr = val.as_ptr() as *const T;
                let count = acc.len() / std::mem::size_of::<T>();
                for i in 0..count {
                    unsafe {
                        let a = *acc_ptr.add(i);
                        let b = *val_ptr.add(i);
                        *acc_ptr.add(i) = std::cmp::min(a, b);
                    }
                }
            });
            (f, true, true)
        }
        ReduceOp::Max => {
            let f = Box::new(move |acc: &mut [u8], val: &[u8]| {
                let acc_ptr = acc.as_mut_ptr() as *mut T;
                let val_ptr = val.as_ptr() as *const T;
                let count = acc.len() / std::mem::size_of::<T>();
                for i in 0..count {
                    unsafe {
                        let a = *acc_ptr.add(i);
                        let b = *val_ptr.add(i);
                        *acc_ptr.add(i) = std::cmp::max(a, b);
                    }
                }
            });
            (f, true, true)
        }
        ReduceOp::Prod => {
            let f = Box::new(move |acc: &mut [u8], val: &[u8]| {
                let acc_ptr = acc.as_mut_ptr() as *mut T;
                let val_ptr = val.as_ptr() as *const T;
                let count = acc.len() / std::mem::size_of::<T>();
                for i in 0..count {
                    unsafe {
                        *acc_ptr.add(i) = *acc_ptr.add(i) * *val_ptr.add(i);
                    }
                }
            });
            (f, true, true)
        }
        _ => {
            return Err(CylonError::new(
                Code::Invalid,
                format!("Unsupported reduction operator {:?}", reduce_op),
            ));
        }
    };

    Ok((func, commutative, associative))
}

fn get_typed_reduce_function_float<T>(
    reduce_op: ReduceOp,
) -> CylonResult<(Box<dyn Fn(&mut [u8], &[u8]) + Send + Sync>, bool, bool)>
where
    T: Copy + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + PartialOrd + 'static,
{
    let (func, associative, commutative): (Box<dyn Fn(&mut [u8], &[u8]) + Send + Sync>, bool, bool) = match reduce_op {
        ReduceOp::Sum => {
            let f = Box::new(move |acc: &mut [u8], val: &[u8]| {
                let acc_ptr = acc.as_mut_ptr() as *mut T;
                let val_ptr = val.as_ptr() as *const T;
                let count = acc.len() / std::mem::size_of::<T>();
                for i in 0..count {
                    unsafe {
                        *acc_ptr.add(i) = *acc_ptr.add(i) + *val_ptr.add(i);
                    }
                }
            });
            (f, true, true)
        }
        ReduceOp::Min => {
            let f = Box::new(move |acc: &mut [u8], val: &[u8]| {
                let acc_ptr = acc.as_mut_ptr() as *mut T;
                let val_ptr = val.as_ptr() as *const T;
                let count = acc.len() / std::mem::size_of::<T>();
                for i in 0..count {
                    unsafe {
                        let a = *acc_ptr.add(i);
                        let b = *val_ptr.add(i);
                        *acc_ptr.add(i) = if a < b { a } else { b };
                    }
                }
            });
            (f, true, true)
        }
        ReduceOp::Max => {
            let f = Box::new(move |acc: &mut [u8], val: &[u8]| {
                let acc_ptr = acc.as_mut_ptr() as *mut T;
                let val_ptr = val.as_ptr() as *const T;
                let count = acc.len() / std::mem::size_of::<T>();
                for i in 0..count {
                    unsafe {
                        let a = *acc_ptr.add(i);
                        let b = *val_ptr.add(i);
                        *acc_ptr.add(i) = if a > b { a } else { b };
                    }
                }
            });
            (f, true, true)
        }
        ReduceOp::Prod => {
            let f = Box::new(move |acc: &mut [u8], val: &[u8]| {
                let acc_ptr = acc.as_mut_ptr() as *mut T;
                let val_ptr = val.as_ptr() as *const T;
                let count = acc.len() / std::mem::size_of::<T>();
                for i in 0..count {
                    unsafe {
                        *acc_ptr.add(i) = *acc_ptr.add(i) * *val_ptr.add(i);
                    }
                }
            });
            (f, true, true)
        }
        _ => {
            return Err(CylonError::new(
                Code::Invalid,
                format!("Unsupported reduction operator {:?}", reduce_op),
            ));
        }
    };

    Ok((func, commutative, associative))
}
