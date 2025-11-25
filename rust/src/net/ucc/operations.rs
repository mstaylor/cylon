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

//! UCC operations implementation
//!
//! Ported from cpp/src/cylon/net/ucc/ucc_operations.hpp/cpp

use std::sync::Arc;
use std::mem;

use crate::error::{CylonError, CylonResult, Code};
use crate::net::ops::base_ops::{TableBcastImpl, TableGatherImpl, TableAllgatherImpl};
use crate::ctx::CylonContext;
use crate::table::Table;
use crate::data_types::{DataType, Type};

use super::ucc_sys::*;

/// Helper function to wait for all UCC collective requests to complete
///
/// Corresponds to C++ WaitAllHelper from ucc_operations.cpp:81-106
unsafe fn wait_all_helper(
    reqs: &mut Vec<ucc_coll_req_h>,
    ctx: ucc_context_h,
) -> CylonResult<()> {
    let mut alldone = false;
    let mut status: ucc_status_t;

    while !alldone {
        alldone = true;
        for r in reqs.iter() {
            // At every iteration progress the context
            ucc_context_progress(ctx);

            status = ucc_collective_test(*r);

            // If an error occurs or the operation is not posted yet, return an error
            if status < 0 || status == UCC_OPERATION_INITIALIZED as i32 {
                let msg = std::ffi::CStr::from_ptr(ucc_status_string(status))
                    .to_string_lossy()
                    .into_owned();
                return Err(CylonError::new(
                    Code::ExecutionError,
                    format!("UCC Failed: {}", msg),
                ));
            }

            // Now status can be OK or IN PROGRESS
            alldone &= status != UCC_INPROGRESS as i32;
        }
    }

    // All done, finalize requests now
    for r in reqs.iter() {
        status = ucc_collective_finalize(*r);
        if status != UCC_OK as i32 {
            let msg = std::ffi::CStr::from_ptr(ucc_status_string(status))
                .to_string_lossy()
                .into_owned();
            return Err(CylonError::new(
                Code::ExecutionError,
                format!("UCC finalize failed: {}", msg),
            ));
        }
    }

    Ok(())
}

/// Convert Cylon DataType to UCC datatype
///
/// Corresponds to C++ GetUccDataType from ucc_operations.cpp:128-182
fn get_ucc_datatype(data_type: &DataType) -> ucc_datatype_t {
    match data_type.get_type() {
        Type::UInt8 => UCC_DT_UINT8,
        Type::Int8 => UCC_DT_INT8,
        Type::UInt16 => UCC_DT_UINT16,
        Type::Int16 => UCC_DT_INT16,
        Type::UInt32 => UCC_DT_UINT32,
        Type::Int32 => UCC_DT_INT32,
        Type::UInt64 => UCC_DT_UINT64,
        Type::Int64 => UCC_DT_INT64,
        Type::Float => UCC_DT_FLOAT32,
        Type::Double => UCC_DT_FLOAT64,
        Type::HalfFloat => UCC_DT_FLOAT16,
        // String types map to UINT8 (byte arrays)
        Type::String | Type::Binary => UCC_DT_UINT8,
        _ => UCC_DT_PREDEFINED_LAST as ucc_datatype_t,
    }
}

/// Reduce operations enum
/// Corresponds to C++ cylon::net::ReduceOp
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    Min,
    Max,
    Prod,
    Land,
    Lor,
    Band,
    Bor,
}

/// Convert Cylon ReduceOp to UCC reduction operation
///
/// Corresponds to C++ GetUccOp from ucc_operations.cpp:184-205
fn get_ucc_op(reduce_op: ReduceOp) -> ucc_reduction_op_t {
    match reduce_op {
        ReduceOp::Sum => UCC_OP_SUM,
        ReduceOp::Min => UCC_OP_MIN,
        ReduceOp::Max => UCC_OP_MAX,
        ReduceOp::Prod => UCC_OP_PROD,
        ReduceOp::Land => UCC_OP_LAND,
        ReduceOp::Lor => UCC_OP_LOR,
        ReduceOp::Band => UCC_OP_BAND,
        ReduceOp::Bor => UCC_OP_BOR,
    }
}

/// UCC implementation of TableAllgatherImpl
///
/// Corresponds to C++ UccTableAllgatherImpl from ucc_operations.hpp:26-49
pub struct UccTableAllgatherImpl {
    ucc_team: ucc_team_h,
    ucc_context: ucc_context_h,
    requests: Vec<ucc_coll_req_h>,
    args: Vec<ucc_coll_args_t>,
    world_size: i32,
}

impl UccTableAllgatherImpl {
    /// Create a new UccTableAllgatherImpl
    ///
    /// Corresponds to C++ UccTableAllgatherImpl::UccTableAllgatherImpl (ucc_operations.cpp:113-121)
    pub fn new(ucc_team: ucc_team_h, ucc_context: ucc_context_h, world_size: i32) -> Self {
        Self {
            ucc_team,
            ucc_context,
            requests: Vec::new(),
            args: Vec::new(),
            world_size,
        }
    }
}

impl TableAllgatherImpl for UccTableAllgatherImpl {
    /// Initialize asynchronous operations for the given number of buffers
    ///
    /// Corresponds to C++ UccTableAllgatherImpl::Init (ucc_operations.cpp:123-126)
    fn init(&mut self, num_buffers: i32) {
        self.requests.resize(num_buffers as usize, std::ptr::null_mut());
        self.args.resize(num_buffers as usize, unsafe { mem::zeroed() });
    }

    /// All-gather buffer sizes
    ///
    /// Corresponds to C++ UccTableAllgatherImpl::AllgatherBufferSizes (ucc_operations.cpp:21-52)
    fn allgather_buffer_sizes(&self, send_data: &[i32], num_buffers: i32, rcv_data: &mut [i32]) -> CylonResult<()> {
        unsafe {
            let mut args: ucc_coll_args_t = mem::zeroed();
            let mut req: ucc_coll_req_h = std::ptr::null_mut();

            args.mask = 0;
            args.coll_type = UCC_COLL_TYPE_ALLGATHER;

            args.src.info.buffer = send_data.as_ptr() as *mut std::ffi::c_void;
            args.src.info.count = num_buffers as u64;
            args.src.info.datatype = UCC_DT_INT32;
            args.src.info.mem_type = UCC_MEMORY_TYPE_HOST;

            args.dst.info.buffer = rcv_data.as_mut_ptr() as *mut std::ffi::c_void;
            args.dst.info.count = (num_buffers * self.world_size) as u64;
            args.dst.info.datatype = UCC_DT_INT32;
            args.dst.info.mem_type = UCC_MEMORY_TYPE_HOST;

            let mut status = ucc_collective_init(&mut args, &mut req, self.ucc_team);
            if status != UCC_OK as i32 {
                let msg = std::ffi::CStr::from_ptr(ucc_status_string(status))
                    .to_string_lossy()
                    .into_owned();
                return Err(CylonError::new(Code::ExecutionError, format!("UCC collective init failed: {}", msg)));
            }

            status = ucc_collective_post(req);
            if status != UCC_OK as i32 {
                let msg = std::ffi::CStr::from_ptr(ucc_status_string(status))
                    .to_string_lossy()
                    .into_owned();
                return Err(CylonError::new(Code::ExecutionError, format!("UCC collective post failed: {}", msg)));
            }

            // Wait for completion
            while UCC_OK as i32 != { status = ucc_collective_test(req); status } {
                if status < 0 {
                    let msg = std::ffi::CStr::from_ptr(ucc_status_string(status))
                        .to_string_lossy()
                        .into_owned();
                    return Err(CylonError::new(Code::ExecutionError, format!("UCC test failed: {}", msg)));
                }
                ucc_context_progress(self.ucc_context);
            }

            status = ucc_collective_finalize(req);
            if status != UCC_OK as i32 {
                let msg = std::ffi::CStr::from_ptr(ucc_status_string(status))
                    .to_string_lossy()
                    .into_owned();
                return Err(CylonError::new(Code::ExecutionError, format!("UCC finalize failed: {}", msg)));
            }

            Ok(())
        }
    }

    /// Non-blocking all-gather of buffer data
    ///
    /// Corresponds to C++ UccTableAllgatherImpl::IallgatherBufferData (ucc_operations.cpp:54-79)
    fn iallgather_buffer_data(
        &mut self,
        buf_idx: i32,
        send_data: &[u8],
        send_count: i32,
        recv_data: &mut [u8],
        recv_count: &[i32],
        displacements: &[i32],
    ) -> CylonResult<()> {
        unsafe {
            let args = &mut self.args[buf_idx as usize];

            args.mask = 0;
            args.coll_type = UCC_COLL_TYPE_ALLGATHERV;

            args.src.info.buffer = send_data.as_ptr() as *mut std::ffi::c_void;
            args.src.info.count = send_count as u64;
            args.src.info.datatype = UCC_DT_UINT8;
            args.src.info.mem_type = UCC_MEMORY_TYPE_HOST;

            args.dst.info_v.buffer = recv_data.as_mut_ptr() as *mut std::ffi::c_void;
            args.dst.info_v.counts = recv_count.as_ptr() as *mut ucc_count_t;
            args.dst.info_v.displacements = displacements.as_ptr() as *mut ucc_aint_t;
            args.dst.info_v.datatype = UCC_DT_UINT8;
            args.dst.info_v.mem_type = UCC_MEMORY_TYPE_HOST;

            let mut status = ucc_collective_init(args, &mut self.requests[buf_idx as usize], self.ucc_team);
            if status != UCC_OK as i32 {
                let msg = std::ffi::CStr::from_ptr(ucc_status_string(status))
                    .to_string_lossy()
                    .into_owned();
                return Err(CylonError::new(Code::ExecutionError, format!("UCC collective init failed: {}", msg)));
            }

            status = ucc_collective_post(self.requests[buf_idx as usize]);
            if status != UCC_OK as i32 {
                let msg = std::ffi::CStr::from_ptr(ucc_status_string(status))
                    .to_string_lossy()
                    .into_owned();
                return Err(CylonError::new(Code::ExecutionError, format!("UCC collective post failed: {}", msg)));
            }

            Ok(())
        }
    }

    /// Wait for all asynchronous operations
    ///
    /// Corresponds to C++ UccTableAllgatherImpl::WaitAll (ucc_operations.cpp:108-111)
    fn wait_all(&mut self, _num_buffers: i32) -> CylonResult<()> {
        unsafe { wait_all_helper(&mut self.requests, self.ucc_context) }
    }

    /// Execute table all-gather operation
    fn execute(&mut self, _table: &Table, _ctx: Arc<CylonContext>) -> CylonResult<Vec<Table>> {
        Err(CylonError::new(
            Code::NotImplemented,
            "UCC TableAllgatherImpl::execute not yet implemented",
        ))
    }
}

/// UCC implementation of TableGatherImpl
///
/// Corresponds to C++ UccTableGatherImpl from ucc_operations.hpp:51-79
pub struct UccTableGatherImpl {
    ucc_team: ucc_team_h,
    ucc_context: ucc_context_h,
    requests: Vec<ucc_coll_req_h>,
    args: Vec<ucc_coll_args_t>,
    world_size: i32,
    rank: i32,
}

impl UccTableGatherImpl {
    /// Create a new UccTableGatherImpl
    ///
    /// Corresponds to C++ UccTableGatherImpl::UccTableGatherImpl (ucc_operations.cpp:247-249)
    pub fn new(ucc_team: ucc_team_h, ucc_context: ucc_context_h, rank: i32, world_size: i32) -> Self {
        Self {
            ucc_team,
            ucc_context,
            requests: Vec::new(),
            args: Vec::new(),
            world_size,
            rank,
        }
    }
}

impl TableGatherImpl for UccTableGatherImpl {
    /// Initialize asynchronous operations
    ///
    /// Corresponds to C++ UccTableGatherImpl::Init (ucc_operations.cpp:251-254)
    fn init(&mut self, num_buffers: i32) {
        self.requests.resize(num_buffers as usize, std::ptr::null_mut());
        self.args.resize(num_buffers as usize, unsafe { mem::zeroed() });
    }

    /// Gather buffer sizes from all processes
    ///
    /// Corresponds to C++ UccTableGatherImpl::GatherBufferSizes (ucc_operations.cpp:256-291)
    fn gather_buffer_sizes(&self, send_data: &[i32], num_buffers: i32, rcv_data: &mut [i32], gather_root: i32) -> CylonResult<()> {
        unsafe {
            let mut args: ucc_coll_args_t = mem::zeroed();
            let mut req: ucc_coll_req_h = std::ptr::null_mut();

            args.mask = 0;
            args.coll_type = UCC_COLL_TYPE_GATHER;
            args.root = gather_root as u64;

            args.src.info.buffer = send_data.as_ptr() as *mut std::ffi::c_void;
            args.src.info.count = num_buffers as u64;
            args.src.info.datatype = UCC_DT_INT32;
            args.src.info.mem_type = UCC_MEMORY_TYPE_HOST;

            if self.rank == gather_root {
                args.dst.info.buffer = rcv_data.as_mut_ptr() as *mut std::ffi::c_void;
                args.dst.info.count = (num_buffers * self.world_size) as u64;
                args.dst.info.datatype = UCC_DT_INT32;
                args.dst.info.mem_type = UCC_MEMORY_TYPE_HOST;
            }

            let mut status = ucc_collective_init(&mut args, &mut req, self.ucc_team);
            if status != UCC_OK as i32 {
                let msg = std::ffi::CStr::from_ptr(ucc_status_string(status))
                    .to_string_lossy()
                    .into_owned();
                return Err(CylonError::new(Code::ExecutionError, format!("UCC collective init failed: {}", msg)));
            }

            status = ucc_collective_post(req);
            if status != UCC_OK as i32 {
                let msg = std::ffi::CStr::from_ptr(ucc_status_string(status))
                    .to_string_lossy()
                    .into_owned();
                return Err(CylonError::new(Code::ExecutionError, format!("UCC collective post failed: {}", msg)));
            }

            // Wait for completion
            while UCC_OK as i32 != { status = ucc_collective_test(req); status } {
                if status < 0 {
                    let msg = std::ffi::CStr::from_ptr(ucc_status_string(status))
                        .to_string_lossy()
                        .into_owned();
                    return Err(CylonError::new(Code::ExecutionError, format!("UCC test failed: {}", msg)));
                }
                ucc_context_progress(self.ucc_context);
            }

            status = ucc_collective_finalize(req);
            if status != UCC_OK as i32 {
                let msg = std::ffi::CStr::from_ptr(ucc_status_string(status))
                    .to_string_lossy()
                    .into_owned();
                return Err(CylonError::new(Code::ExecutionError, format!("UCC finalize failed: {}", msg)));
            }

            Ok(())
        }
    }

    /// Non-blocking gather of buffer data
    ///
    /// Corresponds to C++ UccTableGatherImpl::IgatherBufferData (ucc_operations.cpp:293-323)
    fn igather_buffer_data(
        &mut self,
        buf_idx: i32,
        send_data: &[u8],
        send_count: i32,
        recv_data: &mut [u8],
        recv_count: &[i32],
        displacements: &[i32],
        gather_root: i32,
    ) -> CylonResult<()> {
        unsafe {
            let args = &mut self.args[buf_idx as usize];

            args.mask = 0;
            args.coll_type = UCC_COLL_TYPE_GATHERV;
            args.root = gather_root as u64;

            args.src.info.buffer = send_data.as_ptr() as *mut std::ffi::c_void;
            args.src.info.count = send_count as u64;
            args.src.info.datatype = UCC_DT_UINT8;
            args.src.info.mem_type = UCC_MEMORY_TYPE_HOST;

            if self.rank == gather_root {
                args.dst.info_v.buffer = recv_data.as_mut_ptr() as *mut std::ffi::c_void;
                args.dst.info_v.counts = recv_count.as_ptr() as *mut ucc_count_t;
                args.dst.info_v.displacements = displacements.as_ptr() as *mut ucc_aint_t;
                args.dst.info_v.datatype = UCC_DT_UINT8;
                args.dst.info_v.mem_type = UCC_MEMORY_TYPE_HOST;
            }

            let mut status = ucc_collective_init(args, &mut self.requests[buf_idx as usize], self.ucc_team);
            if status != UCC_OK as i32 {
                let msg = std::ffi::CStr::from_ptr(ucc_status_string(status))
                    .to_string_lossy()
                    .into_owned();
                return Err(CylonError::new(Code::ExecutionError, format!("UCC collective init failed: {}", msg)));
            }

            status = ucc_collective_post(self.requests[buf_idx as usize]);
            if status != UCC_OK as i32 {
                let msg = std::ffi::CStr::from_ptr(ucc_status_string(status))
                    .to_string_lossy()
                    .into_owned();
                return Err(CylonError::new(Code::ExecutionError, format!("UCC collective post failed: {}", msg)));
            }

            Ok(())
        }
    }

    /// Wait for all asynchronous operations
    ///
    /// Corresponds to C++ UccTableGatherImpl::WaitAll (ucc_operations.cpp:325-328)
    fn wait_all(&mut self, _num_buffers: i32) -> CylonResult<()> {
        unsafe { wait_all_helper(&mut self.requests, self.ucc_context) }
    }

    /// Execute table gather operation
    fn execute(
        &mut self,
        _table: &Table,
        _gather_root: i32,
        _gather_from_root: bool,
        _ctx: Arc<CylonContext>,
    ) -> CylonResult<Vec<Table>> {
        Err(CylonError::new(
            Code::NotImplemented,
            "UCC TableGatherImpl::execute not yet implemented",
        ))
    }
}

/// UCC implementation of AllReduceImpl
///
/// Corresponds to C++ UccAllReduceImpl from ucc_operations.hpp:81-93
pub struct UccAllReduceImpl {
    ucc_team: ucc_team_h,
    ucc_context: ucc_context_h,
}

impl UccAllReduceImpl {
    /// Create a new UccAllReduceImpl
    ///
    /// Corresponds to C++ UccAllReduceImpl::UccAllReduceImpl (ucc_operations.cpp:207-208)
    pub fn new(ucc_team: ucc_team_h, ucc_context: ucc_context_h) -> Self {
        Self {
            ucc_team,
            ucc_context,
        }
    }

    /// Perform all-reduce operation
    ///
    /// Corresponds to C++ UccAllReduceImpl::AllReduceBuffer (ucc_operations.cpp:210-245)
    pub fn all_reduce_buffer(
        &self,
        send_buf: &[u8],
        rcv_buf: &mut [u8],
        count: i32,
        data_type: &DataType,
        reduce_op: ReduceOp,
    ) -> CylonResult<()> {
        let dt = get_ucc_datatype(data_type);
        let op = get_ucc_op(reduce_op);

        if dt == UCC_DT_PREDEFINED_LAST as u64 || op == UCC_OP_LAST {
            return Err(CylonError::new(
                Code::NotImplemented,
                "UCC allreduce not implemented for this type or operation",
            ));
        }

        unsafe {
            let mut args: ucc_coll_args_t = mem::zeroed();
            let mut req: ucc_coll_req_h = std::ptr::null_mut();

            args.mask = 0;
            args.coll_type = UCC_COLL_TYPE_ALLREDUCE;

            args.src.info.buffer = send_buf.as_ptr() as *mut std::ffi::c_void;
            args.src.info.count = count as u64;
            args.src.info.datatype = dt;
            args.src.info.mem_type = UCC_MEMORY_TYPE_HOST;

            args.dst.info.buffer = rcv_buf.as_mut_ptr() as *mut std::ffi::c_void;
            args.dst.info.count = count as u64;
            args.dst.info.datatype = dt;
            args.dst.info.mem_type = UCC_MEMORY_TYPE_HOST;

            args.op = op;

            let mut status = ucc_collective_init(&mut args, &mut req, self.ucc_team);
            if status != UCC_OK as i32 {
                let msg = std::ffi::CStr::from_ptr(ucc_status_string(status))
                    .to_string_lossy()
                    .into_owned();
                return Err(CylonError::new(Code::ExecutionError, format!("UCC collective init failed: {}", msg)));
            }

            status = ucc_collective_post(req);
            if status != UCC_OK as i32 {
                let msg = std::ffi::CStr::from_ptr(ucc_status_string(status))
                    .to_string_lossy()
                    .into_owned();
                return Err(CylonError::new(Code::ExecutionError, format!("UCC collective post failed: {}", msg)));
            }

            // Wait for completion
            while UCC_INPROGRESS as i32 == ucc_collective_test(req) {
                ucc_context_progress(self.ucc_context);
            }

            status = ucc_collective_finalize(req);
            if status != UCC_OK as i32 {
                let msg = std::ffi::CStr::from_ptr(ucc_status_string(status))
                    .to_string_lossy()
                    .into_owned();
                return Err(CylonError::new(Code::ExecutionError, format!("UCC finalize failed: {}", msg)));
            }

            Ok(())
        }
    }
}

/// UCC implementation of TableBcastImpl
///
/// Corresponds to C++ UccTableBcastImpl from ucc_operations.hpp:95-112
pub struct UccTableBcastImpl {
    ucc_team: ucc_team_h,
    ucc_context: ucc_context_h,
    reqs: Vec<ucc_coll_req_h>,
    args: Vec<ucc_coll_args_t>,
}

impl UccTableBcastImpl {
    /// Create a new UccTableBcastImpl
    ///
    /// Corresponds to C++ UccTableBcastImpl::UccTableBcastImpl (ucc_operations.cpp:330-331)
    pub fn new(ucc_team: ucc_team_h, ucc_context: ucc_context_h) -> Self {
        Self {
            ucc_team,
            ucc_context,
            reqs: Vec::new(),
            args: Vec::new(),
        }
    }
}

impl TableBcastImpl for UccTableBcastImpl {
    /// Initialize asynchronous operations
    ///
    /// Corresponds to C++ UccTableBcastImpl::Init (ucc_operations.cpp:333-336)
    fn init(&mut self, num_buffers: i32) {
        self.reqs.resize(num_buffers as usize, std::ptr::null_mut());
        self.args.resize(num_buffers as usize, unsafe { mem::zeroed() });
    }

    /// Broadcast buffer size information
    ///
    /// Corresponds to C++ UccTableBcastImpl::BcastBufferSizes (ucc_operations.cpp:338-367)
    fn bcast_buffer_sizes(&self, buffer: &mut [i32], count: i32, bcast_root: i32) -> CylonResult<()> {
        unsafe {
            let mut args: ucc_coll_args_t = mem::zeroed();
            let mut req: ucc_coll_req_h = std::ptr::null_mut();

            args.mask = 0;
            args.coll_type = UCC_COLL_TYPE_BCAST;
            args.root = bcast_root as u64;

            args.src.info.buffer = buffer.as_mut_ptr() as *mut std::ffi::c_void;
            args.src.info.count = count as u64;
            args.src.info.datatype = UCC_DT_INT32;
            args.src.info.mem_type = UCC_MEMORY_TYPE_HOST;

            let mut status = ucc_collective_init(&mut args, &mut req, self.ucc_team);
            if status != UCC_OK as i32 {
                let msg = std::ffi::CStr::from_ptr(ucc_status_string(status))
                    .to_string_lossy()
                    .into_owned();
                return Err(CylonError::new(Code::ExecutionError, format!("UCC collective init failed: {}", msg)));
            }

            status = ucc_collective_post(req);
            if status != UCC_OK as i32 {
                let msg = std::ffi::CStr::from_ptr(ucc_status_string(status))
                    .to_string_lossy()
                    .into_owned();
                return Err(CylonError::new(Code::ExecutionError, format!("UCC collective post failed: {}", msg)));
            }

            // Wait for completion
            while UCC_OK as i32 != { status = ucc_collective_test(req); status } {
                if status < 0 {
                    let msg = std::ffi::CStr::from_ptr(ucc_status_string(status))
                        .to_string_lossy()
                        .into_owned();
                    return Err(CylonError::new(Code::ExecutionError, format!("UCC test failed: {}", msg)));
                }
                ucc_context_progress(self.ucc_context);
            }

            status = ucc_collective_finalize(req);
            if status != UCC_OK as i32 {
                let msg = std::ffi::CStr::from_ptr(ucc_status_string(status))
                    .to_string_lossy()
                    .into_owned();
                return Err(CylonError::new(Code::ExecutionError, format!("UCC finalize failed: {}", msg)));
            }

            Ok(())
        }
    }

    /// Synchronous broadcast of buffer data
    ///
    /// Corresponds to C++ UccTableBcastImpl::BcastBufferData (ucc_operations.cpp:369-398)
    fn bcast_buffer_data(&self, buf_data: &mut [u8], send_count: i32, bcast_root: i32) -> CylonResult<()> {
        unsafe {
            let mut args: ucc_coll_args_t = mem::zeroed();
            let mut req: ucc_coll_req_h = std::ptr::null_mut();

            args.mask = 0;
            args.coll_type = UCC_COLL_TYPE_BCAST;
            args.root = bcast_root as u64;

            args.src.info.buffer = buf_data.as_mut_ptr() as *mut std::ffi::c_void;
            args.src.info.count = send_count as u64;
            args.src.info.datatype = UCC_DT_UINT8;
            args.src.info.mem_type = UCC_MEMORY_TYPE_HOST;

            let mut status = ucc_collective_init(&mut args, &mut req, self.ucc_team);
            if status != UCC_OK as i32 {
                let msg = std::ffi::CStr::from_ptr(ucc_status_string(status))
                    .to_string_lossy()
                    .into_owned();
                return Err(CylonError::new(Code::ExecutionError, format!("UCC collective init failed: {}", msg)));
            }

            status = ucc_collective_post(req);
            if status != UCC_OK as i32 {
                let msg = std::ffi::CStr::from_ptr(ucc_status_string(status))
                    .to_string_lossy()
                    .into_owned();
                return Err(CylonError::new(Code::ExecutionError, format!("UCC collective post failed: {}", msg)));
            }

            // Wait for completion
            while UCC_OK as i32 != { status = ucc_collective_test(req); status } {
                if status < 0 {
                    let msg = std::ffi::CStr::from_ptr(ucc_status_string(status))
                        .to_string_lossy()
                        .into_owned();
                    return Err(CylonError::new(Code::ExecutionError, format!("UCC test failed: {}", msg)));
                }
                ucc_context_progress(self.ucc_context);
            }

            status = ucc_collective_finalize(req);
            if status != UCC_OK as i32 {
                let msg = std::ffi::CStr::from_ptr(ucc_status_string(status))
                    .to_string_lossy()
                    .into_owned();
                return Err(CylonError::new(Code::ExecutionError, format!("UCC finalize failed: {}", msg)));
            }

            Ok(())
        }
    }

    /// Non-blocking broadcast of buffer data
    ///
    /// Corresponds to C++ UccTableBcastImpl::IbcastBufferData (ucc_operations.cpp:400-421)
    fn ibcast_buffer_data(&mut self, buf_idx: i32, buf_data: &mut [u8], send_count: i32, bcast_root: i32) -> CylonResult<()> {
        unsafe {
            let args = &mut self.args[buf_idx as usize];
            let req = &mut self.reqs[buf_idx as usize];

            args.mask = 0;
            args.coll_type = UCC_COLL_TYPE_BCAST;
            args.root = bcast_root as u64;

            args.src.info.buffer = buf_data.as_mut_ptr() as *mut std::ffi::c_void;
            args.src.info.count = send_count as u64;
            args.src.info.datatype = UCC_DT_UINT8;
            args.src.info.mem_type = UCC_MEMORY_TYPE_HOST;

            let mut status = ucc_collective_init(args, req, self.ucc_team);
            if status != UCC_OK as i32 {
                let msg = std::ffi::CStr::from_ptr(ucc_status_string(status))
                    .to_string_lossy()
                    .into_owned();
                return Err(CylonError::new(Code::ExecutionError, format!("UCC collective init failed: {}", msg)));
            }

            status = ucc_collective_post(*req);
            if status != UCC_OK as i32 {
                let msg = std::ffi::CStr::from_ptr(ucc_status_string(status))
                    .to_string_lossy()
                    .into_owned();
                return Err(CylonError::new(Code::ExecutionError, format!("UCC collective post failed: {}", msg)));
            }

            Ok(())
        }
    }

    /// Wait for all asynchronous operations
    ///
    /// Corresponds to C++ UccTableBcastImpl::WaitAll (ucc_operations.cpp:423-426)
    fn wait_all(&mut self, _num_buffers: i32) -> CylonResult<()> {
        unsafe { wait_all_helper(&mut self.reqs, self.ucc_context) }
    }

    /// Execute table broadcast operation
    fn execute(&mut self, _table: &mut Option<Table>, _bcast_root: i32, _ctx: Arc<CylonContext>) -> CylonResult<()> {
        Err(CylonError::new(
            Code::NotImplemented,
            "UCC TableBcastImpl::execute not yet implemented",
        ))
    }
}

/// UCC implementation for AllGatherImpl (raw data, not tables)
///
/// Corresponds to C++ UccAllGatherImpl from ucc_operations.hpp:114-131
pub struct UccAllGatherImpl {
    ucc_team: ucc_team_h,
    ucc_context: ucc_context_h,
    requests: Vec<ucc_coll_req_h>,
    args: Vec<ucc_coll_args_t>,
    world_size: i32,
}

impl UccAllGatherImpl {
    /// Create a new UccAllGatherImpl
    ///
    /// Corresponds to C++ UccAllGatherImpl::UccAllGatherImpl (ucc_operations.cpp:428-430)
    pub fn new(ucc_team: ucc_team_h, ucc_context: ucc_context_h, world_size: i32) -> Self {
        Self {
            ucc_team,
            ucc_context,
            requests: Vec::new(),
            args: Vec::new(),
            world_size,
        }
    }

    /// All-gather buffer sizes
    ///
    /// Corresponds to C++ UccAllGatherImpl::AllgatherBufferSize (ucc_operations.cpp:432-464)
    pub fn allgather_buffer_size(&self, send_data: &[i32], num_buffers: i32, rcv_data: &mut [i32]) -> CylonResult<()> {
        unsafe {
            let mut args: ucc_coll_args_t = mem::zeroed();
            let mut req: ucc_coll_req_h = std::ptr::null_mut();

            args.mask = 0;
            args.coll_type = UCC_COLL_TYPE_ALLGATHER;

            args.src.info.buffer = send_data.as_ptr() as *mut std::ffi::c_void;
            args.src.info.count = num_buffers as u64;
            args.src.info.datatype = UCC_DT_INT32;
            args.src.info.mem_type = UCC_MEMORY_TYPE_HOST;

            args.dst.info.buffer = rcv_data.as_mut_ptr() as *mut std::ffi::c_void;
            args.dst.info.count = (num_buffers * self.world_size) as u64;
            args.dst.info.datatype = UCC_DT_INT32;
            args.dst.info.mem_type = UCC_MEMORY_TYPE_HOST;

            let mut status = ucc_collective_init(&mut args, &mut req, self.ucc_team);
            if status != UCC_OK as i32 {
                let msg = std::ffi::CStr::from_ptr(ucc_status_string(status))
                    .to_string_lossy()
                    .into_owned();
                return Err(CylonError::new(Code::ExecutionError, format!("UCC collective init failed: {}", msg)));
            }

            status = ucc_collective_post(req);
            if status != UCC_OK as i32 {
                let msg = std::ffi::CStr::from_ptr(ucc_status_string(status))
                    .to_string_lossy()
                    .into_owned();
                return Err(CylonError::new(Code::ExecutionError, format!("UCC collective post failed: {}", msg)));
            }

            // Wait for completion
            while UCC_OK as i32 != { status = ucc_collective_test(req); status } {
                if status < 0 {
                    let msg = std::ffi::CStr::from_ptr(ucc_status_string(status))
                        .to_string_lossy()
                        .into_owned();
                    return Err(CylonError::new(Code::ExecutionError, format!("UCC test failed: {}", msg)));
                }
                ucc_context_progress(self.ucc_context);
            }

            status = ucc_collective_finalize(req);
            if status != UCC_OK as i32 {
                let msg = std::ffi::CStr::from_ptr(ucc_status_string(status))
                    .to_string_lossy()
                    .into_owned();
                return Err(CylonError::new(Code::ExecutionError, format!("UCC finalize failed: {}", msg)));
            }

            Ok(())
        }
    }

    /// Non-blocking all-gather of buffer data
    ///
    /// Corresponds to C++ UccAllGatherImpl::IallgatherBufferData (ucc_operations.cpp:466-494)
    pub fn iallgather_buffer_data(
        &mut self,
        buf_idx: i32,
        send_data: &[u8],
        send_count: i32,
        recv_data: &mut [u8],
        recv_count: &[i32],
        displacements: &[i32],
    ) -> CylonResult<()> {
        // Resize if needed (matches C++ line 470-471)
        if self.requests.len() < 3 {
            self.requests.resize(3, std::ptr::null_mut());
            self.args.resize(3, unsafe { mem::zeroed() });
        }

        unsafe {
            let args = &mut self.args[buf_idx as usize];

            args.mask = 0;
            args.coll_type = UCC_COLL_TYPE_ALLGATHERV;

            args.src.info.buffer = send_data.as_ptr() as *mut std::ffi::c_void;
            args.src.info.count = send_count as u64;
            args.src.info.datatype = UCC_DT_UINT8;
            args.src.info.mem_type = UCC_MEMORY_TYPE_HOST;

            args.dst.info_v.buffer = recv_data.as_mut_ptr() as *mut std::ffi::c_void;
            args.dst.info_v.counts = recv_count.as_ptr() as *mut ucc_count_t;
            args.dst.info_v.displacements = displacements.as_ptr() as *mut ucc_aint_t;
            args.dst.info_v.datatype = UCC_DT_UINT8;
            args.dst.info_v.mem_type = UCC_MEMORY_TYPE_HOST;

            let mut status = ucc_collective_init(args, &mut self.requests[buf_idx as usize], self.ucc_team);
            if status != UCC_OK as i32 {
                let msg = std::ffi::CStr::from_ptr(ucc_status_string(status))
                    .to_string_lossy()
                    .into_owned();
                return Err(CylonError::new(Code::ExecutionError, format!("UCC collective init failed: {}", msg)));
            }

            status = ucc_collective_post(self.requests[buf_idx as usize]);
            if status != UCC_OK as i32 {
                let msg = std::ffi::CStr::from_ptr(ucc_status_string(status))
                    .to_string_lossy()
                    .into_owned();
                return Err(CylonError::new(Code::ExecutionError, format!("UCC collective post failed: {}", msg)));
            }

            Ok(())
        }
    }

    /// Wait for all asynchronous operations
    ///
    /// Corresponds to C++ UccAllGatherImpl::WaitAll (ucc_operations.cpp:496-498)
    pub fn wait_all(&mut self) -> CylonResult<()> {
        unsafe { wait_all_helper(&mut self.requests, self.ucc_context) }
    }
}

// Cylon is single-threaded, so these UCC handles can be safely sent/synced
unsafe impl Send for UccTableAllgatherImpl {}
unsafe impl Sync for UccTableAllgatherImpl {}
unsafe impl Send for UccTableGatherImpl {}
unsafe impl Sync for UccTableGatherImpl {}
unsafe impl Send for UccAllReduceImpl {}
unsafe impl Sync for UccAllReduceImpl {}
unsafe impl Send for UccTableBcastImpl {}
unsafe impl Sync for UccTableBcastImpl {}
unsafe impl Send for UccAllGatherImpl {}
unsafe impl Sync for UccAllGatherImpl {}
