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

//! UCX Communicator implementation
//!
//! Ported from cpp/src/cylon/net/ucx/ucx_communicator.hpp/cpp

use std::any::Any;
use std::collections::HashMap;
use std::ptr;

use crate::error::{CylonError, CylonResult, Code};
use crate::net::{CommType, Communicator, Channel};

use super::ucx_sys::*;
use super::operations::{init_context, init_worker};
use super::oob_context::UCXOOBContext;
use super::channel::UCXChannel;

/// UCX Communicator
///
/// Corresponds to C++ UCXCommunicator from ucx_communicator.hpp:75-133
pub struct UCXCommunicator {
    /// Process rank
    rank: i32,
    /// Total number of processes
    world_size: i32,
    /// Whether the communicator has been finalized
    finalized: bool,

    // UCX specific attributes - These need to be passed to the channels created
    // from the communicator

    /// The worker for receiving
    /// Corresponds to C++ ucpRecvWorker (ucx_communicator.hpp:120)
    pub ucp_recv_worker: ucp_worker_h,

    /// The worker for sending
    /// Corresponds to C++ ucpSendWorker (ucx_communicator.hpp:122)
    pub ucp_send_worker: ucp_worker_h,

    /// Endpoint Map
    /// Corresponds to C++ endPointMap (ucx_communicator.hpp:124)
    pub end_point_map: HashMap<i32, ucp_ep_h>,

    /// UCP Context - Holds a UCP communication instance's global information
    /// Corresponds to C++ ucpContext (ucx_communicator.hpp:126)
    ucp_context: ucp_context_h,

    /// OOB context for out-of-band communication
    oob_context: Option<Box<dyn UCXOOBContext>>,
}

impl UCXCommunicator {
    /// Create a new UCX communicator (uninitialized)
    ///
    /// Corresponds to C++ UCXCommunicator::UCXCommunicator (ucx_communicator.cpp:143-144)
    fn new() -> Self {
        Self {
            rank: -1,
            world_size: -1,
            finalized: false,
            ucp_recv_worker: ptr::null_mut(),
            ucp_send_worker: ptr::null_mut(),
            end_point_map: HashMap::new(),
            ucp_context: ptr::null_mut(),
            oob_context: None,
        }
    }

    /// Create a UCX communicator using OOB context (Redis-based)
    ///
    /// Corresponds to C++ UCXCommunicator::MakeOOB (ucx_communicator.cpp:174-262)
    pub fn make_oob(mut oob_context: Box<dyn UCXOOBContext>) -> CylonResult<Self> {
        unsafe {
            let mut comm = Self::new();

            // Initialize OOB
            // Corresponds to C++ line 195
            oob_context.init_oob()?;

            // Get world size and rank
            // Corresponds to C++ lines 197-201
            let (world_size, rank) = oob_context.get_world_size_and_rank()?;
            comm.rank = rank;
            comm.world_size = world_size;

            // Init context
            // Corresponds to C++ lines 204-205
            init_context(&mut comm.ucp_context, ptr::null())?;

            // Init recv worker and get address
            // Corresponds to C++ lines 207-212
            let ucp_recv_worker_addr = init_worker(comm.ucp_context, &mut comm.ucp_recv_worker)?;
            let ucp_send_worker_addr = init_worker(comm.ucp_context, &mut comm.ucp_send_worker)?;

            // Gather all worker addresses
            // Corresponds to C++ lines 214-221
            let addr_size = ucp_recv_worker_addr.addr_size;
            let mut all_addresses = vec![0u8; addr_size * world_size as usize];
            let src_slice = std::slice::from_raw_parts(
                ucp_recv_worker_addr.addr as *const u8,
                addr_size,
            );

            oob_context.oob_allgather(
                src_slice,
                &mut all_addresses,
                addr_size,
                addr_size,
            )?;

            // Iterate and set the endpoints
            // Corresponds to C++ lines 224-255
            comm.end_point_map.reserve(world_size as usize);
            for s_indx in 0..world_size {
                let mut ep_params: ucp_ep_params_t = std::mem::zeroed();
                let mut ep: ucp_ep_h = ptr::null_mut();

                // Get address for this rank
                // Corresponds to C++ lines 229-236
                let address = if rank != s_indx {
                    all_addresses.as_ptr().add(s_indx as usize * addr_size) as *mut ucp_address_t
                } else {
                    ucp_recv_worker_addr.addr
                };

                // Set params for the endpoint
                // Corresponds to C++ lines 238-242
                ep_params.field_mask = (ucp_ep_params_field_UCP_EP_PARAM_FIELD_REMOTE_ADDRESS
                    | ucp_ep_params_field_UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE) as u64;
                ep_params.address = address;
                ep_params.err_mode = ucp_err_handling_mode_t_UCP_ERR_HANDLING_MODE_NONE;

                // Create an endpoint
                // Corresponds to C++ line 245
                let ucx_status = ucp_ep_create(comm.ucp_send_worker, &ep_params, &mut ep);

                comm.end_point_map.insert(s_indx, ep);

                // Check if the endpoint was created properly
                // Corresponds to C++ lines 248-254
                if ucx_status != ucs_status_t_UCS_OK {
                    let msg = std::ffi::CStr::from_ptr(ucs_status_string(ucx_status))
                        .to_string_lossy()
                        .into_owned();
                    return Err(CylonError::new(
                        Code::ExecutionError,
                        format!("Error when creating the endpoint: {}", msg),
                    ));
                }
            }

            // Cleanup worker addresses
            // Corresponds to C++ lines 257-259
            // Note: C++ uses manual delete, we drop the Box
            drop(ucp_recv_worker_addr);
            drop(ucp_send_worker_addr);

            // Store the OOB context
            comm.oob_context = Some(oob_context);

            Ok(comm)
        }
    }

    /// Get the rank of this process
    pub fn get_rank(&self) -> i32 {
        self.rank
    }

    /// Get the world size (total number of processes)
    pub fn get_world_size(&self) -> i32 {
        self.world_size
    }
}

impl Communicator for UCXCommunicator {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn get_rank(&self) -> i32 {
        self.rank
    }

    fn get_world_size(&self) -> i32 {
        self.world_size
    }

    fn get_comm_type(&self) -> CommType {
        CommType::Ucx
    }

    fn is_finalized(&self) -> bool {
        self.finalized
    }

    /// Create a new channel for this communicator
    ///
    /// Corresponds to C++ UCXCommunicator::CreateChannel (ucx_communicator.cpp:102-104)
    fn create_channel(&self) -> CylonResult<Box<dyn Channel>> {
        Ok(Box::new(UCXChannel::new(self)))
    }

    /// Finalize the communicator
    ///
    /// Corresponds to C++ UCXCommunicator::Finalize (ucx_communicator.cpp:363-369)
    fn finalize(&mut self) -> CylonResult<()> {
        if !self.finalized {
            unsafe {
                ucp_cleanup(self.ucp_context);
            }
            if let Some(ref mut oob) = self.oob_context {
                oob.finalize()?;
            }
            self.finalized = true;
        }
        Ok(())
    }

    /// Barrier synchronization
    ///
    /// Corresponds to C++ UCXCommunicator::Barrier (ucx_communicator.cpp:371-373)
    /// Note: C++ uses MPI_Barrier, but we don't have MPI in Redis OOB mode
    /// For now, this is a no-op as UCX doesn't provide native barrier
    fn barrier(&self) -> CylonResult<()> {
        // TODO: Implement barrier using OOB context if needed
        // C++ implementation uses MPI_Barrier, which we don't have in OOB mode
        Ok(())
    }

    fn send(&self, _data: &[u8], _dest: i32, _tag: i32) -> CylonResult<()> {
        Err(CylonError::new(
            Code::NotImplemented,
            "Direct send not implemented for UCX - use channels instead",
        ))
    }

    fn recv(&self, _buffer: &mut Vec<u8>, _source: i32, _tag: i32) -> CylonResult<()> {
        Err(CylonError::new(
            Code::NotImplemented,
            "Direct recv not implemented for UCX - use channels instead",
        ))
    }

    fn all_to_all(&self, _send_data: Vec<Vec<u8>>) -> CylonResult<Vec<Vec<u8>>> {
        Err(CylonError::new(
            Code::NotImplemented,
            "all_to_all not implemented for UCX",
        ))
    }

    fn allgather(&self, _send_data: &[u8]) -> CylonResult<Vec<Vec<u8>>> {
        Err(CylonError::new(
            Code::NotImplemented,
            "allgather not implemented for UCX",
        ))
    }

    fn broadcast(&self, _data: &mut Vec<u8>, _root: i32) -> CylonResult<()> {
        Err(CylonError::new(
            Code::NotImplemented,
            "broadcast not implemented for UCX",
        ))
    }

    /// Broadcast table from root to all processes
    ///
    /// Corresponds to C++ UCXCommunicator::Bcast (ucx_communicator.cpp:126-132)
    fn bcast(
        &self,
        _table: &mut Option<crate::table::Table>,
        _bcast_root: i32,
        _ctx: std::sync::Arc<crate::ctx::CylonContext>,
    ) -> CylonResult<()> {
        Err(CylonError::new(
            Code::NotImplemented,
            "Bcast not implemented for ucx",
        ))
    }

    /// Gather tables from all processes
    ///
    /// Corresponds to C++ UCXCommunicator::Gather (ucx_communicator.cpp:116-124)
    fn gather(
        &self,
        _table: &crate::table::Table,
        _gather_root: i32,
        _gather_from_root: bool,
        _ctx: std::sync::Arc<crate::ctx::CylonContext>,
    ) -> CylonResult<Vec<crate::table::Table>> {
        Err(CylonError::new(
            Code::NotImplemented,
            "Gather not implemented for ucx",
        ))
    }

    /// All-gather tables from all processes
    ///
    /// Corresponds to C++ UCXCommunicator::AllGather (ucx_communicator.cpp:108-114)
    fn all_gather(
        &self,
        _table: &crate::table::Table,
        _ctx: std::sync::Arc<crate::ctx::CylonContext>,
    ) -> CylonResult<Vec<crate::table::Table>> {
        Err(CylonError::new(
            Code::NotImplemented,
            "All gather not implemented for ucx",
        ))
    }

    /// AllReduce on Column
    ///
    /// Corresponds to C++ UCXCommunicator::AllReduce(Column) (ucx_communicator.cpp:134-141)
    fn all_reduce_column(
        &self,
        _values: &crate::table::Column,
        _reduce_op: super::super::comm_operations::ReduceOp,
    ) -> CylonResult<crate::table::Column> {
        Err(CylonError::new(
            Code::NotImplemented,
            "Allreduce not implemented for ucx",
        ))
    }

    /// Allgather Column
    ///
    /// Corresponds to C++ UCXCommunicator::Allgather(Column) (ucx_communicator.cpp:159-165)
    fn allgather_column(
        &self,
        _values: &crate::table::Column,
    ) -> CylonResult<Vec<crate::table::Column>> {
        Err(CylonError::new(
            Code::NotImplemented,
            "Allgather not implemented for ucx",
        ))
    }

    /// AllReduce on Scalar
    ///
    /// Corresponds to C++ UCXCommunicator::AllReduce(Scalar) (ucx_communicator.cpp:150-157)
    fn all_reduce_scalar(
        &self,
        _value: &crate::scalar::Scalar,
        _reduce_op: super::super::comm_operations::ReduceOp,
    ) -> CylonResult<crate::scalar::Scalar> {
        Err(CylonError::new(
            Code::NotImplemented,
            "Allreduce not implemented for ucx",
        ))
    }

    /// Allgather Scalar
    ///
    /// Corresponds to C++ UCXCommunicator::Allgather(Scalar) (ucx_communicator.cpp:167-172)
    fn allgather_scalar(
        &self,
        _value: &crate::scalar::Scalar,
    ) -> CylonResult<crate::table::Column> {
        Err(CylonError::new(
            Code::NotImplemented,
            "Allgather not implemented for ucx",
        ))
    }
}

// Cylon is single-threaded, so these UCX handles can be safely sent/synced
unsafe impl Send for UCXCommunicator {}
unsafe impl Sync for UCXCommunicator {}

// =============================================================================
// UCXUCCCommunicator - Combines UCX (point-to-point) with UCC (collectives)
// =============================================================================

#[cfg(feature = "ucc")]
use crate::net::ucc::ucc_sys::*;
#[cfg(feature = "ucc")]
use crate::net::ucc::operations::{
    UccTableAllgatherImpl, UccTableGatherImpl, UccTableBcastImpl,
    UccAllReduceImpl, UccAllGatherImpl,
};
#[cfg(feature = "ucc")]
use crate::net::ops::base_ops::{TableAllgatherImpl, TableGatherImpl, TableBcastImpl, AllReduceImpl, AllGatherImpl};
#[cfg(feature = "ucc")]
use crate::net::comm_operations::ReduceOp;
#[cfg(feature = "ucc")]
use crate::table::Column;
#[cfg(feature = "ucc")]
use crate::scalar::Scalar;

/// UCX+UCC Communicator
///
/// Combines UCX for point-to-point communication with UCC for collective operations.
/// Corresponds to C++ UCXUCCCommunicator from ucx_communicator.hpp:136-182
#[cfg(feature = "ucc")]
pub struct UCXUCCCommunicator {
    /// Inner UCX communicator for point-to-point communication
    ucx_comm: UCXCommunicator,
    /// UCC team handle for collective operations
    pub ucc_team: ucc_team_h,
    /// UCC context handle
    pub ucc_context: ucc_context_h,
    /// UCC library handle
    ucc_lib: ucc_lib_h,
    /// Whether the communicator has been finalized
    finalized: bool,
}

#[cfg(feature = "ucc")]
impl UCXUCCCommunicator {
    /// Create a new UCX+UCC communicator from an existing UCX communicator
    ///
    /// Corresponds to C++ UCXUCCCommunicator::Make (ucx_communicator.cpp:520-588)
    pub fn new(ucx_comm: UCXCommunicator) -> CylonResult<Self> {
        let rank = ucx_comm.get_rank();
        let world_size = ucx_comm.get_world_size();

        unsafe {
            // Initialize UCC library
            // Corresponds to C++ lines 547-555
            let mut lib: ucc_lib_h = std::ptr::null_mut();
            let mut lib_config: ucc_lib_config_h = std::ptr::null_mut();

            let mut lib_params: ucc_lib_params_t = std::mem::zeroed();
            lib_params.mask = UCC_LIB_PARAM_FIELD_THREAD_MODE as u64;
            lib_params.thread_mode = UCC_THREAD_SINGLE;

            let status = ucc_lib_config_read(std::ptr::null(), std::ptr::null(), &mut lib_config);
            if status != UCC_OK as i32 {
                return Err(CylonError::new(
                    Code::ExecutionError,
                    format!("Failed to read UCC lib config: {}", status),
                ));
            }

            let status = ucc_init(&lib_params, lib_config, &mut lib);
            if status != UCC_OK as i32 {
                return Err(CylonError::new(
                    Code::ExecutionError,
                    format!("Failed to initialize UCC library: {}", status),
                ));
            }
            ucc_lib_config_release(lib_config);

            // Initialize UCC context
            // Corresponds to C++ lines 558-570
            let mut ctx_params: ucc_context_params_t = std::mem::zeroed();
            let mut ctx_config: ucc_context_config_h = std::ptr::null_mut();
            let mut ucc_context: ucc_context_h = std::ptr::null_mut();

            ctx_params.mask = UCC_CONTEXT_PARAM_FIELD_TYPE as u64;
            ctx_params.type_ = UCC_CONTEXT_EXCLUSIVE;

            let status = ucc_context_config_read(lib, std::ptr::null(), &mut ctx_config);
            if status != UCC_OK as i32 {
                return Err(CylonError::new(
                    Code::ExecutionError,
                    format!("Failed to read UCC context config: {}", status),
                ));
            }

            let status = ucc_context_create(lib, &ctx_params, ctx_config, &mut ucc_context);
            if status != UCC_OK as i32 {
                return Err(CylonError::new(
                    Code::ExecutionError,
                    format!("Failed to create UCC context: {}", status),
                ));
            }
            ucc_context_config_release(ctx_config);

            // Initialize UCC team
            // Corresponds to C++ lines 573-587
            let mut team_params: ucc_team_params_t = std::mem::zeroed();
            let mut ucc_team: ucc_team_h = std::ptr::null_mut();

            team_params.mask = UCC_TEAM_PARAM_FIELD_EP as u64 | UCC_TEAM_PARAM_FIELD_EP_RANGE as u64;
            team_params.ep = rank as u64;
            team_params.ep_range = UCC_COLLECTIVE_EP_RANGE_CONTIG;

            let status = ucc_team_create_post(&mut ucc_context, 1, &team_params, &mut ucc_team);
            if status != UCC_OK as i32 {
                return Err(CylonError::new(
                    Code::ExecutionError,
                    format!("Failed to create UCC team: {}", status),
                ));
            }

            // Wait for team creation to complete
            // Corresponds to C++ lines 582-584
            loop {
                let status = ucc_team_create_test(ucc_team);
                if status == UCC_OK as i32 {
                    break;
                }
                if status != UCC_INPROGRESS as i32 {
                    return Err(CylonError::new(
                        Code::ExecutionError,
                        format!("UCC team creation failed: {}", status),
                    ));
                }
            }

            Ok(Self {
                ucx_comm,
                ucc_team,
                ucc_context,
                ucc_lib: lib,
                finalized: false,
            })
        }
    }

    /// Get the UCC team handle
    pub fn get_ucc_team(&self) -> ucc_team_h {
        self.ucc_team
    }

    /// Get the UCC context handle
    pub fn get_ucc_context(&self) -> ucc_context_h {
        self.ucc_context
    }

    /// Get reference to the inner UCX communicator
    pub fn ucx_communicator(&self) -> &UCXCommunicator {
        &self.ucx_comm
    }
}

#[cfg(feature = "ucc")]
impl Communicator for UCXUCCCommunicator {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn get_rank(&self) -> i32 {
        self.ucx_comm.get_rank()
    }

    fn get_world_size(&self) -> i32 {
        self.ucx_comm.get_world_size()
    }

    /// Returns UCX since we use UCX for channels
    /// Corresponds to C++ UCXUCCCommunicator::GetCommType (ucx_communicator.cpp:590)
    fn get_comm_type(&self) -> CommType {
        CommType::Ucx
    }

    fn is_finalized(&self) -> bool {
        self.finalized
    }

    /// Create a new channel - delegates to inner UCX communicator
    /// Corresponds to C++ UCXUCCCommunicator::CreateChannel (ucx_communicator.cpp:592-594)
    fn create_channel(&self) -> CylonResult<Box<dyn Channel>> {
        self.ucx_comm.create_channel()
    }

    /// Finalize the communicator
    /// Corresponds to C++ UCXUCCCommunicator::Finalize (ucx_communicator.cpp:596-626)
    fn finalize(&mut self) -> CylonResult<()> {
        if !self.finalized {
            unsafe {
                // Destroy UCC team
                loop {
                    let status = ucc_team_destroy(self.ucc_team);
                    if status == UCC_OK as i32 {
                        break;
                    }
                    if status != UCC_INPROGRESS as i32 {
                        break;
                    }
                }

                // Destroy UCC context
                ucc_context_destroy(self.ucc_context);

                // Finalize UCC library
                ucc_finalize(self.ucc_lib);
            }

            // Finalize inner UCX communicator
            self.ucx_comm.finalize()?;
            self.finalized = true;
        }
        Ok(())
    }

    /// Barrier synchronization using UCC
    /// Corresponds to C++ UCXUCCCommunicator::Barrier (ucx_communicator.cpp:628-653)
    fn barrier(&self) -> CylonResult<()> {
        unsafe {
            let mut args: ucc_coll_args_t = std::mem::zeroed();
            let mut req: ucc_coll_req_h = std::ptr::null_mut();

            args.mask = 0;
            args.coll_type = UCC_COLL_TYPE_BARRIER;

            let status = ucc_collective_init(&mut args, &mut req, self.ucc_team);
            if status != UCC_OK as i32 {
                return Err(CylonError::new(
                    Code::ExecutionError,
                    format!("UCC barrier init failed: {}", status),
                ));
            }

            let status = ucc_collective_post(req);
            if status != UCC_OK as i32 {
                return Err(CylonError::new(
                    Code::ExecutionError,
                    format!("UCC barrier post failed: {}", status),
                ));
            }

            // Wait for completion
            loop {
                let status = ucc_collective_test(req);
                if status == UCC_OK as i32 {
                    break;
                }
                if status < 0 {
                    return Err(CylonError::new(
                        Code::ExecutionError,
                        format!("UCC barrier test failed: {}", status),
                    ));
                }
                ucc_context_progress(self.ucc_context);
            }

            ucc_collective_finalize(req);
        }
        Ok(())
    }

    fn send(&self, data: &[u8], dest: i32, tag: i32) -> CylonResult<()> {
        self.ucx_comm.send(data, dest, tag)
    }

    fn recv(&self, buffer: &mut Vec<u8>, source: i32, tag: i32) -> CylonResult<()> {
        self.ucx_comm.recv(buffer, source, tag)
    }

    fn all_to_all(&self, send_data: Vec<Vec<u8>>) -> CylonResult<Vec<Vec<u8>>> {
        self.ucx_comm.all_to_all(send_data)
    }

    fn allgather(&self, send_data: &[u8]) -> CylonResult<Vec<Vec<u8>>> {
        self.ucx_comm.allgather(send_data)
    }

    fn broadcast(&self, data: &mut Vec<u8>, root: i32) -> CylonResult<()> {
        self.ucx_comm.broadcast(data, root)
    }

    /// Broadcast table from root to all processes
    /// Corresponds to C++ UCXUCCCommunicator::Bcast (ucx_communicator.cpp:669-675)
    fn bcast(
        &self,
        table: &mut Option<crate::table::Table>,
        bcast_root: i32,
        ctx: std::sync::Arc<crate::ctx::CylonContext>,
    ) -> CylonResult<()> {
        let mut impl_ = UccTableBcastImpl::new(self.ucc_team, self.ucc_context);
        impl_.execute(table, bcast_root, ctx)
    }

    /// Gather tables from all processes to root
    /// Corresponds to C++ UCXUCCCommunicator::Gather (ucx_communicator.cpp:661-667)
    fn gather(
        &self,
        table: &crate::table::Table,
        gather_root: i32,
        gather_from_root: bool,
        ctx: std::sync::Arc<crate::ctx::CylonContext>,
    ) -> CylonResult<Vec<crate::table::Table>> {
        let mut impl_ = UccTableGatherImpl::new(self.ucc_team, self.ucc_context, self.get_rank(), self.get_world_size());
        impl_.execute(table, gather_root, gather_from_root, ctx)
    }

    /// All-gather tables from all processes
    /// Corresponds to C++ UCXUCCCommunicator::AllGather (ucx_communicator.cpp:655-659)
    fn all_gather(
        &self,
        table: &crate::table::Table,
        ctx: std::sync::Arc<crate::ctx::CylonContext>,
    ) -> CylonResult<Vec<crate::table::Table>> {
        let mut impl_ = UccTableAllgatherImpl::new(self.ucc_team, self.ucc_context, self.get_world_size());
        impl_.execute(table, ctx)
    }

    // Column operations

    /// AllReduce on Column
    /// Corresponds to C++ UCXUCCCommunicator::AllReduce(Column) (ucx_communicator.cpp:677-683)
    fn all_reduce_column(
        &self,
        values: &Column,
        reduce_op: ReduceOp,
    ) -> CylonResult<Column> {
        let impl_ = UccAllReduceImpl::new(self.ucc_team, self.ucc_context);
        let result = impl_.execute_column(values, reduce_op)?;
        Ok(Column::new(result.data().clone()))
    }

    /// Allgather Column
    /// Corresponds to C++ UCXUCCCommunicator::Allgather(Column) (ucx_communicator.cpp:693-699)
    fn allgather_column(
        &self,
        values: &Column,
    ) -> CylonResult<Vec<Column>> {
        let mut impl_ = UccAllGatherImpl::new(self.ucc_team, self.ucc_context, self.get_world_size());
        let results = impl_.execute_column(values, self.get_world_size())?;
        Ok(results.into_iter().map(|c| Column::new(c.data().clone())).collect())
    }

    // Scalar operations

    /// AllReduce on Scalar
    /// Corresponds to C++ UCXUCCCommunicator::AllReduce(Scalar) (ucx_communicator.cpp:685-691)
    fn all_reduce_scalar(
        &self,
        value: &Scalar,
        reduce_op: ReduceOp,
    ) -> CylonResult<Scalar> {
        let impl_ = UccAllReduceImpl::new(self.ucc_team, self.ucc_context);
        let result = impl_.execute_scalar(value, reduce_op)?;
        Ok(Scalar::new(result.data().clone()))
    }

    /// Allgather Scalar
    /// Corresponds to C++ UCXUCCCommunicator::Allgather(Scalar) (ucx_communicator.cpp:701-706)
    fn allgather_scalar(
        &self,
        value: &Scalar,
    ) -> CylonResult<Column> {
        let mut impl_ = UccAllGatherImpl::new(self.ucc_team, self.ucc_context, self.get_world_size());
        let result = impl_.execute_scalar(value, self.get_world_size())?;
        Ok(Column::new(result.data().clone()))
    }
}

#[cfg(feature = "ucc")]
unsafe impl Send for UCXUCCCommunicator {}
#[cfg(feature = "ucc")]
unsafe impl Sync for UCXUCCCommunicator {}

#[cfg(feature = "ucc")]
impl Drop for UCXUCCCommunicator {
    fn drop(&mut self) {
        if !self.finalized {
            let _ = self.finalize();
        }
    }
}
