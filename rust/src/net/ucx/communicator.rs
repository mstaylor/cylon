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
}

// Cylon is single-threaded, so these UCX handles can be safely sent/synced
unsafe impl Send for UCXCommunicator {}
unsafe impl Sync for UCXCommunicator {}
