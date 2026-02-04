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

//! Libfabric Communicator implementation
//!
//! Implements the Cylon Communicator trait using libfabric.
//!
//! ## Blocking Pattern
//!
//! Communicator methods (barrier, send, recv, etc.) are blocking by design,
//! matching the C++ implementation. Internally they use spin-polling progress
//! loops, same as UCC:
//!
//! ```cpp
//! // C++ UCC Barrier pattern:
//! ucc_collective_post(req);
//! while (status > UCC_OK) {
//!     status = ucc_context_progress(uccContext);
//!     status = ucc_collective_test(req);
//! }
//! ```
//!
//! The Channel interface provides non-blocking operations with explicit progress.

use std::sync::Arc;
use std::any::Any;

use crate::error::{CylonError, CylonResult, Code};
use crate::net::{CommType, Communicator as CylonCommunicator, Channel};
use crate::net::comm_operations::ReduceOp;
use crate::table::{Table, Column};
use crate::scalar::Scalar;
use crate::ctx::CylonContext;

use super::{
    LibfabricConfig,
    FabricContext,
    Endpoint,
    AddressVector,
    CollectiveOps,
    LibfabricRedisOOB,
};
use super::channel::LibfabricChannel;

/// Libfabric Communicator
///
/// Implements the Cylon Communicator trait using libfabric for high-performance
/// networking with native collective operations.
pub struct LibfabricCommunicator {
    /// This process's rank
    rank: i32,
    /// Total number of processes
    world_size: i32,
    /// Fabric context (fabric, domain, cq)
    ctx: Arc<FabricContext>,
    /// Endpoint
    ep: Arc<Endpoint>,
    /// Address vector
    av: Arc<AddressVector>,
    /// Collective operations manager
    coll_ops: CollectiveOps,
    /// Configuration
    config: LibfabricConfig,
    /// Whether communicator has been finalized
    finalized: bool,
}

impl LibfabricCommunicator {
    /// Create a new LibfabricCommunicator with the given configuration
    pub fn new(config: LibfabricConfig) -> CylonResult<Arc<dyn CylonCommunicator>> {
        // Create OOB context for address exchange
        let mut oob = LibfabricRedisOOB::new(&config)?;

        // Get rank and world size
        let (world_size, rank) = oob.get_world_size_and_rank()?;

        log::info!(
            "LibfabricCommunicator: rank {} of {}, provider: {:?}",
            rank, world_size, config.provider
        );

        // Create fabric context
        let ctx = Arc::new(FabricContext::new(&config)?);

        log::info!("Using libfabric provider: {}", ctx.provider_name());

        // Create address vector
        let av = Arc::new(AddressVector::new(ctx.clone())?);

        // Create endpoint
        let ep = Arc::new(Endpoint::new(ctx.clone(), av.clone())?);

        // Exchange addresses via OOB
        let local_addr = ep.local_addr();
        let addr_size = local_addr.len();

        log::debug!("Local address size: {} bytes", addr_size);

        // Allgather all addresses
        let mut all_addrs = vec![0u8; addr_size * world_size as usize];
        oob.oob_allgather(local_addr, &mut all_addrs, addr_size)?;

        // Insert addresses into AV
        for i in 0..world_size {
            let offset = i as usize * addr_size;
            let peer_addr = &all_addrs[offset..offset + addr_size];
            av.insert(peer_addr, i)?;
        }

        log::debug!("Inserted {} addresses into AV", world_size);

        // Synchronize before proceeding
        oob.barrier("init")?;

        // Create collective operations manager
        let mut coll_ops = CollectiveOps::new(
            ctx.clone(),
            ep.clone(),
            av.clone(),
            rank,
            world_size,
        );

        // Initialize collectives
        coll_ops.init()?;

        Ok(Arc::new(Self {
            rank,
            world_size,
            ctx,
            ep,
            av,
            coll_ops,
            config,
            finalized: false,
        }))
    }

    /// Create from environment variables
    ///
    /// Reads configuration from environment:
    /// - `CYLON_SESSION_ID` - Required session ID
    /// - `CYLON_LIBFABRIC_REDIS_ADDR` - Redis address (default: 127.0.0.1:6379)
    /// - `CYLON_LIBFABRIC_WORLD_SIZE` - World size (required)
    /// - `FI_PROVIDER` - Optional provider override
    pub fn from_env() -> CylonResult<Arc<dyn CylonCommunicator>> {
        let session_id = std::env::var("CYLON_SESSION_ID").map_err(|_| {
            CylonError::new(
                Code::Invalid,
                "CYLON_SESSION_ID environment variable not set"
            )
        })?;

        let redis_addr = std::env::var("CYLON_LIBFABRIC_REDIS_ADDR")
            .unwrap_or_else(|_| "127.0.0.1:6379".to_string());

        let parts: Vec<&str> = redis_addr.split(':').collect();
        let redis_host = parts.get(0).unwrap_or(&"127.0.0.1").to_string();
        let redis_port: u16 = parts.get(1)
            .and_then(|s| s.parse().ok())
            .unwrap_or(6379);

        let world_size: i32 = std::env::var("CYLON_LIBFABRIC_WORLD_SIZE")
            .map_err(|_| CylonError::new(Code::Invalid, "CYLON_LIBFABRIC_WORLD_SIZE not set"))?
            .parse()
            .map_err(|_| CylonError::new(Code::Invalid, "Invalid CYLON_LIBFABRIC_WORLD_SIZE"))?;

        // Check for provider override
        let provider = std::env::var("FI_PROVIDER").ok();

        let mut config = LibfabricConfig::with_redis(&redis_host, redis_port, &session_id, world_size);
        config.provider = provider;

        Self::new(config)
    }

    /// Get the fabric context
    pub fn context(&self) -> &Arc<FabricContext> {
        &self.ctx
    }

    /// Get the endpoint
    pub fn endpoint(&self) -> &Arc<Endpoint> {
        &self.ep
    }

    /// Get the address vector
    pub fn address_vector(&self) -> &Arc<AddressVector> {
        &self.av
    }

    /// Progress all pending operations (non-blocking)
    ///
    /// This is exposed for advanced usage. Normal users should use
    /// the blocking Communicator methods.
    pub fn progress(&self) -> CylonResult<usize> {
        self.coll_ops.progress()
    }

    /// Spin-polling wait for operation completion
    ///
    /// Same pattern as C++ UCC implementation.
    fn wait_for_op(&self, op_id: u64) -> CylonResult<()> {
        loop {
            self.coll_ops.progress()?;
            if self.coll_ops.is_complete(op_id) {
                break;
            }
            std::thread::yield_now();
        }

        if self.coll_ops.has_error(op_id) {
            return Err(CylonError::new(Code::ExecutionError, "Operation failed"));
        }

        Ok(())
    }
}

impl CylonCommunicator for LibfabricCommunicator {
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
        CommType::Libfabric
    }

    fn is_finalized(&self) -> bool {
        self.finalized
    }

    fn create_channel(&self) -> CylonResult<Box<dyn Channel>> {
        Ok(Box::new(LibfabricChannel::new(
            self.ctx.clone(),
            self.ep.clone(),
            self.av.clone(),
            self.rank,
            self.world_size,
        )?))
    }

    fn finalize(&mut self) -> CylonResult<()> {
        if !self.finalized {
            // Progress until all operations complete
            while !self.coll_ops.all_complete() {
                self.coll_ops.progress()?;
                std::thread::yield_now();
            }

            self.finalized = true;
            log::info!("LibfabricCommunicator finalized");
        }
        Ok(())
    }

    fn barrier(&self) -> CylonResult<()> {
        let op_id = self.coll_ops.barrier()?;
        self.wait_for_op(op_id)
    }

    fn send(&self, data: &[u8], dest: i32, _tag: i32) -> CylonResult<()> {
        let dest_addr = self.av.lookup(dest).ok_or_else(|| {
            CylonError::new(Code::ExecutionError, format!("Destination {} not found", dest))
        })?;

        // Post send (retry if EAGAIN)
        loop {
            match self.ep.send(data, dest_addr, std::ptr::null_mut()) {
                Ok(true) => break,
                Ok(false) => {
                    self.progress()?;
                    std::thread::yield_now();
                }
                Err(e) => return Err(e),
            }
        }

        // Wait for send completion
        loop {
            let count = self.progress()?;
            if count > 0 {
                break;
            }
            std::thread::yield_now();
        }

        Ok(())
    }

    fn recv(&self, buffer: &mut Vec<u8>, source: i32, _tag: i32) -> CylonResult<()> {
        let src_addr = self.av.lookup(source).ok_or_else(|| {
            CylonError::new(Code::ExecutionError, format!("Source {} not found", source))
        })?;

        if buffer.is_empty() {
            buffer.resize(65536, 0);
        }

        // Post recv (retry if EAGAIN)
        loop {
            match self.ep.recv(buffer, src_addr, std::ptr::null_mut()) {
                Ok(true) => break,
                Ok(false) => {
                    self.progress()?;
                    std::thread::yield_now();
                }
                Err(e) => return Err(e),
            }
        }

        // Wait for recv completion
        loop {
            let count = self.progress()?;
            if count > 0 {
                break;
            }
            std::thread::yield_now();
        }

        Ok(())
    }

    fn all_to_all(&self, send_data: Vec<Vec<u8>>) -> CylonResult<Vec<Vec<u8>>> {
        let chunk_size = send_data.get(0).map(|v| v.len()).unwrap_or(0);
        let send_buf: Vec<u8> = send_data.into_iter().flatten().collect();
        let mut recv_buf = vec![0u8; send_buf.len()];

        let op_id = self.coll_ops.alltoall(&send_buf, &mut recv_buf)?;
        self.wait_for_op(op_id)?;

        Ok(recv_buf.chunks(chunk_size).map(|c| c.to_vec()).collect())
    }

    fn allgather(&self, data: &[u8]) -> CylonResult<Vec<Vec<u8>>> {
        let mut recv_buf = vec![0u8; data.len() * self.world_size as usize];

        let op_id = self.coll_ops.allgather(data, &mut recv_buf)?;
        self.wait_for_op(op_id)?;

        Ok(recv_buf.chunks(data.len()).map(|c| c.to_vec()).collect())
    }

    fn broadcast(&self, data: &mut Vec<u8>, root: i32) -> CylonResult<()> {
        let op_id = self.coll_ops.broadcast(data.as_mut_slice(), root)?;
        self.wait_for_op(op_id)
    }

    // Table operations - TODO: implement like MPI

    fn bcast(&self, _table: &mut Option<Table>, _bcast_root: i32, _ctx: Arc<CylonContext>) -> CylonResult<()> {
        Err(CylonError::new(Code::NotImplemented, "Table bcast not yet implemented for libfabric"))
    }

    fn gather(&self, _table: &Table, _gather_root: i32, _gather_from_root: bool, _ctx: Arc<CylonContext>) -> CylonResult<Vec<Table>> {
        Err(CylonError::new(Code::NotImplemented, "Table gather not yet implemented for libfabric"))
    }

    fn all_gather(&self, _table: &Table, _ctx: Arc<CylonContext>) -> CylonResult<Vec<Table>> {
        Err(CylonError::new(Code::NotImplemented, "Table all_gather not yet implemented for libfabric"))
    }

    fn all_reduce_column(&self, _values: &Column, _reduce_op: ReduceOp) -> CylonResult<Column> {
        Err(CylonError::new(Code::NotImplemented, "Column all_reduce not yet implemented for libfabric"))
    }

    fn allgather_column(&self, _values: &Column) -> CylonResult<Vec<Column>> {
        Err(CylonError::new(Code::NotImplemented, "Column allgather not yet implemented for libfabric"))
    }

    fn all_reduce_scalar(&self, _value: &Scalar, _reduce_op: ReduceOp) -> CylonResult<Scalar> {
        Err(CylonError::new(Code::NotImplemented, "Scalar all_reduce not yet implemented for libfabric"))
    }

    fn allgather_scalar(&self, _value: &Scalar) -> CylonResult<Column> {
        Err(CylonError::new(Code::NotImplemented, "Scalar allgather not yet implemented for libfabric"))
    }
}
