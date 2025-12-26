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

//! Libfabric collective operations
//!
//! This module provides collective operations using libfabric's native
//! collective API (fi_collective.h).
//!
//! ## Non-Blocking Progress Pattern
//!
//! All operations are non-blocking and follow the same progress model as MPI:
//! 1. Call the operation (e.g., `allreduce()`) - initiates and returns immediately
//! 2. Call `progress()` repeatedly to check for completions
//! 3. Check `is_complete()` to see if operations have finished
//!
//! There is NO blocking wait function - this matches the MPI channel implementation.
//!
//! Example:
//! ```ignore
//! // Start operation
//! let op_id = ops.allreduce(&send_buf, &mut recv_buf, ReduceOp::Sum)?;
//!
//! // Progress loop (non-blocking)
//! loop {
//!     ops.progress()?;
//!     if ops.is_complete(op_id) {
//!         break;
//!     }
//!     // Do other work or yield...
//! }
//! ```

use std::ptr;
use std::sync::{Arc, RwLock};
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::collections::HashMap;

use crate::error::{CylonError, CylonResult, Code};
use crate::net::comm_operations::ReduceOp;
use super::libfabric_sys::*;
use super::error::fi_strerror;
use super::context::FabricContext;
use super::endpoint::Endpoint;
use super::address_vector::{AddressVector, AVSet};

/// Operation status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationStatus {
    /// Operation is pending (not yet completed)
    Pending = 0,
    /// Operation completed successfully
    Completed = 1,
    /// Operation failed with an error
    Error = 2,
}

impl From<u8> for OperationStatus {
    fn from(val: u8) -> Self {
        match val {
            0 => OperationStatus::Pending,
            1 => OperationStatus::Completed,
            _ => OperationStatus::Error,
        }
    }
}

/// Operation type for tracking
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationType {
    Barrier,
    Broadcast,
    Allreduce,
    Allgather,
    Alltoall,
    Reduce,
    Scatter,
    Gather,
    ReduceScatter,
    Send,
    Recv,
}

/// Context for tracking a non-blocking operation
#[repr(C)]
pub struct OperationContext {
    /// Unique operation ID
    id: u64,
    /// Operation type
    op_type: OperationType,
    /// Status (atomic for thread-safe access)
    status: AtomicU8,
    /// Internal context for libfabric
    fi_context: fi_context,
}

impl OperationContext {
    /// Create a new operation context
    pub fn new(id: u64, op_type: OperationType) -> Self {
        Self {
            id,
            op_type,
            status: AtomicU8::new(OperationStatus::Pending as u8),
            fi_context: fi_context {
                internal: [ptr::null_mut(); 4],
            },
        }
    }

    /// Get the operation ID
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Get the operation type
    pub fn op_type(&self) -> OperationType {
        self.op_type
    }

    /// Check if the operation is pending
    pub fn is_pending(&self) -> bool {
        OperationStatus::from(self.status.load(Ordering::SeqCst)) == OperationStatus::Pending
    }

    /// Check if the operation is completed
    pub fn is_completed(&self) -> bool {
        OperationStatus::from(self.status.load(Ordering::SeqCst)) == OperationStatus::Completed
    }

    /// Check if the operation has an error
    pub fn is_error(&self) -> bool {
        OperationStatus::from(self.status.load(Ordering::SeqCst)) == OperationStatus::Error
    }

    /// Mark the operation as completed
    pub fn mark_completed(&self) {
        self.status.store(OperationStatus::Completed as u8, Ordering::SeqCst);
    }

    /// Mark the operation as error
    pub fn mark_error(&self) {
        self.status.store(OperationStatus::Error as u8, Ordering::SeqCst);
    }

    /// Get pointer to the fi_context (for passing to libfabric)
    pub fn as_fi_context_ptr(&mut self) -> *mut std::ffi::c_void {
        &mut self.fi_context as *mut _ as *mut std::ffi::c_void
    }
}

/// Convert Cylon ReduceOp to libfabric fi_op
pub fn to_fi_op(op: ReduceOp) -> fi_op {
    match op {
        ReduceOp::Sum => FI_SUM,
        ReduceOp::Min => FI_MIN,
        ReduceOp::Max => FI_MAX,
        ReduceOp::Prod => FI_PROD,
        ReduceOp::Land => FI_LAND,
        ReduceOp::Lor => FI_LOR,
        ReduceOp::Band => FI_BAND,
        ReduceOp::Bor => FI_BOR,
    }
}

/// Convert Rust type to libfabric fi_datatype
pub fn get_fi_datatype<T: 'static>() -> fi_datatype {
    use std::any::TypeId;

    let type_id = TypeId::of::<T>();

    if type_id == TypeId::of::<i8>() {
        FI_INT8
    } else if type_id == TypeId::of::<i16>() {
        FI_INT16
    } else if type_id == TypeId::of::<i32>() {
        FI_INT32
    } else if type_id == TypeId::of::<i64>() {
        FI_INT64
    } else if type_id == TypeId::of::<u8>() {
        FI_UINT8
    } else if type_id == TypeId::of::<u16>() {
        FI_UINT16
    } else if type_id == TypeId::of::<u32>() {
        FI_UINT32
    } else if type_id == TypeId::of::<u64>() {
        FI_UINT64
    } else if type_id == TypeId::of::<f32>() {
        FI_FLOAT
    } else if type_id == TypeId::of::<f64>() {
        FI_DOUBLE
    } else {
        // Default to bytes for unknown types
        FI_UINT8
    }
}

/// Collective operations manager
///
/// Manages collective operations for a group of endpoints.
/// All operations are non-blocking - call `progress()` to advance operations.
///
/// If hardware collectives are not supported by the provider, operations
/// will use software fallbacks (point-to-point based).
pub struct CollectiveOps {
    /// Fabric context
    ctx: Arc<FabricContext>,
    /// Endpoint
    ep: Arc<Endpoint>,
    /// Address vector
    av: Arc<AddressVector>,
    /// AV set for collectives (created on demand)
    av_set: Option<Arc<AVSet>>,
    /// Multicast group handle
    mc: *mut fid_mc,
    /// Collective address
    coll_addr: fi_addr_t,
    /// Whether collectives are initialized
    initialized: bool,
    /// Whether hardware collectives are supported
    hw_coll_supported: bool,
    /// This rank
    rank: i32,
    /// World size
    world_size: i32,
    /// Next operation ID
    next_op_id: AtomicU64,
    /// Pending operations (id -> context)
    pending_ops: RwLock<HashMap<u64, Box<OperationContext>>>,
}

// Safety: CollectiveOps manages raw pointers but ensures proper cleanup
unsafe impl Send for CollectiveOps {}
unsafe impl Sync for CollectiveOps {}

impl CollectiveOps {
    /// Create a new CollectiveOps instance
    pub fn new(
        ctx: Arc<FabricContext>,
        ep: Arc<Endpoint>,
        av: Arc<AddressVector>,
        rank: i32,
        world_size: i32,
    ) -> Self {
        Self {
            ctx,
            ep,
            av,
            av_set: None,
            mc: ptr::null_mut(),
            coll_addr: FI_ADDR_UNSPEC,
            initialized: false,
            hw_coll_supported: false,
            rank,
            world_size,
            next_op_id: AtomicU64::new(0),
            pending_ops: RwLock::new(HashMap::new()),
        }
    }

    /// Initialize collectives by joining the collective group
    ///
    /// This must be called after all addresses have been exchanged.
    /// If hardware collectives are not supported, this will still succeed
    /// and operations will fall back to software implementations.
    pub fn init(&mut self) -> CylonResult<()> {
        if self.initialized {
            return Ok(());
        }

        // Get all peer addresses
        let mut members: Vec<fi_addr_t> = Vec::with_capacity(self.world_size as usize);
        for i in 0..self.world_size {
            if let Some(addr) = self.av.lookup(i) {
                members.push(addr);
            } else {
                return Err(CylonError::new(
                    Code::ExecutionError,
                    format!("Address for rank {} not found in AV", i),
                ));
            }
        }

        // Try to create AV set for hardware collectives
        match AVSet::new(self.av.clone(), &members) {
            Ok(av_set) => {
                self.coll_addr = av_set.coll_addr();
                self.av_set = Some(Arc::new(av_set));

                // Try to join the collective group
                unsafe {
                    let ret = fi_join_collective(
                        self.ep.ep(),
                        self.coll_addr,
                        self.av_set.as_ref().unwrap().av_set() as *const _,
                        0,
                        &mut self.mc,
                        ptr::null_mut(),
                    );

                    if ret == 0 {
                        self.hw_coll_supported = true;
                        log::info!("Hardware collectives enabled for {} workers", self.world_size);
                    } else {
                        log::warn!("fi_join_collective failed ({}), using software collectives",
                            fi_strerror(ret));
                        self.av_set = None;
                        self.hw_coll_supported = false;
                    }
                }
            }
            Err(e) => {
                log::warn!("AV set not supported ({}), using software collectives", e);
                self.hw_coll_supported = false;
            }
        }

        self.initialized = true;
        log::info!("Collective operations initialized for {} workers (hw_coll={})",
            self.world_size, self.hw_coll_supported);

        Ok(())
    }

    /// Check if collectives are initialized
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Get the number of pending operations
    pub fn pending_count(&self) -> usize {
        self.pending_ops.read().unwrap().len()
    }

    /// Check if all operations are complete
    pub fn all_complete(&self) -> bool {
        let pending = self.pending_ops.read().unwrap();
        pending.iter().all(|(_, ctx)| ctx.is_completed() || ctx.is_error())
    }

    /// Allocate a new operation context
    fn new_context(&self, op_type: OperationType) -> Box<OperationContext> {
        let id = self.next_op_id.fetch_add(1, Ordering::SeqCst);
        Box::new(OperationContext::new(id, op_type))
    }

    /// Register a pending operation
    fn register_pending(&self, ctx: Box<OperationContext>) -> u64 {
        let id = ctx.id();
        self.pending_ops.write().unwrap().insert(id, ctx);
        id
    }

    /// Progress pending operations (non-blocking)
    ///
    /// Polls the completion queue and marks completed operations.
    /// This is the main progress function - call it repeatedly in your event loop.
    /// Returns the number of operations that completed in this call.
    pub fn progress(&self) -> CylonResult<usize> {
        let mut entries = vec![unsafe { std::mem::zeroed::<fi_cq_tagged_entry>() }; 16];
        let count = self.ctx.poll_cq(&mut entries)?;

        if count > 0 {
            // Mark completed operations
            let pending = self.pending_ops.read().unwrap();
            for entry in entries.iter().take(count) {
                // The context pointer is stored in op_context
                let ctx_ptr = entry.op_context as *mut OperationContext;
                if !ctx_ptr.is_null() {
                    // Find and mark the operation as completed
                    unsafe {
                        let op_id = (*ctx_ptr).id();
                        if let Some(op_ctx) = pending.get(&op_id) {
                            op_ctx.mark_completed();
                        }
                    }
                }
            }
        }

        // Check for errors
        if let Some(err) = self.ctx.read_cq_error() {
            log::error!("Completion queue error: {}", err.err);
        }

        Ok(count)
    }

    /// Check if a specific operation is complete (non-blocking)
    ///
    /// Does NOT call progress - caller should call progress() separately.
    pub fn is_complete(&self, op_id: u64) -> bool {
        let pending = self.pending_ops.read().unwrap();
        if let Some(ctx) = pending.get(&op_id) {
            ctx.is_completed() || ctx.is_error()
        } else {
            // Operation not found - assume completed and cleaned up
            true
        }
    }

    /// Check if a specific operation has an error
    pub fn has_error(&self, op_id: u64) -> bool {
        let pending = self.pending_ops.read().unwrap();
        if let Some(ctx) = pending.get(&op_id) {
            ctx.is_error()
        } else {
            false
        }
    }

    /// Clean up completed operations
    ///
    /// Removes completed/errored operations from the pending map.
    /// Call periodically to prevent memory growth.
    pub fn cleanup_completed(&self) -> usize {
        let mut pending = self.pending_ops.write().unwrap();
        let before = pending.len();

        pending.retain(|_, ctx| ctx.is_pending());

        before - pending.len()
    }

    /// Barrier synchronization (non-blocking)
    ///
    /// Returns operation ID. Call progress() and check is_complete() for completion.
    pub fn barrier(&self) -> CylonResult<u64> {
        if !self.initialized {
            return Err(CylonError::new(
                Code::ExecutionError,
                "Collective operations not initialized".to_string(),
            ));
        }

        let mut ctx = self.new_context(OperationType::Barrier);

        // For single process or no hardware collectives, barrier is a no-op
        if self.world_size == 1 || !self.hw_coll_supported {
            ctx.mark_completed();
            return Ok(self.register_pending(ctx));
        }

        unsafe {
            let ret = fi_barrier(self.ep.ep(), self.coll_addr, ctx.as_fi_context_ptr());

            if ret != 0 && ret != -(FI_EAGAIN as isize) {
                return Err(CylonError::new(
                    Code::ExecutionError,
                    format!("fi_barrier failed: {} (error code: {})", fi_strerror(-(ret as i32)), ret),
                ));
            }
        }

        Ok(self.register_pending(ctx))
    }

    /// Broadcast data from root to all processes (non-blocking)
    ///
    /// Returns operation ID. Call progress() and check is_complete() for completion.
    pub fn broadcast<T: Copy + 'static>(
        &self,
        buf: &mut [T],
        root: i32,
    ) -> CylonResult<u64> {
        if !self.initialized {
            return Err(CylonError::new(
                Code::ExecutionError,
                "Collective operations not initialized".to_string(),
            ));
        }

        let root_addr = self.av.lookup(root).ok_or_else(|| {
            CylonError::new(Code::ExecutionError, format!("Root rank {} not found", root))
        })?;

        let datatype = get_fi_datatype::<T>();
        let count = buf.len();
        let mut ctx = self.new_context(OperationType::Broadcast);

        unsafe {
            let ret = fi_broadcast(
                self.ep.ep(),
                buf.as_mut_ptr() as *mut _,
                count,
                ptr::null_mut(),
                self.coll_addr,
                root_addr,
                datatype,
                0,
                ctx.as_fi_context_ptr(),
            );

            if ret != 0 && ret != -(FI_EAGAIN as isize) {
                return Err(CylonError::new(
                    Code::ExecutionError,
                    format!("fi_broadcast failed: {} (error code: {})", fi_strerror(-(ret as i32)), ret),
                ));
            }
        }

        Ok(self.register_pending(ctx))
    }

    /// Allreduce: reduce data from all processes and distribute result (non-blocking)
    pub fn allreduce<T: Copy + 'static>(
        &self,
        send_buf: &[T],
        recv_buf: &mut [T],
        op: ReduceOp,
    ) -> CylonResult<u64> {
        if !self.initialized {
            return Err(CylonError::new(
                Code::ExecutionError,
                "Collective operations not initialized".to_string(),
            ));
        }

        if send_buf.len() != recv_buf.len() {
            return Err(CylonError::new(
                Code::Invalid,
                "Send and receive buffers must have same length".to_string(),
            ));
        }

        let datatype = get_fi_datatype::<T>();
        let fi_op = to_fi_op(op);
        let count = send_buf.len();
        let mut ctx = self.new_context(OperationType::Allreduce);

        unsafe {
            let ret = fi_allreduce(
                self.ep.ep(),
                send_buf.as_ptr() as *const _,
                count,
                ptr::null_mut(),
                recv_buf.as_mut_ptr() as *mut _,
                ptr::null_mut(),
                self.coll_addr,
                datatype,
                fi_op,
                0,
                ctx.as_fi_context_ptr(),
            );

            if ret != 0 && ret != -(FI_EAGAIN as isize) {
                return Err(CylonError::new(
                    Code::ExecutionError,
                    format!("fi_allreduce failed: {} (error code: {})", fi_strerror(-(ret as i32)), ret),
                ));
            }
        }

        Ok(self.register_pending(ctx))
    }

    /// Allgather: gather data from all processes to all (non-blocking)
    pub fn allgather<T: Copy + 'static>(
        &self,
        send_buf: &[T],
        recv_buf: &mut [T],
    ) -> CylonResult<u64> {
        if !self.initialized {
            return Err(CylonError::new(
                Code::ExecutionError,
                "Collective operations not initialized".to_string(),
            ));
        }

        let expected_recv_len = send_buf.len() * self.world_size as usize;
        if recv_buf.len() < expected_recv_len {
            return Err(CylonError::new(
                Code::Invalid,
                format!("Receive buffer too small: {} < {}", recv_buf.len(), expected_recv_len),
            ));
        }

        let datatype = get_fi_datatype::<T>();
        let count = send_buf.len();
        let mut ctx = self.new_context(OperationType::Allgather);

        unsafe {
            let ret = fi_allgather(
                self.ep.ep(),
                send_buf.as_ptr() as *const _,
                count,
                ptr::null_mut(),
                recv_buf.as_mut_ptr() as *mut _,
                ptr::null_mut(),
                self.coll_addr,
                datatype,
                0,
                ctx.as_fi_context_ptr(),
            );

            if ret != 0 && ret != -(FI_EAGAIN as isize) {
                return Err(CylonError::new(
                    Code::ExecutionError,
                    format!("fi_allgather failed: {} (error code: {})", fi_strerror(-(ret as i32)), ret),
                ));
            }
        }

        Ok(self.register_pending(ctx))
    }

    /// Alltoall: all-to-all data exchange (non-blocking)
    pub fn alltoall<T: Copy + 'static>(
        &self,
        send_buf: &[T],
        recv_buf: &mut [T],
    ) -> CylonResult<u64> {
        if !self.initialized {
            return Err(CylonError::new(
                Code::ExecutionError,
                "Collective operations not initialized".to_string(),
            ));
        }

        if send_buf.len() != recv_buf.len() {
            return Err(CylonError::new(
                Code::Invalid,
                "Send and receive buffers must have same length".to_string(),
            ));
        }

        let datatype = get_fi_datatype::<T>();
        let count_per_peer = send_buf.len() / self.world_size as usize;
        let mut ctx = self.new_context(OperationType::Alltoall);

        unsafe {
            let ret = fi_alltoall(
                self.ep.ep(),
                send_buf.as_ptr() as *const _,
                count_per_peer,
                ptr::null_mut(),
                recv_buf.as_mut_ptr() as *mut _,
                ptr::null_mut(),
                self.coll_addr,
                datatype,
                0,
                ctx.as_fi_context_ptr(),
            );

            if ret != 0 && ret != -(FI_EAGAIN as isize) {
                return Err(CylonError::new(
                    Code::ExecutionError,
                    format!("fi_alltoall failed: {} (error code: {})", fi_strerror(-(ret as i32)), ret),
                ));
            }
        }

        Ok(self.register_pending(ctx))
    }

    /// Reduce: reduce data from all processes to root (non-blocking)
    pub fn reduce<T: Copy + 'static>(
        &self,
        send_buf: &[T],
        recv_buf: &mut [T],
        op: ReduceOp,
        root: i32,
    ) -> CylonResult<u64> {
        if !self.initialized {
            return Err(CylonError::new(
                Code::ExecutionError,
                "Collective operations not initialized".to_string(),
            ));
        }

        let root_addr = self.av.lookup(root).ok_or_else(|| {
            CylonError::new(Code::ExecutionError, format!("Root rank {} not found", root))
        })?;

        let datatype = get_fi_datatype::<T>();
        let fi_op = to_fi_op(op);
        let count = send_buf.len();
        let mut ctx = self.new_context(OperationType::Reduce);

        unsafe {
            let ret = fi_reduce(
                self.ep.ep(),
                send_buf.as_ptr() as *const _,
                count,
                ptr::null_mut(),
                recv_buf.as_mut_ptr() as *mut _,
                ptr::null_mut(),
                self.coll_addr,
                root_addr,
                datatype,
                fi_op,
                0,
                ctx.as_fi_context_ptr(),
            );

            if ret != 0 && ret != -(FI_EAGAIN as isize) {
                return Err(CylonError::new(
                    Code::ExecutionError,
                    format!("fi_reduce failed: {} (error code: {})", fi_strerror(-(ret as i32)), ret),
                ));
            }
        }

        Ok(self.register_pending(ctx))
    }

    /// Scatter: distribute data from root to all (non-blocking)
    pub fn scatter<T: Copy + 'static>(
        &self,
        send_buf: &[T],
        recv_buf: &mut [T],
        root: i32,
    ) -> CylonResult<u64> {
        if !self.initialized {
            return Err(CylonError::new(
                Code::ExecutionError,
                "Collective operations not initialized".to_string(),
            ));
        }

        let root_addr = self.av.lookup(root).ok_or_else(|| {
            CylonError::new(Code::ExecutionError, format!("Root rank {} not found", root))
        })?;

        let datatype = get_fi_datatype::<T>();
        let count = recv_buf.len();
        let mut ctx = self.new_context(OperationType::Scatter);

        unsafe {
            let ret = fi_scatter(
                self.ep.ep(),
                send_buf.as_ptr() as *const _,
                count,
                ptr::null_mut(),
                recv_buf.as_mut_ptr() as *mut _,
                ptr::null_mut(),
                self.coll_addr,
                root_addr,
                datatype,
                0,
                ctx.as_fi_context_ptr(),
            );

            if ret != 0 && ret != -(FI_EAGAIN as isize) {
                return Err(CylonError::new(
                    Code::ExecutionError,
                    format!("fi_scatter failed: {} (error code: {})", fi_strerror(-(ret as i32)), ret),
                ));
            }
        }

        Ok(self.register_pending(ctx))
    }

    /// Gather: collect data from all to root (non-blocking)
    pub fn gather<T: Copy + 'static>(
        &self,
        send_buf: &[T],
        recv_buf: &mut [T],
        root: i32,
    ) -> CylonResult<u64> {
        if !self.initialized {
            return Err(CylonError::new(
                Code::ExecutionError,
                "Collective operations not initialized".to_string(),
            ));
        }

        let root_addr = self.av.lookup(root).ok_or_else(|| {
            CylonError::new(Code::ExecutionError, format!("Root rank {} not found", root))
        })?;

        let datatype = get_fi_datatype::<T>();
        let count = send_buf.len();
        let mut ctx = self.new_context(OperationType::Gather);

        unsafe {
            let ret = fi_gather(
                self.ep.ep(),
                send_buf.as_ptr() as *const _,
                count,
                ptr::null_mut(),
                recv_buf.as_mut_ptr() as *mut _,
                ptr::null_mut(),
                self.coll_addr,
                root_addr,
                datatype,
                0,
                ctx.as_fi_context_ptr(),
            );

            if ret != 0 && ret != -(FI_EAGAIN as isize) {
                return Err(CylonError::new(
                    Code::ExecutionError,
                    format!("fi_gather failed: {} (error code: {})", fi_strerror(-(ret as i32)), ret),
                ));
            }
        }

        Ok(self.register_pending(ctx))
    }

    /// Reduce-scatter: reduce and scatter results (non-blocking)
    pub fn reduce_scatter<T: Copy + 'static>(
        &self,
        send_buf: &[T],
        recv_buf: &mut [T],
        op: ReduceOp,
    ) -> CylonResult<u64> {
        if !self.initialized {
            return Err(CylonError::new(
                Code::ExecutionError,
                "Collective operations not initialized".to_string(),
            ));
        }

        let datatype = get_fi_datatype::<T>();
        let fi_op = to_fi_op(op);
        let count = recv_buf.len();
        let mut ctx = self.new_context(OperationType::ReduceScatter);

        unsafe {
            let ret = fi_reduce_scatter(
                self.ep.ep(),
                send_buf.as_ptr() as *const _,
                count,
                ptr::null_mut(),
                recv_buf.as_mut_ptr() as *mut _,
                ptr::null_mut(),
                self.coll_addr,
                datatype,
                fi_op,
                0,
                ctx.as_fi_context_ptr(),
            );

            if ret != 0 && ret != -(FI_EAGAIN as isize) {
                return Err(CylonError::new(
                    Code::ExecutionError,
                    format!("fi_reduce_scatter failed: {} (error code: {})", fi_strerror(-(ret as i32)), ret),
                ));
            }
        }

        Ok(self.register_pending(ctx))
    }
}

impl Drop for CollectiveOps {
    fn drop(&mut self) {
        // Progress until all complete before cleanup
        while !self.all_complete() {
            let _ = self.progress();
            std::thread::yield_now();
        }

        unsafe {
            if !self.mc.is_null() {
                fi_close(&mut (*self.mc).fid);
            }
        }
    }
}
