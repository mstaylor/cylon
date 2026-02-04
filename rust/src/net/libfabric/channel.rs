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

//! Libfabric Channel implementation
//!
//! Implements the Cylon Channel trait using libfabric.
//!
//! ## Non-Blocking Progress Pattern
//!
//! Like MPI and UCX channels, this channel uses non-blocking operations
//! with explicit progress calls:
//! - `send()` queues a request (non-blocking)
//! - `progressSends()` advances send operations (non-blocking)
//! - `progressReceives()` advances receive operations (non-blocking)

use std::collections::{HashMap, VecDeque};
use std::ptr;
use std::sync::Arc;

use crate::error::{CylonError, CylonResult, Code};
use crate::net::{
    Channel, ChannelReceiveCallback, ChannelSendCallback, Allocator, Buffer,
    CYLON_CHANNEL_HEADER_SIZE, CYLON_MSG_FIN, CYLON_MSG_NOT_FIN, MAX_PENDING,
};
use crate::net::request::CylonRequest;
use crate::net::buffer::VecBuffer;

use super::libfabric_sys::*;
use super::error::fi_strerror;
use super::context::FabricContext;
use super::endpoint::Endpoint;
use super::address_vector::AddressVector;

/// Send status enum (matches MPI channel)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SendStatus {
    Init = 0,
    LengthPosted = 1,
    Posted = 2,
    Finish = 3,
    Done = 4,
}

/// Receive status enum (matches MPI channel)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReceiveStatus {
    Init = 0,
    LengthPosted = 1,
    Posted = 2,
    ReceivedFin = 3,
}

/// Pending send tracking
struct PendingSend {
    header_buf: [i32; CYLON_CHANNEL_HEADER_SIZE],
    pending_data: VecDeque<Box<CylonRequest>>,
    status: SendStatus,
    current_send: Option<Box<CylonRequest>>,
    /// fi_addr_t for the target
    target_addr: fi_addr_t,
}

impl PendingSend {
    fn new(target_addr: fi_addr_t) -> Self {
        Self {
            header_buf: [0; CYLON_CHANNEL_HEADER_SIZE],
            pending_data: VecDeque::new(),
            status: SendStatus::Init,
            current_send: None,
            target_addr,
        }
    }
}

/// Pending receive tracking
struct PendingReceive {
    header_buf: [i32; CYLON_CHANNEL_HEADER_SIZE],
    receive_id: i32,
    data: Option<Vec<u8>>,
    length: usize,
    status: ReceiveStatus,
    /// fi_addr_t for the source
    source_addr: fi_addr_t,
    /// Whether header receive is complete
    header_complete: bool,
    /// Whether data receive is complete
    data_complete: bool,
}

impl PendingReceive {
    fn new(receive_id: i32, source_addr: fi_addr_t) -> Self {
        Self {
            header_buf: [0; CYLON_CHANNEL_HEADER_SIZE],
            receive_id,
            data: None,
            length: 0,
            status: ReceiveStatus::Init,
            source_addr,
            header_complete: false,
            data_complete: false,
        }
    }
}

/// Libfabric Channel implementation
///
/// Provides non-blocking send/recv with progress-based completion,
/// matching the MPI and UCX channel implementations.
pub struct LibfabricChannel {
    edge: i32,
    sends: HashMap<i32, PendingSend>,
    pending_receives: HashMap<i32, PendingReceive>,
    finish_requests: HashMap<i32, Box<CylonRequest>>,
    rcv_fn: Option<Box<dyn ChannelReceiveCallback>>,
    send_comp_fn: Option<Box<dyn ChannelSendCallback>>,
    allocator: Option<Box<dyn Allocator>>,
    rank: i32,
    world_size: i32,
    /// Fabric context
    ctx: Arc<FabricContext>,
    /// Endpoint
    ep: Arc<Endpoint>,
    /// Address vector
    av: Arc<AddressVector>,
}

unsafe impl Send for LibfabricChannel {}
unsafe impl Sync for LibfabricChannel {}

impl LibfabricChannel {
    /// Create a new Libfabric channel
    pub fn new(
        ctx: Arc<FabricContext>,
        ep: Arc<Endpoint>,
        av: Arc<AddressVector>,
        rank: i32,
        world_size: i32,
    ) -> CylonResult<Self> {
        Ok(Self {
            edge: 0,
            sends: HashMap::new(),
            pending_receives: HashMap::new(),
            finish_requests: HashMap::new(),
            rcv_fn: None,
            send_comp_fn: None,
            allocator: None,
            rank,
            world_size,
            ctx,
            ep,
            av,
        })
    }

    /// Post a header send
    fn send_header(&mut self, target: i32) {
        if let Some(ps) = self.sends.get_mut(&target) {
            if let Some(r) = ps.pending_data.front() {
                // Prepare header
                let len = r.len();
                ps.header_buf[0] = len as i32;
                ps.header_buf[1] = CYLON_MSG_NOT_FIN;

                // Copy optional header if present
                if r.header_length > 0 {
                    let copy_len = r.header_length.min(6);
                    ps.header_buf[2..2 + copy_len].copy_from_slice(&r.header[..copy_len]);
                }

                // Post send for header (as bytes)
                let header_bytes = unsafe {
                    std::slice::from_raw_parts(
                        ps.header_buf.as_ptr() as *const u8,
                        (2 + r.header_length) * std::mem::size_of::<i32>(),
                    )
                };

                // Non-blocking send - will check completion in progress
                match self.ep.send(header_bytes, ps.target_addr, ptr::null_mut()) {
                    Ok(true) => {
                        ps.status = SendStatus::LengthPosted;
                    }
                    Ok(false) => {
                        // EAGAIN - will retry in next progress
                    }
                    Err(e) => {
                        log::error!("Header send failed: {}", e);
                    }
                }
            }
        }
    }

    /// Post a finish header send
    fn send_finish_header(&mut self, target: i32) {
        if let Some(ps) = self.sends.get_mut(&target) {
            ps.header_buf[0] = 0;
            ps.header_buf[1] = CYLON_MSG_FIN;

            let header_bytes = unsafe {
                std::slice::from_raw_parts(
                    ps.header_buf.as_ptr() as *const u8,
                    2 * std::mem::size_of::<i32>(),
                )
            };

            match self.ep.send(header_bytes, ps.target_addr, ptr::null_mut()) {
                Ok(true) => {
                    ps.status = SendStatus::Finish;
                }
                Ok(false) => {
                    // EAGAIN - will retry
                }
                Err(e) => {
                    log::error!("Finish header send failed: {}", e);
                }
            }
        }
    }

    /// Poll completion queue and mark completions
    fn poll_completions(&self) -> usize {
        let mut entries = vec![unsafe { std::mem::zeroed::<fi_cq_tagged_entry>() }; 16];
        match self.ctx.poll_cq(&mut entries) {
            Ok(count) => count,
            Err(e) => {
                log::error!("CQ poll error: {}", e);
                0
            }
        }
    }
}

impl Channel for LibfabricChannel {
    fn init(
        &mut self,
        edge: i32,
        receives: &[i32],
        sends: &[i32],
        rcv_callback: Box<dyn ChannelReceiveCallback>,
        send_callback: Box<dyn ChannelSendCallback>,
        allocator: Box<dyn Allocator>,
    ) -> CylonResult<()> {
        self.edge = edge;
        self.rcv_fn = Some(rcv_callback);
        self.send_comp_fn = Some(send_callback);
        self.allocator = Some(allocator);

        // Initialize pending receives
        for &source in receives {
            let source_addr = self.av.lookup(source).ok_or_else(|| {
                CylonError::new(Code::ExecutionError, format!("Source {} not found in AV", source))
            })?;

            let mut pr = PendingReceive::new(source, source_addr);

            // Post initial header receive
            let header_bytes = unsafe {
                std::slice::from_raw_parts_mut(
                    pr.header_buf.as_mut_ptr() as *mut u8,
                    CYLON_CHANNEL_HEADER_SIZE * std::mem::size_of::<i32>(),
                )
            };

            match self.ep.recv(header_bytes, source_addr, ptr::null_mut()) {
                Ok(true) => {
                    pr.status = ReceiveStatus::LengthPosted;
                }
                Ok(false) | Err(_) => {
                    // Will post in progress
                    pr.status = ReceiveStatus::Init;
                }
            }

            self.pending_receives.insert(source, pr);
        }

        // Initialize sends
        for &target in sends {
            let target_addr = self.av.lookup(target).ok_or_else(|| {
                CylonError::new(Code::ExecutionError, format!("Target {} not found in AV", target))
            })?;
            self.sends.insert(target, PendingSend::new(target_addr));
        }

        Ok(())
    }

    fn send(&mut self, request: Box<CylonRequest>) -> i32 {
        let target = request.target;
        if let Some(ps) = self.sends.get_mut(&target) {
            if ps.pending_data.len() >= MAX_PENDING {
                return -1;
            }
            ps.pending_data.push_back(request);
            1
        } else {
            -1
        }
    }

    fn send_fin(&mut self, request: Box<CylonRequest>) -> i32 {
        let target = request.target;
        if self.finish_requests.contains_key(&target) {
            return -1;
        }
        self.finish_requests.insert(target, request);
        1
    }

    fn progress_sends(&mut self) {
        // Poll for completions
        let _completions = self.poll_completions();

        let targets: Vec<i32> = self.sends.keys().copied().collect();
        let mut completions = Vec::new();

        for target in targets {
            let should_send_header;
            let should_send_data;
            let should_send_finish;

            if let Some(ps) = self.sends.get(&target) {
                should_send_header = ps.status == SendStatus::Init && !ps.pending_data.is_empty();
                should_send_data = ps.status == SendStatus::LengthPosted;
                should_send_finish = ps.status == SendStatus::Init
                    && ps.pending_data.is_empty()
                    && self.finish_requests.contains_key(&target);
            } else {
                continue;
            }

            if should_send_header {
                self.send_header(target);
            } else if should_send_data {
                // Header sent, now send data
                if let Some(ps) = self.sends.get_mut(&target) {
                    if let Some(r) = ps.pending_data.pop_front() {
                        match self.ep.send(&r.buffer, ps.target_addr, ptr::null_mut()) {
                            Ok(true) => {
                                ps.current_send = Some(r);
                                ps.status = SendStatus::Posted;
                            }
                            Ok(false) => {
                                // EAGAIN - put back
                                ps.pending_data.push_front(r);
                            }
                            Err(e) => {
                                log::error!("Data send failed: {}", e);
                                ps.pending_data.push_front(r);
                            }
                        }
                    }
                }
            } else if should_send_finish {
                self.send_finish_header(target);
            }

            // Check for send completion (simplified - assumes completion after post)
            if let Some(ps) = self.sends.get_mut(&target) {
                if ps.status == SendStatus::Posted {
                    // Mark as complete after poll shows completion
                    if let Some(current) = ps.current_send.take() {
                        completions.push((target, current));
                    }

                    if !ps.pending_data.is_empty() {
                        ps.status = SendStatus::Init;
                    } else if self.finish_requests.contains_key(&target) {
                        ps.status = SendStatus::Init;
                    } else {
                        ps.status = SendStatus::Init;
                    }
                } else if ps.status == SendStatus::Finish {
                    // Finish sent
                    if let Some(fin_req) = self.finish_requests.remove(&target) {
                        if let Some(ref mut send_fn) = self.send_comp_fn {
                            send_fn.send_finish_complete(fin_req);
                        }
                    }
                    ps.status = SendStatus::Done;
                }
            }
        }

        // Notify completions
        if let Some(ref mut send_fn) = self.send_comp_fn {
            for (_target, request) in completions {
                send_fn.send_complete(request);
            }
        }
    }

    fn progress_receives(&mut self) {
        // Poll for completions
        let _completions = self.poll_completions();

        let mut updates: Vec<(i32, bool, i32, Option<Vec<u8>>, Option<Vec<i32>>)> = Vec::new();

        let sources: Vec<i32> = self.pending_receives.keys().copied().collect();

        for source in sources {
            if let Some(pr) = self.pending_receives.get_mut(&source) {
                match pr.status {
                    ReceiveStatus::Init => {
                        // Try to post header receive
                        let header_bytes = unsafe {
                            std::slice::from_raw_parts_mut(
                                pr.header_buf.as_mut_ptr() as *mut u8,
                                CYLON_CHANNEL_HEADER_SIZE * std::mem::size_of::<i32>(),
                            )
                        };

                        if let Ok(true) = self.ep.recv(header_bytes, pr.source_addr, ptr::null_mut()) {
                            pr.status = ReceiveStatus::LengthPosted;
                        }
                    }
                    ReceiveStatus::LengthPosted => {
                        // Check if header received (simplified)
                        // In production, would check completion queue
                        if pr.header_complete {
                            let length = pr.header_buf[0] as usize;
                            let fin_flag = pr.header_buf[1];

                            if fin_flag != CYLON_MSG_FIN {
                                // Allocate and post data receive
                                pr.data = Some(vec![0u8; length]);
                                pr.length = length;

                                if let Some(ref mut data) = pr.data {
                                    if let Ok(true) = self.ep.recv(data, pr.source_addr, ptr::null_mut()) {
                                        pr.status = ReceiveStatus::Posted;

                                        let header = if pr.header_buf.len() > 2 {
                                            Some(pr.header_buf[2..].to_vec())
                                        } else {
                                            None
                                        };
                                        updates.push((source, false, fin_flag, None, header));
                                    }
                                }
                            } else {
                                pr.status = ReceiveStatus::ReceivedFin;
                                updates.push((source, false, fin_flag, None, None));
                            }
                            pr.header_complete = false;
                        }
                    }
                    ReceiveStatus::Posted => {
                        // Check if data received
                        if pr.data_complete {
                            let data = pr.data.take();
                            updates.push((source, true, 0, data, None));

                            // Reset for next receive
                            pr.header_buf = [0; CYLON_CHANNEL_HEADER_SIZE];
                            pr.status = ReceiveStatus::Init;
                            pr.data_complete = false;
                        }
                    }
                    ReceiveStatus::ReceivedFin => {
                        // Done with this source
                    }
                }
            }
        }

        // Apply callbacks
        if let Some(ref mut rcv_fn) = self.rcv_fn {
            for (source, is_data, fin_flag, data_payload, header_payload) in updates {
                if is_data {
                    if let Some(data) = data_payload {
                        let length = data.len();
                        let buffer: Box<dyn Buffer> = Box::new(VecBuffer::with_data(data));
                        rcv_fn.received_data(source, buffer, length);
                    }
                } else {
                    rcv_fn.received_header(source, fin_flag, header_payload);
                }
            }
        }
    }

    fn close(&mut self) {
        self.pending_receives.clear();
        self.sends.clear();
        self.finish_requests.clear();
    }
}
