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

//! Cylon FMI Channel implementation
//!
//! This module corresponds to cpp/src/cylon/net/fmi/fmi_channel.hpp/cpp
//!
//! Provides the FMIChannel class that implements Cylon's Channel trait
//! with a progress-based send/receive model.

use std::collections::HashMap;
use std::sync::Arc;

use crate::error::{CylonError, CylonResult, Code};
use crate::net::{
    Channel, ChannelReceiveCallback, ChannelSendCallback, Allocator, Buffer,
    CYLON_CHANNEL_HEADER_SIZE, CYLON_MSG_FIN, CYLON_MSG_NOT_FIN, MAX_PENDING,
};
use crate::net::request::CylonRequest;

use super::common::{Mode, FmiContext};
use super::communicator::Communicator as FmiCommunicator;

#[cfg(feature = "redis")]
use super::fault_tolerance::HeartbeatWatcher;

/// FMI Send Status (matches cylon::fmi::FMISendStatus)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FMISendStatus {
    SendInit = 0,
    SendLengthPosted = 1,
    SendPosted = 2,
    SendFinish = 3,
    SendDone = 4,
}

/// FMI Receive Status (matches cylon::fmi::FMIReceiveStatus)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FMIReceiveStatus {
    ReceiveInit = 0,
    ReceiveLengthPosted = 1,
    ReceivePosted = 2,
    ReceivedFin = 3,
}

/// Pending send structure (matches cylon::fmi::PendingSend)
struct PendingSend {
    header_buf: [i32; CYLON_CHANNEL_HEADER_SIZE],
    pending_data: std::collections::VecDeque<Box<CylonRequest>>,
    status: FMISendStatus,
    current_send: Option<Box<CylonRequest>>,
    context: FmiContext,
}

impl PendingSend {
    fn new() -> Self {
        Self {
            header_buf: [0; CYLON_CHANNEL_HEADER_SIZE],
            pending_data: std::collections::VecDeque::new(),
            status: FMISendStatus::SendInit,
            current_send: None,
            context: FmiContext::new(),
        }
    }
}

/// Pending receive structure (matches cylon::fmi::PendingReceive)
struct PendingReceive {
    header_buf: [i32; CYLON_CHANNEL_HEADER_SIZE],
    receive_id: i32,
    data: Option<Box<dyn Buffer>>,
    length: usize,
    status: FMIReceiveStatus,
    context: FmiContext,
}

impl PendingReceive {
    fn new(receive_id: i32) -> Self {
        Self {
            header_buf: [0; CYLON_CHANNEL_HEADER_SIZE],
            receive_id,
            data: None,
            length: 0,
            status: FMIReceiveStatus::ReceiveInit,
            context: FmiContext::new(),
        }
    }
}

/// FMI Channel for Cylon (matches cylon::fmi::FMIChannel)
///
/// This struct implements Cylon's Channel trait using the FMI communication layer.
/// It uses a progress-based model with explicit send/receive tracking.
///
/// # Fault Tolerance
///
/// Optionally integrates with [`HeartbeatWatcher`] to detect peer failures during
/// progress operations. When a watcher is set and detects a dead peer, progress
/// methods will notice on the next call (the watcher sets an atomic flag that is
/// checked instantly without blocking I/O).
pub struct FMICylonChannel {
    sends: HashMap<i32, PendingSend>,
    pending_receives: HashMap<i32, PendingReceive>,
    finish_requests: HashMap<i32, Box<CylonRequest>>,
    rcv_fn: Option<Box<dyn ChannelReceiveCallback>>,
    send_comp_fn: Option<Box<dyn ChannelSendCallback>>,
    allocator: Option<Box<dyn Allocator>>,
    rank: i32,
    world_size: i32,
    communicator: Arc<FmiCommunicator>,
    mode: Mode,
    redis_host: String,
    redis_port: i32,
    redis_namespace: String,
    #[cfg(feature = "redis")]
    redis: Option<redis::Client>,
    /// Optional heartbeat watcher for fault detection during progress
    #[cfg(feature = "redis")]
    heartbeat_watcher: Option<Arc<HeartbeatWatcher>>,
    /// Flag indicating if a peer failure was detected
    peer_failure_detected: bool,
}

impl FMICylonChannel {
    /// Create a new FMI Channel
    ///
    /// Matches C++ constructor:
    /// FMIChannel(std::shared_ptr<FMI::Communicator> com, FMI::Utils::Mode mode,
    ///            std::string redis_host, int redis_port, std::string redis_namespace)
    pub fn new(
        communicator: Arc<FmiCommunicator>,
        mode: Mode,
        redis_host: &str,
        redis_port: i32,
        redis_namespace: &str,
    ) -> Self {
        let rank = communicator.get_peer_id();
        let world_size = communicator.get_num_peers();

        Self {
            sends: HashMap::new(),
            pending_receives: HashMap::new(),
            finish_requests: HashMap::new(),
            rcv_fn: None,
            send_comp_fn: None,
            allocator: None,
            rank,
            world_size,
            communicator,
            mode,
            redis_host: redis_host.to_string(),
            redis_port,
            redis_namespace: redis_namespace.to_string(),
            #[cfg(feature = "redis")]
            redis: None,
            #[cfg(feature = "redis")]
            heartbeat_watcher: None,
            peer_failure_detected: false,
        }
    }

    /// Set the heartbeat watcher for fault detection
    ///
    /// When set, the channel will check for peer failures during progress
    /// operations. This is a non-blocking check (just reads an atomic flag).
    #[cfg(feature = "redis")]
    pub fn set_heartbeat_watcher(&mut self, watcher: Arc<HeartbeatWatcher>) {
        self.heartbeat_watcher = Some(watcher);
    }

    /// Check if a peer failure has been detected
    pub fn has_peer_failure(&self) -> bool {
        self.peer_failure_detected
    }

    /// Reset the peer failure flag
    pub fn reset_peer_failure(&mut self) {
        self.peer_failure_detected = false;
    }

    /// Check heartbeat watcher for failures (instant, no I/O)
    #[cfg(feature = "redis")]
    #[inline]
    fn check_heartbeat(&mut self) -> bool {
        if let Some(ref watcher) = self.heartbeat_watcher {
            if watcher.has_peer_failed() || watcher.is_abort_signaled() {
                self.peer_failure_detected = true;
                return true;
            }
        }
        false
    }

    /// Check heartbeat watcher for failures (non-redis stub)
    #[cfg(not(feature = "redis"))]
    #[inline]
    fn check_heartbeat(&mut self) -> bool {
        false
    }

    /// FMI receive (matches FMI_Irecv template)
    /// Uses communicator reference to avoid borrow conflicts
    fn fmi_irecv_with_comm(
        communicator: &FmiCommunicator,
        buf: &mut [u8],
        sender: i32,
        ctx: &mut FmiContext
    ) -> CylonResult<()> {
        ctx.completed = 0;

        // Receive directly into the buffer
        communicator.recv(buf, sender)?;
        ctx.mark_completed();

        Ok(())
    }

    /// FMI send (matches FMI_Isend template)
    fn fmi_isend(&self, buf: &[u8], dest: i32, ctx: &mut FmiContext) -> CylonResult<()> {
        ctx.completed = 0;
        self.communicator.send(buf, dest)?;
        ctx.mark_completed();
        Ok(())
    }

    /// Send header for a request
    fn send_header(&mut self, target: i32) -> CylonResult<()> {
        let ps = self.sends.get_mut(&target).ok_or_else(|| {
            CylonError::new(Code::Invalid, format!("No pending send for target {}", target))
        })?;

        if let Some(req) = ps.pending_data.front() {
            ps.header_buf[0] = req.len() as i32;
            ps.header_buf[1] = CYLON_MSG_NOT_FIN;

            // Copy header from request
            if req.header_length > 0 {
                for i in 0..req.header_length.min(6) {
                    ps.header_buf[2 + i] = req.header[i];
                }
            }

            ps.context = FmiContext::new();
            ps.context.completed = 0;

            let header_bytes = unsafe {
                std::slice::from_raw_parts(
                    ps.header_buf.as_ptr() as *const u8,
                    (2 + req.header_length) * std::mem::size_of::<i32>(),
                )
            };

            self.communicator.send(header_bytes, target)?;
            ps.status = FMISendStatus::SendLengthPosted;
            ps.context.mark_completed();
        }

        Ok(())
    }

    /// Send finish header
    fn send_finish_header(&mut self, target: i32) -> CylonResult<()> {
        let ps = self.sends.get_mut(&target).ok_or_else(|| {
            CylonError::new(Code::Invalid, format!("No pending send for target {}", target))
        })?;

        ps.header_buf[0] = 0;
        ps.header_buf[1] = CYLON_MSG_FIN;

        ps.context = FmiContext::new();
        ps.context.completed = 0;

        let header_bytes = unsafe {
            std::slice::from_raw_parts(
                ps.header_buf.as_ptr() as *const u8,
                CYLON_CHANNEL_HEADER_SIZE * std::mem::size_of::<i32>(),
            )
        };

        self.communicator.send(header_bytes, target)?;
        ps.status = FMISendStatus::SendFinish;
        ps.context.mark_completed();

        Ok(())
    }

    /// Progress sends for blocking mode
    fn progress_send_to(&mut self, peer_id: i32) {
        if peer_id == self.rank {
            self.progress_sends_local(peer_id);
            return;
        }

        let status = {
            let ps = match self.sends.get(&peer_id) {
                Some(ps) => ps,
                None => return,
            };
            ps.status
        };

        if status == FMISendStatus::SendDone {
            return;
        }

        match status {
            FMISendStatus::SendInit => {
                let has_pending = self.sends.get(&peer_id)
                    .map(|ps| !ps.pending_data.is_empty())
                    .unwrap_or(false);
                let has_finish = self.finish_requests.contains_key(&peer_id);

                if has_pending {
                    let _ = self.send_header(peer_id);
                } else if has_finish {
                    let _ = self.send_finish_header(peer_id);
                }
            }
            FMISendStatus::SendLengthPosted => {
                let completed = self.sends.get(&peer_id)
                    .map(|ps| ps.context.is_completed())
                    .unwrap_or(false);

                if completed {
                    // Send the actual data
                    let (buffer, length, target) = {
                        let ps = self.sends.get_mut(&peer_id).unwrap();
                        ps.context = FmiContext::new();
                        ps.context.completed = 0;

                        if let Some(req) = ps.pending_data.pop_front() {
                            let buffer = req.buffer.clone();
                            let target = req.target;
                            let length = req.len();
                            ps.current_send = Some(req);
                            ps.status = FMISendStatus::SendPosted;
                            (Some(buffer), length, target)
                        } else {
                            (None, 0, 0)
                        }
                    };

                    if let Some(buf) = buffer {
                        let _ = self.communicator.send(&buf, target);
                        if let Some(ps) = self.sends.get_mut(&peer_id) {
                            ps.context.mark_completed();
                        }
                    }
                }
            }
            FMISendStatus::SendPosted => {
                let completed = self.sends.get(&peer_id)
                    .map(|ps| ps.context.is_completed())
                    .unwrap_or(false);

                if completed {
                    // Notify send completion
                    let current_send = {
                        let ps = self.sends.get_mut(&peer_id).unwrap();
                        ps.current_send.take()
                    };

                    if let Some(req) = current_send {
                        if let Some(ref mut callback) = self.send_comp_fn {
                            callback.send_complete(req);
                        }
                    }

                    let has_more = self.sends.get(&peer_id)
                        .map(|ps| !ps.pending_data.is_empty())
                        .unwrap_or(false);
                    let has_finish = self.finish_requests.contains_key(&peer_id);

                    if has_more {
                        let _ = self.send_header(peer_id);
                    } else if has_finish {
                        let _ = self.send_finish_header(peer_id);
                    } else {
                        if let Some(ps) = self.sends.get_mut(&peer_id) {
                            ps.status = FMISendStatus::SendInit;
                        }
                    }
                }
            }
            FMISendStatus::SendFinish => {
                let completed = self.sends.get(&peer_id)
                    .map(|ps| ps.context.is_completed())
                    .unwrap_or(false);

                if completed {
                    if let Some(fin_req) = self.finish_requests.remove(&peer_id) {
                        if let Some(ref mut callback) = self.send_comp_fn {
                            callback.send_finish_complete(fin_req);
                        }
                    }
                    if let Some(ps) = self.sends.get_mut(&peer_id) {
                        ps.status = FMISendStatus::SendDone;
                    }
                }
            }
            FMISendStatus::SendDone => {}
        }
    }

    /// Progress sends locally (self-send)
    fn progress_sends_local(&mut self, peer_id: i32) {
        let status = {
            let ps = match self.sends.get(&peer_id) {
                Some(ps) => ps,
                None => return,
            };
            ps.status
        };

        match status {
            FMISendStatus::SendInit => {
                let has_pending = self.sends.get(&peer_id)
                    .map(|ps| !ps.pending_data.is_empty())
                    .unwrap_or(false);
                let has_finish = self.finish_requests.contains_key(&peer_id);

                if has_pending {
                    // Send header locally
                    let header_info = {
                        let ps = self.sends.get(&peer_id).unwrap();
                        if let Some(req) = ps.pending_data.front() {
                            Some((CYLON_MSG_NOT_FIN, req.header.clone(), req.header_length))
                        } else {
                            None
                        }
                    };

                    if let Some((fin_flag, header, header_length)) = header_info {
                        if let Some(ref mut callback) = self.rcv_fn {
                            callback.received_header(self.rank, fin_flag, Some(header.to_vec()));
                        }
                    }

                    if let Some(ps) = self.sends.get_mut(&peer_id) {
                        ps.status = FMISendStatus::SendLengthPosted;
                    }
                } else if has_finish {
                    if let Some(ref mut callback) = self.rcv_fn {
                        callback.received_header(self.rank, CYLON_MSG_FIN, None);
                    }
                    if let Some(ps) = self.sends.get_mut(&peer_id) {
                        ps.status = FMISendStatus::SendFinish;
                    }
                }
            }
            FMISendStatus::SendLengthPosted => {
                // Copy data locally
                let data_info = {
                    let ps = self.sends.get_mut(&peer_id).unwrap();
                    if let Some(req) = ps.pending_data.pop_front() {
                        let buffer = req.buffer.clone();
                        let length = req.len();
                        ps.current_send = Some(req);
                        ps.status = FMISendStatus::SendPosted;
                        Some((buffer, length))
                    } else {
                        None
                    }
                };

                if let Some((buffer, length)) = data_info {
                    // Allocate buffer and copy data
                    if let Some(ref mut alloc) = self.allocator {
                        if let Ok(mut data_buf) = alloc.allocate(length) {
                            data_buf.get_byte_buffer_mut().copy_from_slice(&buffer);
                            if let Some(ref mut callback) = self.rcv_fn {
                                callback.received_data(self.rank, data_buf, length);
                            }
                        }
                    }
                }
            }
            FMISendStatus::SendPosted => {
                let current_send = {
                    let ps = self.sends.get_mut(&peer_id).unwrap();
                    ps.current_send.take()
                };

                if let Some(req) = current_send {
                    if let Some(ref mut callback) = self.send_comp_fn {
                        callback.send_complete(req);
                    }
                }

                let has_more = self.sends.get(&peer_id)
                    .map(|ps| !ps.pending_data.is_empty())
                    .unwrap_or(false);
                let has_finish = self.finish_requests.contains_key(&peer_id);

                if has_more {
                    // Continue with next send
                    let header_info = {
                        let ps = self.sends.get(&peer_id).unwrap();
                        if let Some(req) = ps.pending_data.front() {
                            Some((CYLON_MSG_NOT_FIN, req.header.clone(), req.header_length))
                        } else {
                            None
                        }
                    };

                    if let Some((fin_flag, header, _)) = header_info {
                        if let Some(ref mut callback) = self.rcv_fn {
                            callback.received_header(self.rank, fin_flag, Some(header.to_vec()));
                        }
                    }

                    if let Some(ps) = self.sends.get_mut(&peer_id) {
                        ps.status = FMISendStatus::SendLengthPosted;
                    }
                } else if has_finish {
                    if let Some(ref mut callback) = self.rcv_fn {
                        callback.received_header(self.rank, CYLON_MSG_FIN, None);
                    }
                    if let Some(ps) = self.sends.get_mut(&peer_id) {
                        ps.status = FMISendStatus::SendFinish;
                    }
                } else {
                    if let Some(ps) = self.sends.get_mut(&peer_id) {
                        ps.status = FMISendStatus::SendInit;
                    }
                }
            }
            FMISendStatus::SendFinish => {
                if let Some(fin_req) = self.finish_requests.remove(&peer_id) {
                    if let Some(ref mut callback) = self.send_comp_fn {
                        callback.send_finish_complete(fin_req);
                    }
                }
                if let Some(ps) = self.sends.get_mut(&peer_id) {
                    ps.status = FMISendStatus::SendDone;
                }
            }
            FMISendStatus::SendDone => {}
        }
    }

    /// Progress receives from a peer
    fn progress_receive_from(&mut self, peer_id: i32) {
        if peer_id == self.rank {
            return;
        }

        let status = {
            let pr = match self.pending_receives.get(&peer_id) {
                Some(pr) => pr,
                None => return,
            };
            pr.status
        };

        if status == FMIReceiveStatus::ReceivedFin {
            return;
        }

        match status {
            FMIReceiveStatus::ReceiveInit => {
                // Post header receive
                let header_bytes = {
                    let pr = self.pending_receives.get_mut(&peer_id).unwrap();
                    pr.context = FmiContext::new();
                    pr.context.completed = 0;

                    unsafe {
                        std::slice::from_raw_parts_mut(
                            pr.header_buf.as_mut_ptr() as *mut u8,
                            CYLON_CHANNEL_HEADER_SIZE * std::mem::size_of::<i32>(),
                        )
                    }.to_vec()
                };

                let mut recv_buf = header_bytes;
                let mut ctx = FmiContext::new();
                if let Err(e) = Self::fmi_irecv_with_comm(&self.communicator, &mut recv_buf, peer_id, &mut ctx) {
                    log::error!("Failed to receive header from {}: {:?}", peer_id, e);
                    return;
                }

                // Copy back received header
                if let Some(pr) = self.pending_receives.get_mut(&peer_id) {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            recv_buf.as_ptr() as *const i32,
                            pr.header_buf.as_mut_ptr(),
                            CYLON_CHANNEL_HEADER_SIZE,
                        );
                    }
                    pr.status = FMIReceiveStatus::ReceiveLengthPosted;
                    pr.context.mark_completed();
                }
            }
            FMIReceiveStatus::ReceiveLengthPosted => {
                let completed = self.pending_receives.get(&peer_id)
                    .map(|pr| pr.context.is_completed())
                    .unwrap_or(false);

                if completed {
                    let (length, fin_flag) = {
                        let pr = self.pending_receives.get(&peer_id).unwrap();
                        (pr.header_buf[0] as usize, pr.header_buf[1])
                    };

                    if fin_flag == CYLON_MSG_FIN {
                        if let Some(pr) = self.pending_receives.get_mut(&peer_id) {
                            pr.status = FMIReceiveStatus::ReceivedFin;
                        }
                        if let Some(ref mut callback) = self.rcv_fn {
                            callback.received_header(peer_id, fin_flag, None);
                        }
                        return;
                    }

                    // Allocate buffer and receive data
                    if let Some(ref mut alloc) = self.allocator {
                        if let Ok(data_buf) = alloc.allocate(length) {
                            // Notify header received
                            let header = {
                                let pr = self.pending_receives.get(&peer_id).unwrap();
                                let mut h = vec![0i32; 6];
                                h.copy_from_slice(&pr.header_buf[2..8]);
                                h
                            };

                            if let Some(ref mut callback) = self.rcv_fn {
                                callback.received_header(peer_id, fin_flag, Some(header));
                            }

                            if let Some(pr) = self.pending_receives.get_mut(&peer_id) {
                                pr.data = Some(data_buf);
                                pr.length = length;
                                pr.context = FmiContext::new();
                                pr.context.completed = 0;
                                pr.status = FMIReceiveStatus::ReceivePosted;

                                // Receive the data - use static method to avoid borrow conflict
                                if let Some(ref mut data) = pr.data {
                                    let recv_slice = data.get_byte_buffer_mut();
                                    if let Err(e) = Self::fmi_irecv_with_comm(&self.communicator, recv_slice, peer_id, &mut pr.context) {
                                        log::error!("Failed to receive data from {}: {:?}", peer_id, e);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            FMIReceiveStatus::ReceivePosted => {
                let completed = self.pending_receives.get(&peer_id)
                    .map(|pr| pr.context.is_completed())
                    .unwrap_or(false);

                if completed {
                    let (data, length) = {
                        let pr = self.pending_receives.get_mut(&peer_id).unwrap();
                        (pr.data.take(), pr.length)
                    };

                    if let Some(data_buf) = data {
                        if let Some(ref mut callback) = self.rcv_fn {
                            callback.received_data(peer_id, data_buf, length);
                        }
                    }

                    // Reset for next receive and post next header receive
                    if let Some(pr) = self.pending_receives.get_mut(&peer_id) {
                        pr.header_buf = [0; CYLON_CHANNEL_HEADER_SIZE];
                        pr.context = FmiContext::new();
                        pr.context.completed = 0;
                        pr.status = FMIReceiveStatus::ReceiveLengthPosted;

                        // Post next header receive - use static method to avoid borrow conflict
                        let header_bytes = unsafe {
                            std::slice::from_raw_parts_mut(
                                pr.header_buf.as_mut_ptr() as *mut u8,
                                CYLON_CHANNEL_HEADER_SIZE * std::mem::size_of::<i32>(),
                            )
                        };

                        if let Err(e) = Self::fmi_irecv_with_comm(&self.communicator, header_bytes, peer_id, &mut pr.context) {
                            log::error!("Failed to receive next header from {}: {:?}", peer_id, e);
                        }
                    }
                }
            }
            FMIReceiveStatus::ReceivedFin => {}
        }
    }
}

impl Channel for FMICylonChannel {
    fn init(
        &mut self,
        _edge: i32,
        receives: &[i32],
        sends: &[i32],
        rcv_callback: Box<dyn ChannelReceiveCallback>,
        send_callback: Box<dyn ChannelSendCallback>,
        allocator: Box<dyn Allocator>,
    ) -> CylonResult<()> {
        self.rcv_fn = Some(rcv_callback);
        self.send_comp_fn = Some(send_callback);
        self.allocator = Some(allocator);

        // Initialize pending receives
        for &recv_rank in receives {
            if recv_rank == self.rank {
                continue;
            }
            let mut pr = PendingReceive::new(recv_rank);
            pr.context.completed = if self.mode == Mode::Blocking { 1 } else { 0 };
            pr.status = if self.mode == Mode::Blocking {
                FMIReceiveStatus::ReceiveInit
            } else {
                FMIReceiveStatus::ReceiveLengthPosted
            };
            self.pending_receives.insert(recv_rank, pr);
        }

        // Initialize sends
        for &target in sends {
            self.sends.insert(target, PendingSend::new());
        }

        // Initialize Redis if configured
        #[cfg(feature = "redis")]
        {
            if self.redis_port > 0 && !self.redis_host.is_empty() {
                self.redis = redis::Client::open(
                    format!("redis://{}:{}", self.redis_host, self.redis_port)
                ).ok();
            }
        }

        Ok(())
    }

    fn send(&mut self, request: Box<CylonRequest>) -> i32 {
        let target = request.target;
        if let Some(ps) = self.sends.get_mut(&target) {
            if ps.pending_data.len() > MAX_PENDING {
                return -1;
            }
            ps.pending_data.push_back(request);
            return 1;
        }
        -1
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
        // Check heartbeat at start of progress loop (instant, no I/O)
        if self.check_heartbeat() {
            log::warn!("Peer failure detected during progress_sends");
            return;
        }

        let peers: Vec<i32> = self.sends.keys().copied().collect();
        for peer_id in peers {
            self.progress_send_to(peer_id);
            if self.mode == Mode::Blocking && peer_id != self.rank {
                self.progress_receive_from(peer_id);
            }

            // Check heartbeat after each peer operation
            if self.check_heartbeat() {
                log::warn!("Peer failure detected during progress_sends to peer {}", peer_id);
                return;
            }
        }
    }

    fn progress_receives(&mut self) {
        // Check heartbeat at start of progress loop (instant, no I/O)
        if self.check_heartbeat() {
            log::warn!("Peer failure detected during progress_receives");
            return;
        }

        if self.mode == Mode::NonBlocking {
            let peers: Vec<i32> = self.pending_receives.keys().copied().collect();
            for peer_id in peers {
                self.progress_receive_from(peer_id);

                // Check heartbeat after each peer operation
                if self.check_heartbeat() {
                    log::warn!("Peer failure detected during progress_receives from peer {}", peer_id);
                    return;
                }
            }
        }
    }

    fn notify_completed(&mut self) {
        // Cleanup operations
    }

    fn close(&mut self) {
        self.pending_receives.clear();
        self.sends.clear();
    }

    /// Check if a peer connection is alive
    ///
    /// Combines heartbeat watcher check with underlying communicator check.
    fn is_peer_alive(&self, peer_rank: i32) -> bool {
        // Check heartbeat watcher first (instant, no I/O)
        #[cfg(feature = "redis")]
        if let Some(ref watcher) = self.heartbeat_watcher {
            if watcher.has_peer_failed() {
                // Check if this specific peer is in the dead list
                let dead = watcher.get_dead_peers();
                let peer_id = format!("worker-{}", peer_rank);
                if dead.iter().any(|d| d == &peer_id || d.contains(&peer_rank.to_string())) {
                    return false;
                }
            }
        }

        // If peer_failure_detected flag is set, return false
        if self.peer_failure_detected {
            return false;
        }

        // Fall back to communicator check
        true
    }
}
