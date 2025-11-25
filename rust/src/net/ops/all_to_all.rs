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

//! All-to-all communication pattern
//!
//! Ported from cpp/src/cylon/net/ops/all_to_all.hpp and all_to_all.cpp

use std::collections::{HashSet, VecDeque};
use std::sync::Mutex;

use crate::error::CylonResult;
use crate::net::{Buffer, Channel, ChannelReceiveCallback, ChannelSendCallback};
use crate::net::request::CylonRequest;

/// Callback trait for receiving data in all-to-all operations
/// Corresponds to C++ ReceiveCallback from all_to_all.hpp
pub trait ReceiveCallback: Send + Sync {
    /// Called when data is received
    /// Returns true if we accept this buffer
    fn on_receive(&mut self, source: i32, buffer: Box<dyn Buffer>, length: usize) -> bool;

    /// Called when a header is received (before data)
    /// Returns true if we accept the header
    fn on_receive_header(&mut self, source: i32, finished: i32, header: Option<Vec<i32>>) -> bool;

    /// Called after a buffer is successfully sent
    /// Returns true if send was accepted
    fn on_send_complete(&mut self, target: i32, buffer: &[u8], length: usize) -> bool;
}

/// Send status for all-to-all operations
/// Corresponds to C++ AllToAllSendStatus from all_to_all.hpp
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AllToAllSendStatus {
    Sending = 0,
    FinishSent = 1,
    Finished = 2,
}

/// Tracks sends to a specific target
/// Corresponds to C++ AllToAllSends from all_to_all.hpp
struct AllToAllSends {
    target: i32,
    request_queue: VecDeque<Box<CylonRequest>>,
    pending_queue: VecDeque<Box<CylonRequest>>,
    message_sizes: usize,
    send_status: AllToAllSendStatus,
}

impl AllToAllSends {
    fn new(target: i32) -> Self {
        Self {
            target,
            request_queue: VecDeque::new(),
            pending_queue: VecDeque::new(),
            message_sizes: 0,
            send_status: AllToAllSendStatus::Sending,
        }
    }
}

/// All-to-all communication pattern
/// Corresponds to C++ AllToAll from cpp/src/cylon/net/ops/all_to_all.hpp
///
/// This class manages communication where each process sends/receives data
/// to/from multiple other processes. It:
/// - Queues messages to be sent
/// - Progresses channel operations
/// - Tracks completion state
/// - Forwards callbacks to user code
pub struct AllToAll {
    worker_id: i32,
    sources: Vec<i32>,
    targets: Vec<i32>,
    edge: i32,
    sends: Vec<AllToAllSends>,
    finished_sources: HashSet<i32>,
    finished_targets: HashSet<i32>,
    finish_flag: bool,
    channel: Box<dyn Channel>,
    callback: Box<dyn ReceiveCallback>,
    this_num_targets: usize,
    this_num_sources: usize,
    mutex: Mutex<()>,
}

// Wrapper types to implement channel callbacks using raw pointers to AllToAll
// This matches the C++ design where AllToAll passes `this` to channel->init()
struct AllToAllRecvWrapper(*mut AllToAll);
unsafe impl Send for AllToAllRecvWrapper {}
unsafe impl Sync for AllToAllRecvWrapper {}

struct AllToAllSendWrapper(*mut AllToAll);
unsafe impl Send for AllToAllSendWrapper {}
unsafe impl Sync for AllToAllSendWrapper {}

impl ChannelReceiveCallback for AllToAllRecvWrapper {
    fn received_data(&mut self, receive_id: i32, buffer: Box<dyn Buffer>, length: usize) {
        unsafe { (*self.0).received_data(receive_id, buffer, length) }
    }

    fn received_header(&mut self, receive_id: i32, finished: i32, header: Option<Vec<i32>>) {
        unsafe { (*self.0).received_header(receive_id, finished, header) }
    }
}

impl ChannelSendCallback for AllToAllSendWrapper {
    fn send_complete(&mut self, request: Box<CylonRequest>) {
        unsafe { (*self.0).send_complete(request) }
    }

    fn send_finish_complete(&mut self, request: Box<CylonRequest>) {
        unsafe { (*self.0).send_finish_complete(request) }
    }
}

impl AllToAll {
    /// Create a new AllToAll operation
    ///
    /// # Arguments
    /// * `worker_id` - This process's rank
    /// * `sources` - List of source ranks to receive from
    /// * `targets` - List of target ranks to send to
    /// * `edge_id` - Edge ID for this communication pattern
    /// * `channel` - The channel to use for communication
    /// * `callback` - User callback for receive/send events
    pub fn new(
        worker_id: i32,
        sources: Vec<i32>,
        targets: Vec<i32>,
        edge_id: i32,
        mut channel: Box<dyn Channel>,
        callback: Box<dyn ReceiveCallback>,
        allocator: Box<dyn crate::net::Allocator>,
    ) -> CylonResult<Box<Self>> {
        let edge = edge_id;

        // Initialize sends for each target
        let mut sends = Vec::new();
        for &t in &targets {
            let t_adjusted = (t + worker_id) % (targets.len() as i32);
            sends.push(AllToAllSends::new(t_adjusted));
        }

        // Check if this process is a target or source
        let this_num_targets = if targets.contains(&worker_id) { 1 } else { 0 };
        let this_num_sources = if sources.contains(&worker_id) { 1 } else { 0 };

        // Box first to get a stable address (matches C++ heap allocation)
        let mut all_to_all = Box::new(Self {
            worker_id,
            sources: sources.clone(),
            targets: targets.clone(),
            edge,
            sends,
            finished_sources: HashSet::new(),
            finished_targets: HashSet::new(),
            finish_flag: false,
            channel,
            callback,
            this_num_targets,
            this_num_sources,
            mutex: Mutex::new(()),
        });

        // Initialize channel with self as callbacks (matches C++ design)
        // C++: channel->init(edge_id, srcs, tgts, this, this, alloc);
        // Safety: Box guarantees stable address, raw pointer valid for lifetime of Box
        // The channel will only call these during the lifetime of AllToAll
        unsafe {
            let self_ptr = &mut *all_to_all as *mut Self;
            let recv_callback: Box<dyn ChannelReceiveCallback> = Box::new(AllToAllRecvWrapper(self_ptr));
            let send_callback: Box<dyn ChannelSendCallback> = Box::new(AllToAllSendWrapper(self_ptr));
            all_to_all.channel.init(
                edge_id,
                &sources,
                &targets,
                recv_callback,
                send_callback,
                allocator,
            )?;
        }

        Ok(all_to_all)
    }

    /// Insert a buffer to be sent to a target
    ///
    /// Returns 1 if accepted, -1 if rejected (e.g., after finish() called)
    pub fn insert(&mut self, buffer: Vec<u8>, target: i32) -> i32 {
        if self.finish_flag {
            return -1;
        }

        if target as usize >= self.sends.len() {
            return -1;
        }

        let s = &mut self.sends[target as usize];
        let length = buffer.len();
        let request = Box::new(CylonRequest::new(target, buffer));
        s.request_queue.push_back(request);
        s.message_sizes += length;
        1
    }

    /// Insert a buffer with header to be sent to a target
    ///
    /// Returns 1 if accepted, -1 if rejected
    pub fn insert_with_header(&mut self, buffer: Vec<u8>, target: i32, header: &[i32]) -> i32 {
        if self.finish_flag {
            return -1;
        }

        if header.len() > 6 {
            return -1;
        }

        if target as usize >= self.sends.len() {
            return -1;
        }

        let s = &mut self.sends[target as usize];
        let length = buffer.len();
        let request = Box::new(CylonRequest::new_with_header(target, buffer, header));
        s.request_queue.push_back(request);
        s.message_sizes += length;
        1
    }

    /// Check if the all-to-all operation is complete
    ///
    /// This should be called repeatedly in a loop until it returns true.
    /// It progresses the channel operations and checks completion state.
    pub fn is_complete(&mut self) -> bool {
        let mut all_queues_empty = true;

        // Send queued messages
        for s in &mut self.sends {
            while let Some(request) = s.request_queue.front() {
                if s.send_status == AllToAllSendStatus::FinishSent
                    || s.send_status == AllToAllSendStatus::Finished
                {
                    panic!("Cannot have items to send after finish sent");
                }

                // Try to send through channel
                let result = self.channel.send(Box::new(CylonRequest {
                    buffer: request.buffer.clone(),
                    target: request.target,
                    header: request.header,
                    header_length: request.header_length,
                }));

                if result == 1 {
                    // Request accepted, move to pending queue
                    let req = s.request_queue.pop_front().unwrap();
                    s.pending_queue.push_back(req);
                } else {
                    // Channel can't accept more right now
                    break;
                }
            }

            // Check if we should send finish message
            if s.request_queue.is_empty() && s.pending_queue.is_empty() {
                if self.finish_flag && s.send_status == AllToAllSendStatus::Sending {
                    let fin_request = Box::new(CylonRequest::new_finish(s.target));
                    if self.channel.send_fin(fin_request) == 1 {
                        s.send_status = AllToAllSendStatus::FinishSent;
                    }
                }
            } else {
                all_queues_empty = false;
            }
        }

        // Progress channel operations
        self.channel.progress_sends();
        self.channel.progress_receives();

        // Check if completely done
        let completed = all_queues_empty
            && self.finished_targets.len() == self.targets.len()
            && self.finished_sources.len() == self.sources.len();

        if completed {
            self.channel.notify_completed();
        }

        completed
    }

    /// Signal that no more inserts will be made
    ///
    /// After calling this, is_complete() will send finish messages
    /// once all queued data is sent.
    pub fn finish(&mut self) {
        self.finish_flag = true;
    }

    /// Close the all-to-all operation
    pub fn close(&mut self) {
        self.sends.clear();
        self.channel.close();
    }
}

impl ChannelReceiveCallback for AllToAll {
    fn received_data(&mut self, receive_id: i32, buffer: Box<dyn Buffer>, length: usize) {
        // Forward to user callback
        self.callback.on_receive(receive_id, buffer, length);
    }

    fn received_header(&mut self, receive_id: i32, finished: i32, header: Option<Vec<i32>>) {
        if finished != 0 {
            // Mark source as finished
            self.finished_sources.insert(receive_id);
        }

        // Forward to user callback
        self.callback.on_receive_header(receive_id, finished, header);
    }
}

impl ChannelSendCallback for AllToAll {
    fn send_complete(&mut self, request: Box<CylonRequest>) {
        let target = request.target;
        let length = request.len();
        let buffer = &request.buffer;

        if let Some(s) = self.sends.get_mut(target as usize) {
            // Remove from pending queue
            if !s.pending_queue.is_empty() {
                s.pending_queue.pop_front();
            }

            // Update message size tracking
            s.message_sizes = s.message_sizes.saturating_sub(length);

            // Forward to user callback
            self.callback.on_send_complete(target, buffer, length);
        }
    }

    fn send_finish_complete(&mut self, request: Box<CylonRequest>) {
        let _lock = self.mutex.lock().unwrap();

        let target = request.target;
        self.finished_targets.insert(target);

        if let Some(s) = self.sends.get_mut(target as usize) {
            s.send_status = AllToAllSendStatus::Finished;
        }
    }
}
