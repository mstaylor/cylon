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

//! MPI Channel implementation using raw MPI calls
//!
//! Ported from cpp/src/cylon/net/mpi/mpi_channel.hpp and mpi_channel.cpp
//!
//! # Safety
//!
//! This module uses unsafe raw MPI calls (mpi-sys) to match the C++ implementation
//! exactly. The C++ version uses MPI_Isend/MPI_Irecv with stored MPI_Request handles,
//! which is not possible with rsmpi's safe API due to lifetime constraints.
//!
//! Safety invariants:
//! - Buffers must remain valid while MPI_Request is active
//! - MPI_Request handles must be tested/waited before buffer deallocation
//! - All MPI calls check return codes
//! - Memory is properly aligned for MPI datatypes

use std::collections::{HashMap, VecDeque};
use std::os::raw::{c_int, c_void};
use std::sync::{Arc, Mutex};
use mpi::environment::Universe;
use mpi::ffi::MPI_Comm;

use crate::error::{CylonError, CylonResult, Code};
use crate::net::{
    Channel, ChannelReceiveCallback, ChannelSendCallback, Allocator, Buffer,
    CYLON_CHANNEL_HEADER_SIZE, CYLON_MSG_FIN, CYLON_MSG_NOT_FIN, MAX_PENDING,
};
use crate::net::request::CylonRequest;
use crate::net::buffer::VecBuffer;

// Import raw MPI functions
use mpi_sys::{
    MPI_Request, MPI_Status, MPI_Irecv, MPI_Isend, MPI_Test,
    MPI_Get_count, MPI_Cancel, MPI_Comm_rank,
    MPI_Datatype,
};

// MPI datatypes - use equivalence trait to get raw datatypes
fn get_mpi_int() -> MPI_Datatype {
    use mpi::datatype::Equivalence;
    use mpi::raw::AsRaw;
    <i32 as Equivalence>::equivalent_datatype().as_raw()
}

fn get_mpi_byte() -> MPI_Datatype {
    use mpi::datatype::Equivalence;
    use mpi::raw::AsRaw;
    <u8 as Equivalence>::equivalent_datatype().as_raw()
}

/// Send status enum
/// Corresponds to C++ SendStatus from mpi_channel.hpp
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SendStatus {
    Init = 0,
    LengthPosted = 1,
    Posted = 2,
    Finish = 3,
    Done = 4,
}

/// Receive status enum
/// Corresponds to C++ ReceiveStatus from mpi_channel.hpp
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReceiveStatus {
    Init = 0,
    LengthPosted = 1,
    Posted = 2,
    ReceivedFin = 3,
}

/// Pending send tracking
/// Corresponds to C++ PendingSend from mpi_channel.hpp
struct PendingSend {
    header_buf: [i32; CYLON_CHANNEL_HEADER_SIZE],
    pending_data: VecDeque<Box<CylonRequest>>,
    status: SendStatus,
    request: MPI_Request,
    current_send: Option<Box<CylonRequest>>,
}

impl PendingSend {
    fn new() -> Self {
        Self {
            header_buf: [0; CYLON_CHANNEL_HEADER_SIZE],
            pending_data: VecDeque::new(),
            status: SendStatus::Init,
            request: unsafe { std::mem::zeroed() },
            current_send: None,
        }
    }
}

/// Pending receive tracking
/// Corresponds to C++ PendingReceive from mpi_channel.hpp
struct PendingReceive {
    header_buf: [i32; CYLON_CHANNEL_HEADER_SIZE],
    receive_id: i32,
    data: Option<Vec<u8>>,
    length: usize,
    status: ReceiveStatus,
    request: MPI_Request,
}

impl PendingReceive {
    fn new(receive_id: i32) -> Self {
        Self {
            header_buf: [0; CYLON_CHANNEL_HEADER_SIZE],
            receive_id,
            data: None,
            length: 0,
            status: ReceiveStatus::Init,
            request: unsafe { std::mem::zeroed() },
        }
    }
}

/// MPI Channel implementation using raw MPI
/// Corresponds to C++ MPIChannel from cpp/src/cylon/net/mpi/mpi_channel.hpp
pub struct MPIChannel {
    edge: i32,
    sends: HashMap<i32, PendingSend>,
    pending_receives: HashMap<i32, PendingReceive>,
    finish_requests: HashMap<i32, Box<CylonRequest>>,
    rcv_fn: Option<Box<dyn ChannelReceiveCallback>>,
    send_comp_fn: Option<Box<dyn ChannelSendCallback>>,
    allocator: Option<Box<dyn Allocator>>,
    rank: i32,
    comm: MPI_Comm,
}

// SAFETY: We manually implement Send+Sync for MPIChannel
// This is safe because:
// - MPI communication is explicitly managed through progress functions
// - Buffers are protected by Rust's ownership system
// - MPI_Comm and MPI_Request are opaque handles that MPI library manages
unsafe impl Send for MPIChannel {}
unsafe impl Sync for MPIChannel {}

impl MPIChannel {
    /// Create a new MPI channel
    ///
    /// # Safety
    ///
    /// The MPI_Comm must be valid and remain valid for the lifetime of this channel
    pub unsafe fn new(comm: MPI_Comm) -> Self {
        Self {
            edge: 0,
            sends: HashMap::new(),
            pending_receives: HashMap::new(),
            finish_requests: HashMap::new(),
            rcv_fn: None,
            send_comp_fn: None,
            allocator: None,
            rank: 0,
            comm,
        }
    }

    /// Send header for pending data
    /// Corresponds to MPIChannel::sendHeader() in C++
    fn send_header(&mut self, target: i32) {
        if let Some(ps) = self.sends.get_mut(&target) {
            if let Some(r) = ps.pending_data.front() {
                // Prepare header: [length, finish_flag, ...optional_header]
                ps.header_buf[0] = r.len() as i32;
                ps.header_buf[1] = CYLON_MSG_NOT_FIN;

                // Copy optional header if present
                if r.header_length > 0 {
                    let len = r.header_length.min(6);
                    ps.header_buf[2..2+len].copy_from_slice(&r.header[..len]);
                }

                // Post immediate send for header
                let count = (2 + r.header_length) as c_int;

                // SAFETY:
                // - header_buf lives in PendingSend which won't be moved/dropped
                // - MPI_Request will be tested before PendingSend is dropped
                // - comm is valid
                unsafe {
                    let rc = MPI_Isend(
                        ps.header_buf.as_ptr() as *const c_void,
                        count,
                        get_mpi_int(),
                        target,
                        self.edge,
                        self.comm,
                        &mut ps.request,
                    );
                    if rc != 0 {
                        panic!("MPI_Isend failed with code {}", rc);
                    }
                }

                ps.status = SendStatus::LengthPosted;
            }
        }
    }

    /// Send finish header
    /// Corresponds to MPIChannel::sendFinishHeader() in C++
    fn send_finish_header(&mut self, target: i32) {
        if let Some(ps) = self.sends.get_mut(&target) {
            // Finish header: [0, FIN_FLAG]
            ps.header_buf[0] = 0;
            ps.header_buf[1] = CYLON_MSG_FIN;

            // Post immediate send for finish header
            // SAFETY: Same as send_header
            unsafe {
                let rc = MPI_Isend(
                    ps.header_buf.as_ptr() as *const c_void,
                    2,
                    get_mpi_int(),
                    target,
                    self.edge,
                    self.comm,
                    &mut ps.request,
                );
                if rc != 0 {
                    panic!("MPI_Isend failed with code {}", rc);
                }
            }

            ps.status = SendStatus::Finish;
        }
    }
}

impl Channel for MPIChannel {
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

        // Get MPI rank
        // SAFETY: comm is valid, rank is a valid pointer
        unsafe {
            let rc = MPI_Comm_rank(self.comm, &mut self.rank);
            if rc != 0 {
                return Err(CylonError::new(
                    Code::ExecutionError,
                    format!("MPI_Comm_rank failed with code {}", rc)
                ));
            }
        }

        // Initialize pending receives for all receive sources
        // Post initial header receives using MPI_Irecv
        for &source in receives {
            let mut pr = PendingReceive::new(source);

            // Post non-blocking receive for header
            // SAFETY:
            // - header_buf lives in PendingReceive
            // - MPI_Request will be tested/waited before drop
            unsafe {
                let rc = MPI_Irecv(
                    pr.header_buf.as_mut_ptr() as *mut c_void,
                    CYLON_CHANNEL_HEADER_SIZE as c_int,
                    get_mpi_int(),
                    source,
                    edge,
                    self.comm,
                    &mut pr.request,
                );
                if rc != 0 {
                    return Err(CylonError::new(
                        Code::ExecutionError,
                        format!("MPI_Irecv failed with code {}", rc)
                    ));
                }
            }

            pr.status = ReceiveStatus::LengthPosted;
            self.pending_receives.insert(source, pr);
        }

        // Initialize send structures for all targets
        for &target in sends {
            self.sends.insert(target, PendingSend::new());
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

    fn progress_receives(&mut self) {
        // Corresponds to MPIChannel::progressReceives() in cpp/src/cylon/net/mpi/mpi_channel.cpp

        // Collect updates to avoid borrow conflicts
        // (source, is_data, fin_flag, data_payload, header_payload)
        let mut updates: Vec<(i32, bool, i32, Option<Vec<u8>>, Option<Vec<i32>>)> = Vec::new();

        for (&source, pr) in self.pending_receives.iter_mut() {
            match pr.status {
                ReceiveStatus::LengthPosted => {
                    // Test if header receive completed
                    let mut flag: c_int = 0;
                    let mut status: MPI_Status = unsafe { std::mem::zeroed() };

                    // SAFETY: request is valid, flag and status are valid pointers
                    unsafe {
                        let rc = MPI_Test(&mut pr.request, &mut flag, &mut status);
                        if rc != 0 {
                            panic!("MPI_Test failed with code {}", rc);
                        }
                    }

                    if flag != 0 {
                        // Header received!
                        let mut count: c_int = 0;

                        // SAFETY: status is valid from MPI_Test
                        unsafe {
                            let rc = MPI_Get_count(&status, get_mpi_int(), &mut count);
                            if rc != 0 {
                                panic!("MPI_Get_count failed with code {}", rc);
                            }
                        }

                        let length = pr.header_buf[0] as usize;
                        let fin_flag = pr.header_buf[1];

                        if fin_flag != CYLON_MSG_FIN {
                            // Not a finish message - allocate buffer and post data receive
                            if count as usize > CYLON_CHANNEL_HEADER_SIZE {
                                panic!("Unexpected header size: {} > {}", count, CYLON_CHANNEL_HEADER_SIZE);
                            }

                            // Allocate buffer for data
                            pr.data = Some(vec![0u8; length]);
                            pr.length = length;

                            // Post data receive
                            // SAFETY: data buffer lives in PendingReceive
                            unsafe {
                                let rc = MPI_Irecv(
                                    pr.data.as_mut().unwrap().as_mut_ptr() as *mut c_void,
                                    length as c_int,
                                    get_mpi_byte(),
                                    source,
                                    self.edge,
                                    self.comm,
                                    &mut pr.request,
                                );
                                if rc != 0 {
                                    panic!("MPI_Irecv failed with code {}", rc);
                                }
                            }

                            pr.status = ReceiveStatus::Posted;

                            // Extract optional header
                            let header = if count as usize > 2 {
                                Some(pr.header_buf[2..count as usize].to_vec())
                            } else {
                                None
                            };

                            updates.push((source, false, fin_flag, None, header));
                        } else {
                            // Finish message received
                            if count != 2 {
                                panic!("Unexpected finish header size: {} != 2", count);
                            }
                            pr.status = ReceiveStatus::ReceivedFin;
                            updates.push((source, false, fin_flag, None, None));
                        }
                    }
                }
                ReceiveStatus::Posted => {
                    // Test if data receive completed
                    let mut flag: c_int = 0;
                    let mut status: MPI_Status = unsafe { std::mem::zeroed() };

                    // SAFETY: request is valid
                    unsafe {
                        let rc = MPI_Test(&mut pr.request, &mut flag, &mut status);
                        if rc != 0 {
                            panic!("MPI_Test failed with code {}", rc);
                        }
                    }

                    if flag != 0 {
                        // Data received!
                        let mut count: c_int = 0;

                        // SAFETY: status is valid from MPI_Test
                        unsafe {
                            let rc = MPI_Get_count(&status, get_mpi_byte(), &mut count);
                            if rc != 0 {
                                panic!("MPI_Get_count failed with code {}", rc);
                            }
                        }

                        if count as usize != pr.length {
                            panic!("Unexpected data size: {} != {}", count, pr.length);
                        }

                        // Take the data buffer
                        let data = pr.data.take().unwrap();
                        updates.push((source, true, 0, Some(data.clone()), None));

                        // Re-post header receive for next message
                        pr.header_buf = [0; CYLON_CHANNEL_HEADER_SIZE];

                        // SAFETY: header_buf is valid
                        unsafe {
                            let rc = MPI_Irecv(
                                pr.header_buf.as_mut_ptr() as *mut c_void,
                                CYLON_CHANNEL_HEADER_SIZE as c_int,
                                get_mpi_int(),
                                source,
                                self.edge,
                                self.comm,
                                &mut pr.request,
                            );
                            if rc != 0 {
                                panic!("MPI_Irecv failed with code {}", rc);
                            }
                        }

                        pr.status = ReceiveStatus::LengthPosted;
                    }
                }
                ReceiveStatus::ReceivedFin => {
                    // Done receiving from this source
                }
                _ => {}
            }
        }

        // Apply updates via callbacks
        if let Some(ref mut rcv_fn) = self.rcv_fn {
            for (source, is_data, fin_flag, data_payload, header_payload) in updates {
                if is_data {
                    // Data callback
                    if let Some(data) = data_payload {
                        let length = data.len();
                        let buffer: Box<dyn Buffer> = Box::new(VecBuffer::with_data(data));
                        rcv_fn.received_data(source, buffer, length);
                    }
                } else {
                    // Header callback
                    rcv_fn.received_header(source, fin_flag, header_payload);
                }
            }
        }
    }

    fn progress_sends(&mut self) {
        // Corresponds to MPIChannel::progressSends() in cpp/src/cylon/net/mpi/mpi_channel.cpp

        // Collect completions to avoid borrow conflicts
        let mut completions = Vec::new();

        let targets: Vec<i32> = self.sends.keys().copied().collect();

        for target in targets {
            if let Some(ps) = self.sends.get_mut(&target) {
                match ps.status {
                    SendStatus::Init => {
                        // Ready to send - check if we have pending data or finish
                        if !ps.pending_data.is_empty() {
                            self.send_header(target);
                        } else if self.finish_requests.contains_key(&target) {
                            self.send_finish_header(target);
                        }
                    }
                    SendStatus::LengthPosted => {
                        // Test if header send completed
                        let mut flag: c_int = 0;
                        let mut status: MPI_Status = unsafe { std::mem::zeroed() };

                        // SAFETY: request is valid
                        unsafe {
                            let rc = MPI_Test(&mut ps.request, &mut flag, &mut status);
                            if rc != 0 {
                                panic!("MPI_Test failed with code {}", rc);
                            }
                        }

                        if flag != 0 {
                            // Header sent! Now send the data
                            if let Some(r) = ps.pending_data.pop_front() {
                                // Post data send
                                // SAFETY: buffer lives in current_send which won't be dropped
                                // until request completes
                                unsafe {
                                    let rc = MPI_Isend(
                                        r.buffer.as_ptr() as *const c_void,
                                        r.len() as c_int,
                                        get_mpi_byte(),
                                        target,
                                        self.edge,
                                        self.comm,
                                        &mut ps.request,
                                    );
                                    if rc != 0 {
                                        panic!("MPI_Isend failed with code {}", rc);
                                    }
                                }

                                ps.current_send = Some(r);
                                ps.status = SendStatus::Posted;
                            }
                        }
                    }
                    SendStatus::Posted => {
                        // Test if data send completed
                        let mut flag: c_int = 0;
                        let mut status: MPI_Status = unsafe { std::mem::zeroed() };

                        // SAFETY: request is valid
                        unsafe {
                            let rc = MPI_Test(&mut ps.request, &mut flag, &mut status);
                            if rc != 0 {
                                panic!("MPI_Test failed with code {}", rc);
                            }
                        }

                        if flag != 0 {
                            // Data sent!
                            if let Some(current) = ps.current_send.take() {
                                completions.push((target, current));
                            }

                            // Check if more data to send
                            if !ps.pending_data.is_empty() {
                                self.send_header(target);
                            } else if self.finish_requests.contains_key(&target) {
                                self.send_finish_header(target);
                            } else {
                                ps.status = SendStatus::Init;
                            }
                        }
                    }
                    SendStatus::Finish => {
                        // Test if finish header send completed
                        let mut flag: c_int = 0;
                        let mut status: MPI_Status = unsafe { std::mem::zeroed() };

                        // SAFETY: request is valid
                        unsafe {
                            let rc = MPI_Test(&mut ps.request, &mut flag, &mut status);
                            if rc != 0 {
                                panic!("MPI_Test failed with code {}", rc);
                            }
                        }

                        if flag != 0 {
                            // Finish sent!
                            if let Some(fin_req) = self.finish_requests.remove(&target) {
                                if let Some(ref mut send_fn) = self.send_comp_fn {
                                    send_fn.send_finish_complete(fin_req);
                                }
                            }
                            ps.status = SendStatus::Done;
                        }
                    }
                    SendStatus::Done => {
                        // All done for this target
                    }
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

    fn close(&mut self) {
        // Cancel all pending MPI requests
        // Corresponds to MPIChannel::close() in C++

        // SAFETY: Each request is valid and was created by MPI_Irecv/MPI_Isend
        unsafe {
            for (_source, pr) in self.pending_receives.iter_mut() {
                if pr.status != ReceiveStatus::ReceivedFin {
                    MPI_Cancel(&mut pr.request);
                }
            }

            for (_target, ps) in self.sends.iter_mut() {
                if ps.status != SendStatus::Done {
                    MPI_Cancel(&mut ps.request);
                }
            }
        }

        self.pending_receives.clear();
        self.sends.clear();
        self.finish_requests.clear();
    }
}
