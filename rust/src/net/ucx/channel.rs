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

//! UCX Channel implementation
//!
//! Ported from cpp/src/cylon/net/ucx/ucx_channel.hpp/cpp

use std::collections::{HashMap, VecDeque};
use std::ptr;
use std::ffi::c_void;

use crate::error::{CylonError, CylonResult, Code};
use crate::net::request::CylonRequest;
use crate::net::{ChannelReceiveCallback, ChannelSendCallback, Allocator, Buffer};
use crate::net::{CYLON_CHANNEL_HEADER_SIZE, CYLON_MSG_FIN, MAX_PENDING};

use super::ucx_sys::*;
use super::operations::UcxContext;
use super::communicator::UCXCommunicator;

// Helper functions for UCS_PTR macros (C preprocessor macros not exported by bindgen)
// These correspond to macros defined in ucs/type/status.h

/// Check if a pointer is an error value
/// Corresponds to C macro: UCS_PTR_IS_ERR(_ptr)
#[inline]
unsafe fn ucs_ptr_is_err(ptr: *mut c_void) -> bool {
    (ptr as usize) >= (ucs_status_t_UCS_ERR_LAST as usize)
}

/// Extract status code from an error pointer
/// Corresponds to C macro: UCS_PTR_STATUS(_ptr)
#[inline]
unsafe fn ucs_ptr_status(ptr: *mut c_void) -> ucs_status_t {
    ptr as ucs_status_t
}

/// UCX Send Status
///
/// Corresponds to C++ UCXSendStatus from ucx_channel.hpp:34-40
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum UCXSendStatus {
    SendInit = 0,
    SendLengthPosted = 1,
    SendPosted = 2,
    SendFinish = 3,
    SendDone = 4,
}

/// UCX Receive Status
///
/// Corresponds to C++ UCXReceiveStatus from ucx_channel.hpp:42-47
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum UCXReceiveStatus {
    ReceiveInit = 0,
    ReceiveLengthPosted = 1,
    ReceivePosted = 2,
    ReceivedFin = 3,
}

/// Keep track about the length buffer to receive the length first
///
/// Corresponds to C++ PendingSend from ucx_channel.hpp:52-63
struct PendingSend {
    /// We allow up to 8 ints for the header
    header_buf: [i32; CYLON_CHANNEL_HEADER_SIZE],
    /// Segments of data to be sent
    pending_data: VecDeque<Box<CylonRequest>>,
    /// Status
    status: UCXSendStatus,
    /// The current send, if it is an actual send
    current_send: Option<Box<CylonRequest>>,
    /// UCX context - For tracking the progress of the message
    context: *mut UcxContext,
}

/// Pending receive buffer
///
/// Corresponds to C++ PendingReceive from ucx_channel.hpp:65-76
struct PendingReceive {
    /// We allow up to 8 integer header
    header_buf: [i32; CYLON_CHANNEL_HEADER_SIZE],
    receive_id: i32,
    /// Buffers are untyped: they simply denote a physical memory area
    data: Option<Box<dyn Buffer>>,
    length: i32,
    status: UCXReceiveStatus,
    /// UCX context - For tracking the progress of the message
    context: *mut UcxContext,
}

/// UCX Channel implementation
///
/// Corresponds to C++ UCXChannel from ucx_channel.hpp:83-201
pub struct UCXChannel {
    edge: i32,
    /// Keep track of the length buffers for each receiver
    sends: HashMap<i32, Box<PendingSend>>,
    /// Keep track of the posted receives
    pending_receives: HashMap<i32, Box<PendingReceive>>,
    /// We got finish requests
    finish_requests: HashMap<i32, Box<CylonRequest>>,
    /// Receive callback function
    rcv_fn: Option<Box<dyn ChannelReceiveCallback>>,
    /// Send complete callback function
    send_comp_fn: Option<Box<dyn ChannelSendCallback>>,
    /// Allocator
    allocator: Option<Box<dyn Allocator>>,
    /// MPI rank
    rank: i32,
    /// MPI world size
    world_size: i32,
    /// The worker for receiving
    ucp_recv_worker: ucp_worker_h,
    /// The worker for sending
    ucp_send_worker: ucp_worker_h,
    /// Endpoint Map
    end_point_map: HashMap<i32, ucp_ep_h>,
    /// Tag mask used to match UCX send / receives
    tag_mask: ucp_tag_t,
}

// Cylon is single-threaded, so these UCX handles can be safely sent/synced
unsafe impl Send for UCXChannel {}
unsafe impl Sync for UCXChannel {}

/// Handle the completion of a receive
///
/// Corresponds to C++ recvHandler (ucx_channel.cpp:37-47)
unsafe extern "C" fn recv_handler(
    _request: *mut c_void,
    _status: ucs_status_t,
    _info: *const ucp_tag_recv_info_t,
    ctx: *mut c_void,
) {
    let context = ctx as *mut UcxContext;
    (*context).completed = 1;
}

/// Handle the completion of a send
///
/// Corresponds to C++ sendHandler (ucx_channel.cpp:57-65)
unsafe extern "C" fn send_handler(
    _request: *mut c_void,
    _status: ucs_status_t,
    ctx: *mut c_void,
) {
    let context = ctx as *mut UcxContext;
    (*context).completed = 1;
}

/// Generate the tag used in messaging
///
/// Corresponds to C++ getTag (ucx_channel.cpp:76-78)
///
/// Used to identify the sender, not just the edge
/// Both the edge and sender, which are ints, are combined to make the ucp_tag
fn get_tag(edge: i32, sender: i32) -> u64 {
    ((edge as u64) << 32) + (sender as u64)
}

impl UCXChannel {
    /// Create a new UCX channel
    ///
    /// Corresponds to C++ UCXChannel::UCXChannel (ucx_channel.cpp:80-86)
    pub fn new(com: &UCXCommunicator) -> Self {
        Self {
            edge: 0,
            sends: HashMap::new(),
            pending_receives: HashMap::new(),
            finish_requests: HashMap::new(),
            rcv_fn: None,
            send_comp_fn: None,
            allocator: None,
            rank: com.get_rank(),
            world_size: com.get_world_size(),
            ucp_recv_worker: com.ucp_recv_worker,
            ucp_send_worker: com.ucp_send_worker,
            end_point_map: com.end_point_map.clone(),
            tag_mask: u64::MAX,
        }
    }

    /// UCX Receive
    ///
    /// Corresponds to C++ UCXChannel::UCX_Irecv (ucx_channel.cpp:97-129)
    ///
    /// Modeled after the IRECV function of MPI
    unsafe fn ucx_irecv(
        &self,
        buffer: *mut u8,
        count: usize,
        sender: i32,
        ctx: *mut UcxContext,
    ) -> CylonResult<()> {
        // Corresponds to C++ lines 102-109
        let mut recv_param: ucp_request_param_t = std::mem::zeroed();
        recv_param.op_attr_mask = ucp_op_attr_t_UCP_OP_ATTR_FIELD_CALLBACK
            | ucp_op_attr_t_UCP_OP_ATTR_FIELD_USER_DATA
            | ucp_op_attr_t_UCP_OP_ATTR_FLAG_NO_IMM_CMPL;
        recv_param.cb.recv = Some(recv_handler);
        recv_param.user_data = ctx as *mut c_void;

        // Init completed
        // Corresponds to C++ line 112
        (*ctx).completed = 0;

        // UCP non-blocking tag receive
        // Corresponds to C++ lines 116-121
        let status = ucp_tag_recv_nbx(
            self.ucp_recv_worker,
            buffer as *mut c_void,
            count,
            get_tag(self.edge, sender),
            self.tag_mask,
            &recv_param,
        );

        // Corresponds to C++ lines 123-126
        if ucs_ptr_is_err(status) {
            let err_status = ucs_ptr_status(status);
            return Err(CylonError::new(
                Code::ExecutionError,
                format!("Error in receiving message via UCX: {}", err_status),
            ));
        }

        // Corresponds to C++ line 128
        Ok(())
    }

    /// UCX Send
    ///
    /// Corresponds to C++ UCXChannel::UCX_Isend (ucx_channel.cpp:140-171)
    ///
    /// Modeled after the ISEND function of MPI
    unsafe fn ucx_isend(
        &self,
        buffer: *const u8,
        count: usize,
        ep: ucp_ep_h,
        ctx: *mut UcxContext,
    ) -> CylonResult<()> {
        // Send parameters (Mask, callback, context)
        // Corresponds to C++ lines 148-153
        let mut send_param: ucp_request_param_t = std::mem::zeroed();
        send_param.op_attr_mask = ucp_op_attr_t_UCP_OP_ATTR_FIELD_CALLBACK
            | ucp_op_attr_t_UCP_OP_ATTR_FIELD_USER_DATA
            | ucp_op_attr_t_UCP_OP_ATTR_FLAG_NO_IMM_CMPL;
        send_param.cb.send = Some(send_handler);
        send_param.user_data = ctx as *mut c_void;

        // Init completed
        // Corresponds to C++ line 156
        (*ctx).completed = 0;

        // UCP non-blocking tag send
        // Corresponds to C++ lines 159-163
        let status = ucp_tag_send_nbx(
            ep,
            buffer as *const c_void,
            count,
            get_tag(self.edge, self.rank),
            &send_param,
        );

        // Check if there is an error in the request
        // Corresponds to C++ lines 165-168
        if ucs_ptr_is_err(status) {
            let err_status = ucs_ptr_status(status);
            return Err(CylonError::new(
                Code::ExecutionError,
                format!("Error in sending message via UCX: {}", err_status),
            ));
        }

        // Corresponds to C++ line 170
        Ok(())
    }

    /// Send the length
    ///
    /// Corresponds to C++ UCXChannel::sendHeader (ucx_channel.cpp:429-452)
    unsafe fn send_header(&mut self, target: i32) -> CylonResult<()> {
        let ps = self.sends.get_mut(&target).unwrap();

        // Get the request
        // Corresponds to C++ line 431
        let r = ps.pending_data.front().unwrap();

        // Put the length to the buffer
        // Corresponds to C++ lines 433-434
        ps.header_buf[0] = r.buffer.len() as i32;
        ps.header_buf[1] = 0;

        // Copy data from CylonRequest header to the PendingSend header
        // Corresponds to C++ lines 437-441
        if r.header_length > 0 {
            ps.header_buf[2..2 + r.header_length]
                .copy_from_slice(&r.header[..r.header_length]);
        }

        // Delete old context and create new one
        // Corresponds to C++ lines 442-445
        if !ps.context.is_null() {
            drop(Box::from_raw(ps.context));
        }
        ps.context = Box::into_raw(Box::new(UcxContext { completed: 0 }));

        // Extract values before calling ucx_isend to avoid borrow conflict
        let header_ptr = ps.header_buf.as_ptr() as *const u8;
        let header_size = (2 + r.header_length) * std::mem::size_of::<i32>();
        let ep = *self.end_point_map.get(&target).unwrap();
        let context = ps.context;

        // Drop the mutable borrow before calling ucx_isend
        drop(ps);

        // UCX send of the header
        // Corresponds to C++ lines 446-449
        self.ucx_isend(header_ptr, header_size, ep, context)?;

        // Update status
        // Corresponds to C++ line 451
        let ps = self.sends.get_mut(&target).unwrap();
        ps.status = UCXSendStatus::SendLengthPosted;
        Ok(())
    }

    /// Send finish header
    ///
    /// Corresponds to C++ UCXChannel::sendFinishHeader (ucx_channel.cpp:458-470)
    unsafe fn send_finish_header(&mut self, target: i32) -> CylonResult<()> {
        let ps = self.sends.get_mut(&target).unwrap();

        // For the last header we always send only the first 2 integers
        // Corresponds to C++ lines 460-461
        ps.header_buf[0] = 0;
        ps.header_buf[1] = CYLON_MSG_FIN;

        // Delete old context and create new one
        // Corresponds to C++ lines 462-464
        if !ps.context.is_null() {
            drop(Box::from_raw(ps.context));
        }
        ps.context = Box::into_raw(Box::new(UcxContext { completed: 0 }));

        // Extract values before calling ucx_isend to avoid borrow conflict
        let header_ptr = ps.header_buf.as_ptr() as *const u8;
        let ep = *self.end_point_map.get(&target).unwrap();
        let context = ps.context;

        // Drop the mutable borrow before calling ucx_isend
        drop(ps);

        // UCX send
        // Corresponds to C++ lines 465-468
        self.ucx_isend(header_ptr, 8 * std::mem::size_of::<i32>(), ep, context)?;

        // Update status
        // Corresponds to C++ line 469
        let ps = self.sends.get_mut(&target).unwrap();
        ps.status = UCXSendStatus::SendFinish;
        Ok(())
    }
}

impl crate::net::Channel for UCXChannel {
    /// Initialize the channel
    ///
    /// Corresponds to C++ UCXChannel::init (ucx_channel.cpp:183-231)
    fn init(
        &mut self,
        edge: i32,
        receives: &[i32],
        sends: &[i32],
        rcv_callback: Box<dyn ChannelReceiveCallback>,
        send_callback: Box<dyn ChannelSendCallback>,
        allocator: Box<dyn Allocator>,
    ) -> CylonResult<()> {
        // Storing the parameters given by the Cylon Channel class
        // Corresponds to C++ lines 190-193
        self.edge = edge;
        self.rcv_fn = Some(rcv_callback);
        self.send_comp_fn = Some(send_callback);
        self.allocator = Some(allocator);

        // Get the number of receives and sends to be used in iterations
        // Corresponds to C++ lines 196-197
        let num_recv = receives.len();
        let num_sends = sends.len();

        // Iterate and set the receives
        // Corresponds to C++ lines 202-221
        for &recv_rank in receives {
            // Init a new pending receive for the request
            // Corresponds to C++ line 206
            let mut buf = Box::new(PendingReceive {
                header_buf: [0; CYLON_CHANNEL_HEADER_SIZE],
                receive_id: recv_rank,
                data: None,
                length: 0,
                status: UCXReceiveStatus::ReceiveInit,
                context: ptr::null_mut(),
            });

            // Init context
            // Corresponds to C++ lines 212-213
            buf.context = Box::into_raw(Box::new(UcxContext { completed: 0 }));

            // UCX receive
            // Corresponds to C++ lines 215-218
            unsafe {
                self.ucx_irecv(
                    buf.header_buf.as_mut_ptr() as *mut u8,
                    CYLON_CHANNEL_HEADER_SIZE * std::mem::size_of::<i32>(),
                    recv_rank,
                    buf.context,
                )?;
            }

            // Init status of the receive
            // Corresponds to C++ line 220
            buf.status = UCXReceiveStatus::ReceiveLengthPosted;

            // Add to pendingReceive object to pendingReceives map
            // Corresponds to C++ line 209
            self.pending_receives.insert(recv_rank, buf);
        }

        // Iterate and set the sends
        // Corresponds to C++ lines 224-230
        for &send_rank in sends {
            // Init a new pending send for the request
            // Corresponds to C++ line 229
            self.sends.insert(
                send_rank,
                Box::new(PendingSend {
                    header_buf: [0; CYLON_CHANNEL_HEADER_SIZE],
                    pending_data: VecDeque::new(),
                    status: UCXSendStatus::SendInit,
                    current_send: None,
                    context: ptr::null_mut(),
                }),
            );
        }

        Ok(())
    }

    /// Send a request
    ///
    /// Corresponds to C++ UCXChannel::send (ucx_channel.cpp:238-247)
    fn send(&mut self, request: Box<CylonRequest>) -> i32 {
        // Loads the pending send from sends
        // Corresponds to C++ line 240
        let ps = self.sends.get_mut(&request.target).unwrap();

        // Check if we have too many pending
        // Corresponds to C++ lines 241-243
        if ps.pending_data.len() > MAX_PENDING {
            return -1;
        }

        // pendingData is a queue that has TXRequests
        // Corresponds to C++ line 245
        ps.pending_data.push_back(request);
        1
    }

    /// Send finish message to target
    ///
    /// Corresponds to C++ UCXChannel::sendFin (ucx_channel.cpp:254-264)
    fn send_fin(&mut self, request: Box<CylonRequest>) -> i32 {
        // Checks if the finished request is already in finished req
        // Corresponds to C++ lines 257-259
        if self.finish_requests.contains_key(&request.target) {
            return -1;
        }

        // Add finished req to map
        // Corresponds to C++ line 262
        self.finish_requests.insert(request.target, request);
        1
    }

    /// Progress pending sends
    ///
    /// Corresponds to C++ UCXChannel::progressSends (ucx_channel.cpp:345-423)
    fn progress_sends(&mut self) {
        // Progress the ucp send worker
        // Corresponds to C++ line 347
        unsafe {
            ucp_worker_progress(self.ucp_send_worker);
        }

        // Iterate through the sends
        // Corresponds to C++ line 350
        let targets: Vec<i32> = self.sends.keys().copied().collect();
        for target in targets {
            let status = self.sends.get(&target).unwrap().status;

            // Corresponds to C++ lines 352-376
            if status == UCXSendStatus::SendLengthPosted {
                // If completed
                let completed = unsafe { (*self.sends.get(&target).unwrap().context).completed };
                if completed == 1 {
                    let ps = self.sends.get_mut(&target).unwrap();

                    // Destroy context object
                    // Corresponds to C++ line 357
                    unsafe {
                        if !ps.context.is_null() {
                            drop(Box::from_raw(ps.context));
                            ps.context = ptr::null_mut();
                        }
                    }

                    // Post the actual send
                    // Corresponds to C++ line 360
                    let r = ps.pending_data.pop_front().unwrap();

                    // Send the message
                    // Corresponds to C++ lines 362-367
                    ps.context = Box::into_raw(Box::new(UcxContext { completed: 0 }));

                    // Extract values before calling ucx_isend to avoid borrow conflict
                    let buf_ptr = r.buffer.as_ptr();
                    let buf_len = r.buffer.len();
                    let ep = *self.end_point_map.get(&target).unwrap();
                    let context = ps.context;

                    // Drop the mutable borrow before calling ucx_isend
                    drop(ps);

                    unsafe {
                        let _ = self.ucx_isend(buf_ptr, buf_len, ep, context);
                    }

                    // Update status
                    // Corresponds to C++ line 370
                    let ps = self.sends.get_mut(&target).unwrap();
                    ps.status = UCXSendStatus::SendPosted;

                    // The update the current send in the queue of sends
                    // Corresponds to C++ lines 373-375
                    ps.current_send = Some(r);
                }
            } else if status == UCXSendStatus::SendInit {
                // Corresponds to C++ lines 377-384
                let ps = self.sends.get(&target).unwrap();

                // Send header if no pending data
                if !ps.pending_data.is_empty() {
                    unsafe {
                        let _ = self.send_header(target);
                    }
                } else if self.finish_requests.contains_key(&target) {
                    // If there are finish requests lets send them
                    unsafe {
                        let _ = self.send_finish_header(target);
                    }
                }
            } else if status == UCXSendStatus::SendPosted {
                // Corresponds to C++ lines 385-409
                let completed = unsafe { (*self.sends.get(&target).unwrap().context).completed };
                if completed == 1 {
                    // Check if there are more data to post
                    let has_pending = self.sends.get(&target).unwrap().pending_data.is_empty() == false;

                    if has_pending {
                        // If there are more data to post, post the length buffer now
                        // Corresponds to C++ lines 391-394
                        unsafe {
                            let _ = self.send_header(target);
                        }

                        let ps = self.sends.get_mut(&target).unwrap();
                        // We need to notify about the send completion
                        if let Some(ref mut send_fn) = self.send_comp_fn {
                            if let Some(current_send) = ps.current_send.take() {
                                send_fn.send_complete(current_send);
                            }
                        }
                    } else {
                        // If pending data is empty
                        // Corresponds to C++ lines 397-408
                        let ps = self.sends.get_mut(&target).unwrap();
                        if let Some(ref mut send_fn) = self.send_comp_fn {
                            if let Some(current_send) = ps.current_send.take() {
                                send_fn.send_complete(current_send);
                            }
                        }

                        // Check if request is in finish
                        if self.finish_requests.contains_key(&target) {
                            unsafe {
                                let _ = self.send_finish_header(target);
                            }
                        } else {
                            // If req is not in finish then re-init
                            ps.status = UCXSendStatus::SendInit;
                        }
                    }
                }
            } else if status == UCXSendStatus::SendFinish {
                // Corresponds to C++ lines 410-416
                let completed = unsafe { (*self.sends.get(&target).unwrap().context).completed };
                if completed == 1 {
                    // We are going to send complete
                    let fin_req = self.finish_requests.remove(&target).unwrap();
                    if let Some(ref mut send_fn) = self.send_comp_fn {
                        send_fn.send_finish_complete(fin_req);
                    }
                    self.sends.get_mut(&target).unwrap().status = UCXSendStatus::SendDone;
                }
            }
        }
    }

    /// Progress pending receives
    ///
    /// Corresponds to C++ UCXChannel::progressReceives (ucx_channel.cpp:266-343)
    fn progress_receives(&mut self) {
        // Progress the ucp receive worker
        // Corresponds to C++ line 268
        unsafe {
            ucp_worker_progress(self.ucp_recv_worker);
        }

        // Iterate through the pending receives
        // Corresponds to C++ line 271
        let recv_ids: Vec<i32> = self.pending_receives.keys().copied().collect();
        for recv_id in recv_ids {
            let status = self.pending_receives.get(&recv_id).unwrap().status;

            // Check if the buffer is posted
            // Corresponds to C++ lines 273-317
            if status == UCXReceiveStatus::ReceiveLengthPosted {
                let completed = unsafe {
                    (*self.pending_receives.get(&recv_id).unwrap().context).completed
                };

                // If completed request is completed
                if completed == 1 {
                    let pr = self.pending_receives.get_mut(&recv_id).unwrap();

                    // Get data from the header
                    // Corresponds to C++ lines 277-279
                    let length = pr.header_buf[0];
                    let fin_flag = pr.header_buf[1];

                    // Check whether we are at the end
                    // Corresponds to C++ line 282
                    if fin_flag != CYLON_MSG_FIN {
                        // If not at the end
                        // Corresponds to C++ lines 286-292
                        let mut data = self.allocator.as_ref().unwrap()
                            .allocate(length as usize)
                            .expect("Failed to allocate buffer");

                        // Set the length
                        pr.length = length;

                        // Reset context
                        // Corresponds to C++ lines 295-297
                        unsafe {
                            if !pr.context.is_null() {
                                drop(Box::from_raw(pr.context));
                            }
                            pr.context = Box::into_raw(Box::new(UcxContext { completed: 0 }));

                            // Extract values before calling ucx_irecv to avoid borrow conflict
                            let buf_ptr = data.get_byte_buffer_mut().as_mut_ptr();
                            let context = pr.context;

                            // Drop the mutable borrow before calling ucx_irecv
                            drop(pr);

                            // UCX receive
                            // Corresponds to C++ line 300
                            let _ = self.ucx_irecv(buf_ptr, length as usize, recv_id, context);
                        }

                        // Set the flag to true so we can identify later which buffers are posted
                        // Corresponds to C++ line 302
                        let pr = self.pending_receives.get_mut(&recv_id).unwrap();
                        pr.status = UCXReceiveStatus::ReceivePosted;
                        pr.data = Some(data);

                        // Copy the count - 2 to the buffer
                        // Corresponds to C++ lines 305-307
                        let header: Vec<i32> = pr.header_buf[2..8].to_vec();

                        // Notify the receiver that the destination received the header
                        // Corresponds to C++ line 310
                        if let Some(ref mut rcv_fn) = self.rcv_fn {
                            rcv_fn.received_header(recv_id, fin_flag, Some(header));
                        }
                    } else {
                        // We are not expecting to receive any more
                        // Corresponds to C++ lines 313-315
                        pr.status = UCXReceiveStatus::ReceivedFin;
                        // Notify the receiver
                        if let Some(ref mut rcv_fn) = self.rcv_fn {
                            rcv_fn.received_header(recv_id, fin_flag, None);
                        }
                    }
                }
            } else if status == UCXReceiveStatus::ReceivePosted {
                // Corresponds to C++ lines 318-338
                let completed = unsafe {
                    (*self.pending_receives.get(&recv_id).unwrap().context).completed
                };

                // If request completed
                if completed == 1 {
                    let pr = self.pending_receives.get_mut(&recv_id).unwrap();

                    // Fill header buffer
                    // Corresponds to C++ line 322
                    pr.header_buf.iter_mut().for_each(|x| *x = 0);

                    // Reset the context
                    // Corresponds to C++ lines 325-327
                    unsafe {
                        if !pr.context.is_null() {
                            drop(Box::from_raw(pr.context));
                        }
                        pr.context = Box::into_raw(Box::new(UcxContext { completed: 0 }));

                        // Extract values before calling ucx_irecv to avoid borrow conflict
                        let header_ptr = pr.header_buf.as_mut_ptr() as *mut u8;
                        let context = pr.context;

                        // Drop the mutable borrow before calling ucx_irecv
                        drop(pr);

                        // UCX receive
                        // Corresponds to C++ lines 330-333
                        let _ = self.ucx_irecv(
                            header_ptr,
                            CYLON_CHANNEL_HEADER_SIZE * std::mem::size_of::<i32>(),
                            recv_id,
                            context,
                        );
                    }

                    // Set state
                    // Corresponds to C++ line 335
                    let pr = self.pending_receives.get_mut(&recv_id).unwrap();
                    pr.status = UCXReceiveStatus::ReceiveLengthPosted;

                    // Call the back end
                    // Corresponds to C++ line 337
                    let data = pr.data.take().unwrap();
                    let length = pr.length as usize;
                    if let Some(ref mut rcv_fn) = self.rcv_fn {
                        rcv_fn.received_data(recv_id, data, length);
                    }
                }
            }
        }
    }

    /// Close the channel
    ///
    /// Corresponds to C++ UCXChannel::close (ucx_channel.cpp:475-491)
    fn close(&mut self) {
        // Clear pending receives
        // Corresponds to C++ lines 477-481
        for (_, mut pr) in self.pending_receives.drain() {
            unsafe {
                if !pr.context.is_null() {
                    drop(Box::from_raw(pr.context));
                }
            }
        }

        // Clear the sends
        // Corresponds to C++ lines 484-488
        for (_, mut ps) in self.sends.drain() {
            unsafe {
                if !ps.context.is_null() {
                    drop(Box::from_raw(ps.context));
                }
            }
        }

        // Corresponds to C++ line 490
        self.end_point_map.clear();
    }
}
