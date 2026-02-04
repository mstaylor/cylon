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

//! PeerToPeer channel implementation
//!
//! This module corresponds to cpp/src/cylon/thridparty/fmi/comm/PeerToPeer.hpp/cpp
//!
//! PeerToPeer channels are optimized for direct peer-to-peer communication and
//! implement binomial tree algorithms for collective operations.

use std::cmp::min;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::error::CylonResult;
use super::channel::Channel;
use super::common::*;

/// IOState for tracking non-blocking operations (matches FMI::Comm::IOState)
#[derive(Clone)]
pub struct IOState {
    pub request: Arc<ChannelData>,
    pub processed: usize,
    pub operation: Operation,
    pub context: Option<Arc<Mutex<FmiContext>>>,
    pub dummy: u8,
    pub callback_result: Option<Arc<dyn Fn(NbxStatus, &str, &mut FmiContext) + Send + Sync>>,
    pub callback: Option<Arc<dyn Fn() + Send + Sync>>,
    pub deadline: Instant,
}

impl IOState {
    pub fn new(request: Arc<ChannelData>, operation: Operation, max_timeout_ms: i32) -> Self {
        let deadline = if max_timeout_ms > 0 {
            Instant::now() + Duration::from_millis(max_timeout_ms as u64)
        } else {
            Instant::now() + Duration::from_secs(3600) // 1 hour default
        };

        Self {
            request,
            processed: 0,
            operation,
            context: None,
            dummy: 0,
            callback_result: None,
            callback: None,
            deadline,
        }
    }

    pub fn set_request(&mut self, request: Arc<ChannelData>) {
        self.request = request;
    }

    pub fn set_callback<F>(&mut self, callback: F)
    where
        F: Fn() + Send + Sync + 'static,
    {
        self.callback = Some(Arc::new(callback));
    }

    pub fn set_callback_result<F>(&mut self, callback: F)
    where
        F: Fn(NbxStatus, &str, &mut FmiContext) + Send + Sync + 'static,
    {
        self.callback_result = Some(Arc::new(callback));
    }
}

/// Helper function to transform peer IDs for root-agnostic collective implementations
///
/// This allows implementing collectives as if root were always 0, simplifying
/// the implementation logic. Matches transform_peer_id() in C++.
#[inline]
pub fn transform_peer_id(id: PeerNum, root: PeerNum, num_peers: PeerNum, forward: bool) -> PeerNum {
    if forward {
        // Transform such that root becomes 0
        (id + num_peers - root) % num_peers
    } else {
        // Transform back (0 becomes root)
        (id + root) % num_peers
    }
}

/// Calculate ceil(log2(n)) - number of rounds for binomial tree algorithms
#[inline]
fn ceil_log2(n: PeerNum) -> i32 {
    if n <= 1 {
        0
    } else {
        (32 - (n - 1).leading_zeros()) as i32
    }
}

/// Calculate floor(log2(n))
#[inline]
fn floor_log2(n: PeerNum) -> i32 {
    if n <= 0 {
        0
    } else {
        (31 - n.leading_zeros()) as i32
    }
}

/// PeerToPeer channel trait - extends Channel with direct peer communication
///
/// This trait matches FMI::Comm::PeerToPeer class from the C++ implementation.
/// Channels implementing this trait can address peers directly and use
/// optimized binomial tree algorithms for collectives.
pub trait PeerToPeerChannel: Channel {
    /// Send an object directly to a peer (blocking)
    fn send_object(&self, buf: Arc<ChannelData>, peer_id: PeerNum) -> CylonResult<()>;

    /// Send an object directly to a peer (non-blocking)
    fn send_object_async(
        &self,
        state: Arc<Mutex<IOState>>,
        peer_id: PeerNum,
        mode: Mode,
    ) -> CylonResult<()>;

    /// Receive an object from a peer (blocking)
    fn recv_object(&self, buf: Arc<ChannelData>, peer_id: PeerNum) -> CylonResult<()>;

    /// Receive an object from a peer (non-blocking)
    fn recv_object_async(
        &self,
        state: Arc<Mutex<IOState>>,
        peer_id: PeerNum,
        mode: Mode,
    ) -> CylonResult<()>;
}

// ============================================================================
// Default implementations for PeerToPeer collectives
// These are implemented as free functions that can be used by implementations
// ============================================================================

/// Binomial tree broadcast implementation
///
/// Corresponds to FMI::Comm::PeerToPeer::bcast() in C++
pub fn bcast_binomial<C: PeerToPeerChannel + ?Sized>(
    channel: &C,
    buf: Arc<ChannelData>,
    root: PeerNum,
    mode: Mode,
    callback: Option<NbxCallback>,
) -> CylonResult<()> {
    let peer_id = channel.peer_id();
    let num_peers = channel.num_peers();
    let rounds = ceil_log2(num_peers);
    let trans_peer_id = transform_peer_id(peer_id, root, num_peers, true);

    for i in (0..rounds).rev() {
        let power = 1_i32 << i; // 2^i using bit shift
        let rcpt = trans_peer_id + power;

        if trans_peer_id % (1 << (i + 1)) == 0 && rcpt < num_peers {
            let real_rcpt = transform_peer_id(rcpt, root, num_peers, false);

            if mode == Mode::Blocking {
                channel.send(buf.clone(), real_rcpt)?;
            } else {
                let mut state = IOState::new(buf.clone(), Operation::Send, channel.get_max_timeout());
                if let Some(ref cb) = callback {
                    state.callback_result = Some(cb.clone());
                }
                let state_arc = Arc::new(Mutex::new(state));
                channel.send_object_async(state_arc, real_rcpt, mode)?;
            }
        } else if trans_peer_id % power == 0 && trans_peer_id % (1 << (i + 1)) != 0 {
            let real_src = transform_peer_id(trans_peer_id - power, root, num_peers, false);

            if mode == Mode::Blocking {
                channel.recv(buf.clone(), real_src)?;
            } else {
                let mut state = IOState::new(buf.clone(), Operation::Receive, channel.get_max_timeout());
                if let Some(ref cb) = callback {
                    state.callback_result = Some(cb.clone());
                }
                let state_arc = Arc::new(Mutex::new(state));
                channel.recv_object_async(state_arc, real_src, mode)?;
            }
        }
    }

    Ok(())
}

/// Barrier implementation using allreduce with NOP
///
/// Corresponds to FMI::Comm::PeerToPeer::barrier() in C++
pub fn barrier_impl<C: PeerToPeerChannel + ?Sized>(channel: &C) -> CylonResult<()> {
    let send_data = Arc::new(ChannelData::new(vec![1u8]));
    let recv_data = Arc::new(ChannelData::with_capacity(1));
    let nop = RawFunction::nop();
    allreduce_impl(channel, send_data, recv_data, &nop)
}

/// Binomial tree gather implementation (fixed-size)
///
/// Corresponds to FMI::Comm::PeerToPeer::gather() in C++
pub fn gather_binomial<C: PeerToPeerChannel + ?Sized>(
    channel: &C,
    sendbuf: Arc<ChannelData>,
    recvbuf: Arc<ChannelData>,
    root: PeerNum,
) -> CylonResult<()> {
    let peer_id = channel.peer_id();
    let num_peers = channel.num_peers();
    let rounds = ceil_log2(num_peers);
    let trans_peer_id = transform_peer_id(peer_id, root, num_peers, true);
    let single_buffer_size = sendbuf.len;

    // Copy own data to correct position in recvbuf
    if peer_id != root {
        // Non-root: copy to beginning of local buffer
        let src = sendbuf.as_slice();
        let mut dst = recvbuf.as_mut_slice();
        dst[..single_buffer_size].copy_from_slice(&src[..single_buffer_size]);
    } else {
        // Root: copy to position based on rank
        let src = sendbuf.as_slice();
        let mut dst = recvbuf.as_mut_slice();
        let offset = (root as usize) * single_buffer_size;
        dst[offset..offset + single_buffer_size].copy_from_slice(&src[..single_buffer_size]);
    }

    for i in 0..rounds {
        let power = 1_i32 << i;
        let src = trans_peer_id + power;

        if trans_peer_id % (1 << (i + 1)) == 0 && src < num_peers {
            // Receiver in this round
            let responsible_peers = min(power, num_peers - src) as usize;
            let buf_len = responsible_peers * single_buffer_size;
            let real_src = transform_peer_id(src, root, num_peers, false);

            if peer_id == root {
                let offset = (real_src as usize) * single_buffer_size;

                if offset + buf_len > recvbuf.len {
                    // Need to handle wraparound with temporary buffer
                    let tmp = Arc::new(ChannelData::with_capacity(buf_len));
                    channel.recv(tmp.clone(), real_src)?;

                    let src_data = tmp.as_slice();
                    let mut dst = recvbuf.as_mut_slice();
                    let length_end = recvbuf.len - offset;
                    dst[offset..].copy_from_slice(&src_data[..length_end]);
                    dst[..buf_len - length_end].copy_from_slice(&src_data[length_end..buf_len]);
                } else {
                    // Direct receive into recvbuf
                    let tmp = Arc::new(ChannelData::with_capacity(buf_len));
                    channel.recv(tmp.clone(), real_src)?;

                    let src_data = tmp.as_slice();
                    let mut dst = recvbuf.as_mut_slice();
                    dst[offset..offset + buf_len].copy_from_slice(&src_data[..buf_len]);
                }
            } else {
                // Non-root receiver: receive into local buffer
                let offset = ((src - trans_peer_id) as usize) * single_buffer_size;
                let tmp = Arc::new(ChannelData::with_capacity(buf_len));
                channel.recv(tmp.clone(), real_src)?;

                let src_data = tmp.as_slice();
                let mut dst = recvbuf.as_mut_slice();
                dst[offset..offset + buf_len].copy_from_slice(&src_data[..buf_len]);
            }
        } else if trans_peer_id % power == 0 && trans_peer_id % (1 << (i + 1)) != 0 {
            // Sender in this round
            let responsible_peers = min(power, num_peers - trans_peer_id) as usize;
            let buf_len = responsible_peers * single_buffer_size;
            let real_dst = transform_peer_id(trans_peer_id - power, root, num_peers, false);

            let send_data = {
                let src = recvbuf.as_slice();
                Arc::new(ChannelData::new(src[..buf_len].to_vec()))
            };
            channel.send(send_data, real_dst)?;
        }
    }

    Ok(())
}

/// Variable-size gather implementation
///
/// Corresponds to FMI::Comm::PeerToPeer::gatherv() in C++
pub fn gatherv_binomial<C: PeerToPeerChannel + ?Sized>(
    channel: &C,
    sendbuf: Arc<ChannelData>,
    recvbuf: Arc<ChannelData>,
    root: PeerNum,
    recvcounts: &[i32],
    displs: &[i32],
    mode: Mode,
    callback: Option<NbxCallback>,
) -> CylonResult<()> {
    let peer_id = channel.peer_id();
    let num_peers = channel.num_peers();
    let rounds = ceil_log2(num_peers);
    let trans_peer_id = transform_peer_id(peer_id, root, num_peers, true);

    // Copy own data to correct position
    if peer_id != root {
        let src = sendbuf.as_slice();
        let mut dst = recvbuf.as_mut_slice();
        let copy_len = min(sendbuf.len, dst.len());
        dst[..copy_len].copy_from_slice(&src[..copy_len]);
    } else {
        let src = sendbuf.as_slice();
        let mut dst = recvbuf.as_mut_slice();
        let offset = displs[peer_id as usize] as usize;
        let copy_len = min(sendbuf.len, dst.len() - offset);
        dst[offset..offset + copy_len].copy_from_slice(&src[..copy_len]);
    }

    for i in 0..rounds {
        let power = 1_i32 << i;
        let src_peer = trans_peer_id + power;

        if trans_peer_id % (1 << (i + 1)) == 0 && src_peer < num_peers {
            // Receiver in this round - calculate expected data size
            let responsible_peers = min(power, num_peers - src_peer) as usize;
            let mut buf_len: usize = 0;
            for p in src_peer..(src_peer + responsible_peers as i32) {
                buf_len += recvcounts[p as usize] as usize;
            }

            let real_src = transform_peer_id(src_peer, root, num_peers, false);

            if peer_id == root {
                let offset = displs[real_src as usize] as usize;

                if offset + buf_len > recvbuf.len {
                    // Handle buffer wrap-around with temporary buffer
                    let tmp = Arc::new(ChannelData::with_capacity(buf_len));

                    if mode == Mode::Blocking {
                        channel.recv(tmp.clone(), real_src)?;

                        let src_data = tmp.as_slice();
                        let mut dst = recvbuf.as_mut_slice();
                        let length_end = recvbuf.len - offset;
                        dst[offset..].copy_from_slice(&src_data[..length_end]);
                        dst[..buf_len - length_end].copy_from_slice(&src_data[length_end..buf_len]);
                    } else {
                        let mut state = IOState::new(tmp, Operation::Receive, channel.get_max_timeout());
                        if let Some(ref cb) = callback {
                            state.callback_result = Some(cb.clone());
                        }
                        let state_arc = Arc::new(Mutex::new(state));
                        channel.recv_object_async(state_arc, real_src, mode)?;
                    }
                } else {
                    // Direct receive
                    let request = Arc::new(ChannelData::with_capacity(buf_len));

                    if mode == Mode::Blocking {
                        channel.recv(request.clone(), real_src)?;

                        let src_data = request.as_slice();
                        let mut dst = recvbuf.as_mut_slice();
                        dst[offset..offset + buf_len].copy_from_slice(&src_data[..buf_len]);
                    } else {
                        let mut state = IOState::new(request, Operation::Receive, channel.get_max_timeout());
                        if let Some(ref cb) = callback {
                            state.callback_result = Some(cb.clone());
                        }
                        let state_arc = Arc::new(Mutex::new(state));
                        channel.recv_object_async(state_arc, real_src, mode)?;
                    }
                }
            } else {
                // Non-root receiver
                let request = Arc::new(ChannelData::with_capacity(buf_len));

                if mode == Mode::Blocking {
                    channel.recv(request.clone(), real_src)?;

                    let src_data = request.as_slice();
                    let mut dst = recvbuf.as_mut_slice();
                    // Append after existing data
                    let existing_len = recvcounts[peer_id as usize] as usize;
                    dst[existing_len..existing_len + buf_len].copy_from_slice(&src_data[..buf_len]);
                } else {
                    let mut state = IOState::new(request, Operation::Receive, channel.get_max_timeout());
                    if let Some(ref cb) = callback {
                        state.callback_result = Some(cb.clone());
                    }
                    let state_arc = Arc::new(Mutex::new(state));
                    channel.recv_object_async(state_arc, real_src, mode)?;
                }
            }
        } else if trans_peer_id % power == 0 && trans_peer_id % (1 << (i + 1)) != 0 {
            // Sender in this round
            let responsible_peers = min(power, num_peers - trans_peer_id) as usize;
            let mut buf_len: usize = 0;
            for p in trans_peer_id..(trans_peer_id + responsible_peers as i32) {
                buf_len += recvcounts[p as usize] as usize;
            }

            let real_dst = transform_peer_id(trans_peer_id - power, root, num_peers, false);

            let send_data = {
                let src = recvbuf.as_slice();
                Arc::new(ChannelData::new(src[..buf_len].to_vec()))
            };

            if mode == Mode::Blocking {
                channel.send(send_data, real_dst)?;
            } else {
                let mut state = IOState::new(send_data, Operation::Send, channel.get_max_timeout());
                if let Some(ref cb) = callback {
                    state.callback_result = Some(cb.clone());
                }
                let state_arc = Arc::new(Mutex::new(state));
                channel.send_object_async(state_arc, real_dst, mode)?;
            }
        }
    }

    Ok(())
}

/// Binomial tree scatter implementation
///
/// Corresponds to FMI::Comm::PeerToPeer::scatter() in C++
pub fn scatter_binomial<C: PeerToPeerChannel + ?Sized>(
    channel: &C,
    sendbuf: Arc<ChannelData>,
    recvbuf: Arc<ChannelData>,
    root: PeerNum,
) -> CylonResult<()> {
    let peer_id = channel.peer_id();
    let num_peers = channel.num_peers();
    let rounds = ceil_log2(num_peers);
    let trans_peer_id = transform_peer_id(peer_id, root, num_peers, true);
    let single_buffer_size = recvbuf.len;

    // Use a working buffer for non-root peers
    let sendbuf_cpy = if peer_id == root {
        sendbuf.clone()
    } else {
        Arc::new(ChannelData::with_capacity(sendbuf.len))
    };

    for i in (0..rounds).rev() {
        let power = 1_i32 << i;
        let rcpt = trans_peer_id + power;

        if trans_peer_id % (1 << (i + 1)) == 0 && rcpt < num_peers {
            // Sender in this round
            let responsible_peers = min(power, num_peers - rcpt) as usize;
            let buf_len = responsible_peers * single_buffer_size;
            let real_rcpt = transform_peer_id(rcpt, root, num_peers, false);

            if peer_id == root {
                let offset = (real_rcpt as usize) * single_buffer_size;

                if offset + buf_len > sendbuf.len {
                    // Handle wraparound
                    let src = sendbuf_cpy.as_slice();
                    let length_end = sendbuf_cpy.len - offset;
                    let mut tmp_data = vec![0u8; buf_len];
                    tmp_data[..length_end].copy_from_slice(&src[offset..]);
                    tmp_data[length_end..].copy_from_slice(&src[..buf_len - length_end]);
                    let tmp = Arc::new(ChannelData::new(tmp_data));
                    channel.send(tmp, real_rcpt)?;
                } else {
                    let send_data = {
                        let src = sendbuf_cpy.as_slice();
                        Arc::new(ChannelData::new(src[offset..offset + buf_len].to_vec()))
                    };
                    channel.send(send_data, real_rcpt)?;
                }
            } else {
                let offset = ((rcpt - trans_peer_id) as usize) * single_buffer_size;
                let send_data = {
                    let src = sendbuf_cpy.as_slice();
                    Arc::new(ChannelData::new(src[offset..offset + buf_len].to_vec()))
                };
                channel.send(send_data, real_rcpt)?;
            }
        } else if trans_peer_id % power == 0 && trans_peer_id % (1 << (i + 1)) != 0 {
            // Receiver in this round
            let responsible_peers = min(power, num_peers - trans_peer_id) as usize;
            let buf_len = responsible_peers * single_buffer_size;
            let real_src = transform_peer_id(trans_peer_id - power, root, num_peers, false);

            // Receive into sendbuf_cpy
            let tmp = Arc::new(ChannelData::with_capacity(buf_len));
            channel.recv(tmp.clone(), real_src)?;

            let src = tmp.as_slice();
            let mut dst = sendbuf_cpy.as_mut_slice();
            dst[..buf_len].copy_from_slice(&src[..buf_len]);
        }
    }

    // Copy result to recvbuf
    if peer_id == root {
        let src = sendbuf_cpy.as_slice();
        let mut dst = recvbuf.as_mut_slice();
        let offset = (peer_id as usize) * single_buffer_size;
        dst[..single_buffer_size].copy_from_slice(&src[offset..offset + single_buffer_size]);
    } else {
        let src = sendbuf_cpy.as_slice();
        let mut dst = recvbuf.as_mut_slice();
        dst[..single_buffer_size].copy_from_slice(&src[..single_buffer_size]);
    }

    Ok(())
}

/// Allgather implementation (fixed-size) with gather + broadcast phases
///
/// Corresponds to FMI::Comm::PeerToPeer::allgather() in C++
pub fn allgather_binomial<C: PeerToPeerChannel + ?Sized>(
    channel: &C,
    sendbuf: Arc<ChannelData>,
    recvbuf: Arc<ChannelData>,
    root: PeerNum,
    mode: Mode,
    callback: Option<NbxCallback>,
) -> CylonResult<()> {
    let peer_id = channel.peer_id();
    let num_peers = channel.num_peers();
    let rounds = ceil_log2(num_peers);
    let trans_peer_id = transform_peer_id(peer_id, root, num_peers, true);
    let single_buffer_size = sendbuf.len;
    let total_buffer_size = (num_peers as usize) * single_buffer_size;

    // Copy own data to correct position
    {
        let src = sendbuf.as_slice();
        let mut dst = recvbuf.as_mut_slice();
        let offset = (peer_id as usize) * single_buffer_size;
        dst[offset..offset + single_buffer_size].copy_from_slice(&src[..single_buffer_size]);
    }

    // Phase 1: Gather phase (binomial tree)
    for i in 0..rounds {
        let power = 1_i32 << i;
        let src_peer = trans_peer_id + power;

        if trans_peer_id % (1 << (i + 1)) == 0 && src_peer < num_peers {
            // Receiver
            let responsible_peers = min(power, num_peers - src_peer) as usize;
            let buf_len = responsible_peers * single_buffer_size;
            let real_src = transform_peer_id(src_peer, root, num_peers, false);

            let request = Arc::new(ChannelData::with_capacity(buf_len));

            if mode == Mode::Blocking {
                channel.recv(request.clone(), real_src)?;

                let src_data = request.as_slice();
                let mut dst = recvbuf.as_mut_slice();
                let offset = (real_src as usize) * single_buffer_size;
                dst[offset..offset + buf_len].copy_from_slice(&src_data[..buf_len]);
            } else {
                let mut state = IOState::new(request, Operation::Receive, channel.get_max_timeout());
                if let Some(ref cb) = callback {
                    state.callback_result = Some(cb.clone());
                }
                let state_arc = Arc::new(Mutex::new(state));
                channel.recv_object_async(state_arc, real_src, mode)?;
            }
        } else if trans_peer_id % power == 0 && trans_peer_id % (1 << (i + 1)) != 0 {
            // Sender
            let responsible_peers = min(power, num_peers - trans_peer_id) as usize;
            let buf_len = responsible_peers * single_buffer_size;
            let real_dst = transform_peer_id(trans_peer_id - power, root, num_peers, false);

            let send_data = {
                let src = recvbuf.as_slice();
                let offset = (trans_peer_id as usize) * single_buffer_size;
                Arc::new(ChannelData::new(src[offset..offset + buf_len].to_vec()))
            };

            if mode == Mode::Blocking {
                channel.send(send_data, real_dst)?;
            } else {
                let mut state = IOState::new(send_data, Operation::Send, channel.get_max_timeout());
                if let Some(ref cb) = callback {
                    state.callback_result = Some(cb.clone());
                }
                let state_arc = Arc::new(Mutex::new(state));
                channel.send_object_async(state_arc, real_dst, mode)?;
            }
        }
    }

    // Phase 2: Broadcast phase (so all processes get the gathered data)
    for i in 0..rounds {
        let partner = trans_peer_id ^ (1 << i);

        if partner < num_peers {
            let real_partner = transform_peer_id(partner, root, num_peers, false);

            if (trans_peer_id & (1 << i)) == 0 {
                // Send the full gathered data to partner
                let request = {
                    let src = recvbuf.as_slice();
                    Arc::new(ChannelData::new(src[..total_buffer_size].to_vec()))
                };

                if mode == Mode::Blocking {
                    channel.send(request, real_partner)?;
                } else {
                    let mut state = IOState::new(request, Operation::Send, channel.get_max_timeout());
                    if let Some(ref cb) = callback {
                        state.callback_result = Some(cb.clone());
                    }
                    let state_arc = Arc::new(Mutex::new(state));
                    channel.send_object_async(state_arc, real_partner, mode)?;
                }
            } else {
                // Receive the full gathered data from partner
                let request = Arc::new(ChannelData::with_capacity(total_buffer_size));

                if mode == Mode::Blocking {
                    channel.recv(request.clone(), real_partner)?;

                    let src_data = request.as_slice();
                    let mut dst = recvbuf.as_mut_slice();
                    dst[..total_buffer_size].copy_from_slice(&src_data[..total_buffer_size]);
                } else {
                    let mut state = IOState::new(request, Operation::Receive, channel.get_max_timeout());
                    if let Some(ref cb) = callback {
                        state.callback_result = Some(cb.clone());
                    }
                    let state_arc = Arc::new(Mutex::new(state));
                    channel.recv_object_async(state_arc, real_partner, mode)?;
                }
            }
        }
    }

    Ok(())
}

/// Variable-size allgather implementation
///
/// Corresponds to FMI::Comm::PeerToPeer::allgatherv() in C++
pub fn allgatherv_binomial<C: PeerToPeerChannel + ?Sized>(
    channel: &C,
    sendbuf: Arc<ChannelData>,
    recvbuf: Arc<ChannelData>,
    root: PeerNum,
    recvcounts: &[i32],
    displs: &[i32],
    mode: Mode,
    callback: Option<NbxCallback>,
) -> CylonResult<()> {
    let peer_id = channel.peer_id();
    let num_peers = channel.num_peers();
    let rounds = ceil_log2(num_peers);
    let trans_peer_id = transform_peer_id(peer_id, root, num_peers, true);

    // Calculate total buffer size
    let total_buffer_size: usize = recvcounts.iter().map(|&c| c as usize).sum();

    // Copy own data to correct position
    {
        let src = sendbuf.as_slice();
        let mut dst = recvbuf.as_mut_slice();
        let offset = displs[peer_id as usize] as usize;
        let copy_len = min(sendbuf.len, dst.len() - offset);
        dst[offset..offset + copy_len].copy_from_slice(&src[..copy_len]);
    }

    // Phase 1: Gather phase
    for i in 0..rounds {
        let power = 1_i32 << i;
        let src_peer = trans_peer_id + power;

        if trans_peer_id % (1 << (i + 1)) == 0 && src_peer < num_peers {
            // Calculate total length from responsible peers
            let mut buf_len: usize = 0;
            let end_peer = min(src_peer + power, num_peers);
            for p in src_peer..end_peer {
                buf_len += recvcounts[p as usize] as usize;
            }

            let real_src = transform_peer_id(src_peer, root, num_peers, false);
            let offset = displs[src_peer as usize] as usize;

            let request = Arc::new(ChannelData::with_capacity(buf_len));

            if mode == Mode::Blocking {
                channel.recv(request.clone(), real_src)?;

                let src_data = request.as_slice();
                let mut dst = recvbuf.as_mut_slice();
                dst[offset..offset + buf_len].copy_from_slice(&src_data[..buf_len]);
            } else {
                let mut state = IOState::new(request, Operation::Receive, channel.get_max_timeout());
                if let Some(ref cb) = callback {
                    state.callback_result = Some(cb.clone());
                }
                let state_arc = Arc::new(Mutex::new(state));
                channel.recv_object_async(state_arc, real_src, mode)?;
            }
        } else if trans_peer_id % power == 0 && trans_peer_id % (1 << (i + 1)) != 0 {
            // Calculate data to send
            let mut buf_len: usize = 0;
            let end_peer = min(trans_peer_id + power, num_peers);
            for p in trans_peer_id..end_peer {
                buf_len += recvcounts[p as usize] as usize;
            }

            let real_dst = transform_peer_id(trans_peer_id - power, root, num_peers, false);
            let offset = displs[trans_peer_id as usize] as usize;

            let send_data = {
                let src = recvbuf.as_slice();
                Arc::new(ChannelData::new(src[offset..offset + buf_len].to_vec()))
            };

            if mode == Mode::Blocking {
                channel.send(send_data, real_dst)?;
            } else {
                let mut state = IOState::new(send_data, Operation::Send, channel.get_max_timeout());
                if let Some(ref cb) = callback {
                    state.callback_result = Some(cb.clone());
                }
                let state_arc = Arc::new(Mutex::new(state));
                channel.send_object_async(state_arc, real_dst, mode)?;
            }
        }
    }

    // Phase 2: Broadcast phase
    for i in 0..rounds {
        let partner = trans_peer_id ^ (1 << i);

        if partner < num_peers {
            let real_partner = transform_peer_id(partner, root, num_peers, false);

            if (trans_peer_id & (1 << i)) == 0 {
                // Send full data
                let request = {
                    let src = recvbuf.as_slice();
                    Arc::new(ChannelData::new(src[..total_buffer_size].to_vec()))
                };

                if mode == Mode::Blocking {
                    channel.send(request, real_partner)?;
                } else {
                    let mut state = IOState::new(request, Operation::Send, channel.get_max_timeout());
                    if let Some(ref cb) = callback {
                        state.callback_result = Some(cb.clone());
                    }
                    let state_arc = Arc::new(Mutex::new(state));
                    channel.send_object_async(state_arc, real_partner, mode)?;
                }
            } else {
                // Receive full data
                let request = Arc::new(ChannelData::with_capacity(total_buffer_size));

                if mode == Mode::Blocking {
                    channel.recv(request.clone(), real_partner)?;

                    let src_data = request.as_slice();
                    let mut dst = recvbuf.as_mut_slice();
                    dst[..total_buffer_size].copy_from_slice(&src_data[..total_buffer_size]);
                } else {
                    let mut state = IOState::new(request, Operation::Receive, channel.get_max_timeout());
                    if let Some(ref cb) = callback {
                        state.callback_result = Some(cb.clone());
                    }
                    let state_arc = Arc::new(Mutex::new(state));
                    channel.recv_object_async(state_arc, real_partner, mode)?;
                }
            }
        }
    }

    Ok(())
}

/// Reduce implementation - chooses between LTR and no-order based on function properties
///
/// Corresponds to FMI::Comm::PeerToPeer::reduce() in C++
pub fn reduce_impl<C: PeerToPeerChannel + ?Sized>(
    channel: &C,
    sendbuf: Arc<ChannelData>,
    recvbuf: Arc<ChannelData>,
    root: PeerNum,
    func: &RawFunction,
) -> CylonResult<()> {
    let left_to_right = !(func.commutative && func.associative);
    if left_to_right {
        reduce_ltr(channel, sendbuf, recvbuf, root, func)
    } else {
        reduce_no_order(channel, sendbuf, recvbuf, root, func)
    }
}

/// Left-to-right reduction (for non-associative/non-commutative functions)
fn reduce_ltr<C: PeerToPeerChannel + ?Sized>(
    channel: &C,
    sendbuf: Arc<ChannelData>,
    recvbuf: Arc<ChannelData>,
    root: PeerNum,
    func: &RawFunction,
) -> CylonResult<()> {
    let peer_id = channel.peer_id();
    let num_peers = channel.num_peers();

    if peer_id == root {
        let tmpbuf_len = sendbuf.len * (num_peers as usize);
        let tmpbuf = Arc::new(ChannelData::with_capacity(tmpbuf_len));

        gather_binomial(channel, sendbuf.clone(), tmpbuf.clone(), root)?;

        // Apply function left to right
        {
            let tmp_data = tmpbuf.as_slice();
            let mut result = recvbuf.as_mut_slice();
            result[..sendbuf.len].copy_from_slice(&tmp_data[..sendbuf.len]);

            for i in 1..(num_peers as usize) {
                let offset = i * sendbuf.len;
                (func.f)(&mut result[..sendbuf.len], &tmp_data[offset..offset + sendbuf.len]);
            }
        }
    } else {
        let tmpdata = Arc::new(ChannelData::empty());
        gather_binomial(channel, sendbuf, tmpdata, root)?;
    }

    Ok(())
}

/// Binomial tree reduction (for associative+commutative functions)
fn reduce_no_order<C: PeerToPeerChannel + ?Sized>(
    channel: &C,
    sendbuf: Arc<ChannelData>,
    recvbuf: Arc<ChannelData>,
    root: PeerNum,
    func: &RawFunction,
) -> CylonResult<()> {
    let peer_id = channel.peer_id();
    let num_peers = channel.num_peers();
    let rounds = ceil_log2(num_peers);
    let trans_peer_id = transform_peer_id(peer_id, root, num_peers, true);

    // Working buffer
    let recvbuf_cpy = if peer_id != root {
        Arc::new(ChannelData::with_capacity(sendbuf.len))
    } else {
        recvbuf.clone()
    };

    for i in 0..rounds {
        let power = 1_i32 << i;
        let src = trans_peer_id + power;

        if trans_peer_id % (1 << (i + 1)) == 0 && src < num_peers {
            let real_src = transform_peer_id(src, root, num_peers, false);
            channel.recv(recvbuf_cpy.clone(), real_src)?;

            // Apply function: sendbuf = f(sendbuf, recvbuf_cpy)
            let recv_data = recvbuf_cpy.as_slice();
            let mut send_data = sendbuf.as_mut_slice();
            (func.f)(&mut send_data[..], &recv_data[..]);
        } else if trans_peer_id % power == 0 && trans_peer_id % (1 << (i + 1)) != 0 {
            let real_dst = transform_peer_id(trans_peer_id - power, root, num_peers, false);
            channel.send(sendbuf.clone(), real_dst)?;
        }
    }

    if peer_id == root {
        let src = sendbuf.as_slice();
        let mut dst = recvbuf_cpy.as_mut_slice();
        dst[..sendbuf.len].copy_from_slice(&src[..sendbuf.len]);
    }

    Ok(())
}

/// Allreduce implementation
///
/// Corresponds to FMI::Comm::PeerToPeer::allreduce() in C++
pub fn allreduce_impl<C: PeerToPeerChannel + ?Sized>(
    channel: &C,
    sendbuf: Arc<ChannelData>,
    recvbuf: Arc<ChannelData>,
    func: &RawFunction,
) -> CylonResult<()> {
    let left_to_right = !(func.commutative && func.associative);
    if left_to_right {
        reduce_impl(channel, sendbuf, recvbuf.clone(), 0, func)?;
        bcast_binomial(channel, recvbuf, 0, Mode::Blocking, None)?;
    } else {
        allreduce_no_order(channel, sendbuf, recvbuf, func)?;
    }
    Ok(())
}

/// Recursive doubling allreduce (for associative+commutative functions)
///
/// Corresponds to FMI::Comm::PeerToPeer::allreduce_no_order() in C++
fn allreduce_no_order<C: PeerToPeerChannel + ?Sized>(
    channel: &C,
    sendbuf: Arc<ChannelData>,
    recvbuf: Arc<ChannelData>,
    func: &RawFunction,
) -> CylonResult<()> {
    let peer_id = channel.peer_id();
    let num_peers = channel.num_peers();
    let rounds = floor_log2(num_peers);
    let nearest_power_two = 1_i32 << rounds;

    // Handle non-power-of-two: first receive from processes >= 2^rounds
    if num_peers > nearest_power_two {
        if peer_id < nearest_power_two && peer_id + nearest_power_two < num_peers {
            channel.recv(recvbuf.clone(), peer_id + nearest_power_two)?;
            let recv_data = recvbuf.as_slice();
            let mut send_data = sendbuf.as_mut_slice();
            (func.f)(&mut send_data[..], &recv_data[..]);
        } else if peer_id >= nearest_power_two {
            channel.send(sendbuf.clone(), peer_id - nearest_power_two)?;
        }
    }

    if peer_id < nearest_power_two {
        // Recursive doubling
        for i in 0..rounds {
            let peer = peer_id ^ (1 << i);
            if peer < peer_id {
                channel.send(sendbuf.clone(), peer)?;
                channel.recv(recvbuf.clone(), peer)?;
            } else {
                channel.recv(recvbuf.clone(), peer)?;
                channel.send(sendbuf.clone(), peer)?;
            }
            let recv_data = recvbuf.as_slice();
            let mut send_data = sendbuf.as_mut_slice();
            (func.f)(&mut send_data[..], &recv_data[..]);
        }
    }

    // Send result back to processes >= 2^rounds
    if num_peers > nearest_power_two {
        if peer_id < nearest_power_two && peer_id + nearest_power_two < num_peers {
            channel.send(sendbuf.clone(), peer_id + nearest_power_two)?;
        } else if peer_id >= nearest_power_two {
            channel.recv(sendbuf.clone(), peer_id - nearest_power_two)?;
        }
    }

    // Copy result to recvbuf
    let src = sendbuf.as_slice();
    let mut dst = recvbuf.as_mut_slice();
    dst[..sendbuf.len].copy_from_slice(&src[..sendbuf.len]);

    Ok(())
}

/// Scan implementation
///
/// Corresponds to FMI::Comm::PeerToPeer::scan() in C++
pub fn scan_impl<C: PeerToPeerChannel + ?Sized>(
    channel: &C,
    sendbuf: Arc<ChannelData>,
    recvbuf: Arc<ChannelData>,
    func: &RawFunction,
) -> CylonResult<()> {
    let left_to_right = !(func.commutative && func.associative);
    if left_to_right {
        scan_ltr(channel, sendbuf, recvbuf, func)
    } else {
        scan_no_order(channel, sendbuf, recvbuf, func)
    }
}

/// Linear scan (left-to-right)
fn scan_ltr<C: PeerToPeerChannel + ?Sized>(
    channel: &C,
    sendbuf: Arc<ChannelData>,
    recvbuf: Arc<ChannelData>,
    func: &RawFunction,
) -> CylonResult<()> {
    let peer_id = channel.peer_id();
    let num_peers = channel.num_peers();

    if peer_id == 0 {
        // Improvement: guard against single-peer case
        if num_peers > 1 {
            channel.send(sendbuf.clone(), 1)?;
        }
        let src = sendbuf.as_slice();
        let mut dst = recvbuf.as_mut_slice();
        dst[..sendbuf.len].copy_from_slice(&src[..sendbuf.len]);
    } else {
        channel.recv(recvbuf.clone(), peer_id - 1)?;
        {
            let send_data = sendbuf.as_slice();
            let mut recv_data = recvbuf.as_mut_slice();
            (func.f)(&mut recv_data[..], &send_data[..]);
        }
        if peer_id < num_peers - 1 {
            channel.send(recvbuf.clone(), peer_id + 1)?;
        }
    }

    Ok(())
}

/// Binomial tree scan (up and down phase)
fn scan_no_order<C: PeerToPeerChannel + ?Sized>(
    channel: &C,
    sendbuf: Arc<ChannelData>,
    recvbuf: Arc<ChannelData>,
    func: &RawFunction,
) -> CylonResult<()> {
    let peer_id = channel.peer_id();
    let num_peers = channel.num_peers();
    let rounds = floor_log2(num_peers);

    // Up phase
    for i in 0..rounds {
        let power = 1_i32 << i;
        let power_next = 1_i32 << (i + 1);

        if (peer_id & (power_next - 1)) == power_next - 1 {
            let src = peer_id - power;
            channel.recv(recvbuf.clone(), src)?;
            let recv_data = recvbuf.as_slice();
            let mut send_data = sendbuf.as_mut_slice();
            (func.f)(&mut send_data[..], &recv_data[..]);
        } else if (peer_id & (power - 1)) == power - 1 {
            let dst = peer_id + power;
            if dst < num_peers {
                channel.send(sendbuf.clone(), dst)?;
                // Copy result and return - this peer is done after sending
                let src = sendbuf.as_slice();
                let mut dst = recvbuf.as_mut_slice();
                dst[..sendbuf.len].copy_from_slice(&src[..sendbuf.len]);
                return Ok(());
            }
        }
    }

    // Down phase
    for i in (1..=rounds).rev() {
        let power = 1_i32 << i;
        let power_prev = 1_i32 << (i - 1);

        if (peer_id & (power - 1)) == power - 1 {
            let dst = peer_id + power_prev;
            if dst < num_peers {
                channel.send(sendbuf.clone(), dst)?;
            }
        } else if (peer_id & (power_prev - 1)) == power_prev - 1 {
            let src = peer_id - power_prev;
            if src > 0 {
                channel.recv(recvbuf.clone(), src)?;
                let recv_data = recvbuf.as_slice();
                let mut send_data = sendbuf.as_mut_slice();
                (func.f)(&mut send_data[..], &recv_data[..]);
            }
        }
    }

    // Copy result
    let src = sendbuf.as_slice();
    let mut dst = recvbuf.as_mut_slice();
    dst[..sendbuf.len].copy_from_slice(&src[..sendbuf.len]);

    Ok(())
}
