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

//! Channel base trait and implementations
//!
//! This module corresponds to cpp/src/cylon/thridparty/fmi/comm/Channel.hpp/cpp

use std::sync::Arc;
use crate::error::CylonResult;
use super::common::*;

/// Channel trait - base interface for all FMI communication channels
///
/// This trait matches the FMI::Comm::Channel class from the C++ implementation.
/// Channels provide low-level communication primitives that the FMI Communicator
/// builds upon.
pub trait Channel: Send + Sync {
    // =========================================================================
    // Configuration methods (setters)
    // =========================================================================

    fn set_peer_id(&mut self, peer_id: PeerNum);
    fn set_num_peers(&mut self, num_peers: PeerNum);
    fn set_comm_name(&mut self, comm_name: &str);
    fn set_redis_host(&mut self, host: &str);
    fn set_redis_port(&mut self, port: i32);

    // =========================================================================
    // Configuration methods (getters)
    // =========================================================================

    fn peer_id(&self) -> PeerNum;
    fn num_peers(&self) -> PeerNum;
    fn comm_name(&self) -> &str;

    // =========================================================================
    // Lifecycle methods
    // =========================================================================

    /// Initialize the channel
    fn init(&mut self) -> CylonResult<()> {
        Ok(())
    }

    /// Finalize the channel (cleanup)
    fn finalize(&mut self) -> CylonResult<()> {
        Ok(())
    }

    /// Get maximum timeout in milliseconds
    fn get_max_timeout(&self) -> i32 {
        -1
    }

    // =========================================================================
    // Point-to-point operations
    // =========================================================================

    /// Send data to a peer (blocking)
    fn send(&self, buf: Arc<ChannelData>, dest: PeerNum) -> CylonResult<()>;

    /// Send data to a peer (with mode and callback)
    fn send_async(
        &self,
        buf: Arc<ChannelData>,
        dest: PeerNum,
        context: Option<&mut FmiContext>,
        mode: Mode,
        callback: Option<NbxCallback>,
    ) -> CylonResult<()>;

    /// Receive data from a peer (blocking)
    fn recv(&self, buf: Arc<ChannelData>, src: PeerNum) -> CylonResult<()>;

    /// Receive data from a peer (with mode and callback)
    fn recv_async(
        &self,
        buf: Arc<ChannelData>,
        src: PeerNum,
        context: Option<&mut FmiContext>,
        mode: Mode,
        callback: Option<NbxCallback>,
    ) -> CylonResult<()>;

    /// Check if it's OK to receive from a peer
    fn check_receive(&self, _dest: PeerNum, _mode: Mode) -> bool {
        true
    }

    /// Check if it's OK to send to a peer
    fn check_send(&self, _dest: PeerNum, _mode: Mode) -> bool {
        true
    }

    /// Progress event processing for non-blocking operations
    fn channel_event_progress(&self, _op: Operation) -> EventProcessStatus {
        EventProcessStatus::Noop
    }

    // =========================================================================
    // Collective operations
    // =========================================================================

    /// Broadcast data from root to all peers (blocking)
    fn bcast(&self, buf: Arc<ChannelData>, root: PeerNum) -> CylonResult<()> {
        self.bcast_async(buf, root, Mode::Blocking, None)
    }

    /// Broadcast data from root to all peers (with mode)
    fn bcast_async(
        &self,
        buf: Arc<ChannelData>,
        root: PeerNum,
        mode: Mode,
        callback: Option<NbxCallback>,
    ) -> CylonResult<()>;

    /// Barrier synchronization
    fn barrier(&self) -> CylonResult<()>;

    /// Gather data from all peers to root (blocking)
    ///
    /// Default implementation using send/recv
    fn gather(
        &self,
        sendbuf: Arc<ChannelData>,
        recvbuf: Arc<ChannelData>,
        root: PeerNum,
    ) -> CylonResult<()> {
        let peer_id = self.peer_id();
        let num_peers = self.num_peers();
        let buffer_length = sendbuf.len;

        if peer_id != root {
            self.send(sendbuf, root)?;
        } else {
            // Root: copy own data and receive from others
            {
                let src = sendbuf.as_slice();
                let mut dst = recvbuf.as_mut_slice();
                let offset = (root as usize) * buffer_length;
                dst[offset..offset + buffer_length].copy_from_slice(&src[..buffer_length]);
            }

            for i in 0..num_peers {
                if i != root {
                    let offset = (i as usize) * buffer_length;
                    let peer_data = Arc::new(ChannelData::with_capacity(buffer_length));
                    self.recv(peer_data.clone(), i)?;

                    let src = peer_data.as_slice();
                    let mut dst = recvbuf.as_mut_slice();
                    dst[offset..offset + buffer_length].copy_from_slice(&src[..buffer_length]);
                }
            }
        }
        Ok(())
    }

    /// Gather variable-sized data from all peers to root (blocking)
    fn gatherv(
        &self,
        sendbuf: Arc<ChannelData>,
        recvbuf: Arc<ChannelData>,
        root: PeerNum,
        recvcounts: &[i32],
        displs: &[i32],
    ) -> CylonResult<()> {
        self.gatherv_async(sendbuf, recvbuf, root, recvcounts, displs, Mode::Blocking, None)
    }

    /// Gather variable-sized data (with mode)
    fn gatherv_async(
        &self,
        sendbuf: Arc<ChannelData>,
        recvbuf: Arc<ChannelData>,
        root: PeerNum,
        recvcounts: &[i32],
        displs: &[i32],
        mode: Mode,
        callback: Option<NbxCallback>,
    ) -> CylonResult<()>;

    /// Scatter data from root to all peers (blocking)
    ///
    /// Default implementation using send/recv
    fn scatter(
        &self,
        sendbuf: Arc<ChannelData>,
        recvbuf: Arc<ChannelData>,
        root: PeerNum,
    ) -> CylonResult<()> {
        let peer_id = self.peer_id();
        let num_peers = self.num_peers();
        let buffer_length = recvbuf.len;

        if peer_id == root {
            // Root: send to all peers
            for i in 0..num_peers {
                let offset = (i as usize) * buffer_length;
                let send_slice = {
                    let src = sendbuf.as_slice();
                    src[offset..offset + buffer_length].to_vec()
                };
                let peer_data = Arc::new(ChannelData::new(send_slice));

                if i == root {
                    // Copy to own recvbuf
                    let src = peer_data.as_slice();
                    let mut dst = recvbuf.as_mut_slice();
                    dst[..buffer_length].copy_from_slice(&src[..buffer_length]);
                } else {
                    self.send(peer_data, i)?;
                }
            }
        } else {
            self.recv(recvbuf, root)?;
        }
        Ok(())
    }

    /// Allgather - gather data from all peers and distribute to all (blocking)
    fn allgather(
        &self,
        sendbuf: Arc<ChannelData>,
        recvbuf: Arc<ChannelData>,
        root: PeerNum,
    ) -> CylonResult<()> {
        self.allgather_async(sendbuf, recvbuf, root, Mode::Blocking, None)
    }

    /// Allgather (with mode)
    fn allgather_async(
        &self,
        sendbuf: Arc<ChannelData>,
        recvbuf: Arc<ChannelData>,
        root: PeerNum,
        mode: Mode,
        callback: Option<NbxCallback>,
    ) -> CylonResult<()>;

    /// Allgather variable-sized data (blocking)
    fn allgatherv(
        &self,
        sendbuf: Arc<ChannelData>,
        recvbuf: Arc<ChannelData>,
        root: PeerNum,
        recvcounts: &[i32],
        displs: &[i32],
    ) -> CylonResult<()> {
        self.allgatherv_async(sendbuf, recvbuf, root, recvcounts, displs, Mode::Blocking, None)
    }

    /// Allgather variable-sized data (with mode)
    fn allgatherv_async(
        &self,
        sendbuf: Arc<ChannelData>,
        recvbuf: Arc<ChannelData>,
        root: PeerNum,
        recvcounts: &[i32],
        displs: &[i32],
        mode: Mode,
        callback: Option<NbxCallback>,
    ) -> CylonResult<()>;

    /// Reduce operation - apply function to all data and store result at root
    fn reduce(
        &self,
        sendbuf: Arc<ChannelData>,
        recvbuf: Arc<ChannelData>,
        root: PeerNum,
        func: &RawFunction,
    ) -> CylonResult<()>;

    /// Allreduce - reduce and distribute result to all peers
    ///
    /// Default implementation: reduce to root 0, then broadcast
    fn allreduce(
        &self,
        sendbuf: Arc<ChannelData>,
        recvbuf: Arc<ChannelData>,
        func: &RawFunction,
    ) -> CylonResult<()> {
        self.reduce(sendbuf, recvbuf.clone(), 0, func)?;
        self.bcast(recvbuf, 0)?;
        Ok(())
    }

    /// Inclusive prefix scan
    fn scan(
        &self,
        sendbuf: Arc<ChannelData>,
        recvbuf: Arc<ChannelData>,
        func: &RawFunction,
    ) -> CylonResult<()>;
}
