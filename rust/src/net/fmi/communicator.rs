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

//! FMI Communicator implementation
//!
//! This module corresponds to cpp/src/cylon/thridparty/fmi/Communicator.hpp/cpp
//!
//! The Communicator provides a high-level interface for distributed communication,
//! wrapping the underlying Channel implementation.

use std::sync::Arc;
use crate::error::{CylonError, CylonResult, Code};
use super::common::*;
use super::channel::Channel;
use super::direct::Direct;

/// FMI Communicator
///
/// This struct wraps a Channel and provides high-level communication operations.
/// It matches the FMI::Communicator class from the C++ implementation.
pub struct Communicator {
    peer_id: PeerNum,
    num_peers: PeerNum,
    comm_name: String,
    channel: Box<dyn Channel>,
}

impl Communicator {
    /// Create a new Communicator
    ///
    /// # Arguments
    /// * `peer_id` - Initial peer ID (may be overridden by Redis)
    /// * `num_peers` - Total number of peers
    /// * `backend` - Backend configuration
    /// * `comm_name` - Communicator name
    /// * `redis_host` - Redis host for coordination (empty to disable)
    /// * `redis_port` - Redis port (0 or negative to disable)
    /// * `redis_namespace` - Redis namespace for keys
    pub fn new(
        mut peer_id: PeerNum,
        num_peers: PeerNum,
        backend: &DirectBackend,
        comm_name: &str,
        redis_host: &str,
        redis_port: i32,
        redis_namespace: &str,
    ) -> CylonResult<Self> {
        // Override rank from Redis if configured
        #[cfg(feature = "redis")]
        {
            if redis_port > 0 && !redis_host.is_empty() {
                let client = redis::Client::open(format!("redis://{}:{}", redis_host, redis_port))
                    .map_err(|e| CylonError::new(Code::IoError, format!("Redis connection failed: {}", e)))?;
                let mut conn = client.get_connection()
                    .map_err(|e| CylonError::new(Code::IoError, format!("Redis connection failed: {}", e)))?;

                let key = if !redis_namespace.is_empty() {
                    format!("{}_num_cur_processes", redis_namespace)
                } else {
                    "num_cur_processes".to_string()
                };

                let num_cur_processes: i32 = redis::cmd("INCR")
                    .arg(&key)
                    .query(&mut conn)
                    .map_err(|e| CylonError::new(Code::IoError, format!("Redis INCR failed: {}", e)))?;

                peer_id = num_cur_processes - 1;
                log::info!("Current rank from Redis: {}", peer_id);
            }
        }

        // Create channel based on backend type
        let mut channel: Box<dyn Channel> = match backend.get_backend_type() {
            BackendType::Direct => Box::new(Direct::new(backend)),
            _ => {
                return Err(CylonError::new(
                    Code::Invalid,
                    "Only Direct backend is currently supported".to_string(),
                ));
            }
        };

        // Configure channel
        channel.set_redis_host(redis_host);
        channel.set_redis_port(redis_port);
        channel.set_peer_id(peer_id);
        channel.set_num_peers(num_peers);
        channel.set_comm_name(comm_name);

        // Initialize channel
        channel.init()?;

        Ok(Self {
            peer_id,
            num_peers,
            comm_name: comm_name.to_string(),
            channel,
        })
    }

    /// Create a Communicator with default Redis namespace
    pub fn with_backend(
        peer_id: PeerNum,
        num_peers: PeerNum,
        backend: &DirectBackend,
        comm_name: &str,
    ) -> CylonResult<Self> {
        Self::new(peer_id, num_peers, backend, comm_name, "", -1, "")
    }

    /// Get the peer ID
    pub fn get_peer_id(&self) -> PeerNum {
        self.peer_id
    }

    /// Get the number of peers
    pub fn get_num_peers(&self) -> PeerNum {
        self.num_peers
    }

    /// Get the communicator name
    pub fn get_comm_name(&self) -> &str {
        &self.comm_name
    }

    /// Get a reference to the underlying channel
    pub fn channel(&self) -> &dyn Channel {
        self.channel.as_ref()
    }

    // =========================================================================
    // High-level communication operations
    // =========================================================================

    /// Send data to a peer
    pub fn send(&self, data: &[u8], dest: PeerNum) -> CylonResult<()> {
        let buf = Arc::new(ChannelData::from_slice(data));
        self.channel.send(buf, dest)
    }

    /// Receive data from a peer
    pub fn recv(&self, data: &mut [u8], src: PeerNum) -> CylonResult<()> {
        let buf = Arc::new(ChannelData::with_capacity(data.len()));
        self.channel.recv(buf.clone(), src)?;
        let received = buf.as_slice();
        data.copy_from_slice(&received[..data.len()]);
        Ok(())
    }

    /// Broadcast data from root to all peers
    pub fn bcast(&self, data: &mut [u8], root: PeerNum) -> CylonResult<()> {
        let buf = Arc::new(ChannelData::from_slice(data));
        self.channel.bcast(buf.clone(), root)?;
        if self.peer_id != root {
            let received = buf.as_slice();
            data.copy_from_slice(&received[..data.len()]);
        }
        Ok(())
    }

    /// Barrier synchronization
    pub fn barrier(&self) -> CylonResult<()> {
        self.channel.barrier()
    }

    /// Gather data from all peers to root
    pub fn gather(
        &self,
        sendbuf: &[u8],
        recvbuf: &mut [u8],
        root: PeerNum,
    ) -> CylonResult<()> {
        let send = Arc::new(ChannelData::from_slice(sendbuf));
        let recv = Arc::new(ChannelData::with_capacity(recvbuf.len()));
        self.channel.gather(send, recv.clone(), root)?;
        if self.peer_id == root {
            let received = recv.as_slice();
            recvbuf.copy_from_slice(&received[..recvbuf.len()]);
        }
        Ok(())
    }

    /// Scatter data from root to all peers
    pub fn scatter(
        &self,
        sendbuf: &[u8],
        recvbuf: &mut [u8],
        root: PeerNum,
    ) -> CylonResult<()> {
        let send = Arc::new(ChannelData::from_slice(sendbuf));
        let recv = Arc::new(ChannelData::with_capacity(recvbuf.len()));
        self.channel.scatter(send, recv.clone(), root)?;
        let received = recv.as_slice();
        recvbuf.copy_from_slice(&received[..recvbuf.len()]);
        Ok(())
    }

    /// Allgather data from all peers
    pub fn allgather(
        &self,
        sendbuf: &[u8],
        recvbuf: &mut [u8],
        root: PeerNum,
    ) -> CylonResult<()> {
        let send = Arc::new(ChannelData::from_slice(sendbuf));
        let recv = Arc::new(ChannelData::with_capacity(recvbuf.len()));
        self.channel.allgather(send, recv.clone(), root)?;
        let received = recv.as_slice();
        recvbuf.copy_from_slice(&received[..recvbuf.len()]);
        Ok(())
    }

    /// Gather variable-sized data from all peers to root (blocking)
    pub fn gatherv(
        &self,
        sendbuf: &[u8],
        recvbuf: &mut [u8],
        root: PeerNum,
        recvcounts: &[i32],
        displs: &[i32],
    ) -> CylonResult<()> {
        let send = Arc::new(ChannelData::from_slice(sendbuf));
        let recv = Arc::new(ChannelData::with_capacity(recvbuf.len()));
        self.channel.gatherv(send, recv.clone(), root, recvcounts, displs)?;
        if self.peer_id == root {
            let received = recv.as_slice();
            recvbuf.copy_from_slice(&received[..recvbuf.len()]);
        }
        Ok(())
    }

    /// Gather variable-sized data from all peers to root (with mode and callback)
    pub fn gatherv_async(
        &self,
        sendbuf: &[u8],
        recvbuf: &mut [u8],
        root: PeerNum,
        recvcounts: &[i32],
        displs: &[i32],
        mode: Mode,
        callback: Option<NbxCallback>,
    ) -> CylonResult<()> {
        let send = Arc::new(ChannelData::from_slice(sendbuf));
        let recv = Arc::new(ChannelData::with_capacity(recvbuf.len()));
        self.channel.gatherv_async(send, recv.clone(), root, recvcounts, displs, mode, callback)?;
        if self.peer_id == root {
            let received = recv.as_slice();
            recvbuf.copy_from_slice(&received[..recvbuf.len()]);
        }
        Ok(())
    }

    /// Allgather variable-sized data from all peers (blocking)
    pub fn allgatherv(
        &self,
        sendbuf: &[u8],
        recvbuf: &mut [u8],
        root: PeerNum,
        recvcounts: &[i32],
        displs: &[i32],
    ) -> CylonResult<()> {
        let send = Arc::new(ChannelData::from_slice(sendbuf));
        let recv = Arc::new(ChannelData::with_capacity(recvbuf.len()));
        self.channel.allgatherv(send, recv.clone(), root, recvcounts, displs)?;
        let received = recv.as_slice();
        recvbuf.copy_from_slice(&received[..recvbuf.len()]);
        Ok(())
    }

    /// Allgather variable-sized data from all peers (with mode and callback)
    pub fn allgatherv_async(
        &self,
        sendbuf: &[u8],
        recvbuf: &mut [u8],
        root: PeerNum,
        recvcounts: &[i32],
        displs: &[i32],
        mode: Mode,
        callback: Option<NbxCallback>,
    ) -> CylonResult<()> {
        let send = Arc::new(ChannelData::from_slice(sendbuf));
        let recv = Arc::new(ChannelData::with_capacity(recvbuf.len()));
        self.channel.allgatherv_async(send, recv.clone(), root, recvcounts, displs, mode, callback)?;
        let received = recv.as_slice();
        recvbuf.copy_from_slice(&received[..recvbuf.len()]);
        Ok(())
    }

    /// Broadcast with mode (blocking or non-blocking)
    pub fn bcast_async(
        &self,
        data: &mut [u8],
        root: PeerNum,
        mode: Mode,
        callback: Option<NbxCallback>,
    ) -> CylonResult<()> {
        let buf = Arc::new(ChannelData::from_slice(data));
        self.channel.bcast_async(buf.clone(), root, mode, callback)?;
        if self.peer_id != root {
            let received = buf.as_slice();
            data.copy_from_slice(&received[..data.len()]);
        }
        Ok(())
    }

    /// Progress event processing for non-blocking operations
    pub fn communicator_event_progress(&self, op: Operation) -> EventProcessStatus {
        self.channel.channel_event_progress(op)
    }

    /// Reduce operation
    pub fn reduce<F>(
        &self,
        sendbuf: &[u8],
        recvbuf: &mut [u8],
        root: PeerNum,
        func: F,
        associative: bool,
        commutative: bool,
    ) -> CylonResult<()>
    where
        F: Fn(&mut [u8], &[u8]) + Send + Sync + 'static,
    {
        let send = Arc::new(ChannelData::from_slice(sendbuf));
        let recv = Arc::new(ChannelData::with_capacity(recvbuf.len()));
        let raw_func = RawFunction::new(func, associative, commutative);
        self.channel.reduce(send, recv.clone(), root, &raw_func)?;
        if self.peer_id == root {
            let received = recv.as_slice();
            recvbuf.copy_from_slice(&received[..recvbuf.len()]);
        }
        Ok(())
    }

    /// Allreduce operation
    pub fn allreduce<F>(
        &self,
        sendbuf: &[u8],
        recvbuf: &mut [u8],
        func: F,
        associative: bool,
        commutative: bool,
    ) -> CylonResult<()>
    where
        F: Fn(&mut [u8], &[u8]) + Send + Sync + 'static,
    {
        let send = Arc::new(ChannelData::from_slice(sendbuf));
        let recv = Arc::new(ChannelData::with_capacity(recvbuf.len()));
        let raw_func = RawFunction::new(func, associative, commutative);
        self.channel.allreduce(send, recv.clone(), &raw_func)?;
        let received = recv.as_slice();
        recvbuf.copy_from_slice(&received[..recvbuf.len()]);
        Ok(())
    }

    /// Prefix scan operation
    pub fn scan<F>(
        &self,
        sendbuf: &[u8],
        recvbuf: &mut [u8],
        func: F,
        associative: bool,
        commutative: bool,
    ) -> CylonResult<()>
    where
        F: Fn(&mut [u8], &[u8]) + Send + Sync + 'static,
    {
        let send = Arc::new(ChannelData::from_slice(sendbuf));
        let recv = Arc::new(ChannelData::with_capacity(recvbuf.len()));
        let raw_func = RawFunction::new(func, associative, commutative);
        self.channel.scan(send, recv.clone(), &raw_func)?;
        let received = recv.as_slice();
        recvbuf.copy_from_slice(&received[..recvbuf.len()]);
        Ok(())
    }
}

impl Drop for Communicator {
    fn drop(&mut self) {
        let _ = self.channel.finalize();
    }
}
