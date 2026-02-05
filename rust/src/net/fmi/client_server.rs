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

//! Client-Server channel base implementation
//!
//! This module corresponds to cpp/src/cylon/thridparty/fmi/comm/ClientServer.hpp/cpp
//!
//! Client-Server channels use a storage backend (Redis, S3) for communication.
//! Messages are stored with structured key names based on sender, recipient, and operation.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use crate::error::{CylonError, CylonResult, Code};
use super::channel::Channel;
use super::common::*;

/// Trait for storage backends (Redis, S3)
///
/// Implementations must provide object upload/download/delete operations.
pub trait StorageBackend {
    /// Upload data with a given key (blocking)
    fn upload_object(&self, data: &[u8], name: &str) -> CylonResult<()>;

    /// Download data with a given key (blocking). Returns Ok(true) if found, Ok(false) if not exists.
    fn download_object(&self, buf: &mut [u8], name: &str) -> CylonResult<bool>;

    /// Delete an object by key
    fn delete_object(&self, name: &str) -> CylonResult<()>;

    /// List all object names (keys) in the storage
    fn get_object_names(&self) -> CylonResult<Vec<String>>;

    /// Start an async upload operation
    fn upload_object_async(
        &self,
        data: Arc<ChannelData>,
        name: String,
        context: Option<Arc<FmiContext>>,
        callback: Option<NbxCallback>,
    ) -> CylonResult<()>;

    /// Start an async download operation
    fn download_object_async(
        &self,
        buf: Arc<ChannelData>,
        name: String,
        context: Option<Arc<FmiContext>>,
        callback: Option<NbxCallback>,
    ) -> CylonResult<()>;

    /// Process pending async operations (non-blocking poll)
    fn process_pending_operations(&self) -> EventProcessStatus;

    /// Check if there are pending operations
    fn has_pending_operations(&self) -> bool;

    /// Get timeout in milliseconds (backoff between retries)
    fn get_timeout(&self) -> i32;

    /// Get max timeout in milliseconds
    fn get_max_timeout(&self) -> i32;
}

/// Client-Server channel implementation
///
/// This struct wraps a storage backend and provides the Channel trait implementation.
/// Single-threaded design matching C++ FMI model.
pub struct ClientServer<S: StorageBackend> {
    // Configuration
    peer_id: PeerNum,
    num_peers: PeerNum,
    comm_name: String,
    redis_host: String,
    redis_port: i32,

    // Storage backend
    storage: S,

    // Operation counters for unique key generation
    num_operations: HashMap<String, u32>,

    // Track created objects for cleanup
    created_objects: Vec<String>,
}

impl<S: StorageBackend> ClientServer<S> {
    /// Create a new ClientServer channel with the given storage backend
    pub fn new(storage: S) -> Self {
        Self {
            peer_id: -1,
            num_peers: 0,
            comm_name: String::new(),
            redis_host: String::new(),
            redis_port: -1,
            storage,
            num_operations: HashMap::new(),
            created_objects: Vec::new(),
        }
    }

    /// Generate key name for send operation
    fn process_sends(&mut self, dest: PeerNum) -> String {
        let key = format!("send{}", dest);
        let op_num = self.num_operations.entry(key).or_insert(0);
        let file_name = format!("{}{}_{}_{}",
            self.comm_name, self.peer_id, dest, *op_num);
        *op_num += 1;
        file_name
    }

    /// Generate key name for recv operation
    fn process_received(&mut self, src: PeerNum) -> String {
        let key = format!("recv{}", src);
        let op_num = self.num_operations.entry(key).or_insert(0);
        let file_name = format!("{}{}_{}_{}",
            self.comm_name, src, self.peer_id, *op_num);
        *op_num += 1;
        file_name
    }

    /// Upload data and track for cleanup (blocking)
    fn upload(&mut self, buf: Arc<ChannelData>, name: &str) -> CylonResult<()> {
        self.created_objects.push(name.to_string());
        let data = buf.as_slice();
        self.storage.upload_object(&data, name)
    }

    /// Download data with retry until found or timeout (blocking)
    fn download(&self, buf: Arc<ChannelData>, name: &str) -> CylonResult<()> {
        let timeout = self.storage.get_timeout() as u64;
        let max_timeout = self.storage.get_max_timeout() as u64;
        let mut elapsed_time = 0u64;

        while elapsed_time < max_timeout {
            let mut data = buf.as_mut_slice();
            if self.storage.download_object(&mut data, name)? {
                return Ok(());
            }
            elapsed_time += timeout;
            thread::sleep(Duration::from_millis(timeout));
        }

        Err(CylonError::new(
            Code::ExecutionError,
            format!("Timeout waiting for object: {}", name),
        ))
    }

    /// Get storage backend reference
    pub fn storage(&self) -> &S {
        &self.storage
    }

    /// Get mutable storage backend reference
    pub fn storage_mut(&mut self) -> &mut S {
        &mut self.storage
    }
}

impl<S: StorageBackend + Send + Sync + 'static> Channel for ClientServer<S> {
    fn set_peer_id(&mut self, peer_id: PeerNum) {
        self.peer_id = peer_id;
    }

    fn set_num_peers(&mut self, num_peers: PeerNum) {
        self.num_peers = num_peers;
    }

    fn set_comm_name(&mut self, comm_name: &str) {
        self.comm_name = comm_name.to_string();
    }

    fn set_redis_host(&mut self, host: &str) {
        self.redis_host = host.to_string();
    }

    fn set_redis_port(&mut self, port: i32) {
        self.redis_port = port;
    }

    fn peer_id(&self) -> PeerNum {
        self.peer_id
    }

    fn num_peers(&self) -> PeerNum {
        self.num_peers
    }

    fn comm_name(&self) -> &str {
        &self.comm_name
    }

    fn get_max_timeout(&self) -> i32 {
        self.storage.get_max_timeout()
    }

    fn send(&self, buf: Arc<ChannelData>, dest: PeerNum) -> CylonResult<()> {
        // Note: This is a limitation - we need &mut self for process_sends
        // For now, we construct the filename directly
        let key = format!("send{}", dest);
        let op_num = 0u32; // This won't track properly without mut
        let file_name = format!("{}{}_{}_{}",
            self.comm_name, self.peer_id, dest, op_num);

        let data = buf.as_slice();
        self.storage.upload_object(&data, &file_name)
    }

    fn send_async(
        &self,
        buf: Arc<ChannelData>,
        dest: PeerNum,
        context: Option<Arc<FmiContext>>,
        _mode: Mode,
        callback: Option<NbxCallback>,
    ) -> CylonResult<()> {
        let key = format!("send{}", dest);
        let op_num = 0u32;
        let file_name = format!("{}{}_{}_{}",
            self.comm_name, self.peer_id, dest, op_num);

        self.storage.upload_object_async(buf, file_name, context, callback)
    }

    fn recv(&self, buf: Arc<ChannelData>, src: PeerNum) -> CylonResult<()> {
        let key = format!("recv{}", src);
        let op_num = 0u32;
        let file_name = format!("{}{}_{}_{}",
            self.comm_name, src, self.peer_id, op_num);

        self.download(buf, &file_name)
    }

    fn recv_async(
        &self,
        buf: Arc<ChannelData>,
        src: PeerNum,
        context: Option<Arc<FmiContext>>,
        _mode: Mode,
        callback: Option<NbxCallback>,
    ) -> CylonResult<()> {
        let key = format!("recv{}", src);
        let op_num = 0u32;
        let file_name = format!("{}{}_{}_{}",
            self.comm_name, src, self.peer_id, op_num);

        self.storage.download_object_async(buf, file_name, context, callback)
    }

    fn channel_event_progress(&self, _op: Operation) -> EventProcessStatus {
        self.storage.process_pending_operations()
    }

    fn bcast_async(
        &self,
        buf: Arc<ChannelData>,
        root: PeerNum,
        _mode: Mode,
        _callback: Option<NbxCallback>,
    ) -> CylonResult<()> {
        let op_num = 0u32;
        let file_name = format!("{}{}_bcast_{}", self.comm_name, root, op_num);

        if self.peer_id == root {
            let data = buf.as_slice();
            self.storage.upload_object(&data, &file_name)
        } else {
            self.download(buf, &file_name)
        }
    }

    fn barrier(&self) -> CylonResult<()> {
        let timeout = self.storage.get_timeout() as u64;
        let max_timeout = self.storage.get_max_timeout() as u64;

        let barrier_num = 0u32;
        let barrier_suffix = format!("_barrier_{}", barrier_num);
        let file_name = format!("{}{}{}", self.comm_name, self.peer_id, barrier_suffix);

        // Upload marker
        let marker = Arc::new(ChannelData::new(vec![1u8]));
        let data = marker.as_slice();
        self.storage.upload_object(&data, &file_name)?;

        // Wait for all peers
        let mut elapsed_time = 0u64;
        while elapsed_time < max_timeout {
            let objects = self.storage.get_object_names()?;
            let num_arrived = objects.iter()
                .filter(|s| s.ends_with(&barrier_suffix))
                .count();

            if num_arrived >= self.num_peers as usize {
                return Ok(());
            }

            elapsed_time += timeout;
            thread::sleep(Duration::from_millis(timeout));
        }

        Err(CylonError::new(Code::ExecutionError, "Barrier timeout"))
    }

    fn gatherv_async(
        &self,
        sendbuf: Arc<ChannelData>,
        recvbuf: Arc<ChannelData>,
        root: PeerNum,
        recvcounts: &[i32],
        displs: &[i32],
        _mode: Mode,
        _callback: Option<NbxCallback>,
    ) -> CylonResult<()> {
        if self.peer_id != root {
            self.send(sendbuf, root)?;
        } else {
            // Root: copy own data
            {
                let src = sendbuf.as_slice();
                let mut dst = recvbuf.as_mut_slice();
                let offset = displs[root as usize] as usize;
                let count = recvcounts[root as usize] as usize;
                dst[offset..offset + count].copy_from_slice(&src[..count]);
            }

            // Receive from all other peers
            for i in 0..self.num_peers {
                if i != root {
                    let offset = displs[i as usize] as usize;
                    let count = recvcounts[i as usize] as usize;
                    let peer_data = Arc::new(ChannelData::with_capacity(count));
                    self.recv(peer_data.clone(), i)?;

                    let src = peer_data.as_slice();
                    let mut dst = recvbuf.as_mut_slice();
                    dst[offset..offset + count].copy_from_slice(&src[..count]);
                }
            }
        }
        Ok(())
    }

    fn allgather_async(
        &self,
        sendbuf: Arc<ChannelData>,
        recvbuf: Arc<ChannelData>,
        root: PeerNum,
        _mode: Mode,
        _callback: Option<NbxCallback>,
    ) -> CylonResult<()> {
        self.gather(sendbuf, recvbuf.clone(), root)?;
        self.bcast(recvbuf, root)
    }

    fn allgatherv_async(
        &self,
        sendbuf: Arc<ChannelData>,
        recvbuf: Arc<ChannelData>,
        root: PeerNum,
        recvcounts: &[i32],
        displs: &[i32],
        mode: Mode,
        callback: Option<NbxCallback>,
    ) -> CylonResult<()> {
        self.gatherv_async(sendbuf, recvbuf.clone(), root, recvcounts, displs, mode, callback)?;
        self.bcast(recvbuf, root)
    }

    fn reduce(
        &self,
        sendbuf: Arc<ChannelData>,
        recvbuf: Arc<ChannelData>,
        root: PeerNum,
        func: &RawFunction,
    ) -> CylonResult<()> {
        let timeout = self.storage.get_timeout() as u64;
        let max_timeout = self.storage.get_max_timeout() as u64;
        let reduce_num = 0u32;

        if self.peer_id == root {
            let left_to_right = !(func.commutative && func.associative);
            let buffer_length = sendbuf.len;
            let mut received = vec![false; self.num_peers as usize];
            let mut applied = vec![false; self.num_peers as usize];
            let mut data = vec![0u8; buffer_length * self.num_peers as usize];

            // Copy own data
            {
                let src = sendbuf.as_slice();
                let mut dst = recvbuf.as_mut_slice();
                dst[..buffer_length].copy_from_slice(&src[..buffer_length]);
            }
            received[root as usize] = true;
            applied[root as usize] = true;

            let mut elapsed_time = 0u64;
            while elapsed_time < max_timeout && applied.iter().any(|&v| !v) {
                // Try to receive from all peers
                for i in 0..self.num_peers {
                    if received[i as usize] {
                        continue;
                    }
                    let file_name = format!("{}{}_reduce_{}", self.comm_name, i, reduce_num);
                    let offset = (i as usize) * buffer_length;
                    let download_slice = &mut data[offset..offset + buffer_length];

                    if self.storage.download_object(download_slice, &file_name)? {
                        received[i as usize] = true;
                    }
                }

                // Apply function where possible
                let mut all_left_applied = true;
                for i in 0..self.num_peers as usize {
                    if received[i] && !applied[i] && (!left_to_right || all_left_applied) {
                        let offset = i * buffer_length;
                        let src = &data[offset..offset + buffer_length];
                        let mut dst = recvbuf.as_mut_slice();
                        (func.f)(&mut dst, src);
                        applied[i] = true;
                    } else if !received[i] {
                        all_left_applied = false;
                    }
                }

                elapsed_time += timeout;
                thread::sleep(Duration::from_millis(timeout));
            }

            if applied.iter().any(|&v| !v) {
                return Err(CylonError::new(Code::ExecutionError, "Reduce timeout"));
            }
        } else {
            let file_name = format!("{}{}_reduce_{}", self.comm_name, self.peer_id, reduce_num);
            let data = sendbuf.as_slice();
            self.storage.upload_object(&data, &file_name)?;
        }

        Ok(())
    }

    fn scan(
        &self,
        sendbuf: Arc<ChannelData>,
        recvbuf: Arc<ChannelData>,
        func: &RawFunction,
    ) -> CylonResult<()> {
        let timeout = self.storage.get_timeout() as u64;
        let max_timeout = self.storage.get_max_timeout() as u64;
        let scan_num = 0u32;

        // Upload own data (except last peer)
        if self.peer_id != self.num_peers - 1 {
            let file_name = format!("{}{}_scan_{}", self.comm_name, self.peer_id, scan_num);
            let data = sendbuf.as_slice();
            self.storage.upload_object(&data, &file_name)?;
        }

        let left_to_right = !(func.commutative && func.associative);
        let num_data = (self.peer_id + 1) as usize;
        let buffer_length = sendbuf.len;
        let mut received = vec![false; num_data];
        let mut applied = vec![false; num_data];
        let mut data = vec![0u8; buffer_length * num_data];

        // Copy own data
        {
            let src = sendbuf.as_slice();
            let mut dst = recvbuf.as_mut_slice();
            dst[..buffer_length].copy_from_slice(&src[..buffer_length]);
        }
        received[self.peer_id as usize] = true;
        applied[self.peer_id as usize] = true;

        let mut elapsed_time = 0u64;
        while elapsed_time < max_timeout && applied.iter().any(|&v| !v) {
            // Receive all values
            for i in 0..num_data {
                if received[i] {
                    continue;
                }
                let file_name = format!("{}{}_scan_{}", self.comm_name, i, scan_num);
                let offset = i * buffer_length;
                let download_slice = &mut data[offset..offset + buffer_length];

                if self.storage.download_object(download_slice, &file_name)? {
                    received[i] = true;
                }
            }

            // Apply function
            let mut all_left_applied = true;
            for i in 0..num_data {
                if received[i] && !applied[i] && (!left_to_right || all_left_applied) {
                    let offset = i * buffer_length;
                    let src = &data[offset..offset + buffer_length];
                    let mut dst = recvbuf.as_mut_slice();
                    (func.f)(&mut dst, src);
                    applied[i] = true;
                } else if !received[i] {
                    all_left_applied = false;
                }
            }

            elapsed_time += timeout;
            thread::sleep(Duration::from_millis(timeout));
        }

        if applied.iter().any(|&v| !v) {
            return Err(CylonError::new(Code::ExecutionError, "Scan timeout"));
        }

        Ok(())
    }

    fn finalize(&mut self) -> CylonResult<()> {
        for name in &self.created_objects {
            let _ = self.storage.delete_object(name);
        }
        self.created_objects.clear();
        Ok(())
    }
}