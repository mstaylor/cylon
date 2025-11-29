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

//! Direct channel implementation using TCPunch
//!
//! This module corresponds to cpp/src/cylon/thridparty/fmi/comm/Direct.hpp/cpp
//!
//! The Direct channel uses TCPunch for TCP NAT hole punching to establish
//! direct peer-to-peer connections.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

use crate::error::{CylonError, CylonResult, Code};
use super::common::*;
use super::channel::Channel;
use super::peer_to_peer::*;
use super::tcpunch;

/// Connection timeout for hole punching (120 seconds)
const HOLEPUNCH_CONNECT_TIMEOUT: u64 = 120000;

/// Maximum TCPunch retry attempts
const MAX_TCPUNCH_TRIES: i32 = 6;

/// Create a vector of None options for TcpStream (can't use vec![None; n] since TcpStream doesn't implement Clone)
fn create_socket_vec(n: usize) -> Vec<Option<TcpStream>> {
    (0..n).map(|_| None).collect()
}

/// Direct channel implementation
///
/// This channel uses TCPunch for NAT hole punching to establish direct
/// TCP connections between peers.
pub struct Direct {
    // Configuration
    peer_id: PeerNum,
    num_peers: PeerNum,
    comm_name: String,
    hostname: String,
    port: i32,
    mode: Mode,
    enable_ping: bool,
    max_timeout: i32,
    resolve_host_dns: bool,

    // Redis configuration (for coordination)
    redis_host: String,
    redis_port: i32,

    // Socket management
    /// Sockets by mode: Mode -> Vec<socket per peer>
    sockets: RwLock<HashMap<Mode, Vec<Option<TcpStream>>>>,

    /// IO states for non-blocking operations
    io_states: RwLock<HashMap<Operation, HashMap<i32, Arc<Mutex<IOState>>>>>,
}

impl Direct {
    /// Create a new Direct channel from backend configuration
    pub fn new(backend: &DirectBackend) -> Self {
        let mut hostname = backend.get_host().to_string();

        // Resolve DNS if configured
        if backend.resolve_host_dns() {
            if let Ok(addrs) = std::net::ToSocketAddrs::to_socket_addrs(
                &format!("{}:0", hostname)
            ) {
                if let Some(addr) = addrs.into_iter().next() {
                    hostname = addr.ip().to_string();
                    log::info!("Resolved rendezvous DNS: {}", hostname);
                }
            }
        }

        Self {
            peer_id: -1,
            num_peers: 0,
            comm_name: String::new(),
            hostname,
            port: backend.get_port(),
            mode: backend.get_blocking_mode(),
            enable_ping: backend.enable_host_ping(),
            max_timeout: backend.get_max_timeout(),
            resolve_host_dns: backend.resolve_host_dns(),
            redis_host: String::new(),
            redis_port: -1,
            sockets: RwLock::new(HashMap::new()),
            io_states: RwLock::new(HashMap::new()),
        }
    }

    /// Get pairing name for TCPunch
    fn get_pairing_name(&self, a: PeerNum, b: PeerNum, mode: Mode) -> String {
        let min_id = std::cmp::min(a, b);
        let max_id = std::cmp::max(a, b);
        let mode_str = match mode {
            Mode::Blocking => "BLOCKING",
            Mode::NonBlocking => "NONBLOCKING",
        };
        format!("fmi_pair{}_{}{}", min_id, max_id, mode_str)
    }

    /// Ensure socket is connected to partner (lazy connection establishment)
    fn check_socket(&self, partner_id: PeerNum, pair_name: &str) -> CylonResult<()> {
        let mut current_try = 0;

        while current_try < MAX_TCPUNCH_TRIES {
            // Check if socket already exists
            {
                let sockets = self.sockets.read().unwrap();
                if let Some(mode_sockets) = sockets.get(&Mode::Blocking) {
                    if let Some(Some(_)) = mode_sockets.get(partner_id as usize) {
                        return Ok(());
                    }
                }
            }

            // Initialize socket vector if needed
            {
                let mut sockets = self.sockets.write().unwrap();
                let mode_sockets = sockets.entry(Mode::Blocking).or_insert_with(|| {
                    create_socket_vec(self.num_peers as usize)
                });

                if mode_sockets.get(partner_id as usize).map(|s| s.is_some()).unwrap_or(false) {
                    return Ok(());
                }
            }

            // Try to establish connection via TCPunch
            match tcpunch::pair(
                pair_name,
                &self.hostname,
                self.port as u16,
                HOLEPUNCH_CONNECT_TIMEOUT,
            ) {
                Ok(mut stream) => {
                    log::info!("Paired partnerId: {} to pair_name: {}", partner_id, pair_name);

                    // Configure socket
                    let timeout = Duration::from_millis(self.max_timeout as u64);
                    stream.set_read_timeout(Some(timeout)).ok();
                    stream.set_write_timeout(Some(timeout)).ok();
                    stream.set_nodelay(true).ok();

                    // Store socket
                    let mut sockets = self.sockets.write().unwrap();
                    let mode_sockets = sockets.entry(Mode::Blocking).or_insert_with(|| {
                        create_socket_vec(self.num_peers as usize)
                    });
                    mode_sockets[partner_id as usize] = Some(stream);

                    return Ok(());
                }
                Err(e) => {
                    log::warn!(
                        "Socket pairing failed: {} pairName: {} partnerId: {}",
                        e, pair_name, partner_id
                    );

                    // Try to remove stale pairing
                    let _ = tcpunch::remove_pair(
                        &format!("remove_pair_{}", pair_name),
                        &self.hostname,
                        self.port as u16,
                        HOLEPUNCH_CONNECT_TIMEOUT,
                    );

                    current_try += 1;
                    if current_try == MAX_TCPUNCH_TRIES {
                        return Err(CylonError::new(
                            Code::IoError,
                            format!("Failed to connect to peer {} after {} tries", partner_id, MAX_TCPUNCH_TRIES),
                        ));
                    }

                    std::thread::sleep(Duration::from_millis(500));
                }
            }
        }

        Err(CylonError::new(
            Code::IoError,
            format!("Failed to connect to peer {}", partner_id),
        ))
    }

    /// Check socket for non-blocking mode
    fn check_socket_nbx(&self, partner_id: PeerNum, pair_name: &str) -> CylonResult<()> {
        let mut current_try = 0;

        while current_try < MAX_TCPUNCH_TRIES {
            // Check if socket already exists
            {
                let sockets = self.sockets.read().unwrap();
                if let Some(mode_sockets) = sockets.get(&Mode::NonBlocking) {
                    if let Some(Some(_)) = mode_sockets.get(partner_id as usize) {
                        return Ok(());
                    }
                }
            }

            // Initialize socket vector if needed
            {
                let mut sockets = self.sockets.write().unwrap();
                sockets.entry(Mode::NonBlocking).or_insert_with(|| {
                    create_socket_vec(self.num_peers as usize)
                });
            }

            // Try to establish connection
            log::info!("Trying to pair partnerId: {} to pair_name: {}", partner_id, pair_name);

            match tcpunch::pair(
                pair_name,
                &self.hostname,
                self.port as u16,
                self.max_timeout as u64,
            ) {
                Ok(stream) => {
                    log::info!("Paired partnerId: {} to pair_name: {}", partner_id, pair_name);

                    // Set non-blocking
                    stream.set_nonblocking(true).map_err(|e| {
                        CylonError::new(Code::IoError, format!("Failed to set non-blocking: {}", e))
                    })?;

                    // Configure socket
                    let timeout = Duration::from_millis(self.max_timeout as u64);
                    stream.set_read_timeout(Some(timeout)).ok();
                    stream.set_write_timeout(Some(timeout)).ok();
                    stream.set_nodelay(true).ok();

                    // Store socket
                    let mut sockets = self.sockets.write().unwrap();
                    let mode_sockets = sockets.get_mut(&Mode::NonBlocking).unwrap();
                    mode_sockets[partner_id as usize] = Some(stream);

                    return Ok(());
                }
                Err(e) => {
                    log::warn!(
                        "Socket pairing failed: {} pairName: {} partnerId: {}",
                        e, pair_name, partner_id
                    );

                    current_try += 1;
                    if current_try == MAX_TCPUNCH_TRIES {
                        log::warn!("Max tries reached for partnerId: {} pair_name: {}", partner_id, pair_name);
                        return Ok(()); // Don't fail, just log
                    }

                    // Exponential backoff
                    let backoff = 500 * (1 << std::cmp::min(current_try, 4));
                    std::thread::sleep(Duration::from_millis(backoff as u64));
                }
            }
        }

        Ok(())
    }

    /// Get socket for peer (must be connected)
    fn get_socket(&self, partner_id: PeerNum, mode: Mode) -> CylonResult<std::sync::RwLockReadGuard<'_, HashMap<Mode, Vec<Option<TcpStream>>>>> {
        let sockets = self.sockets.read().unwrap();
        Ok(sockets)
    }
}

impl Channel for Direct {
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

    fn init(&mut self) -> CylonResult<()> {
        if self.num_peers > 0 {
            for i in 0..self.num_peers {
                if i == self.peer_id {
                    continue;
                }

                if self.mode == Mode::NonBlocking {
                    let pair_name = self.get_pairing_name(self.peer_id, i, Mode::NonBlocking);
                    self.check_socket_nbx(i, &pair_name)?;
                }
            }
        }
        Ok(())
    }

    fn finalize(&mut self) -> CylonResult<()> {
        // Sockets will be closed when dropped
        let mut sockets = self.sockets.write().unwrap();
        sockets.clear();
        Ok(())
    }

    fn get_max_timeout(&self) -> i32 {
        self.max_timeout
    }

    fn send(&self, buf: Arc<ChannelData>, dest: PeerNum) -> CylonResult<()> {
        let pair_name = self.get_pairing_name(self.peer_id, dest, Mode::Blocking);
        self.check_socket(dest, &pair_name)?;

        let mut sockets = self.sockets.write().unwrap();
        let mode_sockets = sockets.get_mut(&Mode::Blocking).ok_or_else(|| {
            CylonError::new(Code::IoError, "No blocking sockets initialized".to_string())
        })?;

        let stream = mode_sockets[dest as usize].as_mut().ok_or_else(|| {
            CylonError::new(Code::IoError, format!("No socket for peer {}", dest))
        })?;

        let data = buf.as_slice();
        stream.write_all(&data[..buf.len]).map_err(|e| {
            if e.kind() == std::io::ErrorKind::WouldBlock {
                CylonError::new(Code::IoError, "Send timeout".to_string())
            } else {
                CylonError::new(Code::IoError, format!("Send error: {}", e))
            }
        })?;

        Ok(())
    }

    fn send_async(
        &self,
        buf: Arc<ChannelData>,
        dest: PeerNum,
        _context: Option<&mut FmiContext>,
        mode: Mode,
        _callback: Option<NbxCallback>,
    ) -> CylonResult<()> {
        if mode == Mode::Blocking {
            self.send(buf, dest)
        } else {
            let pair_name = self.get_pairing_name(self.peer_id, dest, Mode::NonBlocking);
            self.check_socket_nbx(dest, &pair_name)?;

            // Store state for later processing
            let state = IOState::new(buf, Operation::Send, self.max_timeout);
            let state_arc = Arc::new(Mutex::new(state));

            let mut io_states = self.io_states.write().unwrap();
            let send_states = io_states.entry(Operation::Send).or_insert_with(HashMap::new);

            // Use socket fd as key (simplified - use dest for now)
            send_states.insert(dest, state_arc);

            Ok(())
        }
    }

    fn recv(&self, buf: Arc<ChannelData>, src: PeerNum) -> CylonResult<()> {
        let pair_name = self.get_pairing_name(self.peer_id, src, Mode::Blocking);
        self.check_socket(src, &pair_name)?;

        let mut sockets = self.sockets.write().unwrap();
        let mode_sockets = sockets.get_mut(&Mode::Blocking).ok_or_else(|| {
            CylonError::new(Code::IoError, "No blocking sockets initialized".to_string())
        })?;

        let stream = mode_sockets[src as usize].as_mut().ok_or_else(|| {
            CylonError::new(Code::IoError, format!("No socket for peer {}", src))
        })?;

        let mut data = buf.as_mut_slice();
        stream.read_exact(&mut data[..buf.len]).map_err(|e| {
            if e.kind() == std::io::ErrorKind::WouldBlock {
                CylonError::new(Code::IoError, "Recv timeout".to_string())
            } else {
                CylonError::new(Code::IoError, format!("Recv error: {}", e))
            }
        })?;

        Ok(())
    }

    fn recv_async(
        &self,
        buf: Arc<ChannelData>,
        src: PeerNum,
        _context: Option<&mut FmiContext>,
        mode: Mode,
        _callback: Option<NbxCallback>,
    ) -> CylonResult<()> {
        if mode == Mode::Blocking {
            self.recv(buf, src)
        } else {
            let pair_name = self.get_pairing_name(self.peer_id, src, Mode::NonBlocking);
            self.check_socket_nbx(src, &pair_name)?;

            // Store state for later processing
            let state = IOState::new(buf, Operation::Receive, self.max_timeout);
            let state_arc = Arc::new(Mutex::new(state));

            let mut io_states = self.io_states.write().unwrap();
            let recv_states = io_states.entry(Operation::Receive).or_insert_with(HashMap::new);
            recv_states.insert(src, state_arc);

            Ok(())
        }
    }

    fn channel_event_progress(&self, op: Operation) -> EventProcessStatus {
        let io_states = self.io_states.read().unwrap();

        if op == Operation::Default {
            // Process all operations
            let mut status = EventProcessStatus::Empty;
            for (_operation, _states) in io_states.iter() {
                // TODO: Process pending operations
                status = EventProcessStatus::Processing;
            }
            status
        } else {
            if let Some(_states) = io_states.get(&op) {
                // TODO: Process pending operations for this op
                EventProcessStatus::Processing
            } else {
                EventProcessStatus::Empty
            }
        }
    }

    fn bcast_async(
        &self,
        buf: Arc<ChannelData>,
        root: PeerNum,
        mode: Mode,
        callback: Option<NbxCallback>,
    ) -> CylonResult<()> {
        bcast_binomial(self, buf, root, mode, callback)
    }

    fn barrier(&self) -> CylonResult<()> {
        barrier_impl(self)
    }

    fn gatherv_async(
        &self,
        _sendbuf: Arc<ChannelData>,
        _recvbuf: Arc<ChannelData>,
        _root: PeerNum,
        _recvcounts: &[i32],
        _displs: &[i32],
        _mode: Mode,
        _callback: Option<NbxCallback>,
    ) -> CylonResult<()> {
        // TODO: Implement gatherv
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
        // Default implementation: gather then broadcast
        gather_binomial(self, sendbuf, recvbuf.clone(), root)?;
        bcast_binomial(self, recvbuf, root, Mode::Blocking, None)
    }

    fn allgatherv_async(
        &self,
        _sendbuf: Arc<ChannelData>,
        _recvbuf: Arc<ChannelData>,
        _root: PeerNum,
        _recvcounts: &[i32],
        _displs: &[i32],
        _mode: Mode,
        _callback: Option<NbxCallback>,
    ) -> CylonResult<()> {
        // TODO: Implement allgatherv
        Ok(())
    }

    fn reduce(
        &self,
        sendbuf: Arc<ChannelData>,
        recvbuf: Arc<ChannelData>,
        root: PeerNum,
        func: &RawFunction,
    ) -> CylonResult<()> {
        reduce_impl(self, sendbuf, recvbuf, root, func)
    }

    fn scan(
        &self,
        sendbuf: Arc<ChannelData>,
        recvbuf: Arc<ChannelData>,
        func: &RawFunction,
    ) -> CylonResult<()> {
        scan_impl(self, sendbuf, recvbuf, func)
    }
}

impl PeerToPeerChannel for Direct {
    fn send_object(&self, buf: Arc<ChannelData>, peer_id: PeerNum) -> CylonResult<()> {
        self.send(buf, peer_id)
    }

    fn send_object_async(
        &self,
        state: Arc<Mutex<IOState>>,
        peer_id: PeerNum,
        mode: Mode,
    ) -> CylonResult<()> {
        let buf = {
            let s = state.lock().unwrap();
            s.request.clone()
        };
        self.send_async(buf, peer_id, None, mode, None)
    }

    fn recv_object(&self, buf: Arc<ChannelData>, peer_id: PeerNum) -> CylonResult<()> {
        self.recv(buf, peer_id)
    }

    fn recv_object_async(
        &self,
        state: Arc<Mutex<IOState>>,
        peer_id: PeerNum,
        mode: Mode,
    ) -> CylonResult<()> {
        let buf = {
            let s = state.lock().unwrap();
            s.request.clone()
        };
        self.recv_async(buf, peer_id, None, mode, None)
    }
}
