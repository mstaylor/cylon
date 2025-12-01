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

//! TCPunch - TCP NAT Hole Punching Library (Protocol v2)
//!
//! This module implements TCP NAT hole punching for establishing direct
//! peer-to-peer connections between nodes that may be behind NAT.
//!
//! ## Protocol v2 Changes
//!
//! - Fixed-size request (141 bytes) and response (51 bytes)
//! - Reconnection support via UUID token
//! - Explicit status codes (WAITING, PAIRED, TIMEOUT, ERROR)
//!
//! The technique works by:
//! 1. Both peers connect to a rendezvous server
//! 2. The server exchanges each peer's public IP:port information
//! 3. Both peers simultaneously attempt to connect to each other
//! 4. Using SO_REUSEADDR/SO_REUSEPORT allows binding multiple sockets to the same port
//! 5. One connection succeeds (either active connect or passive accept)

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream, SocketAddr, Ipv4Addr};
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use crate::error::{CylonError, CylonResult, Code};

// ============================================================================
// Protocol v2 Constants
// ============================================================================

/// Maximum length of pairing name
pub const MAX_PAIRING_NAME: usize = 100;

/// Length of reconnection token (UUID string)
pub const TOKEN_LENGTH: usize = 37;

/// Client request size (100 + 37 + 4 = 141 bytes)
pub const CLIENT_REQUEST_SIZE: usize = 141;

/// Server response size (1 + 4 + 2 + 4 + 2 + 37 + 1 = 51 bytes)
pub const SERVER_RESPONSE_SIZE: usize = 51;

/// Magic number for validation handshake
const VALIDATION_MAGIC: u32 = 0xDEADBEEF;

/// Default timeout for connection attempts (30 seconds)
const DEFAULT_TIMEOUT_MS: u64 = 30000;

/// Validation timeout (15 seconds)
const VALIDATION_TIMEOUT_SECS: u64 = 15;

/// Default max retries for reconnection
const DEFAULT_MAX_RETRIES: u32 = 3;

// ============================================================================
// Protocol v2 Types
// ============================================================================

/// Pairing status returned by server
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PairingStatus {
    /// Registered, waiting for peer
    Waiting = 0,
    /// Peer found, proceed to hole punching
    Paired = 1,
    /// Server-side timeout, reconnect with token
    Timeout = 2,
    /// Invalid request/token, start fresh
    Error = 3,
}

impl From<u8> for PairingStatus {
    fn from(v: u8) -> Self {
        match v {
            0 => Self::Waiting,
            1 => Self::Paired,
            2 => Self::Timeout,
            _ => Self::Error,
        }
    }
}

/// Peer information (IP and port)
#[derive(Debug, Clone, Copy)]
pub struct PeerInfo {
    pub ip: Ipv4Addr,
    pub port: u16,
}

impl PeerInfo {
    pub fn to_socket_addr(&self) -> SocketAddr {
        SocketAddr::new(std::net::IpAddr::V4(self.ip), self.port)
    }

    pub fn is_empty(&self) -> bool {
        self.ip.is_unspecified() && self.port == 0
    }
}

/// Server response (51 bytes)
#[derive(Debug)]
pub struct ServerResponse {
    pub status: PairingStatus,
    pub your_info: PeerInfo,
    pub peer_info: Option<PeerInfo>,
    pub token: String,
}

/// Legacy peer connection data (for compatibility)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PeerConnectionData {
    pub ip: u32,        // IPv4 address in network byte order
    pub port: u16,      // Port in network byte order
}

/// Validation message for handshake
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ValidationMsg {
    pub magic: u32,
    pub peer_id: u32,
    pub timestamp: u32,
}

impl ValidationMsg {
    pub fn new() -> Self {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as u32;

        Self {
            magic: VALIDATION_MAGIC,
            peer_id: 0,
            timestamp,
        }
    }

    pub fn to_bytes(&self) -> [u8; 12] {
        let mut bytes = [0u8; 12];
        bytes[0..4].copy_from_slice(&self.magic.to_ne_bytes());
        bytes[4..8].copy_from_slice(&self.peer_id.to_ne_bytes());
        bytes[8..12].copy_from_slice(&self.timestamp.to_ne_bytes());
        bytes
    }

    pub fn from_bytes(bytes: &[u8; 12]) -> Self {
        Self {
            magic: u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            peer_id: u32::from_ne_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
            timestamp: u32::from_ne_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]),
        }
    }
}

impl Default for ValidationMsg {
    fn default() -> Self {
        Self::new()
    }
}

impl PeerConnectionData {
    pub fn to_bytes(&self) -> [u8; 6] {
        let mut bytes = [0u8; 6];
        bytes[0..4].copy_from_slice(&self.ip.to_ne_bytes());
        bytes[4..6].copy_from_slice(&self.port.to_ne_bytes());
        bytes
    }

    pub fn from_bytes(bytes: &[u8; 6]) -> Self {
        Self {
            ip: u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            port: u16::from_ne_bytes([bytes[4], bytes[5]]),
        }
    }

    pub fn to_socket_addr(&self) -> SocketAddr {
        let ip = Ipv4Addr::from(u32::from_be(self.ip));
        let port = u16::from_be(self.port);
        SocketAddr::new(std::net::IpAddr::V4(ip), port)
    }
}

// ============================================================================
// Protocol v2 Functions
// ============================================================================

/// Build a client request (141 bytes)
pub fn build_request(pairing_name: &str, token: Option<&str>) -> [u8; CLIENT_REQUEST_SIZE] {
    let mut buf = [0u8; CLIENT_REQUEST_SIZE];

    // Write pairing name (offset 0, max 99 chars + null)
    let name_bytes = pairing_name.as_bytes();
    let len = name_bytes.len().min(MAX_PAIRING_NAME - 1);
    buf[..len].copy_from_slice(&name_bytes[..len]);

    // Write reconnect token if present (offset 100, max 36 chars + null)
    if let Some(t) = token {
        let token_bytes = t.as_bytes();
        let len = token_bytes.len().min(TOKEN_LENGTH - 1);
        buf[MAX_PAIRING_NAME..MAX_PAIRING_NAME + len].copy_from_slice(&token_bytes[..len]);
    }

    // Flags at offset 137 (4 bytes) - reserved, set to 0
    // Already zero-initialized

    buf
}

/// Parse server response (51 bytes)
pub fn parse_response(buf: &[u8; SERVER_RESPONSE_SIZE]) -> ServerResponse {
    // Status (1 byte at offset 0)
    let status = PairingStatus::from(buf[0]);

    // Your IP (4 bytes at offset 1, network byte order)
    let your_ip = Ipv4Addr::new(buf[1], buf[2], buf[3], buf[4]);
    // Your port (2 bytes at offset 5, network byte order)
    let your_port = u16::from_be_bytes([buf[5], buf[6]]);

    // Peer IP (4 bytes at offset 7, network byte order)
    let peer_ip = Ipv4Addr::new(buf[7], buf[8], buf[9], buf[10]);
    // Peer port (2 bytes at offset 11, network byte order)
    let peer_port = u16::from_be_bytes([buf[11], buf[12]]);

    // Token (37 bytes at offset 13)
    let token_end = buf[13..50].iter().position(|&b| b == 0).unwrap_or(37);
    let token = String::from_utf8_lossy(&buf[13..13 + token_end]).to_string();

    // Determine if peer info is valid
    let peer_info = if peer_ip.is_unspecified() && peer_port == 0 {
        None
    } else {
        Some(PeerInfo { ip: peer_ip, port: peer_port })
    };

    ServerResponse {
        status,
        your_info: PeerInfo { ip: your_ip, port: your_port },
        peer_info,
        token,
    }
}

// ============================================================================
// Socket Configuration
// ============================================================================

/// Configure socket with reuse options
#[cfg(unix)]
fn configure_socket_reuse(socket: &socket2::Socket) -> CylonResult<()> {
    socket.set_reuse_address(true).map_err(|e| {
        CylonError::new(Code::IoError, format!("Failed to set SO_REUSEADDR: {}", e))
    })?;

    // SO_REUSEPORT is available on Linux and most Unix platforms
    // socket2 provides this on all Unix platforms
    #[cfg(all(unix, not(any(target_os = "solaris", target_os = "illumos"))))]
    {
        socket.set_reuse_port(true).map_err(|e| {
            CylonError::new(Code::IoError, format!("Failed to set SO_REUSEPORT: {}", e))
        })?;
    }

    Ok(())
}

/// Listener thread function - accepts incoming connections
fn peer_listen(
    local_port: u16,
    connection_established: Arc<AtomicBool>,
    accepting_socket: Arc<AtomicI32>,
) -> CylonResult<()> {
    use socket2::{Domain, Protocol, Socket, Type};

    // Create socket with reuse options
    let socket = Socket::new(Domain::IPV4, Type::STREAM, Some(Protocol::TCP))
        .map_err(|e| CylonError::new(Code::IoError, format!("Socket creation failed: {}", e)))?;

    configure_socket_reuse(&socket)?;

    // Set accept timeout (3 minutes for cloud environments)
    socket.set_read_timeout(Some(Duration::from_secs(180)))
        .map_err(|e| CylonError::new(Code::IoError, format!("Failed to set timeout: {}", e)))?;

    // Bind to local port
    let addr: SocketAddr = format!("0.0.0.0:{}", local_port).parse().unwrap();
    socket.bind(&addr.into())
        .map_err(|e| CylonError::new(Code::IoError, format!("Could not bind to local port: {}", e)))?;

    // Listen
    socket.listen(1)
        .map_err(|e| CylonError::new(Code::IoError, format!("Listen failed: {}", e)))?;

    let listener: TcpListener = socket.into();
    listener.set_nonblocking(true)
        .map_err(|e| CylonError::new(Code::IoError, format!("Failed to set non-blocking: {}", e)))?;

    let mut error_count = 0;

    loop {
        if connection_established.load(Ordering::SeqCst) {
            break;
        }

        match listener.accept() {
            Ok((stream, _peer_addr)) => {
                log::info!("Successfully connected to peer via accept");

                // Store the raw fd
                #[cfg(unix)]
                {
                    use std::os::unix::io::AsRawFd;
                    accepting_socket.store(stream.as_raw_fd(), Ordering::SeqCst);
                    // Prevent the stream from being dropped (fd will be managed by caller)
                    std::mem::forget(stream);
                }

                connection_established.store(true, Ordering::SeqCst);
                return Ok(());
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                // No connection yet, sleep briefly and retry
                thread::sleep(Duration::from_millis(10));
                continue;
            }
            Err(e) => {
                log::debug!("Accept error: {}", e);
                error_count += 1;
                if error_count > 5 {
                    let backoff_delay = std::cmp::min(100 * (1 << (error_count - 5)), 5000);
                    thread::sleep(Duration::from_millis(backoff_delay));
                }
            }
        }
    }

    Ok(())
}

/// Perform hole punching after receiving peer info
fn do_hole_punch(
    your_info: &PeerInfo,
    peer_info: &PeerInfo,
    timeout_ms: u64,
) -> CylonResult<TcpStream> {
    use socket2::{Domain, Protocol, Socket, Type};

    let local_port = your_info.port;
    let peer_addr = peer_info.to_socket_addr();

    log::info!("Starting hole punch: local port {}, peer {}", local_port, peer_addr);

    // Start listener thread
    let connection_established = Arc::new(AtomicBool::new(false));
    let accepting_socket = Arc::new(AtomicI32::new(-1));

    let conn_established_clone = connection_established.clone();
    let accepting_socket_clone = accepting_socket.clone();

    let listener_handle = thread::spawn(move || {
        peer_listen(local_port, conn_established_clone, accepting_socket_clone)
    });

    // Create socket for active connection attempts
    let peer_socket = Socket::new(Domain::IPV4, Type::STREAM, Some(Protocol::TCP))
        .map_err(|e| CylonError::new(Code::IoError, format!("Socket creation failed: {}", e)))?;

    configure_socket_reuse(&peer_socket)?;
    peer_socket.set_nonblocking(true)
        .map_err(|e| CylonError::new(Code::IoError, format!("Failed to set non-blocking: {}", e)))?;

    // Bind to same local port
    let local_addr: SocketAddr = format!("0.0.0.0:{}", local_port).parse().unwrap();
    peer_socket.bind(&local_addr.into())
        .map_err(|e| CylonError::new(Code::IoError,
            format!("Binding to same port failed: {}", e)))?;

    // Attempt connection with retries
    let start_time = Instant::now();
    let max_connection_time = Duration::from_millis(timeout_ms);
    let mut attempt_count = 0;
    let mut connected = false;

    while !connection_established.load(Ordering::SeqCst) {
        if start_time.elapsed() > max_connection_time {
            connection_established.store(true, Ordering::SeqCst); // Signal listener to stop
            return Err(CylonError::new(Code::IoError, "Connection timeout".to_string()));
        }

        match peer_socket.connect(&peer_addr.into()) {
            Ok(()) => {
                log::info!("Successfully connected to peer via connect()");
                connected = true;
                break;
            }
            Err(ref e) if e.raw_os_error() == Some(libc::EISCONN) => {
                log::info!("Successfully connected to peer (EISCONN)");
                connected = true;
                break;
            }
            Err(ref e) if e.raw_os_error() == Some(libc::EALREADY)
                       || e.raw_os_error() == Some(libc::EAGAIN)
                       || e.raw_os_error() == Some(libc::EINPROGRESS) => {
                attempt_count += 1;
                thread::sleep(Duration::from_millis(10));
                continue;
            }
            Err(_e) => {
                let base_delay = 100;
                let backoff_delay = base_delay * (1 + attempt_count / 10);
                thread::sleep(Duration::from_millis(std::cmp::min(backoff_delay, 1000)));
                attempt_count += 1;
                continue;
            }
        }
    }

    // Determine which connection to use
    let mut peer_stream = if connection_established.load(Ordering::SeqCst) && !connected {
        // Listener accepted connection
        let _ = listener_handle.join();
        let fd = accepting_socket.load(Ordering::SeqCst);
        if fd < 0 {
            return Err(CylonError::new(Code::IoError, "No valid socket from listener".to_string()));
        }

        #[cfg(unix)]
        unsafe {
            use std::os::unix::io::FromRawFd;
            TcpStream::from_raw_fd(fd)
        }
        #[cfg(not(unix))]
        {
            return Err(CylonError::new(Code::IoError, "Platform not supported".to_string()));
        }
    } else {
        // Active connection succeeded
        connection_established.store(true, Ordering::SeqCst);
        let _ = listener_handle.join();

        peer_socket.set_nonblocking(false)
            .map_err(|e| CylonError::new(Code::IoError,
                format!("Failed to set blocking: {}", e)))?;
        peer_socket.into()
    };

    // Perform validation handshake
    peer_stream.set_read_timeout(Some(Duration::from_secs(VALIDATION_TIMEOUT_SECS)))
        .map_err(|e| CylonError::new(Code::IoError, format!("Failed to set timeout: {}", e)))?;
    peer_stream.set_write_timeout(Some(Duration::from_secs(VALIDATION_TIMEOUT_SECS)))
        .map_err(|e| CylonError::new(Code::IoError, format!("Failed to set timeout: {}", e)))?;

    // Send validation message
    let validation_msg = ValidationMsg::new();
    peer_stream.write_all(&validation_msg.to_bytes())
        .map_err(|e| {
            log::error!("Validation handshake failed: could not send validation message: {}", e);
            CylonError::new(Code::IoError, "Validation handshake failed: send".to_string())
        })?;

    // Receive peer's validation message
    let mut peer_validation_bytes = [0u8; 12];
    peer_stream.read_exact(&mut peer_validation_bytes)
        .map_err(|e| {
            log::error!("Validation handshake failed: could not receive validation: {}", e);
            CylonError::new(Code::IoError, "Validation handshake failed: receive".to_string())
        })?;

    let peer_validation = ValidationMsg::from_bytes(&peer_validation_bytes);
    if peer_validation.magic != VALIDATION_MAGIC {
        log::error!("Validation handshake failed: invalid magic number");
        return Err(CylonError::new(Code::IoError,
            "Validation handshake failed: invalid magic".to_string()));
    }

    log::info!("Validation handshake completed successfully");

    // Clear timeouts for normal operation
    peer_stream.set_read_timeout(None).ok();
    peer_stream.set_write_timeout(None).ok();

    Ok(peer_stream)
}

/// Establish a peer-to-peer connection using TCP NAT hole punching (Protocol v2)
///
/// # Arguments
/// * `pairing_name` - Unique name for this pairing (both peers must use the same name)
/// * `server_address` - IP address of the rendezvous server
/// * `port` - Port of the rendezvous server (default: 10000)
/// * `timeout_ms` - Connection timeout in milliseconds (0 for default 30s)
///
/// # Returns
/// * `Ok(TcpStream)` - Successfully established connection
/// * `Err(CylonError)` - Connection failed (timeout, validation failure, etc.)
pub fn pair(
    pairing_name: &str,
    server_address: &str,
    port: u16,
    timeout_ms: u64,
) -> CylonResult<TcpStream> {
    pair_with_retries(pairing_name, server_address, port, timeout_ms, DEFAULT_MAX_RETRIES)
}

/// Establish a peer-to-peer connection with configurable retries (Protocol v2)
///
/// # Arguments
/// * `pairing_name` - Unique name for this pairing (both peers must use the same name)
/// * `server_address` - IP address of the rendezvous server
/// * `port` - Port of the rendezvous server
/// * `timeout_ms` - Connection timeout in milliseconds (0 for default 30s)
/// * `max_retries` - Maximum number of reconnection attempts
///
/// # Returns
/// * `Ok(TcpStream)` - Successfully established connection
/// * `Err(CylonError)` - Connection failed after all retries
pub fn pair_with_retries(
    pairing_name: &str,
    server_address: &str,
    port: u16,
    timeout_ms: u64,
    max_retries: u32,
) -> CylonResult<TcpStream> {
    use socket2::{Domain, Protocol, Socket, Type};

    let timeout_ms = if timeout_ms == 0 { DEFAULT_TIMEOUT_MS } else { timeout_ms };
    let timeout = Duration::from_millis(timeout_ms);

    let server_addr: SocketAddr = format!("{}:{}", server_address, port)
        .parse()
        .map_err(|e| CylonError::new(Code::Invalid, format!("Invalid server address: {}", e)))?;

    let mut reconnect_token: Option<String> = None;

    for attempt in 0..max_retries {
        log::debug!("Pairing attempt {} of {} for '{}'", attempt + 1, max_retries, pairing_name);

        // Connect to rendezvous server
        let socket = Socket::new(Domain::IPV4, Type::STREAM, Some(Protocol::TCP))
            .map_err(|e| CylonError::new(Code::IoError, format!("Socket creation failed: {}", e)))?;

        configure_socket_reuse(&socket)?;

        socket.set_read_timeout(Some(timeout))
            .map_err(|e| CylonError::new(Code::IoError, format!("Failed to set timeout: {}", e)))?;
        socket.set_write_timeout(Some(timeout))
            .map_err(|e| CylonError::new(Code::IoError, format!("Failed to set timeout: {}", e)))?;

        socket.connect(&server_addr.into())
            .map_err(|e| CylonError::new(Code::IoError,
                format!("Connection to rendezvous server failed: {}", e)))?;

        let mut stream: TcpStream = socket.into();

        // Send request (141 bytes)
        let request = build_request(pairing_name, reconnect_token.as_deref());
        stream.write_all(&request)
            .map_err(|e| CylonError::new(Code::IoError,
                format!("Failed to send request: {}", e)))?;

        // Receive response (51 bytes)
        let mut resp_buf = [0u8; SERVER_RESPONSE_SIZE];
        stream.read_exact(&mut resp_buf)
            .map_err(|e| CylonError::new(Code::IoError,
                format!("Failed to receive response: {}", e)))?;

        let resp = parse_response(&resp_buf);

        // Save token for potential reconnection
        if !resp.token.is_empty() {
            reconnect_token = Some(resp.token.clone());
        }

        match resp.status {
            PairingStatus::Paired => {
                // Got peer immediately
                let peer = resp.peer_info.ok_or_else(|| {
                    CylonError::new(Code::IoError, "No peer info in PAIRED response".to_string())
                })?;

                log::info!("Paired immediately with peer at {}:{}", peer.ip, peer.port);
                return do_hole_punch(&resp.your_info, &peer, timeout_ms);
            }

            PairingStatus::Waiting => {
                log::debug!("Registered, waiting for peer (token: {})", resp.token);

                // Wait for second response with peer info
                match stream.read_exact(&mut resp_buf) {
                    Ok(()) => {
                        let resp2 = parse_response(&resp_buf);

                        if resp2.status == PairingStatus::Paired {
                            let peer = resp2.peer_info.ok_or_else(|| {
                                CylonError::new(Code::IoError, "No peer info in PAIRED response".to_string())
                            })?;

                            log::info!("Peer found: {}:{}", peer.ip, peer.port);
                            return do_hole_punch(&resp.your_info, &peer, timeout_ms);
                        } else {
                            log::warn!("Unexpected status after WAITING: {:?}", resp2.status);
                            // Fall through to retry
                        }
                    }
                    Err(e) => {
                        log::warn!("Timeout waiting for peer (attempt {}): {}", attempt + 1, e);
                        // Fall through to retry with token
                    }
                }
            }

            PairingStatus::Timeout => {
                log::warn!("Server timeout (attempt {}), will retry with token", attempt + 1);
                // Token is already saved, retry
                thread::sleep(Duration::from_millis(1000));
                continue;
            }

            PairingStatus::Error => {
                log::warn!("Server error (attempt {}), clearing token and retrying", attempt + 1);
                reconnect_token = None;
                thread::sleep(Duration::from_millis(1000));
                continue;
            }
        }
    }

    Err(CylonError::new(
        Code::IoError,
        format!("Failed to pair '{}' after {} retries", pairing_name, max_retries),
    ))
}

/// Remove a pairing from the rendezvous server (Protocol v2)
///
/// Note: In Protocol v2, there is no explicit "remove" operation.
/// The server automatically cleans up pairings after timeout.
/// This function sends a request with an empty token which effectively
/// creates a new registration that will timeout if no peer connects.
///
/// For immediate cleanup, clients should simply disconnect and let
/// the server handle cleanup via its internal timeout mechanism.
pub fn remove_pair(
    pairing_name: &str,
    server_address: &str,
    port: u16,
    timeout_ms: u64,
) -> CylonResult<()> {
    let timeout = Duration::from_millis(if timeout_ms == 0 { DEFAULT_TIMEOUT_MS } else { timeout_ms });

    let server_addr: SocketAddr = format!("{}:{}", server_address, port)
        .parse()
        .map_err(|e| CylonError::new(Code::Invalid, format!("Invalid server address: {}", e)))?;

    let mut stream = TcpStream::connect_timeout(&server_addr, timeout)
        .map_err(|e| CylonError::new(Code::IoError,
            format!("Connection to rendezvous server failed: {}", e)))?;

    stream.set_write_timeout(Some(timeout))
        .map_err(|e| CylonError::new(Code::IoError, format!("Failed to set timeout: {}", e)))?;

    // Send a Protocol v2 request (141 bytes) with no reconnect token
    // The server will register this and clean it up after timeout
    let request = build_request(pairing_name, None);
    stream.write_all(&request)
        .map_err(|e| CylonError::new(Code::IoError,
            format!("Failed to send request: {}", e)))?;

    // Immediately close the connection - server will clean up
    drop(stream);

    log::debug!("Sent remove request for pairing '{}'", pairing_name);

    Ok(())
}
