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

//! TCPunch - TCP NAT Hole Punching Library
//!
//! This module implements TCP NAT hole punching for establishing direct
//! peer-to-peer connections between nodes that may be behind NAT.
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

/// Magic number for validation handshake
const VALIDATION_MAGIC: u32 = 0xDEADBEEF;

/// Default timeout for connection attempts (30 seconds)
const DEFAULT_TIMEOUT_MS: u64 = 30000;

/// Validation timeout (15 seconds)
const VALIDATION_TIMEOUT_SECS: u64 = 15;

/// Peer connection data received from rendezvous server
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

/// Establish a peer-to-peer connection using TCP NAT hole punching
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
    use socket2::{Domain, Protocol, Socket, Type};

    let timeout_ms = if timeout_ms == 0 { DEFAULT_TIMEOUT_MS } else { timeout_ms };
    let timeout = Duration::from_millis(timeout_ms);

    // Connect to rendezvous server
    let server_addr: SocketAddr = format!("{}:{}", server_address, port)
        .parse()
        .map_err(|e| CylonError::new(Code::Invalid, format!("Invalid server address: {}", e)))?;

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

    let mut rendezvous_stream: TcpStream = socket.into();

    // Send pairing name
    rendezvous_stream.write_all(pairing_name.as_bytes())
        .map_err(|e| CylonError::new(Code::IoError,
            format!("Failed to send pairing name: {}", e)))?;

    // Receive our public info
    let mut public_info_bytes = [0u8; 6];
    rendezvous_stream.read_exact(&mut public_info_bytes)
        .map_err(|e| CylonError::new(Code::IoError,
            format!("Failed to receive public info: {}", e)))?;
    let public_info = PeerConnectionData::from_bytes(&public_info_bytes);
    let local_port = u16::from_be(public_info.port);

    // Start listener thread
    let connection_established = Arc::new(AtomicBool::new(false));
    let accepting_socket = Arc::new(AtomicI32::new(-1));

    let conn_established_clone = connection_established.clone();
    let accepting_socket_clone = accepting_socket.clone();

    let listener_handle = thread::spawn(move || {
        peer_listen(local_port, conn_established_clone, accepting_socket_clone)
    });

    // Receive peer info
    let mut peer_info_bytes = [0u8; 6];
    rendezvous_stream.read_exact(&mut peer_info_bytes)
        .map_err(|e| CylonError::new(Code::IoError,
            format!("Failed to receive peer info: {}", e)))?;
    let peer_info = PeerConnectionData::from_bytes(&peer_info_bytes);
    let peer_addr = peer_info.to_socket_addr();

    log::info!("Peer address: {}", peer_addr);

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

    log::info!("Validation handshake completed successfully for pair: {}", pairing_name);

    // Clear timeouts for normal operation
    peer_stream.set_read_timeout(None).ok();
    peer_stream.set_write_timeout(None).ok();

    Ok(peer_stream)
}

/// Remove a pairing from the rendezvous server
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

    stream.write_all(pairing_name.as_bytes())
        .map_err(|e| CylonError::new(Code::IoError,
            format!("Failed to send pairing name: {}", e)))?;

    Ok(())
}
