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

//! Peer socket establishment for the direct-redis channel.
//!
//! Each rank publishes its own reachable `host:port` into a Redis hash keyed by
//! `<comm_name>:direct_redis_addrs`. Ranks below `num_peers - 1` bind a listening
//! socket and accept inbound connections; higher ranks read a peer's address from
//! Redis and dial out. This requires an environment where peers can bind and
//! listen — Fargate or ECS, not Lambda.

use std::collections::HashMap;
use std::net::TcpListener;
use std::net::TcpStream;
use std::sync::atomic::AtomicBool;
use std::sync::{Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use redis::{Client, Commands};

use crate::error::{Code, CylonError, CylonResult};
use super::common::{Mode, PeerNum};

const POLL_INTERVAL_MS: u64 = 200;
const HANDSHAKE_TIMEOUT_MS: u64 = 5000;

const MODE_BYTE_BLOCKING: u8 = 0;
const MODE_BYTE_NONBLOCKING: u8 = 1;

pub fn encode_mode_byte(mode: Mode) -> u8 {
    match mode {
        Mode::NonBlocking => MODE_BYTE_NONBLOCKING,
        Mode::Blocking => MODE_BYTE_BLOCKING,
    }
}

pub fn decode_mode_byte(mode_byte: u8) -> Option<Mode> {
    match mode_byte {
        MODE_BYTE_BLOCKING => Some(Mode::Blocking),
        MODE_BYTE_NONBLOCKING => Some(Mode::NonBlocking),
        _ => None,
    }
}

pub fn peer_and_mode_key(peer_id: PeerNum, mode: Mode) -> i32 {
    peer_id * 2 + encode_mode_byte(mode) as i32
}

pub fn addr_ttl_seconds() -> u64 {
    std::env::var("CYLON_KEY_TTL")
        .unwrap_or_else(|_| "3600".to_string())
        .parse()
        .unwrap_or(3600)
}

pub fn split_metadata_uri(url: &str) -> CylonResult<(String, u16, String)> {
    let scheme_pos = url.find("://").ok_or_else(|| {
        CylonError::new(
            Code::Invalid,
            format!(
                "direct-redis: ECS_CONTAINER_METADATA_URI_V4 is malformed (no scheme separator): {}",
                url
            ),
        )
    })?;
    let scheme_end = scheme_pos + 3;
    let path_start = url[scheme_end..].find('/').map(|i| i + scheme_end).ok_or_else(|| {
        CylonError::new(
            Code::Invalid,
            format!(
                "direct-redis: ECS_CONTAINER_METADATA_URI_V4 is malformed (no path component): {}",
                url
            ),
        )
    })?;
    let authority = &url[scheme_end..path_start];
    let (host, port) = match authority.rfind(':') {
        Some(i) => {
            let port: u16 = authority[i + 1..].parse().map_err(|_| {
                CylonError::new(
                    Code::Invalid,
                    format!(
                        "direct-redis: ECS_CONTAINER_METADATA_URI_V4 has an unparseable port: {}",
                        url
                    ),
                )
            })?;
            (authority[..i].to_string(), port)
        }
        None => (authority.to_string(), 80),
    };
    Ok((host, port, url[path_start..].to_string()))
}

pub fn parse_ipv4_from_metadata(body: &str) -> CylonResult<String> {
    let parsed: serde_json::Value = serde_json::from_str(body).map_err(|e| {
        CylonError::new(
            Code::Invalid,
            format!("direct-redis: ECS metadata response is not valid JSON: {}", e),
        )
    })?;
    parsed
        .get("Networks")
        .and_then(|n| n.as_array())
        .and_then(|networks| {
            networks.iter().find_map(|net| {
                net.get("IPv4Addresses")
                    .and_then(|a| a.as_array())
                    .and_then(|addrs| addrs.first())
                    .and_then(|first| first.as_str())
            })
        })
        .map(|s| s.to_string())
        .ok_or_else(|| {
            CylonError::new(
                Code::Invalid,
                "direct-redis: could not find IPv4Addresses in ECS metadata response".to_string(),
            )
        })
}

fn http_get(host: &str, port: u16, path: &str) -> CylonResult<String> {
    use std::io::{Read, Write};

    let mut stream = TcpStream::connect((host, port)).map_err(|e| {
        CylonError::new(
            Code::IoError,
            format!("direct-redis: ECS metadata connect to {}:{} failed: {}", host, port, e),
        )
    })?;
    stream.set_read_timeout(Some(Duration::from_secs(2))).ok();
    stream.set_write_timeout(Some(Duration::from_secs(2))).ok();

    let request = format!(
        "GET {} HTTP/1.1\r\nHost: {}\r\nConnection: close\r\n\r\n",
        path, host
    );
    stream.write_all(request.as_bytes()).map_err(|e| {
        CylonError::new(Code::IoError, format!("direct-redis: ECS metadata request failed: {}", e))
    })?;

    let mut response = String::new();
    stream.read_to_string(&mut response).map_err(|e| {
        CylonError::new(Code::IoError, format!("direct-redis: ECS metadata read failed: {}", e))
    })?;

    response
        .find("\r\n\r\n")
        .map(|i| response[i + 4..].to_string())
        .ok_or_else(|| {
            CylonError::new(
                Code::Invalid,
                "direct-redis: ECS metadata response had no header/body separator".to_string(),
            )
        })
}

/// Establishes peer sockets by exchanging listen addresses through Redis.
#[derive(Default)]
pub struct RedisDirectEstablisher {
    redis_host: String,
    redis_port: i32,
    redis_namespace: String,
    comm_name: String,
    self_rank: PeerNum,
    pub listen_port: i32,
    host_override: String,
    initialized: bool,
    listener: Option<TcpListener>,
    running: std::sync::Arc<AtomicBool>,
    accept_thread: Option<JoinHandle<()>>,
    accepted: std::sync::Arc<(Mutex<HashMap<i32, TcpStream>>, Condvar)>,
}

impl RedisDirectEstablisher {
    fn addr_key(&self) -> String {
        if self.redis_namespace.is_empty() {
            format!("{}:direct_redis_addrs", self.comm_name)
        } else {
            format!("{}:{}:direct_redis_addrs", self.redis_namespace, self.comm_name)
        }
    }

    fn redis_client(&self) -> CylonResult<Client> {
        Client::open(format!("redis://{}:{}/", self.redis_host, self.redis_port)).map_err(|e| {
            CylonError::new(
                Code::IoError,
                format!(
                    "direct-redis: rank {} (comm_name={}) could not open Redis at {}:{}: {}",
                    self.self_rank, self.comm_name, self.redis_host, self.redis_port, e
                ),
            )
        })
    }

    fn publish_own_address(&self, own_addr: &str) -> CylonResult<()> {
        let client = self.redis_client()?;
        let mut conn = client.get_connection().map_err(|e| {
            CylonError::new(Code::IoError, format!("direct-redis: Redis connect failed: {}", e))
        })?;
        let key = self.addr_key();
        let _: () = conn
            .hset(&key, self.self_rank.to_string(), own_addr)
            .map_err(|e| {
                CylonError::new(
                    Code::IoError,
                    format!("direct-redis: HSET {} failed: {}", key, e),
                )
            })?;
        let _: () = conn.expire(&key, addr_ttl_seconds() as i64).map_err(|e| {
            CylonError::new(Code::IoError, format!("direct-redis: EXPIRE {} failed: {}", key, e))
        })?;
        log::info!(
            "direct-redis: published rank {} address {}",
            self.self_rank,
            own_addr
        );
        Ok(())
    }

    fn lookup_peer_address(&self, partner_id: PeerNum, timeout_ms: i32) -> CylonResult<String> {
        let client = self.redis_client()?;
        let mut conn = client.get_connection().map_err(|e| {
            CylonError::new(Code::IoError, format!("direct-redis: Redis connect failed: {}", e))
        })?;
        let key = self.addr_key();
        let mut waited_ms: i32 = 0;
        while waited_ms < timeout_ms {
            let val: Option<String> = conn.hget(&key, partner_id.to_string()).map_err(|e| {
                CylonError::new(
                    Code::IoError,
                    format!("direct-redis: HGET {} failed: {}", key, e),
                )
            })?;
            if let Some(addr) = val {
                return Ok(addr);
            }
            std::thread::sleep(Duration::from_millis(POLL_INTERVAL_MS));
            waited_ms += POLL_INTERVAL_MS as i32;
        }
        log::warn!(
            "direct-redis: partner {} never published its address within {}ms (key={})",
            partner_id,
            timeout_ms,
            key
        );
        Err(CylonError::new(
            Code::IoError,
            format!(
                "direct-redis: timed out waiting for partner {} address (key={})",
                partner_id, key
            ),
        ))
    }

    /// Test-only accessor for the peer address lookup.
    pub fn lookup_peer_address_for_test(
        &self,
        partner_id: PeerNum,
        timeout_ms: i32,
    ) -> CylonResult<String> {
        self.lookup_peer_address(partner_id, timeout_ms)
    }

    /// Test-only accessor for the own-address resolution.
    pub fn resolve_own_address_for_test(&self) -> CylonResult<String> {
        self.resolve_own_address()
    }

    /// Test-only setter for the listen port, without going through `init()`.
    pub fn set_listen_port_for_test(&mut self, port: i32) {
        self.listen_port = port;
    }

    fn delete_own_address(&self) -> CylonResult<()> {
        let client = self.redis_client()?;
        let mut conn = client.get_connection().map_err(|e| {
            CylonError::new(Code::IoError, format!("direct-redis: Redis connect failed: {}", e))
        })?;
        let key = self.addr_key();
        let _: () = conn.hdel(&key, self.self_rank.to_string()).map_err(|e| {
            CylonError::new(Code::IoError, format!("direct-redis: HDEL {} failed: {}", key, e))
        })?;
        Ok(())
    }

    /// Release the listening socket and any accepted-but-unconsumed connections.
    pub fn finalize(&mut self) {
        self.running.store(false, std::sync::atomic::Ordering::SeqCst);
        if let Some(handle) = self.accept_thread.take() {
            if let Err(e) = handle.join() {
                log::error!(
                    "direct-redis: accept thread for rank {} panicked: {:?}",
                    self.self_rank,
                    e
                );
            }
        }
        self.listener = None;
        let (lock, _) = &*self.accepted;
        lock.lock().unwrap_or_else(|e| e.into_inner()).clear();

        if self.initialized {
            if let Err(e) = self.delete_own_address() {
                log::warn!("direct-redis: failed to clean up published address for rank {}: {}", self.self_rank, e);
            }
        }
    }

    /// Bind a listening socket (ranks below `num_peers - 1`), publish this rank's
    /// address, and begin accepting peer connections. Must be called exactly once.
    #[allow(clippy::too_many_arguments)]
    pub fn init(
        &mut self,
        redis_host: String,
        redis_port: i32,
        redis_namespace: String,
        comm_name: String,
        self_rank: PeerNum,
        num_peers: PeerNum,
        listen_port: i32,
        host_override: String,
    ) -> CylonResult<()> {
        if self.initialized {
            return Err(CylonError::new(
                Code::InvalidState,
                format!(
                    "direct-redis: init() called more than once on rank {} (comm_name={})",
                    self.self_rank, self.comm_name
                ),
            ));
        }
        self.initialized = true;
        self.redis_host = redis_host;
        self.redis_port = redis_port;
        self.redis_namespace = redis_namespace;
        self.comm_name = comm_name;
        self.self_rank = self_rank;
        self.listen_port = listen_port;
        self.host_override = host_override;

        if self.self_rank < num_peers - 1 {
            let listener = TcpListener::bind(("0.0.0.0", self.listen_port as u16)).map_err(|e| {
                CylonError::new(
                    Code::IoError,
                    format!(
                        "direct-redis: bind() on port {} failed: {}",
                        self.listen_port, e
                    ),
                )
            })?;
            listener.set_nonblocking(true).map_err(|e| {
                CylonError::new(
                    Code::IoError,
                    format!("direct-redis: set_nonblocking on listener failed: {}", e),
                )
            })?;

            self.running.store(true, std::sync::atomic::Ordering::SeqCst);
            let running = self.running.clone();
            let accepted = self.accepted.clone();
            let thread_listener = listener.try_clone().map_err(|e| {
                CylonError::new(
                    Code::IoError,
                    format!("direct-redis: listener try_clone failed: {}", e),
                )
            })?;
            self.listener = Some(listener);
            self.accept_thread = Some(std::thread::spawn(move || {
                accept_loop(thread_listener, running, accepted);
            }));
        }

        let own_addr = self.resolve_own_address()?;
        self.publish_own_address(&own_addr)
    }

    /// Obtain a socket to `partner_id`. Ranks above the partner dial out; ranks
    /// below wait for the partner's inbound connection.
    pub fn connect(
        &self,
        self_rank: PeerNum,
        partner_id: PeerNum,
        timeout_ms: i32,
        mode: Mode,
    ) -> CylonResult<TcpStream> {
        if partner_id < self_rank {
            return self.dial_peer(self_rank, partner_id, timeout_ms, mode);
        }
        self.await_peer(partner_id, timeout_ms, mode)
    }

    fn dial_peer(
        &self,
        self_rank: PeerNum,
        partner_id: PeerNum,
        timeout_ms: i32,
        mode: Mode,
    ) -> CylonResult<TcpStream> {
        let addr = self.lookup_peer_address(partner_id, timeout_ms)?;
        let colon = addr.rfind(':').ok_or_else(|| {
            CylonError::new(
                Code::Invalid,
                format!(
                    "direct-redis: rank {} (comm_name={}) read a malformed address for partner {} — no host:port separator in \"{}\"",
                    self_rank, self.comm_name, partner_id, addr
                ),
            )
        })?;
        let peer_host = &addr[..colon];
        let peer_port: u16 = addr[colon + 1..].parse().map_err(|_| {
            CylonError::new(
                Code::Invalid,
                format!(
                    "direct-redis: rank {} (comm_name={}) read a malformed address for partner {} — unparseable port in \"{}\"",
                    self_rank, self.comm_name, partner_id, addr
                ),
            )
        })?;

        let mut stream = TcpStream::connect((peer_host, peer_port)).map_err(|e| {
            log::warn!(
                "direct-redis: connect() to peer {} at {} failed: {}",
                partner_id,
                addr,
                e
            );
            CylonError::new(
                Code::IoError,
                format!("direct-redis: connect to peer {} at {} failed: {}", partner_id, addr, e),
            )
        })?;

        use std::io::Write;
        let rank_net = (self_rank as u32).to_be_bytes();
        stream.write_all(&rank_net).map_err(|e| {
            CylonError::new(Code::IoError, format!("direct-redis: rank handshake send failed: {}", e))
        })?;
        stream.write_all(&[encode_mode_byte(mode)]).map_err(|e| {
            CylonError::new(Code::IoError, format!("direct-redis: mode handshake send failed: {}", e))
        })?;
        Ok(stream)
    }

    fn await_peer(
        &self,
        partner_id: PeerNum,
        timeout_ms: i32,
        mode: Mode,
    ) -> CylonResult<TcpStream> {
        let key = peer_and_mode_key(partner_id, mode);
        let (lock, cv) = &*self.accepted;
        let guard = lock.lock().unwrap_or_else(|e| e.into_inner());
        let deadline = Duration::from_millis(timeout_ms as u64);
        let (mut guard, timeout) = cv
            .wait_timeout_while(guard, deadline, |map| !map.contains_key(&key))
            .unwrap();
        if timeout.timed_out() {
            return Err(CylonError::new(
                Code::IoError,
                format!(
                    "direct-redis: timed out waiting for peer {} to connect (mode key {})",
                    partner_id, key
                ),
            ));
        }
        guard.remove(&key).ok_or_else(|| {
            CylonError::new(
                Code::InvalidState,
                format!("direct-redis: accepted socket for peer {} vanished", partner_id),
            )
        })
    }

    fn resolve_own_address(&self) -> CylonResult<String> {
        if !self.host_override.is_empty() {
            return Ok(format!("{}:{}", self.host_override, self.listen_port));
        }
        let metadata_uri = std::env::var("ECS_CONTAINER_METADATA_URI_V4").map_err(|_| {
            CylonError::new(
                Code::Invalid,
                "direct-redis: no host_override and ECS_CONTAINER_METADATA_URI_V4 is not set — cannot resolve own address".to_string(),
            )
        })?;
        let (host, port, path) = split_metadata_uri(&metadata_uri)?;
        let body = http_get(&host, port, &path)?;
        let ip = parse_ipv4_from_metadata(&body)?;
        Ok(format!("{}:{}", ip, self.listen_port))
    }
}

fn recv_exactly(stream: &mut TcpStream, dst: &mut [u8]) -> bool {
    use std::io::Read;
    stream.read_exact(dst).is_ok()
}

fn accept_loop(
    listener: TcpListener,
    running: std::sync::Arc<AtomicBool>,
    accepted: std::sync::Arc<(Mutex<HashMap<i32, TcpStream>>, Condvar)>,
) {
    while running.load(std::sync::atomic::Ordering::SeqCst) {
        let stream = match listener.accept() {
            Ok((stream, _)) => stream,
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(POLL_INTERVAL_MS));
                continue;
            }
            Err(e) => {
                log::warn!(
                    "direct-redis: accept() failed: {} — continuing to accept further peers",
                    e
                );
                continue;
            }
        };
        if !running.load(std::sync::atomic::Ordering::SeqCst) {
            return;
        }

        let accepted = accepted.clone();
        std::thread::spawn(move || {
            let mut stream = stream;
            if stream.set_nonblocking(false).is_err() {
                return;
            }
            stream
                .set_read_timeout(Some(Duration::from_millis(HANDSHAKE_TIMEOUT_MS)))
                .ok();

            let mut rank_buf = [0u8; 4];
            let mut mode_buf = [0u8; 1];
            if !recv_exactly(&mut stream, &mut rank_buf) || !recv_exactly(&mut stream, &mut mode_buf) {
                log::warn!(
                    "direct-redis: rank/mode handshake failed on an accepted connection, dropping it and continuing to accept further peers"
                );
                return;
            }
            let from_peer = u32::from_be_bytes(rank_buf) as i32;
            let from_mode = match decode_mode_byte(mode_buf[0]) {
                Some(m) => m,
                None => {
                    log::warn!(
                        "direct-redis: rank/mode handshake failed on an accepted connection, dropping it and continuing to accept further peers"
                    );
                    return;
                }
            };

            let key = peer_and_mode_key(from_peer, from_mode);
            let (lock, cv) = &*accepted;
            {
                let mut map = lock.lock().unwrap_or_else(|e| e.into_inner());
                if map.contains_key(&key) {
                    log::warn!(
                        "direct-redis: a second {:?} connection from peer {} arrived before the first was consumed — closing the superseded one",
                        from_mode,
                        from_peer
                    );
                }
                map.insert(key, stream);
            }
            log::info!(
                "direct-redis: accepted {:?} connection from peer {}",
                from_mode,
                from_peer
            );
            cv.notify_all();
        });
    }
}

impl Drop for RedisDirectEstablisher {
    fn drop(&mut self) {
        self.finalize();
    }
}
