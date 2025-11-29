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

//! Core types for the FMI communication layer

use std::sync::Arc;

/// Type for peer IDs / numbers
pub type PeerNum = i32;

/// Communication mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Mode {
    Blocking,
    NonBlocking,
}

/// Status codes for non-blocking operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NbxStatus {
    Success,
    SendFailed,
    ReceiveFailed,
    DummySendFailed,
    ConnectionClosedByPeer,
    SocketCreateFailed,
    TcpNoDelayFailed,
    FcntlGetFailed,
    FcntlSetFailed,
    AddEventFailed,
    EpollWaitFailed,
    SocketPairFailed,
    SocketSetRcvTimeoFailed,
    SocketSetSndTimeoFailed,
    SocketSetTcpNoDelayFailed,
    SocketSetNonBlockingFailed,
    NbxTimeout,
}

/// Event processing status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventProcessStatus {
    Processing,
    Empty,
    Noop,
}

/// List of supported collective operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum Operation {
    Send = 0x001,
    Receive = 0x002,
    Bcast = 0x003,
    Barrier = 0x004,
    Allgather = 0x005,
    Allgatherv = 0x006,
    Gather = 0x007,
    Gatherv = 0x008,
    Scatter = 0x009,
    Reduce = 0x010,
    Allreduce = 0x011,
    Scan = 0x012,
    Default = 0x020,
}

/// Optimization hint for channel policy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hint {
    Fast,
    Cheap,
}

/// Backend types for FMI communication
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendType {
    S3,
    Redis,
    Direct,
}

/// Completion context for tracking async operations
#[derive(Debug, Clone)]
pub struct FmiContext {
    pub completed: bool,
}

impl FmiContext {
    pub fn new() -> Self {
        Self { completed: false }
    }

    pub fn is_completed(&self) -> bool {
        self.completed
    }

    pub fn mark_completed(&mut self) {
        self.completed = true;
    }
}

impl Default for FmiContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Information about an operation (for channel policy decisions)
#[derive(Debug, Clone)]
pub struct OperationInfo {
    pub op: Operation,
    pub data_size: usize,
    pub left_to_right: bool,
}

/// Raw function for reduction operations
/// The function takes two pointers (accumulator, value) and applies value to accumulator
pub type RawFunc = Box<dyn Fn(&mut [u8], &[u8]) + Send + Sync>;

/// Raw function with associativity/commutativity information
pub struct RawFunction {
    pub func: RawFunc,
    pub associative: bool,
    pub commutative: bool,
}

impl RawFunction {
    pub fn new<F>(func: F, associative: bool, commutative: bool) -> Self
    where
        F: Fn(&mut [u8], &[u8]) + Send + Sync + 'static,
    {
        Self {
            func: Box::new(func),
            associative,
            commutative,
        }
    }
}

/// Data buffer for channel operations
#[derive(Clone)]
pub struct ChannelData {
    buf: Arc<Vec<u8>>,
    offset: usize,
    len: usize,
}

impl ChannelData {
    /// Create new channel data from a vector
    pub fn new(data: Vec<u8>) -> Self {
        let len = data.len();
        Self {
            buf: Arc::new(data),
            offset: 0,
            len,
        }
    }

    /// Create channel data with specified capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buf: Arc::new(vec![0u8; capacity]),
            offset: 0,
            len: capacity,
        }
    }

    /// Create channel data from a slice (copies data)
    pub fn from_slice(data: &[u8]) -> Self {
        Self::new(data.to_vec())
    }

    /// Get the data as a slice
    pub fn as_slice(&self) -> &[u8] {
        &self.buf[self.offset..self.offset + self.len]
    }

    /// Get the data as a mutable slice (requires unique ownership)
    pub fn as_mut_slice(&mut self) -> Option<&mut [u8]> {
        Arc::get_mut(&mut self.buf).map(|v| &mut v[self.offset..self.offset + self.len])
    }

    /// Get the length of the data
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the data is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get the underlying pointer (for FFI)
    pub fn as_ptr(&self) -> *const u8 {
        self.buf[self.offset..].as_ptr()
    }

    /// Convert to owned Vec (may clone if shared)
    pub fn into_vec(self) -> Vec<u8> {
        match Arc::try_unwrap(self.buf) {
            Ok(mut v) => {
                if self.offset > 0 || self.len < v.len() {
                    v.drain(..self.offset);
                    v.truncate(self.len);
                }
                v
            }
            Err(arc) => arc[self.offset..self.offset + self.len].to_vec(),
        }
    }
}

impl std::fmt::Debug for ChannelData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChannelData")
            .field("len", &self.len)
            .field("offset", &self.offset)
            .finish()
    }
}

/// Backend configuration base
#[derive(Debug, Clone)]
pub struct BackendConfig {
    pub enabled: bool,
    pub host: String,
    pub port: i32,
    pub max_timeout: i32,
    pub timeout: i32,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            host: String::new(),
            port: -1,
            max_timeout: -1,
            timeout: -1,
        }
    }
}

impl BackendConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    pub fn with_host(mut self, host: &str) -> Self {
        self.host = host.to_string();
        self
    }

    pub fn with_port(mut self, port: i32) -> Self {
        self.port = port;
        self
    }

    pub fn with_max_timeout(mut self, max_timeout: i32) -> Self {
        self.max_timeout = max_timeout;
        self
    }

    pub fn with_timeout(mut self, timeout: i32) -> Self {
        self.timeout = timeout;
        self
    }
}

/// Direct backend configuration (uses TCPunch for NAT hole punching)
#[derive(Debug, Clone)]
pub struct DirectBackendConfig {
    pub base: BackendConfig,
    pub resolve_host_dns: bool,
    pub enable_host_ping: bool,
    pub blocking_mode: Mode,
}

impl Default for DirectBackendConfig {
    fn default() -> Self {
        Self {
            base: BackendConfig::default(),
            resolve_host_dns: false,
            enable_host_ping: false,
            blocking_mode: Mode::Blocking,
        }
    }
}

impl DirectBackendConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_enabled(mut self, enabled: bool) -> Self {
        self.base.enabled = enabled;
        self
    }

    pub fn with_host(mut self, host: &str) -> Self {
        self.base.host = host.to_string();
        self
    }

    pub fn with_port(mut self, port: i32) -> Self {
        self.base.port = port;
        self
    }

    pub fn with_max_timeout(mut self, max_timeout: i32) -> Self {
        self.base.max_timeout = max_timeout;
        self
    }

    pub fn set_resolve_dns(mut self, resolve: bool) -> Self {
        self.resolve_host_dns = resolve;
        self
    }

    pub fn set_enable_ping(mut self, enable: bool) -> Self {
        self.enable_host_ping = enable;
        self
    }

    pub fn set_blocking_mode(mut self, mode: Mode) -> Self {
        self.blocking_mode = mode;
        self
    }

    pub fn backend_type(&self) -> BackendType {
        BackendType::Direct
    }
}

/// Redis backend configuration
#[derive(Debug, Clone)]
pub struct RedisBackendConfig {
    pub base: BackendConfig,
    pub namespace: String,
}

impl Default for RedisBackendConfig {
    fn default() -> Self {
        Self {
            base: BackendConfig::default(),
            namespace: String::new(),
        }
    }
}

impl RedisBackendConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_enabled(mut self, enabled: bool) -> Self {
        self.base.enabled = enabled;
        self
    }

    pub fn with_host(mut self, host: &str) -> Self {
        self.base.host = host.to_string();
        self
    }

    pub fn with_port(mut self, port: i32) -> Self {
        self.base.port = port;
        self
    }

    pub fn with_namespace(mut self, namespace: &str) -> Self {
        self.namespace = namespace.to_string();
        self
    }

    pub fn backend_type(&self) -> BackendType {
        BackendType::Redis
    }
}
