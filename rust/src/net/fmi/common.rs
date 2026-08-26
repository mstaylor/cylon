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

//! Common types for FMI communication
//!
//! This module corresponds to:
//! - cpp/src/cylon/thridparty/fmi/utils/Common.hpp
//! - cpp/src/cylon/thridparty/fmi/utils/Backends.hpp
//! - cpp/src/cylon/thridparty/fmi/utils/DirectBackend.hpp

use std::sync::Arc;
use std::sync::atomic::{AtomicI32, Ordering};

/// Type for peer IDs / numbers (matches FMI::Utils::peer_num)
pub type PeerNum = i32;

/// Communication mode (matches FMI::Utils::Mode)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Mode {
    Blocking,
    NonBlocking,
}

/// Status codes for non-blocking operations (matches FMI::Utils::NbxStatus)
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

/// Event processing status (matches FMI::Utils::EventProcessStatus)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventProcessStatus {
    Processing,
    Empty,
    Noop,
}

/// List of supported collective operations (matches FMI::Utils::Operation)
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

/// Optimization hint for channel policy (matches FMI::Utils::Hint)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hint {
    Fast,
    Cheap,
}

/// Backend types for FMI communication (matches FMI::Utils::BackendType)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendType {
    S3,
    Redis,
    Direct,
}

/// Completion context for tracking async operations (matches FMI::Utils::fmiContext)
/// Uses AtomicI32 for lock-free, non-blocking completion signaling.
#[derive(Debug)]
pub struct FmiContext {
    pub completed: AtomicI32,
}

impl FmiContext {
    pub fn new() -> Self {
        Self { completed: AtomicI32::new(0) }
    }

    pub fn is_completed(&self) -> bool {
        self.completed.load(Ordering::Acquire) != 0
    }

    /// Mark as completed (non-blocking, uses atomic store)
    pub fn mark_completed(&self) {
        self.completed.store(1, Ordering::Release);
    }

    /// Reset to incomplete state
    pub fn reset(&self) {
        self.completed.store(0, Ordering::Release);
    }
}

impl Default for FmiContext {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for FmiContext {
    fn clone(&self) -> Self {
        Self {
            completed: AtomicI32::new(self.completed.load(Ordering::Acquire)),
        }
    }
}

/// Information about an operation (matches FMI::Utils::OperationInfo)
#[derive(Debug, Clone)]
pub struct OperationInfo {
    pub op: Operation,
    pub data_size: usize,
    pub left_to_right: bool,
}

/// Raw function type for reduction operations (matches raw_function)
pub struct RawFunction {
    /// The reduction function: f(accumulator, value) -> modifies accumulator
    pub f: Box<dyn Fn(&mut [u8], &[u8]) + Send + Sync>,
    /// Whether the function is associative
    pub associative: bool,
    /// Whether the function is commutative
    pub commutative: bool,
}

impl RawFunction {
    pub fn new<F>(f: F, associative: bool, commutative: bool) -> Self
    where
        F: Fn(&mut [u8], &[u8]) + Send + Sync + 'static,
    {
        Self {
            f: Box::new(f),
            associative,
            commutative,
        }
    }

    /// Create a NOP function for barrier operations
    pub fn nop() -> Self {
        Self::new(|_, _| {}, true, true)
    }
}

/// Channel data buffer (matches channel_data struct)
///
/// This struct holds a buffer for channel operations. It uses shared ownership
/// to allow zero-copy sharing between operations.
#[derive(Clone)]
pub struct ChannelData {
    pub buf: Arc<std::sync::RwLock<Vec<u8>>>,
    pub len: usize,
}

impl ChannelData {
    /// Create new channel data from a vector
    pub fn new(data: Vec<u8>) -> Self {
        let len = data.len();
        Self {
            buf: Arc::new(std::sync::RwLock::new(data)),
            len,
        }
    }

    /// Create channel data with specified capacity (zeroed)
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buf: Arc::new(std::sync::RwLock::new(vec![0u8; capacity])),
            len: capacity,
        }
    }

    /// Create channel data from a raw pointer and length (for compatibility)
    ///
    /// # Safety
    /// The caller must ensure the pointer is valid for `len` bytes
    pub unsafe fn from_raw(ptr: *mut u8, len: usize) -> Self {
        let data = std::slice::from_raw_parts(ptr, len).to_vec();
        Self::new(data)
    }

    /// Create channel data from a slice (copies data)
    pub fn from_slice(data: &[u8]) -> Self {
        Self::new(data.to_vec())
    }

    /// Create an empty channel data (for non-root peers in gather)
    pub fn empty() -> Self {
        Self {
            buf: Arc::new(std::sync::RwLock::new(Vec::new())),
            len: 0,
        }
    }

    /// Get the data as a slice (read lock)
    pub fn as_slice(&self) -> std::sync::RwLockReadGuard<'_, Vec<u8>> {
        self.buf.read().unwrap()
    }

    /// Get the data as a mutable slice (write lock)
    pub fn as_mut_slice(&self) -> std::sync::RwLockWriteGuard<'_, Vec<u8>> {
        self.buf.write().unwrap()
    }

    /// Get a raw pointer to the data
    pub fn as_ptr(&self) -> *const u8 {
        self.buf.read().unwrap().as_ptr()
    }

    /// Get a mutable raw pointer to the data
    pub fn as_mut_ptr(&self) -> *mut u8 {
        self.buf.write().unwrap().as_mut_ptr()
    }
}

impl std::fmt::Debug for ChannelData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChannelData")
            .field("len", &self.len)
            .finish()
    }
}

/// Callback type for non-blocking operations
/// Using Arc for easy cloning across async operations
/// Takes &FmiContext (immutable) since FmiContext uses atomics internally
pub type NbxCallback = Arc<dyn Fn(NbxStatus, &str, &FmiContext) + Send + Sync>;

// ============================================================================
// Backend Configuration (matches Backends.hpp, DirectBackend.hpp)
// ============================================================================

/// Base backend configuration (matches FMI::Utils::Backends)
#[derive(Debug, Clone)]
pub struct Backends {
    enabled: bool,
    host: String,
    port: i32,
    max_timeout: i32,
    timeout: i32,
}

impl Default for Backends {
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

impl Backends {
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

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn get_host(&self) -> &str {
        &self.host
    }

    pub fn get_port(&self) -> i32 {
        self.port
    }

    pub fn get_timeout(&self) -> i32 {
        self.timeout
    }

    pub fn get_max_timeout(&self) -> i32 {
        self.max_timeout
    }

    pub fn get_backend_type(&self) -> BackendType {
        BackendType::Direct
    }

    pub fn get_name(&self) -> &str {
        "direct"
    }
}

/// Direct backend configuration (matches FMI::Utils::DirectBackend)
#[derive(Debug, Clone)]
pub struct DirectBackend {
    base: Backends,
    resolve_host_dns: bool,
    enable_host_ping: bool,
    use_direct_redis: bool,
    blocking_mode: Mode,
    advertise_host: Option<String>,
}

impl Default for DirectBackend {
    fn default() -> Self {
        Self {
            base: Backends::default(),
            resolve_host_dns: false,
            enable_host_ping: false,
            use_direct_redis: false,
            blocking_mode: Mode::Blocking,
            advertise_host: None,
        }
    }
}

impl DirectBackend {
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

    pub fn with_advertise_host(mut self, host: &str) -> Self {
        self.advertise_host = Some(host.to_string());
        self
    }

    pub fn advertise_host(&self) -> Option<&str> {
        self.advertise_host.as_deref()
    }

    pub fn with_port(mut self, port: i32) -> Self {
        self.base.port = port;
        self
    }

    pub fn with_max_timeout(mut self, max_timeout: i32) -> Self {
        self.base.max_timeout = max_timeout;
        self
    }

    pub fn with_timeout(mut self, timeout: i32) -> Self {
        self.base.timeout = timeout;
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

    /// Opt this backend into the direct-redis channel, which establishes peer
    /// sockets by publishing addresses through Redis rather than hole punching.
    pub fn set_use_direct_redis(mut self, use_it: bool) -> Self {
        self.use_direct_redis = use_it;
        self
    }

    pub fn set_blocking_mode(mut self, mode: Mode) -> Self {
        self.blocking_mode = mode;
        self
    }

    pub fn is_enabled(&self) -> bool {
        self.base.enabled
    }

    pub fn get_host(&self) -> &str {
        &self.base.host
    }

    pub fn get_port(&self) -> i32 {
        self.base.port
    }

    pub fn get_timeout(&self) -> i32 {
        self.base.timeout
    }

    pub fn get_max_timeout(&self) -> i32 {
        self.base.max_timeout
    }

    pub fn resolve_host_dns(&self) -> bool {
        self.resolve_host_dns
    }

    pub fn enable_host_ping(&self) -> bool {
        self.enable_host_ping
    }

    pub fn use_direct_redis(&self) -> bool {
        self.use_direct_redis
    }

    pub fn get_blocking_mode(&self) -> Mode {
        self.blocking_mode
    }

    pub fn get_backend_type(&self) -> BackendType {
        BackendType::Direct
    }

    pub fn get_name(&self) -> &str {
        "direct"
    }
}

/// Redis backend configuration (matches FMI::Utils::RedisBackend)
#[derive(Debug, Clone)]
pub struct RedisBackend {
    base: Backends,
}

impl Default for RedisBackend {
    fn default() -> Self {
        Self {
            base: Backends::default()
                .with_host("localhost")
                .with_port(6379)
                .with_timeout(100)
                .with_max_timeout(30000),
        }
    }
}

impl RedisBackend {
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

    pub fn with_timeout(mut self, timeout: i32) -> Self {
        self.base.timeout = timeout;
        self
    }

    pub fn is_enabled(&self) -> bool {
        self.base.enabled
    }

    pub fn get_host(&self) -> &str {
        &self.base.host
    }

    pub fn get_port(&self) -> i32 {
        self.base.port
    }

    pub fn get_timeout(&self) -> i32 {
        self.base.timeout
    }

    pub fn get_max_timeout(&self) -> i32 {
        self.base.max_timeout
    }

    pub fn get_backend_type(&self) -> BackendType {
        BackendType::Redis
    }

    pub fn get_name(&self) -> &str {
        "redis"
    }
}

/// S3 backend configuration (matches FMI::Utils::S3Backend)
#[derive(Debug, Clone)]
pub struct S3Backend {
    base: Backends,
    bucket_name: String,
    region: String,
    endpoint: Option<String>,
}

impl Default for S3Backend {
    fn default() -> Self {
        Self {
            base: Backends::default()
                .with_timeout(100)
                .with_max_timeout(30000),
            bucket_name: String::new(),
            region: "us-east-1".to_string(),
            endpoint: None,
        }
    }
}

impl S3Backend {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_enabled(mut self, enabled: bool) -> Self {
        self.base.enabled = enabled;
        self
    }

    pub fn with_bucket_name(mut self, bucket: &str) -> Self {
        self.bucket_name = bucket.to_string();
        self
    }

    pub fn with_region(mut self, region: &str) -> Self {
        self.region = region.to_string();
        self
    }

    pub fn with_endpoint(mut self, endpoint: &str) -> Self {
        self.endpoint = Some(endpoint.to_string());
        self
    }

    pub fn with_max_timeout(mut self, max_timeout: i32) -> Self {
        self.base.max_timeout = max_timeout;
        self
    }

    pub fn with_timeout(mut self, timeout: i32) -> Self {
        self.base.timeout = timeout;
        self
    }

    pub fn is_enabled(&self) -> bool {
        self.base.enabled
    }

    pub fn get_bucket_name(&self) -> &str {
        &self.bucket_name
    }

    pub fn get_region(&self) -> &str {
        &self.region
    }

    pub fn get_endpoint(&self) -> Option<&str> {
        self.endpoint.as_deref()
    }

    pub fn get_timeout(&self) -> i32 {
        self.base.timeout
    }

    pub fn get_max_timeout(&self) -> i32 {
        self.base.max_timeout
    }

    pub fn get_backend_type(&self) -> BackendType {
        BackendType::S3
    }

    pub fn get_name(&self) -> &str {
        "s3"
    }
}
