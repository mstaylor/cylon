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

//! Cylon FMI Communicator implementation
//!
//! This module corresponds to cpp/src/cylon/net/fmi/fmi_communicator.hpp/cpp
//!
//! Provides the FMIConfig and FMICommunicator classes that integrate
//! the FMI communication layer with Cylon's Communicator trait.

use std::sync::Arc;

use crate::error::{CylonError, CylonResult, Code};
use crate::net::comm_config::CommConfig;
use crate::net::communicator::Communicator as CylonCommunicator;
use crate::net::{Channel, CommType};

use super::common::{DirectBackend, Mode};
use super::communicator::Communicator as FmiCommunicator;
use super::cylon_channel::FMICylonChannel;

/// Channel type as named at the configuration boundary.
///
/// Parsing is case-insensitive so that one spelling — `direct-redis` — works
/// identically across an `FMI_CHANNEL_TYPE` environment variable, a config file,
/// and the C++ and Cython clients. Internals use this typed value, never a string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelType {
    Direct,
    DirectRedis,
    Redis,
    S3,
}

impl std::str::FromStr for ChannelType {
    type Err = CylonError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "direct" => Ok(ChannelType::Direct),
            "direct-redis" => Ok(ChannelType::DirectRedis),
            "redis" => Ok(ChannelType::Redis),
            "s3" => Ok(ChannelType::S3),
            other => Err(CylonError::new(
                Code::Invalid,
                format!(
                    "unknown FMI channel type \"{}\" — expected one of: direct, direct-redis, redis, s3",
                    other
                ),
            )),
        }
    }
}

/// FMI Configuration (matches cylon::net::FMIConfig)
///
/// Configuration for creating an FMI-based communicator.
///
/// # Example using builder pattern
/// ```ignore
/// let config = FMIConfig::builder()
///     .rank(0)
///     .world_size(4)
///     .host("localhost")
///     .port(8080)
///     .build();
/// ```
#[derive(Debug, Clone)]
pub struct FMIConfig {
    rank: i32,
    world_size: i32,
    comm_name: String,
    backend: DirectBackend,
    nonblocking: bool,
    redis_host: String,
    redis_port: i32,
    redis_namespace: String,
}

/// Builder for FMIConfig with sensible defaults
#[derive(Debug, Clone)]
pub struct FMIConfigBuilder {
    rank: i32,
    world_size: i32,
    host: String,
    port: i32,
    max_timeout: i32,
    resolve_ip: bool,
    comm_name: String,
    nonblocking: bool,
    enable_ping: bool,
    use_direct_redis: bool,
    redis_host: String,
    redis_port: i32,
    redis_namespace: String,
    advertise_host: Option<String>,
}

impl Default for FMIConfigBuilder {
    fn default() -> Self {
        Self {
            rank: 0,
            world_size: 1,
            host: "localhost".to_string(),
            port: 8080,
            max_timeout: 30000,      // 30 seconds
            resolve_ip: false,
            comm_name: "cylon".to_string(),
            nonblocking: true,       // Default to non-blocking
            enable_ping: true,
            use_direct_redis: false,
            redis_host: "localhost".to_string(),
            redis_port: 6379,
            redis_namespace: "cylon".to_string(),
            advertise_host: None,
        }
    }
}

impl FMIConfigBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the rank of this process (required)
    pub fn rank(mut self, rank: i32) -> Self {
        self.rank = rank;
        self
    }

    /// Set the total number of processes (required)
    pub fn world_size(mut self, world_size: i32) -> Self {
        self.world_size = world_size;
        self
    }

    /// Set the TCPunch server host (default: "localhost")
    pub fn host(mut self, host: &str) -> Self {
        self.host = host.to_string();
        self
    }

    /// Explicit address to advertise to peers for the direct-redis channel (bypasses
    /// ECS metadata auto-discovery). Leave unset on Fargate/ECS to let
    /// `resolve_own_address` discover the task's real address automatically.
    pub fn advertise_host(mut self, host: &str) -> Self {
        self.advertise_host = Some(host.to_string());
        self
    }

    /// Set the TCPunch server port (default: 8080). TCPunch: the rendezvous
    /// server's port (a remote address). direct-redis: this rank's own listen
    /// port (a local bind).
    pub fn port(mut self, port: i32) -> Self {
        self.port = port;
        self
    }

    /// Set the maximum timeout in milliseconds (default: 30000)
    pub fn max_timeout(mut self, max_timeout: i32) -> Self {
        self.max_timeout = max_timeout;
        self
    }

    /// Set whether to resolve DNS (default: false)
    pub fn resolve_ip(mut self, resolve_ip: bool) -> Self {
        self.resolve_ip = resolve_ip;
        self
    }

    /// Set the communicator name (default: "cylon")
    pub fn comm_name(mut self, comm_name: &str) -> Self {
        self.comm_name = comm_name.to_string();
        self
    }

    /// Set whether to use non-blocking mode (default: true)
    pub fn nonblocking(mut self, nonblocking: bool) -> Self {
        self.nonblocking = nonblocking;
        self
    }

    /// Set whether to enable ping (default: true)
    pub fn enable_ping(mut self, enable_ping: bool) -> Self {
        self.enable_ping = enable_ping;
        self
    }

    /// Opt into the direct-redis channel: peers exchange listen addresses through
    /// Redis and connect directly, with no rendezvous server. Requires an
    /// environment where peers can bind and listen.
    pub fn use_direct_redis(mut self, use_it: bool) -> Self {
        self.use_direct_redis = use_it;
        self
    }

    /// Configuration-boundary entry point: set the channel from a `ChannelType`
    /// parsed via `ChannelType::from_str` (e.g. from an `FMI_CHANNEL_TYPE` env
    /// var or config file), threading it into `use_direct_redis`. Fails for
    /// `Redis`/`S3` since `FMIConfigBuilder` only constructs `Direct`/`DirectRedis`
    /// backends — those two channel types are not selectable through this builder.
    pub fn channel_type(self, ct: ChannelType) -> CylonResult<Self> {
        match ct {
            ChannelType::DirectRedis => Ok(self.use_direct_redis(true)),
            ChannelType::Direct => Ok(self.use_direct_redis(false)),
            ChannelType::Redis | ChannelType::S3 => Err(CylonError::new(
                Code::Invalid,
                format!(
                    "FMIConfigBuilder::channel_type does not support {:?} — only Direct and DirectRedis are constructible through this builder",
                    ct
                ),
            )),
        }
    }

    /// Set the Redis host for session management (default: "localhost")
    pub fn redis_host(mut self, redis_host: &str) -> Self {
        self.redis_host = redis_host.to_string();
        self
    }

    /// Set the Redis port (default: 6379)
    pub fn redis_port(mut self, redis_port: i32) -> Self {
        self.redis_port = redis_port;
        self
    }

    /// Set the Redis namespace (default: "cylon")
    pub fn redis_namespace(mut self, redis_namespace: &str) -> Self {
        self.redis_namespace = redis_namespace.to_string();
        self
    }

    /// Build the FMIConfig
    pub fn build(self) -> FMIConfig {
        let mode = if self.nonblocking { Mode::NonBlocking } else { Mode::Blocking };
        let mut backend = DirectBackend::new()
            .with_host(&self.host)
            .with_port(self.port)
            .with_max_timeout(self.max_timeout)
            .set_resolve_dns(self.resolve_ip)
            .set_blocking_mode(mode)
            .set_enable_ping(self.enable_ping)
            .set_use_direct_redis(self.use_direct_redis);

        if let Some(ref host) = self.advertise_host {
            backend = backend.with_advertise_host(host);
        }

        FMIConfig {
            rank: self.rank,
            world_size: self.world_size,
            comm_name: self.comm_name,
            backend,
            nonblocking: self.nonblocking,
            redis_host: self.redis_host,
            redis_port: self.redis_port,
            redis_namespace: self.redis_namespace,
        }
    }
}

impl FMIConfig {
    /// Create a builder for FMIConfig with sensible defaults
    ///
    /// # Example
    /// ```ignore
    /// let config = FMIConfig::builder()
    ///     .rank(0)
    ///     .world_size(4)
    ///     .host("tcpunch.example.com")
    ///     .port(8080)
    ///     .build();
    /// ```
    pub fn builder() -> FMIConfigBuilder {
        FMIConfigBuilder::new()
    }

    /// Create a new FMIConfig with explicit backend
    ///
    /// Matches C++ constructor:
    /// FMIConfig(int rank, int world_size, std::shared_ptr<FMI::Utils::Backends> backend,
    ///           std::string comm_name, bool nonblocking,
    ///           std::string redis_host, int redis_port, std::string redis_namespace)
    pub fn new(
        rank: i32,
        world_size: i32,
        backend: DirectBackend,
        comm_name: &str,
        nonblocking: bool,
        redis_host: &str,
        redis_port: i32,
        redis_namespace: &str,
    ) -> Self {
        Self {
            rank,
            world_size,
            comm_name: comm_name.to_string(),
            backend,
            nonblocking,
            redis_host: redis_host.to_string(),
            redis_port,
            redis_namespace: redis_namespace.to_string(),
        }
    }

    /// Create a new FMIConfig with host/port configuration
    ///
    /// Matches C++ constructor:
    /// FMIConfig(int rank, int world_size, std::string host, int port, int maxtimeout,
    ///           bool resolveIp, std::string comm_name, bool nonblocking)
    pub fn with_host(
        rank: i32,
        world_size: i32,
        host: &str,
        port: i32,
        max_timeout: i32,
        resolve_ip: bool,
        comm_name: &str,
        nonblocking: bool,
    ) -> Self {
        let mode = if nonblocking { Mode::NonBlocking } else { Mode::Blocking };
        let backend = DirectBackend::new()
            .with_host(host)
            .with_port(port)
            .with_max_timeout(max_timeout)
            .set_resolve_dns(resolve_ip)
            .set_blocking_mode(mode);

        Self {
            rank,
            world_size,
            comm_name: comm_name.to_string(),
            backend,
            nonblocking,
            redis_host: String::new(),
            redis_port: -1,
            redis_namespace: String::new(),
        }
    }

    /// Create a new FMIConfig with host/port and ping configuration
    ///
    /// Matches C++ constructor:
    /// FMIConfig(int rank, int world_size, std::string host, int port, int maxtimeout,
    ///           bool resolveIp, std::string comm_name, bool nonblocking, bool enablePing)
    pub fn with_host_and_ping(
        rank: i32,
        world_size: i32,
        host: &str,
        port: i32,
        max_timeout: i32,
        resolve_ip: bool,
        comm_name: &str,
        nonblocking: bool,
        enable_ping: bool,
    ) -> Self {
        let mode = if nonblocking { Mode::NonBlocking } else { Mode::Blocking };
        let backend = DirectBackend::new()
            .with_host(host)
            .with_port(port)
            .with_max_timeout(max_timeout)
            .set_resolve_dns(resolve_ip)
            .set_blocking_mode(mode)
            .set_enable_ping(enable_ping);

        Self {
            rank,
            world_size,
            comm_name: comm_name.to_string(),
            backend,
            nonblocking,
            redis_host: String::new(),
            redis_port: -1,
            redis_namespace: String::new(),
        }
    }

    /// Create a new FMIConfig with full configuration including Redis
    ///
    /// Matches C++ constructor:
    /// FMIConfig(int rank, int world_size, std::string host, int port, int maxtimeout,
    ///           bool resolveIp, std::string comm_name, bool nonblocking,
    ///           bool enablePing, std::string redis_host, int redis_port, std::string redis_namespace)
    pub fn with_redis(
        rank: i32,
        world_size: i32,
        host: &str,
        port: i32,
        max_timeout: i32,
        resolve_ip: bool,
        comm_name: &str,
        nonblocking: bool,
        enable_ping: bool,
        redis_host: &str,
        redis_port: i32,
        redis_namespace: &str,
    ) -> Self {
        let mode = if nonblocking { Mode::NonBlocking } else { Mode::Blocking };
        let backend = DirectBackend::new()
            .with_host(host)
            .with_port(port)
            .with_max_timeout(max_timeout)
            .set_resolve_dns(resolve_ip)
            .set_blocking_mode(mode)
            .set_enable_ping(enable_ping);

        Self {
            rank,
            world_size,
            comm_name: comm_name.to_string(),
            backend,
            nonblocking,
            redis_host: redis_host.to_string(),
            redis_port,
            redis_namespace: redis_namespace.to_string(),
        }
    }

    /// Static factory method matching C++ Make()
    pub fn make(
        rank: i32,
        world_size: i32,
        backend: DirectBackend,
        comm_name: &str,
        nonblocking: bool,
        redis_host: &str,
        redis_port: i32,
        redis_namespace: &str,
    ) -> Arc<Self> {
        Arc::new(Self::new(
            rank, world_size, backend, comm_name, nonblocking,
            redis_host, redis_port, redis_namespace,
        ))
    }

    pub fn get_rank(&self) -> i32 {
        self.rank
    }

    pub fn get_world_size(&self) -> i32 {
        self.world_size
    }

    pub fn get_comm_name(&self) -> &str {
        &self.comm_name
    }

    pub fn get_backend(&self) -> &DirectBackend {
        &self.backend
    }

    pub fn backend(&self) -> &DirectBackend {
        &self.backend
    }

    pub fn is_nonblocking(&self) -> bool {
        self.nonblocking
    }

    pub fn get_redis_host(&self) -> &str {
        &self.redis_host
    }

    pub fn get_redis_port(&self) -> i32 {
        self.redis_port
    }

    pub fn get_redis_namespace(&self) -> &str {
        &self.redis_namespace
    }
}

impl CommConfig for FMIConfig {
    fn get_type(&self) -> CommType {
        CommType::Fmi
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// FMI Communicator for Cylon (matches cylon::net::FMICommunicator)
///
/// This struct wraps the FMI Communicator and implements Cylon's Communicator trait.
pub struct FMICommunicator {
    rank: i32,
    world_size: i32,
    fmi_comm: Arc<FmiCommunicator>,
    nonblocking: bool,
    redis_host: String,
    redis_port: i32,
    redis_namespace: String,
}

impl FMICommunicator {
    /// Create a new FMICommunicator
    ///
    /// Matches C++ constructor:
    /// FMICommunicator(MemoryPool *pool, int32_t rank, int32_t world_size,
    ///                 const std::shared_ptr<FMI::Communicator> &fmi_comm, bool nonblocking)
    pub fn new(
        rank: i32,
        world_size: i32,
        fmi_comm: Arc<FmiCommunicator>,
        nonblocking: bool,
    ) -> Self {
        Self {
            rank,
            world_size,
            fmi_comm,
            nonblocking,
            redis_host: String::new(),
            redis_port: -1,
            redis_namespace: String::new(),
        }
    }

    /// Create a new FMICommunicator with Redis configuration
    ///
    /// Matches C++ constructor:
    /// FMICommunicator(MemoryPool *pool, int32_t rank, int32_t world_size,
    ///                 const std::shared_ptr<FMI::Communicator> &fmi_comm, bool nonblocking,
    ///                 std::string redis_host, int redis_port, std::string redis_namespace)
    pub fn with_redis(
        rank: i32,
        world_size: i32,
        fmi_comm: Arc<FmiCommunicator>,
        nonblocking: bool,
        redis_host: &str,
        redis_port: i32,
        redis_namespace: &str,
    ) -> Self {
        Self {
            rank,
            world_size,
            fmi_comm,
            nonblocking,
            redis_host: redis_host.to_string(),
            redis_port,
            redis_namespace: redis_namespace.to_string(),
        }
    }

    /// Create FMICommunicator from config
    ///
    /// Matches C++ static method:
    /// Status Make(const std::shared_ptr<CommConfig> &config, MemoryPool *pool,
    ///             std::shared_ptr<Communicator> *out)
    pub fn make(config: &FMIConfig) -> CylonResult<Arc<Self>> {
        // Create the underlying FMI communicator
        let fmi_comm = FmiCommunicator::new(
            config.get_rank(),
            config.get_world_size(),
            config.get_backend(),
            config.get_comm_name(),
            config.get_redis_host(),
            config.get_redis_port(),
            config.get_redis_namespace(),
        )?;

        let rank = fmi_comm.get_peer_id();
        let world_size = fmi_comm.get_num_peers();

        if rank < 0 || world_size < 0 || rank >= world_size {
            return Err(CylonError::new(
                Code::ExecutionError,
                format!("Malformed rank: {} or world size: {}", rank, world_size),
            ));
        }

        Ok(Arc::new(Self::with_redis(
            rank,
            world_size,
            Arc::new(fmi_comm),
            config.is_nonblocking(),
            config.get_redis_host(),
            config.get_redis_port(),
            config.get_redis_namespace(),
        )))
    }

    /// Get the blocking mode
    ///
    /// Matches C++ method: FMI::Utils::Mode getBlockingMode() const
    pub fn get_blocking_mode(&self) -> Mode {
        if self.nonblocking {
            Mode::NonBlocking
        } else {
            Mode::Blocking
        }
    }

    /// Get the underlying FMI communicator
    ///
    /// Matches C++ method: std::shared_ptr<FMI::Communicator> fmi_comm() const
    pub fn fmi_comm(&self) -> Arc<FmiCommunicator> {
        self.fmi_comm.clone()
    }
}

impl CylonCommunicator for FMICommunicator {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn get_rank(&self) -> i32 {
        self.rank
    }

    fn get_world_size(&self) -> i32 {
        self.world_size
    }

    fn get_comm_type(&self) -> CommType {
        CommType::Fmi
    }

    fn is_finalized(&self) -> bool {
        false
    }

    fn create_channel(&self) -> CylonResult<Box<dyn Channel>> {
        Ok(Box::new(FMICylonChannel::new(
            self.fmi_comm.clone(),
            self.get_blocking_mode(),
            &self.redis_host,
            self.redis_port,
            &self.redis_namespace,
        )))
    }

    fn finalize(&mut self) -> CylonResult<()> {
        // FMI cleanup is handled by Drop
        Ok(())
    }

    fn barrier(&self) -> CylonResult<()> {
        self.fmi_comm.barrier()
    }

    fn send(&self, data: &[u8], dest: i32, _tag: i32) -> CylonResult<()> {
        self.fmi_comm.send(data, dest)
    }

    fn recv(&self, buffer: &mut Vec<u8>, source: i32, _tag: i32) -> CylonResult<()> {
        self.fmi_comm.recv(buffer, source)
    }

    fn all_to_all(&self, send_data: Vec<Vec<u8>>) -> CylonResult<Vec<Vec<u8>>> {
        // FMI doesn't have native all-to-all, implement using point-to-point
        let world_size = self.world_size as usize;
        let rank = self.rank as usize;
        let mut recv_data = vec![Vec::new(); world_size];

        // Copy own data
        if rank < send_data.len() {
            recv_data[rank] = send_data[rank].clone();
        }

        // Exchange with all other processes
        for i in 0..world_size {
            if i != rank {
                // Send to process i
                if i < send_data.len() {
                    self.fmi_comm.send(&send_data[i], i as i32)?;
                }
                // Receive from process i
                let mut buf = vec![0u8; 1024 * 1024]; // 1MB buffer
                self.fmi_comm.recv(&mut buf, i as i32)?;
                recv_data[i] = buf;
            }
        }

        Ok(recv_data)
    }

    fn allgather(&self, data: &[u8]) -> CylonResult<Vec<Vec<u8>>> {
        let world_size = self.world_size as usize;
        let mut result = vec![Vec::new(); world_size];

        // Use FMI allgather
        let recv_buf_size = data.len() * world_size;
        let mut recv_buf = vec![0u8; recv_buf_size];
        self.fmi_comm.allgather(data, &mut recv_buf, 0)?;

        // Split into individual buffers
        let chunk_size = data.len();
        for i in 0..world_size {
            result[i] = recv_buf[i * chunk_size..(i + 1) * chunk_size].to_vec();
        }

        Ok(result)
    }

    fn broadcast(&self, data: &mut Vec<u8>, root: i32) -> CylonResult<()> {
        self.fmi_comm.bcast(data, root)
    }

    /// Native FMI even scatter over the FMI channel (real `Channel::scatter`, not
    /// the all_to_all emulation). On the Direct channel this is the O(log P) binomial
    /// tree (`scatter_binomial`), matching reduce/bcast/allgather and the C++ path.
    ///
    /// Every rank's chunk is the same length, so the chunk length is broadcast
    /// from `root` first; then `root`'s concatenated `partitions` are scattered.
    fn scatter_bytes(&self, partitions: Vec<Vec<u8>>, root: i32) -> CylonResult<Vec<u8>> {
        let world_size = self.world_size as usize;
        let rank = self.rank;

        let mut len_buf = if rank == root {
            if partitions.len() != world_size {
                return Err(CylonError::new(
                    crate::error::Code::ValueError,
                    format!(
                        "scatter_bytes requires {} partitions on root, got {}",
                        world_size,
                        partitions.len()
                    ),
                ));
            }
            let chunk_len = partitions.first().map(|p| p.len()).unwrap_or(0);
            if let Some(bad) = partitions.iter().find(|p| p.len() != chunk_len) {
                return Err(CylonError::new(
                    crate::error::Code::ValueError,
                    format!(
                        "scatter_bytes requires equal-length partitions ({} vs {}); use scatterv_bytes",
                        chunk_len,
                        bad.len()
                    ),
                ));
            }
            (chunk_len as u32).to_le_bytes().to_vec()
        } else {
            vec![0u8; 4]
        };
        self.fmi_comm.bcast(&mut len_buf, root)?;
        let chunk_len = u32::from_le_bytes(len_buf[..4].try_into().unwrap()) as usize;

        let sendbuf: Vec<u8> = if rank == root {
            let mut s = Vec::with_capacity(world_size * chunk_len);
            for p in &partitions {
                s.extend_from_slice(p);
            }
            s
        } else {
            Vec::new()
        };
        let mut recvbuf = vec![0u8; chunk_len];
        self.fmi_comm.scatter(&sendbuf, &mut recvbuf, root)?;
        Ok(recvbuf)
    }

    /// Native FMI uneven scatter (scatterv) over the FMI channel's real
    /// `Channel::scatterv`. On the Direct channel this is the O(log P) binomial tree
    /// (`scatterv_binomial`), like `scatter_bytes`. The per-rank byte counts are
    /// broadcast from `root`, displacements are the prefix sum, then `root`'s
    /// concatenated `partitions` are scattered with those counts.
    fn scatterv_bytes(&self, partitions: Vec<Vec<u8>>, root: i32) -> CylonResult<Vec<u8>> {
        let world_size = self.world_size as usize;
        let rank = self.rank;

        let mut counts_buf = if rank == root {
            if partitions.len() != world_size {
                return Err(CylonError::new(
                    crate::error::Code::ValueError,
                    format!(
                        "scatterv_bytes requires {} partitions on root, got {}",
                        world_size,
                        partitions.len()
                    ),
                ));
            }
            let mut b = Vec::with_capacity(world_size * 4);
            for p in &partitions {
                b.extend_from_slice(&(p.len() as i32).to_le_bytes());
            }
            b
        } else {
            vec![0u8; world_size * 4]
        };
        self.fmi_comm.bcast(&mut counts_buf, root)?;

        let sendcounts: Vec<i32> = (0..world_size)
            .map(|i| i32::from_le_bytes(counts_buf[i * 4..i * 4 + 4].try_into().unwrap()))
            .collect();
        let mut displs = vec![0i32; world_size];
        let mut acc = 0i32;
        for i in 0..world_size {
            displs[i] = acc;
            acc += sendcounts[i];
        }

        let sendbuf: Vec<u8> = if rank == root {
            let mut s = Vec::with_capacity(acc as usize);
            for p in &partitions {
                s.extend_from_slice(p);
            }
            s
        } else {
            Vec::new()
        };
        let my_len = sendcounts[rank as usize] as usize;
        let mut recvbuf = vec![0u8; my_len];
        self.fmi_comm
            .scatterv(&sendbuf, &mut recvbuf, root, &sendcounts, &displs)?;
        Ok(recvbuf)
    }

    /// Native FMI element-wise reduce (binomial tree on the Direct channel) instead
    /// of the allgather+fold emulation. The fold semantics come from the shared
    /// `apply_reduce`, so a native and an emulated reduce produce identical results.
    fn reduce_bytes(
        &self,
        data: &[u8],
        root: i32,
        op: crate::net::comm_operations::ReduceOp,
        dtype: crate::net::communicator::ReduceDtype,
    ) -> CylonResult<Vec<u8>> {
        crate::net::communicator::validate_reduce_bytes(data, op, dtype)?;
        let mut recvbuf = vec![0u8; data.len()];
        self.fmi_comm.reduce(
            data,
            &mut recvbuf,
            root,
            move |acc: &mut [u8], other: &[u8]| {
                // Validated at the boundary above, so this never errors for the
                // op/dtype pairs that reach here.
                let _ = crate::net::communicator::apply_reduce(acc, other, op, dtype);
            },
            true, // associative
            true, // commutative
        )?;
        if self.rank == root {
            Ok(recvbuf)
        } else {
            Ok(Vec::new())
        }
    }

    fn bcast(
        &self,
        table: &mut Option<crate::table::Table>,
        bcast_root: i32,
        ctx: std::sync::Arc<crate::ctx::CylonContext>,
    ) -> CylonResult<()> {
        use super::cylon_operations::FmiTableBcastImpl;
        use crate::net::ops::TableBcastImpl;

        let mut impl_ = FmiTableBcastImpl::new(self.fmi_comm.clone(), self.get_blocking_mode());
        impl_.execute(table, bcast_root, ctx)
    }

    fn gather(
        &self,
        table: &crate::table::Table,
        gather_root: i32,
        gather_from_root: bool,
        ctx: std::sync::Arc<crate::ctx::CylonContext>,
    ) -> CylonResult<Vec<crate::table::Table>> {
        use super::cylon_operations::FmiTableGatherImpl;
        use crate::net::ops::TableGatherImpl;

        let mut impl_ = FmiTableGatherImpl::new(self.fmi_comm.clone(), self.get_blocking_mode());
        impl_.execute(table, gather_root, gather_from_root, ctx)
    }

    fn all_gather(
        &self,
        table: &crate::table::Table,
        ctx: std::sync::Arc<crate::ctx::CylonContext>,
    ) -> CylonResult<Vec<crate::table::Table>> {
        use super::cylon_operations::FmiTableAllgatherImpl;
        use crate::net::ops::TableAllgatherImpl;

        let mut impl_ = FmiTableAllgatherImpl::new(self.fmi_comm.clone(), self.get_blocking_mode());
        impl_.execute(table, ctx)
    }

    fn all_reduce_column(
        &self,
        _values: &crate::table::Column,
        _reduce_op: crate::net::comm_operations::ReduceOp,
    ) -> CylonResult<crate::table::Column> {
        Err(CylonError::new(
            crate::error::Code::NotImplemented,
            "all_reduce_column not yet implemented for FMI",
        ))
    }

    fn allgather_column(
        &self,
        _values: &crate::table::Column,
    ) -> CylonResult<Vec<crate::table::Column>> {
        Err(CylonError::new(
            crate::error::Code::NotImplemented,
            "allgather_column not yet implemented for FMI",
        ))
    }

    fn all_reduce_scalar(
        &self,
        _value: &crate::scalar::Scalar,
        _reduce_op: crate::net::comm_operations::ReduceOp,
    ) -> CylonResult<crate::scalar::Scalar> {
        Err(CylonError::new(
            crate::error::Code::NotImplemented,
            "all_reduce_scalar not yet implemented for FMI",
        ))
    }

    fn allgather_scalar(
        &self,
        _value: &crate::scalar::Scalar,
    ) -> CylonResult<crate::table::Column> {
        Err(CylonError::new(
            crate::error::Code::NotImplemented,
            "allgather_scalar not yet implemented for FMI",
        ))
    }
}
