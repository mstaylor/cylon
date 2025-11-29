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
    redis_host: String,
    redis_port: i32,
    redis_namespace: String,
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
            redis_host: "localhost".to_string(),
            redis_port: 6379,
            redis_namespace: "cylon".to_string(),
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

    /// Set the TCPunch server port (default: 8080)
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
        let backend = DirectBackend::new()
            .with_host(&self.host)
            .with_port(self.port)
            .with_max_timeout(self.max_timeout)
            .set_resolve_dns(self.resolve_ip)
            .set_blocking_mode(mode)
            .set_enable_ping(self.enable_ping);

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
}
