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

//! Libfabric (Open Fabrics Interfaces) networking components
//!
//! This module provides a communication layer using libfabric for high-performance
//! networking with support for multiple providers (EFA, TCP, shared memory, etc.)
//! and native collective operations.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    LibfabricCommunicator                        │
//! │         (implements Cylon Communicator trait)                   │
//! └─────────────────────────────────────────────────────────────────┘
//!                                │
//!                                ▼
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                      FabricContext                              │
//! │    (fabric, domain, endpoint, completion queue, AV)            │
//! └─────────────────────────────────────────────────────────────────┘
//!                                │
//!                                ▼
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    Libfabric FFI Bindings                       │
//! └─────────────────────────────────────────────────────────────────┘
//!                                │
//!                                ▼
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                      Provider Layer                              │
//! │  ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐   │
//! │  │  efa  │ │  shm  │ │  tcp  │ │ verbs │ │ psm2  │ │sockets│   │
//! │  └───────┘ └───────┘ └───────┘ └───────┘ └───────┘ └───────┘   │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

use std::any::Any;
use crate::net::{CommConfig, CommType};

pub mod libfabric_sys;
pub mod error;
pub mod context;
pub mod endpoint;
pub mod address_vector;
pub mod oob;
pub mod operations;
pub mod communicator;
pub mod channel;

pub use error::*;
pub use context::*;
pub use endpoint::*;
pub use address_vector::*;
pub use oob::*;
pub use operations::*;
pub use communicator::*;
pub use channel::*;

/// Endpoint type for libfabric
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EndpointType {
    /// Reliable datagram - connectionless, reliable delivery
    /// Most common choice for collective operations
    #[default]
    ReliableDatagram,
    /// Message - connection-oriented
    Message,
}

/// Progress mode for libfabric operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ProgressMode {
    /// Automatic progress - provider handles progress internally
    #[default]
    Auto,
    /// Manual progress - application must call progress functions
    Manual,
}

/// Out-of-band communication type for address exchange
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OOBType {
    /// Redis-based OOB (default)
    #[default]
    Redis,
    /// MPI-based OOB (when running under MPI)
    Mpi,
}

/// Libfabric Configuration
///
/// Configuration for creating a Libfabric-based communicator.
#[derive(Debug, Clone)]
pub struct LibfabricConfig {
    /// Force specific provider (None = auto-select based on environment)
    /// Examples: "efa", "tcp", "shm", "verbs", "sockets"
    pub provider: Option<String>,

    /// Endpoint type
    pub endpoint_type: EndpointType,

    /// Completion queue size
    pub cq_size: usize,

    /// Address vector size (maximum number of peers)
    pub av_size: usize,

    /// Progress mode
    pub progress_mode: ProgressMode,

    /// OOB type for address exchange
    pub oob_type: OOBType,

    /// Redis host for OOB communication
    pub redis_host: String,

    /// Redis port for OOB communication
    pub redis_port: u16,

    /// Session ID for namespace isolation in Redis
    pub session_id: String,

    /// World size (number of processes)
    pub world_size: i32,

    /// Maximum message size for eager protocol
    pub max_eager_size: usize,

    /// Timeout for operations in milliseconds
    pub timeout_ms: u64,
}

impl Default for LibfabricConfig {
    fn default() -> Self {
        Self {
            provider: None, // Auto-select
            endpoint_type: EndpointType::ReliableDatagram,
            cq_size: 1024,
            av_size: 256,
            progress_mode: ProgressMode::Auto,
            oob_type: OOBType::Redis,
            redis_host: "127.0.0.1".to_string(),
            redis_port: 6379,
            session_id: uuid::Uuid::new_v4().to_string(),
            world_size: 0, // Will be determined from Redis
            max_eager_size: 8192,
            timeout_ms: 30000,
        }
    }
}

impl LibfabricConfig {
    /// Create a new LibfabricConfig with Redis OOB
    pub fn with_redis(
        redis_host: &str,
        redis_port: u16,
        session_id: &str,
        world_size: i32,
    ) -> Self {
        Self {
            oob_type: OOBType::Redis,
            redis_host: redis_host.to_string(),
            redis_port,
            session_id: session_id.to_string(),
            world_size,
            ..Default::default()
        }
    }

    /// Create a new LibfabricConfig with a specific provider
    pub fn with_provider(provider: &str) -> Self {
        Self {
            provider: Some(provider.to_string()),
            ..Default::default()
        }
    }

    /// Builder pattern: set provider
    pub fn provider(mut self, provider: &str) -> Self {
        self.provider = Some(provider.to_string());
        self
    }

    /// Builder pattern: set endpoint type
    pub fn endpoint_type(mut self, ep_type: EndpointType) -> Self {
        self.endpoint_type = ep_type;
        self
    }

    /// Builder pattern: set completion queue size
    pub fn cq_size(mut self, size: usize) -> Self {
        self.cq_size = size;
        self
    }

    /// Builder pattern: set address vector size
    pub fn av_size(mut self, size: usize) -> Self {
        self.av_size = size;
        self
    }

    /// Builder pattern: set Redis connection
    pub fn redis(mut self, host: &str, port: u16) -> Self {
        self.redis_host = host.to_string();
        self.redis_port = port;
        self.oob_type = OOBType::Redis;
        self
    }

    /// Builder pattern: set session ID
    pub fn session_id(mut self, id: &str) -> Self {
        self.session_id = id.to_string();
        self
    }

    /// Builder pattern: set world size
    pub fn world_size(mut self, size: i32) -> Self {
        self.world_size = size;
        self
    }

    /// Builder pattern: set timeout
    pub fn timeout_ms(mut self, ms: u64) -> Self {
        self.timeout_ms = ms;
        self
    }
}

impl CommConfig for LibfabricConfig {
    fn get_type(&self) -> CommType {
        CommType::Libfabric
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
