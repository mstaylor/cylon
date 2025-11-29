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

//! UCX (Unified Communication X) networking components
//!
//! Ported from cpp/src/cylon/net/ucx/

use std::any::Any;
use crate::net::{CommConfig, CommType};

pub mod ucx_sys;
pub mod oob_context;
pub mod redis_oob;
pub mod operations;
pub mod channel;
pub mod communicator;

pub use oob_context::*;
pub use redis_oob::*;
pub use operations::*;
pub use channel::*;
pub use communicator::*;

/// Out-of-band communication type
/// Corresponds to C++ OOBType from oob_type.hpp
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OOBType {
    /// MPI-based OOB
    Mpi,
    /// Redis-based OOB
    Redis,
}

/// UCX Configuration
///
/// Configuration for creating a UCX-based communicator.
/// Corresponds to C++ UCXConfig from ucx_communicator.hpp
#[derive(Debug, Clone)]
pub struct UCXConfig {
    /// OOB type (MPI or Redis)
    pub oob_type: OOBType,
    /// Redis host for OOB communication
    pub redis_host: String,
    /// Redis port for OOB communication
    pub redis_port: u16,
    /// Session ID for Redis-based OOB
    pub session_id: String,
    /// World size (number of processes)
    pub world_size: i32,
}

impl UCXConfig {
    /// Create a new UCXConfig with Redis OOB
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
        }
    }

    /// Create a new UCXConfig with MPI OOB
    pub fn with_mpi() -> Self {
        Self {
            oob_type: OOBType::Mpi,
            redis_host: String::new(),
            redis_port: 0,
            session_id: String::new(),
            world_size: 0, // Will be determined by MPI
        }
    }
}

impl CommConfig for UCXConfig {
    fn get_type(&self) -> CommType {
        CommType::Ucx
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
