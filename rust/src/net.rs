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

//! Networking and communication components
//!
//! Ported from cpp/src/cylon/net/

use async_trait::async_trait;
use std::sync::Arc;

use crate::error::CylonResult;

pub mod buffer;
pub mod channel;
pub mod comm_config;
pub mod comm_operations;
pub mod communicator;

#[cfg(feature = "mpi")]
pub mod mpi;

#[cfg(feature = "gloo")]
pub mod gloo;

// Re-exports for convenience
pub use comm_config::*;
pub use communicator::Communicator;

/// Communication type enum corresponding to C++ CommType
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommType {
    Local,
    #[cfg(feature = "mpi")]
    Mpi,
    #[cfg(feature = "gloo")]
    Gloo,
    Ucx,
    Redis,
}

/// Buffer trait for network communication
/// Corresponds to C++ Buffer interface
pub trait Buffer: Send + Sync {
    fn data(&self) -> &[u8];
    fn data_mut(&mut self) -> &mut [u8];
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Channel trait for point-to-point communication
/// Corresponds to C++ Channel interface
#[async_trait]
pub trait Channel: Send + Sync {
    async fn send(&self, buffer: &dyn Buffer, target: i32) -> CylonResult<()>;
    async fn receive(&self, buffer: &mut dyn Buffer, source: i32) -> CylonResult<()>;
    async fn isend(&self, buffer: &dyn Buffer, target: i32) -> CylonResult<Box<dyn Request>>;
    async fn ireceive(&self, buffer: &mut dyn Buffer, source: i32) -> CylonResult<Box<dyn Request>>;
}

/// Request trait for non-blocking operations
#[async_trait]
pub trait Request: Send + Sync {
    async fn wait(&mut self) -> CylonResult<()>;
    fn test(&mut self) -> CylonResult<bool>;
}

// The Communicator trait is now defined in communicator.rs to match C++ structure
// (cpp/src/cylon/net/communicator.hpp defines the base Communicator class,
//  cpp/src/cylon/net/mpi/mpi_communicator.cpp implements it)