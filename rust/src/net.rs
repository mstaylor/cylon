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
pub use communicator::*;

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

/// Communicator trait - main interface for distributed operations
/// Corresponds to C++ Communicator interface
#[async_trait]
pub trait Communicator: Send + Sync {
    fn get_rank(&self) -> i32;
    fn get_world_size(&self) -> i32;
    fn get_comm_type(&self) -> CommType;

    async fn barrier(&self) -> CylonResult<()>;
    async fn finalize(&self) -> CylonResult<()>;

    // Collective operations
    async fn allreduce(&self, send_buf: &[u8], recv_buf: &mut [u8]) -> CylonResult<()>;
    async fn allgather(&self, send_buf: &[u8], recv_buf: &mut [u8]) -> CylonResult<()>;
    async fn alltoall(&self, send_buf: &[u8], recv_buf: &mut [u8]) -> CylonResult<()>;
    async fn broadcast(&self, buf: &mut [u8], root: i32) -> CylonResult<()>;
    async fn gather(&self, send_buf: &[u8], recv_buf: &mut [u8], root: i32) -> CylonResult<()>;
    async fn scatter(&self, send_buf: &[u8], recv_buf: &mut [u8], root: i32) -> CylonResult<()>;
}