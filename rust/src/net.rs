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
/// Corresponds to C++ Communicator class from cpp/src/cylon/net/communicator.hpp
pub trait Communicator: Send + Sync {
    fn get_rank(&self) -> i32;
    fn get_world_size(&self) -> i32;
    fn get_comm_type(&self) -> CommType;
    fn is_finalized(&self) -> bool;

    // TODO: Add create_channel when Channel is fully implemented
    // fn create_channel(&self) -> CylonResult<Box<dyn Channel>>;

    fn finalize(&mut self) -> CylonResult<()>;
    fn barrier(&self) -> CylonResult<()>;

    // Table operations - these work with Cylon Table objects
    // TODO: Implement when Table operations are ported
    // fn all_gather(&self, table: &crate::table::Table) -> CylonResult<Vec<crate::table::Table>>;
    // fn gather(&self, table: &crate::table::Table, gather_root: i32, gather_from_root: bool) -> CylonResult<Vec<crate::table::Table>>;
    // fn bcast(&self, table: &mut Option<crate::table::Table>, bcast_root: i32, ctx: &crate::ctx::CylonContext) -> CylonResult<()>;

    // Column operations - these work with Cylon Column objects
    // TODO: Implement when Column operations are ported
    // fn all_reduce_column(&self, values: &crate::table::Column, reduce_op: comm_operations::ReduceOp) -> CylonResult<crate::table::Column>;
    // fn allgather_column(&self, values: &crate::table::Column) -> CylonResult<Vec<crate::table::Column>>;

    // Scalar operations - these work with Cylon Scalar objects
    // TODO: Implement when Scalar operations are ported
    // fn all_reduce_scalar(&self, value: &crate::scalar::Scalar, reduce_op: comm_operations::ReduceOp) -> CylonResult<crate::scalar::Scalar>;
    // fn allgather_scalar(&self, value: &crate::scalar::Scalar) -> CylonResult<crate::table::Column>;
}