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

//! Communicator trait and related types
//!
//! Ported from cpp/src/cylon/net/communicator.hpp
//!
//! This module defines the base Communicator trait that all communication
//! backends (MPI, Gloo, etc.) must implement.

use crate::error::CylonResult;
use std::any::Any;

use super::CommType;

/// Communicator trait - main interface for distributed operations
/// Corresponds to C++ Communicator class from cpp/src/cylon/net/communicator.hpp
pub trait Communicator: Send + Sync {
    /// Enable downcasting to concrete communicator types
    fn as_any(&self) -> &dyn Any;
    fn get_rank(&self) -> i32;
    fn get_world_size(&self) -> i32;
    fn get_comm_type(&self) -> CommType;
    fn is_finalized(&self) -> bool;

    // TODO: Add create_channel when Channel is fully implemented
    // fn create_channel(&self) -> CylonResult<Box<dyn Channel>>;

    fn finalize(&mut self) -> CylonResult<()>;
    fn barrier(&self) -> CylonResult<()>;

    // Point-to-point communication primitives

    /// Send data to a specific process
    ///
    /// # Arguments
    /// * `data` - The data to send
    /// * `dest` - The destination process rank
    /// * `tag` - Message tag for identification
    fn send(&self, data: &[u8], dest: i32, tag: i32) -> CylonResult<()>;

    /// Receive data from a specific process
    ///
    /// # Arguments
    /// * `buffer` - Buffer to store received data
    /// * `source` - The source process rank
    /// * `tag` - Message tag for identification
    fn recv(&self, buffer: &mut Vec<u8>, source: i32, tag: i32) -> CylonResult<()>;

    // Collective communication primitives
    //
    // NOTE: The following byte-level operations do NOT exist in the C++ Communicator interface.
    // C++ only has Table/Column/Scalar level operations. These are provided for low-level
    // operations but may be removed in the future to match C++ API exactly.

    /// All-to-all communication: each process sends different data to each process
    ///
    /// # Arguments
    /// * `send_data` - Vector of data to send, indexed by destination rank
    ///
    /// # Returns
    /// Vector of data received from each process, indexed by source rank
    fn all_to_all(&self, send_data: Vec<Vec<u8>>) -> CylonResult<Vec<Vec<u8>>>;

    // NOTE: Byte-level gather doesn't exist in C++ Communicator interface
    // C++ only has Table/Column/Scalar level operations
    // Removed to avoid name collision with table-level gather below

    /// Gather data from all processes to all processes
    ///
    /// # Arguments
    /// * `data` - Data to send from this process
    ///
    /// # Returns
    /// Vector containing data from all processes, indexed by source rank
    fn allgather(&self, data: &[u8]) -> CylonResult<Vec<Vec<u8>>>;

    /// Broadcast data from root to all processes
    ///
    /// # Arguments
    /// * `data` - Data buffer (input on root, output on other processes)
    /// * `root` - The rank of the root process
    fn broadcast(&self, data: &mut Vec<u8>, root: i32) -> CylonResult<()>;

    // Table operations - these work with Cylon Table objects

    /// Broadcast a table from root process to all other processes
    ///
    /// # Arguments
    /// * `table` - Table to broadcast (Some on root, None on non-root initially).
    ///            After execution, all processes will have the same table.
    /// * `bcast_root` - The rank of the root process
    /// * `ctx` - CylonContext
    ///
    /// Corresponds to C++ Communicator::Bcast() in cpp/src/cylon/net/communicator.hpp
    fn bcast(&self, table: &mut Option<crate::table::Table>, bcast_root: i32, ctx: std::sync::Arc<crate::ctx::CylonContext>) -> CylonResult<()>;

    /// Gather tables from all processes to root process
    ///
    /// # Arguments
    /// * `table` - Table to gather
    /// * `gather_root` - The rank of the root process
    /// * `gather_from_root` - If true, root's table is included in results
    /// * `ctx` - CylonContext
    ///
    /// # Returns
    /// Vector of tables (only populated on root process)
    ///
    /// Corresponds to C++ Communicator::Gather() in cpp/src/cylon/net/communicator.hpp
    fn gather(&self, table: &crate::table::Table, gather_root: i32, gather_from_root: bool, ctx: std::sync::Arc<crate::ctx::CylonContext>) -> CylonResult<Vec<crate::table::Table>>;

    // TODO: Implement when operations are ported
    // fn all_gather(&self, table: &crate::table::Table) -> CylonResult<Vec<crate::table::Table>>;

    // Column operations - these work with Cylon Column objects
    // TODO: Implement when Column operations are ported
    // fn all_reduce_column(&self, values: &crate::table::Column, reduce_op: comm_operations::ReduceOp) -> CylonResult<crate::table::Column>;
    // fn allgather_column(&self, values: &crate::table::Column) -> CylonResult<Vec<crate::table::Column>>;

    // Scalar operations - these work with Cylon Scalar objects
    // TODO: Implement when Scalar operations are ported
    // fn all_reduce_scalar(&self, value: &crate::scalar::Scalar, reduce_op: comm_operations::ReduceOp) -> CylonResult<crate::scalar::Scalar>;
    // fn allgather_scalar(&self, value: &crate::scalar::Scalar) -> CylonResult<crate::table::Column>;
}
