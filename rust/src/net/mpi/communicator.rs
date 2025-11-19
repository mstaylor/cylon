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

//! MPI Communicator implementation
//!
//! Ported from cpp/src/cylon/net/mpi/mpi_communicator.hpp and mpi_communicator.cpp
//!
//! Updated for rsmpi 0.8 API

use mpi::environment::Universe;
use mpi::traits::*; // Import all MPI traits
use std::sync::{Arc, Mutex};

use crate::error::{Code, CylonError, CylonResult};
use crate::net::{CommType, Communicator as CylonCommunicator}; // Alias our trait
use crate::net::ops::{TableBcastImpl, TableGatherImpl, TableAllgatherImpl, MpiTableBcastImpl, MpiTableGatherImpl, MpiTableAllgatherImpl};
use crate::table::Table;
use crate::ctx::CylonContext;

/// MPI Communicator
/// Corresponds to C++ MPICommunicator class from cpp/src/cylon/net/mpi/mpi_communicator.hpp
/// Updated for rsmpi 0.8 - stores Universe instead of raw MPI_Comm
pub struct MPICommunicator {
    rank: i32,
    world_size: i32,
    // MPI universe is stored to keep MPI initialized
    universe: Arc<Mutex<Option<Universe>>>,
    finalized: bool,
}

impl MPICommunicator {
    /// Get the raw MPI_Comm for low-level operations
    /// This is needed for operations like creating MPIChannel that require raw MPI_Comm
    #[cfg(feature = "mpi")]
    pub fn get_raw_comm(&self) -> CylonResult<mpi_sys::MPI_Comm> {
        if let Some(ref universe) = *self.universe.lock().unwrap() {
            use mpi::traits::*;
            Ok(universe.world().as_communicator().as_raw())
        } else {
            Err(CylonError::new(Code::Invalid, "MPI not initialized".to_string()))
        }
    }

    /// Create a new MPICommunicator (equivalent to C++ Make())
    /// Corresponds to MPICommunicator::Make() in cpp/src/cylon/net/mpi/mpi_communicator.cpp
    /// Updated for rsmpi 0.8 API
    pub fn make() -> CylonResult<Arc<dyn CylonCommunicator>> {
        // Initialize MPI using rsmpi 0.8 API
        let universe = mpi::initialize()
            .ok_or_else(|| CylonError::new(
                Code::Invalid,
                "Failed to initialize MPI (already initialized or MPI library not found)".to_string()
            ))?;

        // Get world communicator and query rank/size
        let world = universe.world();
        let rank = world.rank();
        let world_size = world.size();

        // Validate rank and world size
        if rank < 0 || world_size < 0 || rank >= world_size {
            return Err(CylonError::new(
                Code::ExecutionError,
                format!("Malformed rank: {} or world size: {}", rank, world_size)
            ));
        }

        Ok(Arc::new(Self {
            rank,
            world_size,
            universe: Arc::new(Mutex::new(Some(universe))),
            finalized: false,
        }))
    }
}

impl CylonCommunicator for MPICommunicator {
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
        CommType::Mpi
    }

    fn is_finalized(&self) -> bool {
        self.finalized
    }

    fn finalize(&mut self) -> CylonResult<()> {
        // Finalize MPI by dropping the universe
        // Corresponds to MPICommunicator::Finalize() in cpp/src/cylon/net/mpi/mpi_communicator.cpp
        if !self.finalized {
            let mut universe_lock = self.universe.lock().unwrap();
            *universe_lock = None; // Drop the universe, which finalizes MPI
            self.finalized = true;
        }
        Ok(())
    }

    fn barrier(&self) -> CylonResult<()> {
        // Barrier synchronization - all processes wait here
        // Corresponds to MPICommunicator::Barrier() in C++
        if let Some(ref universe) = *self.universe.lock().unwrap() {
            universe.world().barrier();
            Ok(())
        } else {
            Err(CylonError::new(Code::Invalid, "MPI not initialized"))
        }
    }

    fn send(&self, data: &[u8], dest: i32, _tag: i32) -> CylonResult<()> {
        // Point-to-point send using rsmpi 0.8 API
        // Note: rsmpi 0.8 doesn't support tags in the simple API
        if let Some(ref universe) = *self.universe.lock().unwrap() {
            let world = universe.world();
            world.process_at_rank(dest).send(data);
            Ok(())
        } else {
            Err(CylonError::new(Code::Invalid, "MPI not initialized"))
        }
    }

    fn recv(&self, buffer: &mut Vec<u8>, source: i32, _tag: i32) -> CylonResult<()> {
        // Point-to-point receive using rsmpi 0.8 API
        // Note: rsmpi 0.8 doesn't support tags in the simple API
        if let Some(ref universe) = *self.universe.lock().unwrap() {
            let world = universe.world();
            let (msg, _status) = world.process_at_rank(source).receive_vec::<u8>();
            *buffer = msg;
            Ok(())
        } else {
            Err(CylonError::new(Code::Invalid, "MPI not initialized"))
        }
    }

    fn all_to_all(&self, _send_data: Vec<Vec<u8>>) -> CylonResult<Vec<Vec<u8>>> {
        // NOTE: This byte-level operation doesn't exist in C++ Communicator
        // C++ only has Table/Column/Scalar level operations
        // TODO: Implement when needed for distributed operations
        Err(CylonError::new(
            Code::NotImplemented,
            "all_to_all not implemented - use Table-level operations"
        ))
    }

    fn allgather(&self, _data: &[u8]) -> CylonResult<Vec<Vec<u8>>> {
        // NOTE: This byte-level operation doesn't exist in C++ Communicator
        // C++ only has Table/Column/Scalar level operations
        // TODO: Implement when needed for distributed operations
        Err(CylonError::new(
            Code::NotImplemented,
            "allgather not implemented - use Table-level operations"
        ))
    }

    fn broadcast(&self, _data: &mut Vec<u8>, _root: i32) -> CylonResult<()> {
        // NOTE: This byte-level operation doesn't exist in C++ Communicator
        // C++ only has Table/Column/Scalar level operations
        // TODO: Implement when needed for distributed operations
        Err(CylonError::new(
            Code::NotImplemented,
            "broadcast not implemented - use Table-level operations (bcast)"
        ))
    }

    // Table operations

    fn bcast(&self, table: &mut Option<Table>, bcast_root: i32, ctx: Arc<CylonContext>) -> CylonResult<()> {
        // Create MPI table broadcast implementation and execute
        // Corresponds to MPICommunicator::Bcast() in cpp/src/cylon/net/mpi/mpi_communicator.cpp
        let mut impl_obj = MpiTableBcastImpl::new(self.universe.clone(), self.rank);
        impl_obj.execute(table, bcast_root, ctx)
    }

    fn gather(&self, table: &Table, gather_root: i32, gather_from_root: bool, ctx: Arc<CylonContext>) -> CylonResult<Vec<Table>> {
        // Create MPI table gather implementation and execute
        // Corresponds to MPICommunicator::Gather() in cpp/src/cylon/net/mpi/mpi_communicator.cpp
        let mut impl_obj = MpiTableGatherImpl::new(self.universe.clone(), self.rank, self.world_size);
        impl_obj.execute(table, gather_root, gather_from_root, ctx)
    }

    fn all_gather(&self, table: &Table, ctx: Arc<CylonContext>) -> CylonResult<Vec<Table>> {
        // Create MPI table allgather implementation and execute
        // Corresponds to MPICommunicator::AllGather() in cpp/src/cylon/net/mpi/mpi_communicator.cpp
        let mut impl_obj = MpiTableAllgatherImpl::new(self.universe.clone(), self.world_size);
        impl_obj.execute(table, ctx)
    }

    // Column and Scalar operations are TODO until those types are fully ported
    // The C++ implementation has:
    // - AllReduce(Column) -> Column
    // - Allgather(Column) -> vector<Column>
    // - AllReduce(Scalar) -> Scalar
    // - Allgather(Scalar) -> Column
}