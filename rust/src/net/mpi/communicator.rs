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

use mpi::environment::Universe;
use mpi::ffi::{MPI_Comm, MPI_COMM_NULL, MPI_COMM_WORLD};
use mpi::topology::{Communicator as MpiCommunicator, SystemCommunicator};
use std::sync::{Arc, Mutex};

use crate::error::{Code, CylonError, CylonResult};
use crate::net::{CommType, Communicator};
use super::config::MPIConfig;

/// MPI Communicator
/// Corresponds to C++ MPICommunicator class from cpp/src/cylon/net/mpi/mpi_communicator.hpp
pub struct MPICommunicator {
    rank: i32,
    world_size: i32,
    mpi_comm: MPI_Comm,
    // MPI universe is stored to keep MPI initialized
    universe: Arc<Mutex<Option<Universe>>>,
    externally_init: bool,
    finalized: bool,
}

impl MPICommunicator {
    /// Create a new MPICommunicator (equivalent to C++ Make())
    /// Corresponds to MPICommunicator::Make() in cpp/src/cylon/net/mpi/mpi_communicator.cpp
    pub fn make(config: &MPIConfig) -> CylonResult<Arc<dyn Communicator>> {
        // Get MPI communicator from config
        let mut mpi_comm = config.get_mpi_comm();

        // Check if MPI is already initialized
        let externally_init = mpi::is_initialized();

        // If comm is not MPI_COMM_NULL and MPI is not initialized, return error
        // Corresponds to C++ lines 64-66
        if mpi_comm != MPI_COMM_NULL && !externally_init {
            return Err(CylonError::new(
                Code::Invalid,
                "non-null MPI_Comm passed without initializing MPI".to_string()
            ));
        }

        // Initialize MPI if not already initialized
        // Corresponds to C++ lines 68-70
        let universe = if !externally_init {
            if let Some(univ) = mpi::initialize() {
                Some(univ)
            } else {
                return Err(CylonError::new(
                    Code::Invalid,
                    "Failed to initialize MPI".to_string()
                ));
            }
        } else {
            None
        };

        // If comm is MPI_COMM_NULL, use MPI_COMM_WORLD
        // Corresponds to C++ lines 72-74
        if mpi_comm == MPI_COMM_NULL {
            mpi_comm = MPI_COMM_WORLD;
        }

        // Get world communicator for rank and size
        let world = if let Some(ref univ) = universe {
            univ.world()
        } else {
            // When externally initialized, we can still access the world
            unsafe { SystemCommunicator::from_raw(MPI_COMM_WORLD) }
        };

        let rank = world.rank();
        let world_size = world.size();

        // Validate rank and world size
        // Corresponds to C++ lines 82-85
        if rank < 0 || world_size < 0 || rank >= world_size {
            return Err(CylonError::new(
                Code::ExecutionError,
                format!("Malformed rank: {} or world size: {}", rank, world_size)
            ));
        }

        Ok(Arc::new(Self {
            rank,
            world_size,
            mpi_comm,
            universe: Arc::new(Mutex::new(universe)),
            externally_init,
            finalized: false,
        }))
    }

    /// Get the MPI communicator
    /// Corresponds to C++ mpi_comm() const
    pub fn mpi_comm(&self) -> MPI_Comm {
        self.mpi_comm
    }
}

impl Communicator for MPICommunicator {
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
        // Finalize only if we initialized MPI (not externally initialized)
        // Corresponds to MPICommunicator::Finalize() in cpp/src/cylon/net/mpi/mpi_communicator.cpp
        if !self.externally_init && !self.finalized {
            let mut universe_lock = self.universe.lock().unwrap();
            *universe_lock = None; // Drop the universe, which finalizes MPI
            self.finalized = true;
        }
        Ok(())
    }

    fn barrier(&self) -> CylonResult<()> {
        // Corresponds to MPICommunicator::Barrier() in C++
        if let Some(ref universe) = *self.universe.lock().unwrap() {
            universe.world().barrier();
            Ok(())
        } else {
            Err(CylonError::new(Code::Invalid, "MPI universe not available"))
        }
    }

    // Table, Column, and Scalar operations are TODO until those types are fully ported
    // The C++ implementation has:
    // - AllGather(Table) -> vector<Table>
    // - Gather(Table) -> vector<Table>
    // - Bcast(Table)
    // - AllReduce(Column) -> Column
    // - Allgather(Column) -> vector<Column>
    // - AllReduce(Scalar) -> Scalar
    // - Allgather(Scalar) -> Column
}