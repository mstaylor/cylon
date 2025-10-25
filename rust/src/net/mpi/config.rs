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

//! MPI configuration
//!
//! Ported from cpp/src/cylon/net/mpi/mpi_communicator.hpp (MPIConfig class)

use mpi::ffi::MPI_Comm;
use std::sync::Arc;

use crate::net::{CommConfig, CommType};

/// MPI configuration
/// Corresponds to C++ MPIConfig class
pub struct MPIConfig {
    /// MPI communicator (MPI_COMM_NULL by default)
    /// Corresponds to C++ member: MPI_Comm comm_
    comm: MPI_Comm,
}

impl MPIConfig {
    /// Create a new MPIConfig with the given MPI communicator
    /// Corresponds to C++ MPIConfig(MPI_Comm comm = MPI_COMM_NULL)
    pub fn new(comm: MPI_Comm) -> Self {
        Self { comm }
    }

    /// Get the MPI communicator
    /// Corresponds to C++ GetMPIComm() const
    pub fn get_mpi_comm(&self) -> MPI_Comm {
        self.comm
    }

    /// Create MPIConfig with MPI_COMM_NULL (equivalent to C++ Make(MPI_COMM comm = MPI_COMM_NULL))
    pub fn make(comm: MPI_Comm) -> Arc<Self> {
        Arc::new(Self::new(comm))
    }
}

impl Default for MPIConfig {
    fn default() -> Self {
        // Use null pointer as default (not used in rsmpi 0.8 anyway)
        Self::new(std::ptr::null_mut())
    }
}

// NOTE: CommConfig implementation removed because:
// 1. MPIConfig is not used in rsmpi 0.8 implementation
// 2. MPI_Comm raw pointer doesn't implement Send/Sync
// 3. C++ code is single-threaded, so Rust port should be too
//
// impl CommConfig for MPIConfig {
//     fn get_type(&self) -> CommType {
//         CommType::Mpi
//     }
// }