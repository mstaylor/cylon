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

//! MPI implementation of table gather and allgather
//!
//! Ported from cpp/src/cylon/net/ops/gather.cpp

use std::sync::Arc;

#[cfg(feature = "mpi")]
use std::sync::Mutex;
#[cfg(feature = "mpi")]
use mpi::environment::Universe;
#[cfg(feature = "mpi")]
use mpi::traits::*;

use crate::error::{Code, CylonError, CylonResult};
use crate::ctx::CylonContext;
use crate::table::Table;
use crate::net::ops::{TableGatherImpl, TableAllgatherImpl};
use crate::net::serialize::{serialize_table, deserialize_table};

/// MPI implementation of table gather
/// Corresponds to C++ MpiTableGatherImpl from cpp/src/cylon/net/mpi/mpi_operations.hpp
#[cfg(feature = "mpi")]
pub struct MpiTableGatherImpl {
    /// MPI universe reference
    universe: Arc<Mutex<Option<Universe>>>,
    /// Rank of this process
    rank: i32,
    /// World size
    world_size: i32,
}

#[cfg(feature = "mpi")]
impl MpiTableGatherImpl {
    /// Create a new MpiTableGatherImpl
    ///
    /// # Arguments
    /// * `universe` - MPI universe
    /// * `rank` - Rank of this process
    /// * `world_size` - Total number of processes
    pub fn new(universe: Arc<Mutex<Option<Universe>>>, rank: i32, world_size: i32) -> Self {
        Self { universe, rank, world_size }
    }
}

#[cfg(feature = "mpi")]
impl TableGatherImpl for MpiTableGatherImpl {
    fn init(&mut self, _num_buffers: i32) {
        // For our simplified implementation, we don't need async operations yet
        // The C++ version initializes request/status vectors here
    }

    fn gather_buffer_sizes(
        &self,
        send_data: &[i32],
        _num_buffers: i32,
        rcv_data: &mut [i32],
        gather_root: i32,
    ) -> CylonResult<()> {
        if let Some(ref universe) = *self.universe.lock().unwrap() {
            let world = universe.world();
            let root_process = world.process_at_rank(gather_root);

            if self.rank == gather_root {
                // We're the root, gather into rcv_data
                root_process.gather_into_root(send_data, rcv_data);
            } else {
                // We're not the root, just send
                root_process.gather_into(send_data);
            }
            Ok(())
        } else {
            Err(CylonError::new(Code::Invalid, "MPI not initialized"))
        }
    }

    fn igather_buffer_data(
        &mut self,
        _buf_idx: i32,
        send_data: &[u8],
        send_count: i32,
        recv_data: &mut [u8],
        recv_count: &[i32],
        displacements: &[i32],
        gather_root: i32,
    ) -> CylonResult<()> {
        // For now, use synchronous gather (gatherv)
        // TODO: Implement non-blocking version with MPI_Igatherv when rsmpi supports it
        if let Some(ref universe) = *self.universe.lock().unwrap() {
            let world = universe.world();
            let root_process = world.process_at_rank(gather_root);

            let send_slice = &send_data[..send_count as usize];

            if self.rank == gather_root {
                // Root process: gather variable-length data
                // Create partition for variable-length receive
                // Note: rsmpi expects counts and displacements as references to slices
                let mut partition = mpi::datatype::PartitionMut::new(recv_data, recv_count, displacements);
                root_process.gather_varcount_into_root(send_slice, &mut partition);
            } else {
                // Non-root process: just send
                root_process.gather_varcount_into(send_slice);
            }
            Ok(())
        } else {
            Err(CylonError::new(Code::Invalid, "MPI not initialized"))
        }
    }

    fn wait_all(&mut self, _num_buffers: i32) -> CylonResult<()> {
        // Since we're using synchronous operations for now, nothing to wait for
        Ok(())
    }

    fn execute(
        &mut self,
        table: &Table,
        gather_root: i32,
        gather_from_root: bool,
        ctx: Arc<CylonContext>,
    ) -> CylonResult<Vec<Table>> {
        let is_root = self.rank == gather_root;

        // 1. Serialize local table
        let serialized_data = serialize_table(table)?;
        let send_size = serialized_data.len() as i32;

        // 2. Gather buffer sizes
        let mut all_sizes = if is_root {
            vec![0i32; self.world_size as usize]
        } else {
            vec![]
        };

        if is_root {
            self.gather_buffer_sizes(&[send_size], 1, &mut all_sizes, gather_root)?;
        } else {
            self.gather_buffer_sizes(&[send_size], 1, &mut [], gather_root)?;
        }

        // 3. Calculate displacements and total size (root only)
        let (total_size, displacements) = if is_root {
            let mut disps = vec![0i32; self.world_size as usize];
            let mut cumulative = 0i32;
            for i in 0..self.world_size as usize {
                disps[i] = cumulative;
                cumulative += all_sizes[i];
            }
            (cumulative as usize, disps)
        } else {
            (0, vec![])
        };

        // 4. Gather actual data
        let mut recv_buffer = if is_root {
            vec![0u8; total_size]
        } else {
            vec![]
        };

        if is_root {
            self.igather_buffer_data(
                0,
                &serialized_data,
                send_size,
                &mut recv_buffer,
                &all_sizes,
                &displacements,
                gather_root,
            )?;
        } else {
            self.igather_buffer_data(
                0,
                &serialized_data,
                send_size,
                &mut [],
                &[],
                &[],
                gather_root,
            )?;
        }

        self.wait_all(1)?;

        // 5. Deserialize tables (root only)
        let mut result = Vec::new();
        if is_root {
            let start_idx = if gather_from_root { 0 } else { 1 };
            for i in start_idx..self.world_size as usize {
                let start = displacements[i] as usize;
                let end = start + all_sizes[i] as usize;
                let table_data = &recv_buffer[start..end];
                let deserialized_table = deserialize_table(ctx.clone(), table_data)?;
                result.push(deserialized_table);
            }
        }

        Ok(result)
    }
}

/// MPI implementation of table allgather
/// Corresponds to C++ MpiTableAllgatherImpl from cpp/src/cylon/net/mpi/mpi_operations.hpp
#[cfg(feature = "mpi")]
pub struct MpiTableAllgatherImpl {
    /// MPI universe reference
    universe: Arc<Mutex<Option<Universe>>>,
    /// World size
    world_size: i32,
}

#[cfg(feature = "mpi")]
impl MpiTableAllgatherImpl {
    /// Create a new MpiTableAllgatherImpl
    ///
    /// # Arguments
    /// * `universe` - MPI universe
    /// * `world_size` - Total number of processes
    pub fn new(universe: Arc<Mutex<Option<Universe>>>, world_size: i32) -> Self {
        Self { universe, world_size }
    }
}

#[cfg(feature = "mpi")]
impl TableAllgatherImpl for MpiTableAllgatherImpl {
    fn init(&mut self, _num_buffers: i32) {
        // For our simplified implementation, we don't need async operations yet
    }

    fn allgather_buffer_sizes(
        &self,
        send_data: &[i32],
        _num_buffers: i32,
        rcv_data: &mut [i32],
    ) -> CylonResult<()> {
        if let Some(ref universe) = *self.universe.lock().unwrap() {
            let world = universe.world();
            world.all_gather_into(send_data, rcv_data);
            Ok(())
        } else {
            Err(CylonError::new(Code::Invalid, "MPI not initialized"))
        }
    }

    fn iallgather_buffer_data(
        &mut self,
        _buf_idx: i32,
        send_data: &[u8],
        send_count: i32,
        recv_data: &mut [u8],
        recv_count: &[i32],
        displacements: &[i32],
    ) -> CylonResult<()> {
        // For now, use synchronous allgatherv
        // TODO: Implement non-blocking version when rsmpi supports it
        if let Some(ref universe) = *self.universe.lock().unwrap() {
            let world = universe.world();

            let send_slice = &send_data[..send_count as usize];

            // Create partition for variable-length receive
            // Note: rsmpi expects counts and displacements as references to slices
            let mut partition = mpi::datatype::PartitionMut::new(recv_data, recv_count, displacements);
            world.all_gather_varcount_into(send_slice, &mut partition);

            Ok(())
        } else {
            Err(CylonError::new(Code::Invalid, "MPI not initialized"))
        }
    }

    fn wait_all(&mut self, _num_buffers: i32) -> CylonResult<()> {
        // Since we're using synchronous operations for now, nothing to wait for
        Ok(())
    }

    fn execute(&mut self, table: &Table, ctx: Arc<CylonContext>) -> CylonResult<Vec<Table>> {
        // 1. Serialize local table
        let serialized_data = serialize_table(table)?;
        let send_size = serialized_data.len() as i32;

        // 2. Allgather buffer sizes
        let mut all_sizes = vec![0i32; self.world_size as usize];
        self.allgather_buffer_sizes(&[send_size], 1, &mut all_sizes)?;

        // 3. Calculate displacements and total size
        let mut displacements = vec![0i32; self.world_size as usize];
        let mut cumulative = 0i32;
        for i in 0..self.world_size as usize {
            displacements[i] = cumulative;
            cumulative += all_sizes[i];
        }
        let total_size = cumulative as usize;

        // 4. Allgather actual data
        let mut recv_buffer = vec![0u8; total_size];
        self.iallgather_buffer_data(
            0,
            &serialized_data,
            send_size,
            &mut recv_buffer,
            &all_sizes,
            &displacements,
        )?;

        self.wait_all(1)?;

        // 5. Deserialize all tables
        let mut result = Vec::new();
        for i in 0..self.world_size as usize {
            let start = displacements[i] as usize;
            let end = start + all_sizes[i] as usize;
            let table_data = &recv_buffer[start..end];
            let deserialized_table = deserialize_table(ctx.clone(), table_data)?;
            result.push(deserialized_table);
        }

        Ok(result)
    }
}
