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

//! MPI implementation of table broadcast
//!
//! Ported from cpp/src/cylon/net/mpi/mpi_operations.hpp (MpiTableBcastImpl)

use std::sync::Arc;
use mpi::environment::Universe;
use mpi::traits::*;

use crate::error::{Code, CylonError, CylonResult};
use crate::ctx::CylonContext;
use crate::table::Table;
use crate::net::ops::TableBcastImpl;
use crate::net::serialize::{serialize_table, deserialize_table};

/// MPI implementation of table broadcast
/// Corresponds to C++ MpiTableBcastImpl from cpp/src/cylon/net/mpi/mpi_operations.hpp
pub struct MpiTableBcastImpl {
    /// MPI universe reference
    universe: Arc<std::sync::Mutex<Option<Universe>>>,
    /// Rank of this process
    rank: i32,
}

impl MpiTableBcastImpl {
    /// Create a new MpiTableBcastImpl
    ///
    /// # Arguments
    /// * `universe` - MPI universe
    /// * `rank` - Rank of this process
    pub fn new(universe: Arc<std::sync::Mutex<Option<Universe>>>, rank: i32) -> Self {
        Self { universe, rank }
    }
}

impl TableBcastImpl for MpiTableBcastImpl {
    fn init(&mut self, _num_buffers: i32) {
        // For our simplified implementation, we don't need async operations yet
        // The C++ version initializes request/status vectors here
    }

    fn bcast_buffer_sizes(&self, buffer: &mut [i32], _count: i32, bcast_root: i32) -> CylonResult<()> {
        if let Some(ref universe) = *self.universe.lock().unwrap() {
            let world = universe.world();
            let root_process = world.process_at_rank(bcast_root);
            root_process.broadcast_into(buffer);
            Ok(())
        } else {
            Err(CylonError::new(Code::Invalid, "MPI not initialized"))
        }
    }

    fn bcast_buffer_data(&self, buf_data: &mut [u8], _send_count: i32, bcast_root: i32) -> CylonResult<()> {
        if let Some(ref universe) = *self.universe.lock().unwrap() {
            let world = universe.world();
            let root_process = world.process_at_rank(bcast_root);
            root_process.broadcast_into(buf_data);
            Ok(())
        } else {
            Err(CylonError::new(Code::Invalid, "MPI not initialized"))
        }
    }

    fn ibcast_buffer_data(&mut self, _buf_idx: i32, buf_data: &mut [u8], _send_count: i32, bcast_root: i32) -> CylonResult<()> {
        // For now, use synchronous broadcast
        // TODO: Implement non-blocking version with MPI_Ibcast when rsmpi supports it
        self.bcast_buffer_data(buf_data, _send_count, bcast_root)
    }

    fn wait_all(&mut self, _num_buffers: i32) -> CylonResult<()> {
        // Since we're using synchronous operations for now, nothing to wait for
        Ok(())
    }

    fn execute(&mut self, table: &mut Option<Table>, bcast_root: i32, ctx: Arc<CylonContext>) -> CylonResult<()> {
        let is_root = self.rank == bcast_root;

        // Simplified implementation using our existing serialization:
        // 1. Serialize on root
        // 2. Broadcast size
        // 3. Broadcast data
        // 4. Deserialize on non-root

        let mut data = if is_root {
            if let Some(ref t) = table {
                serialize_table(t)?
            } else {
                return Err(CylonError::new(
                    Code::Invalid,
                    "Root process must have a table to broadcast".to_string()
                ));
            }
        } else {
            Vec::new()
        };

        // Broadcast size first
        let mut size = data.len() as i32;
        self.bcast_buffer_sizes(std::slice::from_mut(&mut size), 1, bcast_root)?;

        // Resize buffer on non-root
        if !is_root {
            data.resize(size as usize, 0);
        }

        // Broadcast actual data
        if size > 0 {
            self.bcast_buffer_data(&mut data, size, bcast_root)?;
        }

        // Deserialize on non-root
        if !is_root {
            *table = Some(deserialize_table(ctx, &data)?);
        }

        Ok(())
    }
}
