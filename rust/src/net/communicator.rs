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
use super::comm_operations::ReduceOp;

/// Element data type for the byte-level numeric reduce (`reduce_bytes`).
///
/// The byte buffer is interpreted as a flat little-endian array of this type.
/// This is deliberately a small, closed set — the numeric widths the operator
/// payloads use (float32 embeddings/scores, int counts) — not a general Arrow
/// type map.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceDtype {
    F32,
    F64,
    I32,
    I64,
}

impl ReduceDtype {
    /// Byte width of one element.
    pub fn size(self) -> usize {
        match self {
            ReduceDtype::F32 | ReduceDtype::I32 => 4,
            ReduceDtype::F64 | ReduceDtype::I64 => 8,
        }
    }
}

/// Whether `apply_reduce` supports the given op/dtype pair.
///
/// The byte-level reduce implements the arithmetic ops (Sum/Min/Max/Prod) on all
/// four numeric widths; the bitwise/logical ops (Band/Bor/Land/Lor) are not
/// implemented here because no operator payload needs them.
pub fn apply_reduce_supported(op: ReduceOp) -> bool {
    matches!(op, ReduceOp::Sum | ReduceOp::Min | ReduceOp::Max | ReduceOp::Prod)
}

/// Validate a `reduce_bytes` request before it hits the transport.
///
/// Fails fast at the boundary (per the "fail fast" rule) so the reduce closure
/// downstream never has to signal an error mid-collective.
pub fn validate_reduce_bytes(
    data: &[u8],
    op: ReduceOp,
    dtype: ReduceDtype,
) -> CylonResult<()> {
    if !apply_reduce_supported(op) {
        return Err(crate::error::CylonError::new(
            crate::error::Code::NotImplemented,
            format!("reduce_bytes does not implement op {:?} (arithmetic ops only)", op),
        ));
    }
    if data.len() % dtype.size() != 0 {
        return Err(crate::error::CylonError::new(
            crate::error::Code::ValueError,
            format!(
                "reduce_bytes: buffer of {} bytes is not a whole number of {:?} elements ({} bytes each)",
                data.len(),
                dtype,
                dtype.size()
            ),
        ));
    }
    Ok(())
}

/// Combine `other` into `acc` element-wise for a byte-level numeric reduce.
///
/// Both slices are little-endian arrays of `dtype` with the same length. Used
/// both by the emulated default (`reduce_bytes` local fold) and by the native
/// FMI reduce closure, so the fold semantics stay identical across backends.
/// Callers must have passed `validate_reduce_bytes` first; an unsupported op
/// returns an error rather than silently mutating `acc`.
pub fn apply_reduce(
    acc: &mut [u8],
    other: &[u8],
    op: ReduceOp,
    dtype: ReduceDtype,
) -> CylonResult<()> {
    macro_rules! fold {
        ($ty:ty, $combine:expr) => {{
            let sz = std::mem::size_of::<$ty>();
            let n = acc.len() / sz;
            let combine: fn($ty, $ty) -> $ty = $combine;
            for i in 0..n {
                let a = <$ty>::from_le_bytes(acc[i * sz..(i + 1) * sz].try_into().unwrap());
                let b = <$ty>::from_le_bytes(other[i * sz..(i + 1) * sz].try_into().unwrap());
                let r = combine(a, b);
                acc[i * sz..(i + 1) * sz].copy_from_slice(&r.to_le_bytes());
            }
        }};
    }

    match (dtype, op) {
        (ReduceDtype::F32, ReduceOp::Sum) => fold!(f32, |a, b| a + b),
        (ReduceDtype::F32, ReduceOp::Prod) => fold!(f32, |a, b| a * b),
        (ReduceDtype::F32, ReduceOp::Max) => fold!(f32, |a, b| a.max(b)),
        (ReduceDtype::F32, ReduceOp::Min) => fold!(f32, |a, b| a.min(b)),
        (ReduceDtype::F64, ReduceOp::Sum) => fold!(f64, |a, b| a + b),
        (ReduceDtype::F64, ReduceOp::Prod) => fold!(f64, |a, b| a * b),
        (ReduceDtype::F64, ReduceOp::Max) => fold!(f64, |a, b| a.max(b)),
        (ReduceDtype::F64, ReduceOp::Min) => fold!(f64, |a, b| a.min(b)),
        (ReduceDtype::I32, ReduceOp::Sum) => fold!(i32, |a, b| a.wrapping_add(b)),
        (ReduceDtype::I32, ReduceOp::Prod) => fold!(i32, |a, b| a.wrapping_mul(b)),
        (ReduceDtype::I32, ReduceOp::Max) => fold!(i32, |a, b| a.max(b)),
        (ReduceDtype::I32, ReduceOp::Min) => fold!(i32, |a, b| a.min(b)),
        (ReduceDtype::I64, ReduceOp::Sum) => fold!(i64, |a, b| a.wrapping_add(b)),
        (ReduceDtype::I64, ReduceOp::Prod) => fold!(i64, |a, b| a.wrapping_mul(b)),
        (ReduceDtype::I64, ReduceOp::Max) => fold!(i64, |a, b| a.max(b)),
        (ReduceDtype::I64, ReduceOp::Min) => fold!(i64, |a, b| a.min(b)),
        _ => {
            return Err(crate::error::CylonError::new(
                crate::error::Code::NotImplemented,
                format!("apply_reduce: op {:?} not supported (arithmetic ops only)", op),
            ));
        }
    }
    Ok(())
}

/// Communicator trait - main interface for distributed operations
/// Corresponds to C++ Communicator class from cpp/src/cylon/net/communicator.hpp
pub trait Communicator: Send + Sync {
    /// Enable downcasting to concrete communicator types
    fn as_any(&self) -> &dyn Any;
    fn get_rank(&self) -> i32;
    fn get_world_size(&self) -> i32;
    fn get_comm_type(&self) -> CommType;
    fn is_finalized(&self) -> bool;

    /// Create a new channel for this communicator
    ///
    /// Corresponds to C++ Communicator::CreateChannel() (communicator.hpp:44)
    fn create_channel(&self) -> CylonResult<Box<dyn super::Channel>>;

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

    /// Scatter equal-sized byte chunks from `root` to all ranks.
    ///
    /// `partitions` is meaningful only on `root`: exactly `world_size` chunks of
    /// equal length, one per rank. Every rank returns its own chunk. Byte-level
    /// (not in the C++ Communicator interface, like `all_to_all`/`allgather`).
    ///
    /// The default emulates the collective via `all_to_all`; backends with a
    /// native scatter (FMI) override this to use the binomial/direct path.
    fn scatter_bytes(&self, partitions: Vec<Vec<u8>>, root: i32) -> CylonResult<Vec<u8>> {
        let world_size = self.get_world_size() as usize;
        let send_data: Vec<Vec<u8>> = if self.get_rank() == root {
            if partitions.len() != world_size {
                return Err(crate::error::CylonError::new(
                    crate::error::Code::ValueError,
                    format!(
                        "scatter_bytes requires {} partitions on root, got {}",
                        world_size,
                        partitions.len()
                    ),
                ));
            }
            partitions
        } else {
            vec![Vec::new(); world_size]
        };
        let results = self.all_to_all(send_data)?;
        Ok(results.get(root as usize).cloned().unwrap_or_default())
    }

    /// Scatter variable-sized byte chunks (scatterv) from `root` to all ranks.
    ///
    /// Same contract as `scatter_bytes` but chunk lengths may differ per rank.
    /// The default emulates via `all_to_all` (which is size-agnostic); backends
    /// with a native scatterv (FMI) override this with the binomial/direct path.
    fn scatterv_bytes(&self, partitions: Vec<Vec<u8>>, root: i32) -> CylonResult<Vec<u8>> {
        // all_to_all already tolerates variable-length partitions, so the even and
        // uneven emulated paths are identical here; only native backends differ.
        self.scatter_bytes(partitions, root)
    }

    /// Element-wise numeric reduce of a byte buffer to `root`.
    ///
    /// `data` is a flat little-endian array of `dtype` elements; every rank passes
    /// the same element count. Returns the reduced buffer on `root` and an empty
    /// vector elsewhere. Only the arithmetic ops (Sum/Min/Max/Prod) are supported
    /// — the ops the five operators actually use.
    ///
    /// The default emulates via `allgather` + a local fold; backends with a native
    /// reduce (FMI) override this to use the binomial tree.
    fn reduce_bytes(
        &self,
        data: &[u8],
        root: i32,
        op: crate::net::comm_operations::ReduceOp,
        dtype: ReduceDtype,
    ) -> CylonResult<Vec<u8>> {
        validate_reduce_bytes(data, op, dtype)?;
        let all = self.allgather(data)?;
        if self.get_rank() != root {
            return Ok(Vec::new());
        }
        let mut acc = all.first().cloned().unwrap_or_default();
        for chunk in all.iter().skip(1) {
            apply_reduce(&mut acc, chunk, op, dtype)?;
        }
        Ok(acc)
    }

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

    /// AllGather tables from all processes to all processes
    ///
    /// Each process sends its table to all other processes.
    /// After this operation, every process has tables from all processes.
    ///
    /// # Arguments
    /// * `table` - Table from this process
    /// * `ctx` - CylonContext
    ///
    /// # Returns
    /// Vector of tables from all processes (index = source rank) on every process
    ///
    /// Corresponds to C++ Communicator::AllGather() in cpp/src/cylon/net/communicator.hpp
    fn all_gather(&self, table: &crate::table::Table, ctx: std::sync::Arc<crate::ctx::CylonContext>) -> CylonResult<Vec<crate::table::Table>>;

    // Column operations - these work with Cylon Column objects

    /// AllReduce on a Column
    ///
    /// # Arguments
    /// * `values` - Column to reduce
    /// * `reduce_op` - Reduction operation
    ///
    /// # Returns
    /// Reduced Column
    ///
    /// Corresponds to C++ Communicator::AllReduce(Column) in cpp/src/cylon/net/communicator.hpp
    fn all_reduce_column(
        &self,
        values: &crate::table::Column,
        reduce_op: super::comm_operations::ReduceOp,
    ) -> CylonResult<crate::table::Column>;

    /// Allgather Columns from all processes
    ///
    /// # Arguments
    /// * `values` - Column from this process
    ///
    /// # Returns
    /// Vector of Columns from all processes
    ///
    /// Corresponds to C++ Communicator::Allgather(Column) in cpp/src/cylon/net/communicator.hpp
    fn allgather_column(
        &self,
        values: &crate::table::Column,
    ) -> CylonResult<Vec<crate::table::Column>>;

    // Scalar operations - these work with Cylon Scalar objects

    /// AllReduce on a Scalar
    ///
    /// # Arguments
    /// * `value` - Scalar to reduce
    /// * `reduce_op` - Reduction operation
    ///
    /// # Returns
    /// Reduced Scalar
    ///
    /// Corresponds to C++ Communicator::AllReduce(Scalar) in cpp/src/cylon/net/communicator.hpp
    fn all_reduce_scalar(
        &self,
        value: &crate::scalar::Scalar,
        reduce_op: super::comm_operations::ReduceOp,
    ) -> CylonResult<crate::scalar::Scalar>;

    /// Allgather Scalars from all processes
    ///
    /// # Arguments
    /// * `value` - Scalar from this process
    ///
    /// # Returns
    /// Column containing scalars from all processes
    ///
    /// Corresponds to C++ Communicator::Allgather(Scalar) in cpp/src/cylon/net/communicator.hpp
    fn allgather_scalar(
        &self,
        value: &crate::scalar::Scalar,
    ) -> CylonResult<crate::table::Column>;
}
