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

//! Out-of-band context traits for UCX/UCC
//!
//! Ported from cpp/src/cylon/net/ucx/ucx_oob_context.hpp and ucc_oob_context.hpp

use crate::error::CylonResult;
use super::OOBType;

/// UCX Out-of-Band context trait
///
/// Corresponds to C++ UCXOOBContext from ucx_oob_context.hpp
///
/// This trait defines the interface for out-of-band communication
/// used to exchange UCX worker addresses during initialization.
pub trait UCXOOBContext: Send + Sync {
    /// Initialize the OOB context
    fn init_oob(&mut self) -> CylonResult<()>;

    /// Get world size and rank
    ///
    /// Returns (world_size, rank)
    fn get_world_size_and_rank(&mut self) -> CylonResult<(i32, i32)>;

    /// Perform an allgather operation for OOB address exchange
    ///
    /// # Arguments
    /// * `src` - Source buffer containing this process's data
    /// * `dst` - Destination buffer to receive all processes' data
    /// * `src_size` - Size of source data
    /// * `dst_size` - Total size of destination buffer
    fn oob_allgather(
        &mut self,
        src: &[u8],
        dst: &mut [u8],
        src_size: usize,
        dst_size: usize,
    ) -> CylonResult<()>;

    /// Finalize the OOB context
    fn finalize(&mut self) -> CylonResult<()>;
}

/// UCC Out-of-Band context trait
///
/// Corresponds to C++ UCCOOBContext from ucc_oob_context.hpp
///
/// This trait defines the interface for out-of-band communication
/// used by UCC for team creation and collective operations.
#[cfg(feature = "ucc")]
pub trait UCCOOBContext: Send + Sync {
    /// Initialize OOB with the given rank
    fn init_oob(&mut self, rank: i32);

    /// Create a UCX OOB context
    fn make_ucx_oob_context(&self) -> Box<dyn UCXOOBContext>;

    /// Get the collective info pointer for UCC
    ///
    /// This is used as the coll_info parameter in UCC OOB functions
    fn get_coll_info(&self) -> *mut std::ffi::c_void;

    /// Get the OOB type
    fn oob_type(&self) -> OOBType;

    /// Get the world size
    fn get_world_size(&self) -> i32;

    /// Get the rank
    fn get_rank(&self) -> i32;

    /// Set the rank
    fn set_rank(&mut self, rank: i32);

    /// Finalize the OOB context
    fn finalize(&mut self) -> CylonResult<()>;
}
