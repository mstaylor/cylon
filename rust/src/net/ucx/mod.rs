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

//! UCX (Unified Communication X) networking components
//!
//! Ported from cpp/src/cylon/net/ucx/

pub mod ucx_sys;
pub mod oob_context;
pub mod redis_oob;
pub mod operations;
pub mod channel;
pub mod communicator;

pub use oob_context::*;
pub use redis_oob::*;
pub use operations::*;
pub use channel::*;
pub use communicator::*;

/// Out-of-band communication type
/// Corresponds to C++ OOBType from oob_type.hpp
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OOBType {
    /// MPI-based OOB
    Mpi,
    /// Redis-based OOB
    Redis,
}
