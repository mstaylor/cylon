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

//! Distributed table operations
//!
//! Ported from cpp/src/cylon/net/ops/

pub mod base_ops;
pub mod all_to_all;

#[cfg(feature = "mpi")]
pub mod bcast;
#[cfg(feature = "mpi")]
pub mod gather;

pub use base_ops::{Buffer, TableBcastImpl, TableGatherImpl, TableAllgatherImpl, AllReduceImpl, AllGatherImpl};
pub use all_to_all::{AllToAll, ReceiveCallback};
#[cfg(feature = "mpi")]
pub use bcast::MpiTableBcastImpl;
#[cfg(feature = "mpi")]
pub use gather::{MpiTableGatherImpl, MpiTableAllgatherImpl};
