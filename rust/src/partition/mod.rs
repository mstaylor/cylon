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

//! Partitioning operations
//!
//! Ported from cpp/src/cylon/partition/
//!
//! This module provides partitioning functionality similar to C++ Cylon's partition namespace.

pub mod hash_partition;

// Re-export main functions for convenience
pub use hash_partition::hash_partition;

// TODO: Port from cpp/src/cylon/partition/
// - MapToHashPartitions (hash_partition.rs implements this inline)
// - Split (hash_partition.rs implements this inline)
// - Repartition operations (partition.hpp/cpp)
