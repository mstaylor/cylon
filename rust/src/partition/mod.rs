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
//!
//! ## Hash Partitioning
//! Use [`hash_partition`] for distributing data based on hash values of columns.
//! This is useful for operations like shuffle-based joins and groupby.
//!
//! ## Range Partitioning
//! Use [`map_to_sort_partitions`] for distributing data based on value ranges.
//! This is used by distributed sort to ensure globally sorted output.

pub mod hash_partition;
pub mod range_partition;

// Re-export main functions for convenience
pub use hash_partition::hash_partition;
pub use range_partition::{
    map_to_sort_partitions,
    split_by_partition,
    PartitionMapping,
    RangePartitionOptions,
};
