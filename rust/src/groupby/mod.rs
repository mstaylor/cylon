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

//! GroupBy operations
//!
//! Ported from cpp/src/cylon/groupby/
//!
//! This module provides groupby functionality similar to C++ Cylon's groupby namespace.
//! Implements both local hash-based groupby and distributed groupby.

pub mod hash_groupby;
pub mod groupby;

// Re-export main functions for convenience
pub use hash_groupby::{hash_groupby, hash_groupby_single};
pub use groupby::{distributed_hash_groupby, distributed_hash_groupby_single};

// TODO: Port from cpp/src/cylon/groupby/
// - DistributedPipelineGroupBy (groupby.hpp/cpp) - not used in Python bindings
// - Pipeline groupby (pipeline_groupby.hpp/cpp) - not used in Python bindings