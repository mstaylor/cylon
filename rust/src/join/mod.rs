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

//! Join operations for Cylon tables
//!
//! Ported from cpp/src/cylon/join/

pub mod config;
mod hash_join;
mod sort_join;
mod utils;

pub use config::{JoinConfig, JoinType, JoinAlgorithm};
use crate::table::Table;
use crate::error::CylonResult;

/// Join two tables based on the provided configuration
///
/// Corresponds to C++ JoinTables function
pub fn join(
    left: &Table,
    right: &Table,
    config: &JoinConfig,
) -> CylonResult<Table> {
    match config.algorithm() {
        JoinAlgorithm::Hash => {
            hash_join::hash_join(left, right, config)
        }
        JoinAlgorithm::Sort => {
            sort_join::sort_join(left, right, config)
        }
    }
}
