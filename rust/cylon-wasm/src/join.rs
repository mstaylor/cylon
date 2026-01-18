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

//! Hash join implementation for WASM
//!
//! This module provides a thin wrapper around the core cylon join implementation,
//! adding JSON serialization support for the WASM API.

use serde::{Deserialize, Serialize};

use cylon::join::{hash_join_batches, JoinType as CylonJoinType};

use crate::table::Table;
use crate::error::{WasmError, WasmResult};

/// Join type enumeration (serializable for JSON API)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JoinType {
    Inner,
    Left,
    Right,
    FullOuter,
}

impl JoinType {
    /// Convert to cylon's JoinType
    fn to_cylon(self) -> CylonJoinType {
        match self {
            JoinType::Inner => CylonJoinType::Inner,
            JoinType::Left => CylonJoinType::Left,
            JoinType::Right => CylonJoinType::Right,
            JoinType::FullOuter => CylonJoinType::FullOuter,
        }
    }
}

/// Join configuration (serializable for JSON API)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JoinConfig {
    pub join_type: JoinType,
    pub left_on: Vec<usize>,
    pub right_on: Vec<usize>,
    #[serde(default = "default_left_suffix")]
    pub left_suffix: String,
    #[serde(default = "default_right_suffix")]
    pub right_suffix: String,
}

fn default_left_suffix() -> String { "_l".to_string() }
fn default_right_suffix() -> String { "_r".to_string() }

impl JoinConfig {
    pub fn new(join_type: JoinType, left_on: Vec<usize>, right_on: Vec<usize>) -> Self {
        Self {
            join_type,
            left_on,
            right_on,
            left_suffix: default_left_suffix(),
            right_suffix: default_right_suffix(),
        }
    }

    pub fn with_suffixes(mut self, left_suffix: String, right_suffix: String) -> Self {
        self.left_suffix = left_suffix;
        self.right_suffix = right_suffix;
        self
    }
}

/// Perform hash join operation using the core cylon implementation
pub fn hash_join(left: &Table, right: &Table, config: &JoinConfig) -> WasmResult<Table> {
    if config.left_on.len() != config.right_on.len() {
        return Err(WasmError::invalid("left_on and right_on must have same length"));
    }
    if config.left_on.is_empty() {
        return Err(WasmError::invalid("Must specify at least one join column"));
    }

    let left_batch = left.batch();
    let right_batch = right.batch();

    // Use the core cylon hash_join_batches implementation
    let result_batch = hash_join_batches(
        left_batch,
        right_batch,
        &config.left_on,
        &config.right_on,
        config.join_type.to_cylon(),
        &config.left_suffix,
        &config.right_suffix,
    ).map_err(|e| WasmError::execution_error(e.to_string()))?;

    Ok(Table::new(result_batch))
}