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

//! Join configuration
//!
//! Ported from cpp/src/cylon/join/join_config.hpp

use crate::error::{CylonError, CylonResult, Code};

/// Type of join operation
/// Corresponds to C++ JoinType enum
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    Inner,
    Left,
    Right,
    FullOuter,
}

/// Join algorithm to use
/// Corresponds to C++ JoinAlgorithm enum
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinAlgorithm {
    Sort,
    Hash,
}

/// Configuration for join operations
/// Corresponds to C++ JoinConfig class
#[derive(Debug, Clone)]
pub struct JoinConfig {
    join_type: JoinType,
    algorithm: JoinAlgorithm,
    left_column_indices: Vec<usize>,
    right_column_indices: Vec<usize>,
    left_table_suffix: String,
    right_table_suffix: String,
}

impl JoinConfig {
    /// Create a new join configuration
    pub fn new(
        join_type: JoinType,
        left_column_indices: Vec<usize>,
        right_column_indices: Vec<usize>,
        algorithm: JoinAlgorithm,
        left_table_suffix: String,
        right_table_suffix: String,
    ) -> CylonResult<Self> {
        if left_column_indices.len() != right_column_indices.len() {
            return Err(CylonError::new(
                Code::Invalid,
                "left and right column indices sizes are not equal".to_string(),
            ));
        }

        Ok(Self {
            join_type,
            algorithm,
            left_column_indices,
            right_column_indices,
            left_table_suffix,
            right_table_suffix,
        })
    }

    /// Create an inner join configuration
    pub fn inner_join(
        left_column_idx: usize,
        right_column_idx: usize,
    ) -> Self {
        Self {
            join_type: JoinType::Inner,
            algorithm: JoinAlgorithm::Hash,
            left_column_indices: vec![left_column_idx],
            right_column_indices: vec![right_column_idx],
            left_table_suffix: String::new(),
            right_table_suffix: String::new(),
        }
    }

    /// Create an inner join with multiple columns
    pub fn inner_join_multi(
        left_column_indices: Vec<usize>,
        right_column_indices: Vec<usize>,
    ) -> CylonResult<Self> {
        Self::new(
            JoinType::Inner,
            left_column_indices,
            right_column_indices,
            JoinAlgorithm::Hash,
            String::new(),
            String::new(),
        )
    }

    /// Create a left join configuration
    pub fn left_join(
        left_column_idx: usize,
        right_column_idx: usize,
    ) -> Self {
        Self {
            join_type: JoinType::Left,
            algorithm: JoinAlgorithm::Hash,
            left_column_indices: vec![left_column_idx],
            right_column_indices: vec![right_column_idx],
            left_table_suffix: String::new(),
            right_table_suffix: String::new(),
        }
    }

    /// Create a left join with multiple columns
    pub fn left_join_multi(
        left_column_indices: Vec<usize>,
        right_column_indices: Vec<usize>,
    ) -> CylonResult<Self> {
        Self::new(
            JoinType::Left,
            left_column_indices,
            right_column_indices,
            JoinAlgorithm::Hash,
            String::new(),
            String::new(),
        )
    }

    /// Create a right join configuration
    pub fn right_join(
        left_column_idx: usize,
        right_column_idx: usize,
    ) -> Self {
        Self {
            join_type: JoinType::Right,
            algorithm: JoinAlgorithm::Hash,
            left_column_indices: vec![left_column_idx],
            right_column_indices: vec![right_column_idx],
            left_table_suffix: String::new(),
            right_table_suffix: String::new(),
        }
    }

    /// Create a right join with multiple columns
    pub fn right_join_multi(
        left_column_indices: Vec<usize>,
        right_column_indices: Vec<usize>,
    ) -> CylonResult<Self> {
        Self::new(
            JoinType::Right,
            left_column_indices,
            right_column_indices,
            JoinAlgorithm::Hash,
            String::new(),
            String::new(),
        )
    }

    /// Create a full outer join configuration
    pub fn full_outer_join(
        left_column_idx: usize,
        right_column_idx: usize,
    ) -> Self {
        Self {
            join_type: JoinType::FullOuter,
            algorithm: JoinAlgorithm::Hash,
            left_column_indices: vec![left_column_idx],
            right_column_indices: vec![right_column_idx],
            left_table_suffix: String::new(),
            right_table_suffix: String::new(),
        }
    }

    /// Create a full outer join with multiple columns
    pub fn full_outer_join_multi(
        left_column_indices: Vec<usize>,
        right_column_indices: Vec<usize>,
    ) -> CylonResult<Self> {
        Self::new(
            JoinType::FullOuter,
            left_column_indices,
            right_column_indices,
            JoinAlgorithm::Hash,
            String::new(),
            String::new(),
        )
    }

    /// Set the join algorithm
    pub fn with_algorithm(mut self, algorithm: JoinAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Set table suffixes for duplicate column names
    pub fn with_suffixes(mut self, left_suffix: String, right_suffix: String) -> Self {
        self.left_table_suffix = left_suffix;
        self.right_table_suffix = right_suffix;
        self
    }

    /// Get the join type
    pub fn join_type(&self) -> JoinType {
        self.join_type
    }

    /// Get the algorithm
    pub fn algorithm(&self) -> JoinAlgorithm {
        self.algorithm
    }

    /// Get left column indices
    pub fn left_column_indices(&self) -> &[usize] {
        &self.left_column_indices
    }

    /// Get right column indices
    pub fn right_column_indices(&self) -> &[usize] {
        &self.right_column_indices
    }

    /// Get left table suffix
    pub fn left_table_suffix(&self) -> &str {
        &self.left_table_suffix
    }

    /// Get right table suffix
    pub fn right_table_suffix(&self) -> &str {
        &self.right_table_suffix
    }

    /// Check if this is a multi-column join
    pub fn is_multi_column(&self) -> bool {
        self.left_column_indices.len() > 1
    }
}
