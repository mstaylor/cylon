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

//! Arrow index types
//!
//! Ported from cpp/src/cylon/indexing/index.hpp

use std::sync::Arc;
use arrow::array::Array;
use crate::error::CylonResult;

/// Indexing types
/// Corresponds to C++ IndexingType enum (index.hpp:36-42)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexingType {
    Range = 0,
    Linear = 1,
    Hash = 2,
    BinaryTree = 3,
    BTree = 4,
}

/// Base trait for Arrow indices
/// Corresponds to C++ BaseArrowIndex class (index.hpp:44-95)
pub trait BaseArrowIndex: Send + Sync {
    /// Get the column ID this index is based on
    fn get_col_id(&self) -> i32;

    /// Get the size of the index
    fn get_size(&self) -> usize;

    /// Get the indexing type
    fn get_indexing_type(&self) -> IndexingType;

    /// Get the index as an Arrow array
    fn get_index_array(&self) -> CylonResult<Arc<dyn Array>>;

    /// Set the index array
    fn set_index_array(&mut self, index_arr: Arc<dyn Array>) -> CylonResult<()>;

    /// Check if index values are unique
    fn is_unique(&self) -> bool;
}

/// Range index - generates indices on-the-fly
/// Corresponds to C++ ArrowRangeIndex (index.hpp:391-421)
#[derive(Debug, Clone)]
pub struct ArrowRangeIndex {
    start: i64,
    size: usize,
    step: i64,
}

impl ArrowRangeIndex {
    /// Create a new range index
    /// Corresponds to C++ ArrowRangeIndex constructor (index.cpp:309)
    pub fn new(start: i64, size: usize, step: i64) -> Self {
        Self { start, size, step }
    }

    /// Get the start value
    pub fn start(&self) -> i64 {
        self.start
    }

    /// Get the step value
    pub fn step(&self) -> i64 {
        self.step
    }
}

impl BaseArrowIndex for ArrowRangeIndex {
    fn get_col_id(&self) -> i32 {
        0 // Range index doesn't correspond to a specific column
    }

    fn get_size(&self) -> usize {
        self.size
    }

    fn get_indexing_type(&self) -> IndexingType {
        IndexingType::Range
    }

    fn get_index_array(&self) -> CylonResult<Arc<dyn Array>> {
        use arrow::array::Int64Array;

        // Generate the range on-the-fly
        let indices: Vec<i64> = (0..self.size)
            .map(|i| self.start + (i as i64) * self.step)
            .collect();

        Ok(Arc::new(Int64Array::from(indices)))
    }

    fn set_index_array(&mut self, _index_arr: Arc<dyn Array>) -> CylonResult<()> {
        // Range index is computed, not stored
        Err(crate::error::CylonError::new(
            crate::error::Code::Invalid,
            "Cannot set index array on a RangeIndex".to_string(),
        ))
    }

    fn is_unique(&self) -> bool {
        self.step != 0 // Unique if step is non-zero
    }
}

/// Linear index - stores indices in an array
/// Corresponds to C++ ArrowLinearIndex (index.hpp:423-462)
#[derive(Debug, Clone)]
pub struct ArrowLinearIndex {
    col_id: i32,
    index_array: Arc<dyn Array>,
}

impl ArrowLinearIndex {
    /// Create a new linear index
    pub fn new(col_id: i32, index_array: Arc<dyn Array>) -> Self {
        Self { col_id, index_array }
    }
}

impl BaseArrowIndex for ArrowLinearIndex {
    fn get_col_id(&self) -> i32 {
        self.col_id
    }

    fn get_size(&self) -> usize {
        self.index_array.len()
    }

    fn get_indexing_type(&self) -> IndexingType {
        IndexingType::Linear
    }

    fn get_index_array(&self) -> CylonResult<Arc<dyn Array>> {
        Ok(self.index_array.clone())
    }

    fn set_index_array(&mut self, index_arr: Arc<dyn Array>) -> CylonResult<()> {
        self.index_array = index_arr;
        Ok(())
    }

    fn is_unique(&self) -> bool {
        // For now, assume not unique
        // Full implementation would check for uniqueness
        false
    }
}
