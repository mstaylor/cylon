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

//! Table - main data structure for Cylon
//!
//! Ported from cpp/src/cylon/table.hpp

use std::sync::Arc;
use arrow::array::RecordBatch;
use arrow::datatypes::Schema;

use crate::ctx::CylonContext;
use crate::error::CylonResult;

pub mod column;
pub use column::Column;

/// Table provides the main API for using cylon for data processing
/// Corresponds to C++ Table class from cpp/src/cylon/table.hpp
pub struct Table {
    ctx: Arc<CylonContext>,
    // Using Arrow RecordBatch internally (similar to C++ using arrow::Table)
    batches: Vec<RecordBatch>,
    retain: bool,
}

impl Table {
    /// Create a table from Arrow RecordBatch
    pub fn from_record_batch(
        ctx: Arc<CylonContext>,
        batch: RecordBatch,
    ) -> CylonResult<Self> {
        Ok(Self {
            ctx,
            batches: vec![batch],
            retain: true,
        })
    }

    /// Create a table from multiple Arrow RecordBatches
    pub fn from_record_batches(
        ctx: Arc<CylonContext>,
        batches: Vec<RecordBatch>,
    ) -> CylonResult<Self> {
        Ok(Self {
            ctx,
            batches,
            retain: true,
        })
    }

    /// Get the number of columns
    pub fn columns(&self) -> i32 {
        if let Some(batch) = self.batches.first() {
            batch.num_columns() as i32
        } else {
            0
        }
    }

    /// Get the number of rows
    pub fn rows(&self) -> i64 {
        self.batches.iter().map(|b| b.num_rows() as i64).sum()
    }

    /// Check if table is empty
    pub fn is_empty(&self) -> bool {
        self.rows() == 0
    }

    /// Get the context
    pub fn get_context(&self) -> Arc<CylonContext> {
        self.ctx.clone()
    }

    /// Get the schema
    pub fn schema(&self) -> Option<Arc<Schema>> {
        self.batches.first().map(|b| b.schema())
    }

    /// Get column names
    pub fn column_names(&self) -> Vec<String> {
        if let Some(schema) = self.schema() {
            schema.fields().iter().map(|f| f.name().clone()).collect()
        } else {
            Vec::new()
        }
    }

    /// Set retention flag
    pub fn retain_memory(&mut self, retain: bool) {
        self.retain = retain;
    }

    /// Check if table retains memory
    pub fn is_retain(&self) -> bool {
        self.retain
    }

    /// Get the number of batches in the table
    pub fn num_batches(&self) -> usize {
        self.batches.len()
    }

    /// Get a reference to a specific batch
    pub fn batch(&self, index: usize) -> Option<&RecordBatch> {
        self.batches.get(index)
    }

    /// Get all batches
    pub fn batches(&self) -> &[RecordBatch] {
        &self.batches
    }
}

// TODO: Port table operations from cpp/src/cylon/table.hpp:
// - FromCSV
// - WriteCSV
// - Merge
// - Join, DistributedJoin
// - Union, DistributedUnion
// - Subtract, DistributedSubtract
// - Intersect, DistributedIntersect
// - Shuffle
// - HashPartition
// - Sort, DistributedSort
// - Select
// - Project
// - Unique, DistributedUnique
// - Slice, DistributedSlice
// - Head, Tail
// etc.