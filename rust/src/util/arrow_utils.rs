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

//! Utility functions for working with Arrow arrays and data
//!
//! Ported from cpp/src/cylon/util/arrow_utils.hpp and arrow_utils.cpp

use arrow::array::{Array, ArrayRef};
use arrow::record_batch::RecordBatch;
use arrow::datatypes::Schema;

/// Get the size in bytes of an array
pub fn array_size_bytes(array: &dyn Array) -> usize {
    array.get_array_memory_size()
}

/// Convert Arrow RecordBatch to Vec<ArrayRef>
pub fn record_batch_to_arrays(batch: &RecordBatch) -> Vec<ArrayRef> {
    batch.columns().to_vec()
}

/// Get column names from schema
pub fn schema_column_names(schema: &Schema) -> Vec<String> {
    schema.fields().iter().map(|f| f.name().clone()).collect()
}

/// Check if schema has column
pub fn schema_has_column(schema: &Schema, name: &str) -> bool {
    schema.field_with_name(name).is_ok()
}

use crate::{CylonResult, table::Table};
/// Sample a table uniformly.
/// This is a placeholder implementation.
pub fn sample_table_uniform(table: &Table, num_samples: usize) -> CylonResult<Table> {
    table.head(num_samples)
}

// TODO: Port additional functions from cpp/src/cylon/util/arrow_utils.cpp:
// - SortTable
// - SortTableMultiColumns
// - copy_array_by_indices
// - free_table
// - Duplicate (ChunkedArray and Table)
// - SampleArray
// - SampleTableUniform
// - GetChunkOrEmptyArray
// - GetNumberSplitsToFitInCache
// - GetBytesAndElements
// - CreateEmptyTable
// - MakeEmptyArrowTable
// - CheckArrowTableContainsChunks
// - MakeDummyArray
// - WrapNumericVector
