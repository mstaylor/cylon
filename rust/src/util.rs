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

//! Utility functions and helpers
//!
//! Ported from various files in cpp/src/cylon/util/

pub mod uuid;
pub mod logging;

pub use self::uuid::*;

/// Utility functions for working with Arrow arrays and data
/// Leverages the Apache Arrow Rust implementation
pub mod arrow_utils {
    use arrow::array::{Array, ArrayRef};
    use arrow::record_batch::RecordBatch;
    use arrow::datatypes::Schema;
    use crate::error::CylonResult;

    /// Check if two arrays are equal
    pub fn arrays_equal(left: &dyn Array, right: &dyn Array) -> bool {
        arrow::util::data_gen::arrays_equal(left, right)
    }

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
}

/// Built-in functions and operations
pub mod builtins {
    /// Check if a string represents a numeric value
    pub fn is_numeric(s: &str) -> bool {
        s.parse::<f64>().is_ok()
    }

    /// Parse a string to a numeric value
    pub fn parse_numeric(s: &str) -> Option<f64> {
        s.parse().ok()
    }
}

/// Sorting utilities
pub mod sort {
    use std::cmp::Ordering;

    /// Generic comparison function type
    pub type CompareFn<T> = fn(&T, &T) -> Ordering;

    /// Sort a vector using the provided comparison function
    pub fn sort_by<T, F>(vec: &mut [T], compare: F)
    where
        F: FnMut(&T, &T) -> Ordering,
    {
        vec.sort_by(compare);
    }

    /// Check if a slice is sorted
    pub fn is_sorted<T: Ord>(slice: &[T]) -> bool {
        slice.windows(2).all(|w| w[0] <= w[1])
    }
}

/// String conversion utilities
pub mod to_string {
    use std::fmt::Display;

    /// Convert any Display type to string
    pub fn to_string<T: Display>(value: T) -> String {
        value.to_string()
    }

    /// Convert option to string with default
    pub fn option_to_string<T: Display>(opt: Option<T>, default: &str) -> String {
        opt.map_or_else(|| default.to_string(), |v| v.to_string())
    }
}