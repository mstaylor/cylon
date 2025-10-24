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
pub mod arrow_utils;
pub mod to_string;

pub use self::uuid::*;

/// Built-in functions and operations
/// Ported from cpp/src/cylon/util/builtins.hpp
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
/// Ported from cpp/src/cylon/util/sort.hpp
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