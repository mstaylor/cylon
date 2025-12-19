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

    /// Insertion sort that co-sorts data and indices
    /// Corresponds to C++ insertion_sort (sort.hpp:61-75)
    #[inline]
    fn insertion_sort<T: PartialOrd + Copy>(data: &mut [T], indices: &mut [i64]) {
        let length = data.len();
        for i in 1..length {
            let value = data[i];
            let value2 = indices[i];
            let mut j = i as isize - 1;

            while j >= 0 && data[j as usize] > value {
                data[(j + 1) as usize] = data[j as usize];
                indices[(j + 1) as usize] = indices[j as usize];
                j -= 1;
            }
            data[(j + 1) as usize] = value;
            indices[(j + 1) as usize] = value2;
        }
    }

    /// Heapify for heap sort
    /// Corresponds to C++ heapify (sort.hpp:77-93)
    #[inline]
    fn heapify<T: PartialOrd + Copy>(arr: &mut [T], n: usize, i: usize, indices: &mut [i64]) {
        let mut largest = i;
        let l = 2 * i + 1;
        let r = 2 * i + 2;

        if l < n && arr[l] > arr[largest] {
            largest = l;
        }
        if r < n && arr[r] > arr[largest] {
            largest = r;
        }
        if largest != i {
            arr.swap(i, largest);
            indices.swap(i, largest);
            heapify(arr, n, largest, indices);
        }
    }

    /// Heap sort that co-sorts data and indices
    /// Corresponds to C++ heap_sort (sort.hpp:95-106)
    #[inline]
    fn heap_sort<T: PartialOrd + Copy>(arr: &mut [T], indices: &mut [i64]) {
        let length = arr.len();
        for i in (0..=length / 2).rev() {
            heapify(arr, length, i, indices);
        }

        for i in (1..length).rev() {
            arr.swap(0, i);
            indices.swap(0, i);
            heapify(arr, i, 0, indices);
        }
    }

    /// Partition for quicksort
    /// Corresponds to C++ partition (sort.hpp:109-123)
    #[inline]
    fn partition<T: PartialOrd + Copy>(data: &mut [T], indices: &mut [i64], start: usize, end: usize) -> usize {
        let pivot = data[end];
        let mut i = start as isize - 1;

        for j in start..end {
            if data[j] <= pivot {
                i += 1;
                indices.swap(i as usize, j);
                data.swap(i as usize, j);
            }
        }
        data.swap((i + 1) as usize, end);
        indices.swap((i + 1) as usize, end);
        (i + 1) as usize
    }

    /// Introsort implementation
    /// Corresponds to C++ introsort_impl (sort.hpp:125-138)
    fn introsort_impl<T: PartialOrd + Copy>(data: &mut [T], indices: &mut [i64], start: usize, end: usize, maxdepth: usize) {
        if end <= start {
            return;
        }
        let len = end - start + 1;
        if len < 32 {
            insertion_sort(&mut data[start..=end], &mut indices[start..=end]);
        } else if maxdepth == 0 {
            heap_sort(&mut data[start..=end], &mut indices[start..=end]);
        } else {
            let p = partition(data, indices, start, end);
            if p > 0 {
                introsort_impl(data, indices, start, p - 1, maxdepth - 1);
            }
            introsort_impl(data, indices, p + 1, end, maxdepth - 1);
        }
    }

    /// Introsort - hybrid sorting algorithm that co-sorts data and indices
    /// Corresponds to C++ introsort (sort.hpp:140-144)
    ///
    /// This sorts the data array in place while simultaneously maintaining
    /// the corresponding permutation in the indices array.
    pub fn introsort<T: PartialOrd + Copy>(data: &mut [T], indices: &mut [i64]) {
        let length = data.len();
        if length <= 1 {
            return;
        }
        let depth = ((length as f64).ln() * 2.0) as usize;
        introsort_impl(data, indices, 0, length - 1, depth);
    }
}