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

//! Arrow kernel operations
//!
//! Ported from cpp/src/cylon/arrow/arrow_kernels.cpp

use std::sync::Arc;
use arrow::array::{
    Array, ArrayRef,
    Int8Array, Int16Array, Int32Array, Int64Array,
    UInt8Array, UInt16Array, UInt32Array, UInt64Array,
    Float32Array, Float64Array,
};
use arrow::datatypes::DataType;

use crate::error::{CylonResult, CylonError, Code};
use crate::util::sort::introsort;

/// Sort array data in place and return indices
///
/// Corresponds to C++ SortIndicesInPlace (arrow_kernels.cpp:496-504)
/// and NumericInplaceIndexSortKernel::Sort (arrow_kernels.cpp:435-459)
///
/// # Returns
/// A tuple of (sorted_array, original_indices)
/// - sorted_array: The array with data now in sorted order
/// - original_indices: Maps sorted_position -> original_position
///
/// # Memory
/// This function extracts array data, sorts it with introsort, and creates a new array.
/// The original array can be dropped after this call, achieving similar memory
/// efficiency to C++ in-place sorting.
pub fn sort_indices_inplace(
    array: &ArrayRef,
) -> CylonResult<(ArrayRef, UInt64Array)> {
    let length = array.len();
    if length == 0 {
        return Ok((array.clone(), UInt64Array::from(Vec::<u64>::new())));
    }

    // Create indices array (arrow_kernels.cpp:452-455)
    let mut indices: Vec<i64> = (0..length as i64).collect();

    // Type dispatch (arrow_kernels.cpp:476-491)
    match array.data_type() {
        DataType::Int8 => {
            let arr = array.as_any().downcast_ref::<Int8Array>().unwrap();
            let mut data: Vec<i8> = arr.values().iter().copied().collect();
            introsort(&mut data, &mut indices);
            let sorted = Int8Array::from(data);
            let offsets: Vec<u64> = indices.iter().map(|&i| i as u64).collect();
            Ok((Arc::new(sorted), UInt64Array::from(offsets)))
        }
        DataType::Int16 => {
            let arr = array.as_any().downcast_ref::<Int16Array>().unwrap();
            let mut data: Vec<i16> = arr.values().iter().copied().collect();
            introsort(&mut data, &mut indices);
            let sorted = Int16Array::from(data);
            let offsets: Vec<u64> = indices.iter().map(|&i| i as u64).collect();
            Ok((Arc::new(sorted), UInt64Array::from(offsets)))
        }
        DataType::Int32 => {
            let arr = array.as_any().downcast_ref::<Int32Array>().unwrap();
            let mut data: Vec<i32> = arr.values().iter().copied().collect();
            introsort(&mut data, &mut indices);
            let sorted = Int32Array::from(data);
            let offsets: Vec<u64> = indices.iter().map(|&i| i as u64).collect();
            Ok((Arc::new(sorted), UInt64Array::from(offsets)))
        }
        DataType::Int64 => {
            let arr = array.as_any().downcast_ref::<Int64Array>().unwrap();
            let mut data: Vec<i64> = arr.values().iter().copied().collect();
            introsort(&mut data, &mut indices);
            let sorted = Int64Array::from(data);
            let offsets: Vec<u64> = indices.iter().map(|&i| i as u64).collect();
            Ok((Arc::new(sorted), UInt64Array::from(offsets)))
        }
        DataType::UInt8 => {
            let arr = array.as_any().downcast_ref::<UInt8Array>().unwrap();
            let mut data: Vec<u8> = arr.values().iter().copied().collect();
            introsort(&mut data, &mut indices);
            let sorted = UInt8Array::from(data);
            let offsets: Vec<u64> = indices.iter().map(|&i| i as u64).collect();
            Ok((Arc::new(sorted), UInt64Array::from(offsets)))
        }
        DataType::UInt16 => {
            let arr = array.as_any().downcast_ref::<UInt16Array>().unwrap();
            let mut data: Vec<u16> = arr.values().iter().copied().collect();
            introsort(&mut data, &mut indices);
            let sorted = UInt16Array::from(data);
            let offsets: Vec<u64> = indices.iter().map(|&i| i as u64).collect();
            Ok((Arc::new(sorted), UInt64Array::from(offsets)))
        }
        DataType::UInt32 => {
            let arr = array.as_any().downcast_ref::<UInt32Array>().unwrap();
            let mut data: Vec<u32> = arr.values().iter().copied().collect();
            introsort(&mut data, &mut indices);
            let sorted = UInt32Array::from(data);
            let offsets: Vec<u64> = indices.iter().map(|&i| i as u64).collect();
            Ok((Arc::new(sorted), UInt64Array::from(offsets)))
        }
        DataType::UInt64 => {
            let arr = array.as_any().downcast_ref::<UInt64Array>().unwrap();
            let mut data: Vec<u64> = arr.values().iter().copied().collect();
            introsort(&mut data, &mut indices);
            let sorted = UInt64Array::from(data);
            let offsets: Vec<u64> = indices.iter().map(|&i| i as u64).collect();
            Ok((Arc::new(sorted), UInt64Array::from(offsets)))
        }
        DataType::Float32 => {
            let arr = array.as_any().downcast_ref::<Float32Array>().unwrap();
            let mut data: Vec<f32> = arr.values().iter().copied().collect();
            introsort(&mut data, &mut indices);
            let sorted = Float32Array::from(data);
            let offsets: Vec<u64> = indices.iter().map(|&i| i as u64).collect();
            Ok((Arc::new(sorted), UInt64Array::from(offsets)))
        }
        DataType::Float64 => {
            let arr = array.as_any().downcast_ref::<Float64Array>().unwrap();
            let mut data: Vec<f64> = arr.values().iter().copied().collect();
            introsort(&mut data, &mut indices);
            let sorted = Float64Array::from(data);
            let offsets: Vec<u64> = indices.iter().map(|&i| i as u64).collect();
            Ok((Arc::new(sorted), UInt64Array::from(offsets)))
        }
        dt => Err(CylonError::new(
            Code::Invalid,
            format!("SortIndicesInPlace: unsupported type {:?}", dt),
        )),
    }
}
