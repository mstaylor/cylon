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

//! Arrow partition kernels for hashing and partitioning
//!
//! Ported from cpp/src/cylon/arrow/arrow_partition_kernels.hpp/.cpp

use std::hash::{Hash, Hasher};
use std::sync::Arc;
use arrow::array::{Array, ArrayRef};
use arrow::datatypes::DataType;
use crate::error::{CylonError, CylonResult, Code};

/// Trait for hash partition kernels
/// Ported from cpp/src/cylon/arrow/arrow_partition_kernels.hpp HashPartitionKernel (lines 55-80)
pub trait HashPartitionKernel: Send + Sync {
    /// Update hash values for all rows in a column
    /// Ported from cpp/src/cylon/arrow/arrow_partition_kernels.hpp line 69-70
    ///
    /// This allows building composite hashes across multiple columns
    ///
    /// # Arguments
    /// * `idx_col` - The column to hash
    /// * `partial_hashes` - Hash values to update (must be same length as column)
    fn update_hash(&self, idx_col: &ArrayRef, partial_hashes: &mut [u32]) -> CylonResult<()>;

    /// Compute hash for a single value at an index
    /// Ported from cpp/src/cylon/arrow/arrow_partition_kernels.hpp line 79
    ///
    /// # Arguments
    /// * `values` - The array containing the value
    /// * `index` - Index of the value to hash
    fn to_hash(&self, values: &ArrayRef, index: i64) -> u32;
}

/// Hash a value using ahash (similar to MurmurHash3 in C++)
#[inline]
fn hash_value<T: Hash>(value: &T) -> u32 {
    let mut hasher = ahash::AHasher::default();
    value.hash(&mut hasher);
    hasher.finish() as u32
}

/// Numeric hash partition kernel for integer and floating point types
/// Ported from cpp/src/cylon/arrow/arrow_partition_kernels.cpp NumericHashPartitionKernel (lines 147-233)
struct NumericHashPartitionKernel<T>
where
    T: arrow::datatypes::ArrowPrimitiveType,
{
    _phantom: std::marker::PhantomData<T>,
}

impl<T> NumericHashPartitionKernel<T>
where
    T: arrow::datatypes::ArrowPrimitiveType + Send + Sync,
{
    fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T> HashPartitionKernel for NumericHashPartitionKernel<T>
where
    T: arrow::datatypes::ArrowPrimitiveType + Send + Sync,
{
    /// Ported from cpp/src/cylon/arrow/arrow_partition_kernels.cpp lines 198-215
    fn update_hash(&self, idx_col: &ArrayRef, partial_hashes: &mut [u32]) -> CylonResult<()> {
        use arrow::array::PrimitiveArray;

        if partial_hashes.len() != idx_col.len() {
            return Err(CylonError::new(
                Code::Invalid,
                format!("partial hashes size {} != column length {}",
                    partial_hashes.len(), idx_col.len()),
            ));
        }

        let array = idx_col.as_any()
            .downcast_ref::<PrimitiveArray<T>>()
            .ok_or_else(|| CylonError::new(
                Code::Invalid,
                "Failed to downcast to primitive array".to_string(),
            ))?;

        for i in 0..array.len() {
            if !array.is_null(i) {
                let value = array.value(i);
                // Hash the bytes of the value (works for all primitive types including floats)
                let hash = {
                    let bytes = unsafe {
                        std::slice::from_raw_parts(
                            &value as *const T::Native as *const u8,
                            std::mem::size_of::<T::Native>(),
                        )
                    };
                    hash_value(&bytes)
                };
                // Combine hash: hash + 31 * partial_hash (C++ line 209)
                partial_hashes[i] = hash.wrapping_add(31u32.wrapping_mul(partial_hashes[i]));
            }
            // Null values don't change the hash (C++ line 212-214)
        }

        Ok(())
    }

    /// Ported from cpp/src/cylon/arrow/arrow_partition_kernels.cpp lines 217-227
    fn to_hash(&self, values: &ArrayRef, index: i64) -> u32 {
        use arrow::array::PrimitiveArray;

        if values.is_null(index as usize) {
            return 0; // C++ line 219
        }

        let array = values.as_any()
            .downcast_ref::<PrimitiveArray<T>>()
            .expect("Failed to downcast array");

        let value = array.value(index as usize);
        // Hash the bytes of the value
        let bytes = unsafe {
            std::slice::from_raw_parts(
                &value as *const T::Native as *const u8,
                std::mem::size_of::<T::Native>(),
            )
        };
        hash_value(&bytes) // C++ line 224
    }
}

/// String/Binary hash partition kernel
/// Ported from cpp/src/cylon/arrow/arrow_partition_kernels.cpp BinaryHashPartitionKernel (lines 322-408)
struct StringHashPartitionKernel;

impl HashPartitionKernel for StringHashPartitionKernel {
    fn update_hash(&self, idx_col: &ArrayRef, partial_hashes: &mut [u32]) -> CylonResult<()> {
        use arrow::array::StringArray;

        if partial_hashes.len() != idx_col.len() {
            return Err(CylonError::new(
                Code::Invalid,
                format!("partial hashes size {} != column length {}",
                    partial_hashes.len(), idx_col.len()),
            ));
        }

        let array = idx_col.as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| CylonError::new(
                Code::Invalid,
                "Failed to downcast to string array".to_string(),
            ))?;

        for i in 0..array.len() {
            if !array.is_null(i) {
                let value = array.value(i);
                let hash = hash_value(&value);
                partial_hashes[i] = hash.wrapping_add(31u32.wrapping_mul(partial_hashes[i]));
            }
        }

        Ok(())
    }

    fn to_hash(&self, values: &ArrayRef, index: i64) -> u32 {
        use arrow::array::StringArray;

        if values.is_null(index as usize) {
            return 0;
        }

        let array = values.as_any()
            .downcast_ref::<StringArray>()
            .expect("Failed to downcast array");

        let value = array.value(index as usize);
        hash_value(&value)
    }
}

/// Large string hash partition kernel
struct LargeStringHashPartitionKernel;

impl HashPartitionKernel for LargeStringHashPartitionKernel {
    fn update_hash(&self, idx_col: &ArrayRef, partial_hashes: &mut [u32]) -> CylonResult<()> {
        use arrow::array::LargeStringArray;

        if partial_hashes.len() != idx_col.len() {
            return Err(CylonError::new(
                Code::Invalid,
                format!("partial hashes size {} != column length {}",
                    partial_hashes.len(), idx_col.len()),
            ));
        }

        let array = idx_col.as_any()
            .downcast_ref::<LargeStringArray>()
            .ok_or_else(|| CylonError::new(
                Code::Invalid,
                "Failed to downcast to large string array".to_string(),
            ))?;

        for i in 0..array.len() {
            if !array.is_null(i) {
                let value = array.value(i);
                let hash = hash_value(&value);
                partial_hashes[i] = hash.wrapping_add(31u32.wrapping_mul(partial_hashes[i]));
            }
        }

        Ok(())
    }

    fn to_hash(&self, values: &ArrayRef, index: i64) -> u32 {
        use arrow::array::LargeStringArray;

        if values.is_null(index as usize) {
            return 0;
        }

        let array = values.as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("Failed to downcast array");

        let value = array.value(index as usize);
        hash_value(&value)
    }
}

/// Boolean hash partition kernel
struct BooleanHashPartitionKernel;

impl HashPartitionKernel for BooleanHashPartitionKernel {
    fn update_hash(&self, idx_col: &ArrayRef, partial_hashes: &mut [u32]) -> CylonResult<()> {
        use arrow::array::BooleanArray;

        if partial_hashes.len() != idx_col.len() {
            return Err(CylonError::new(
                Code::Invalid,
                format!("partial hashes size {} != column length {}",
                    partial_hashes.len(), idx_col.len()),
            ));
        }

        let array = idx_col.as_any()
            .downcast_ref::<BooleanArray>()
            .ok_or_else(|| CylonError::new(
                Code::Invalid,
                "Failed to downcast to boolean array".to_string(),
            ))?;

        for i in 0..array.len() {
            if !array.is_null(i) {
                let value = array.value(i);
                let hash = hash_value(&value);
                partial_hashes[i] = hash.wrapping_add(31u32.wrapping_mul(partial_hashes[i]));
            }
        }

        Ok(())
    }

    fn to_hash(&self, values: &ArrayRef, index: i64) -> u32 {
        use arrow::array::BooleanArray;

        if values.is_null(index as usize) {
            return 0;
        }

        let array = values.as_any()
            .downcast_ref::<BooleanArray>()
            .expect("Failed to downcast array");

        let value = array.value(index as usize);
        hash_value(&value)
    }
}

/// Create a hash partition kernel for a given data type
/// Ported from cpp/src/cylon/arrow/arrow_partition_kernels.cpp CreateHashPartitionKernel (lines 410-440)
pub fn create_hash_partition_kernel(data_type: &DataType) -> CylonResult<Box<dyn HashPartitionKernel>> {
    use arrow::datatypes::*;

    match data_type {
        // Boolean - C++ line 413
        DataType::Boolean => Ok(Box::new(BooleanHashPartitionKernel)),

        // Integers - C++ lines 414-421
        DataType::UInt8 => Ok(Box::new(NumericHashPartitionKernel::<UInt8Type>::new())),
        DataType::Int8 => Ok(Box::new(NumericHashPartitionKernel::<Int8Type>::new())),
        DataType::UInt16 => Ok(Box::new(NumericHashPartitionKernel::<UInt16Type>::new())),
        DataType::Int16 => Ok(Box::new(NumericHashPartitionKernel::<Int16Type>::new())),
        DataType::UInt32 => Ok(Box::new(NumericHashPartitionKernel::<UInt32Type>::new())),
        DataType::Int32 => Ok(Box::new(NumericHashPartitionKernel::<Int32Type>::new())),
        DataType::UInt64 => Ok(Box::new(NumericHashPartitionKernel::<UInt64Type>::new())),
        DataType::Int64 => Ok(Box::new(NumericHashPartitionKernel::<Int64Type>::new())),

        // Floats - C++ lines 422-423
        DataType::Float32 => Ok(Box::new(NumericHashPartitionKernel::<Float32Type>::new())),
        DataType::Float64 => Ok(Box::new(NumericHashPartitionKernel::<Float64Type>::new())),

        // Strings - C++ lines 424-426
        DataType::Utf8 => Ok(Box::new(StringHashPartitionKernel)),
        DataType::LargeUtf8 => Ok(Box::new(LargeStringHashPartitionKernel)),

        // Dates and times - C++ lines 431-435
        DataType::Date32 => Ok(Box::new(NumericHashPartitionKernel::<Date32Type>::new())),
        DataType::Date64 => Ok(Box::new(NumericHashPartitionKernel::<Date64Type>::new())),
        DataType::Timestamp(_, _) => Ok(Box::new(NumericHashPartitionKernel::<TimestampMicrosecondType>::new())),
        DataType::Time32(_) => Ok(Box::new(NumericHashPartitionKernel::<Time32MillisecondType>::new())),
        DataType::Time64(_) => Ok(Box::new(NumericHashPartitionKernel::<Time64MicrosecondType>::new())),

        // Unsupported - C++ lines 436-438
        _ => Err(CylonError::new(
            Code::NotImplemented,
            format!("Unsupported hash partition kernel data type: {:?}", data_type),
        )),
    }
}
