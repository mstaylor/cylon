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

//! Arrow row comparators and hash functions for set operations
//!
//! Ported from cpp/src/cylon/arrow/arrow_comparator.hpp/.cpp

use std::sync::Arc;
use arrow::array::{Array, ArrayRef};
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;
use crate::error::{CylonError, CylonResult, Code};
use super::arrow_partition_kernels::{create_hash_partition_kernel, HashPartitionKernel};

/// Bit manipulation utilities for dual-table row indexing
/// Ported from cpp/src/cylon/util/arrow_utils.hpp lines 55-63
///
/// Uses bit 63 to distinguish rows from two different tables:
/// - Left table rows: index stored as-is (bit 63 = 0)
/// - Right table rows: index with bit 63 set (bit 63 = 1)

/// Set bit 63 to mark an index as belonging to the right table
/// Ported from cpp/src/cylon/util/arrow_utils.hpp line 55-57
#[inline]
pub fn set_bit(v: i64) -> i64 {
    v | (1i64 << 63)
}

/// Clear bit 63 to get the actual row index
/// Ported from cpp/src/cylon/util/arrow_utils.hpp line 58-60
#[inline]
pub fn clear_bit(v: i64) -> i64 {
    v & 0x7FFF_FFFF_FFFF_FFFF
}

/// Check bit 63 to determine which table (0 = left, 1 = right)
/// Ported from cpp/src/cylon/util/arrow_utils.hpp line 61-63
#[inline]
pub fn check_bit(v: i64) -> usize {
    ((v >> 63) & 1) as usize
}

/// Hash function for a single table's rows
/// Ported from cpp/src/cylon/arrow/arrow_comparator.hpp TableRowIndexHash (lines 210-237)
pub struct TableRowIndexHash {
    /// Pre-computed hash values for each row
    hashes: Vec<u32>,
}

impl TableRowIndexHash {
    /// Create a hash function for a RecordBatch
    /// Ported from cpp/src/cylon/arrow/arrow_comparator.cpp TableRowIndexHash::Make (lines 841-858)
    pub fn new(batch: &RecordBatch) -> CylonResult<Self> {
        let all_columns: Vec<usize> = (0..batch.num_columns()).collect();
        Self::new_with_columns(batch, &all_columns)
    }

    /// Create a hash function for specific columns of a RecordBatch
    /// Ported from cpp/src/cylon/arrow/arrow_comparator.cpp TableRowIndexHash::Make (lines 841-858)
    pub fn new_with_columns(batch: &RecordBatch, col_indices: &[usize]) -> CylonResult<Self> {
        let num_rows = batch.num_rows();
        let mut hashes = vec![0u32; num_rows];

        // Compute hash for each specified column and combine (C++ lines 852-856)
        for &col_idx in col_indices {
            if col_idx >= batch.num_columns() {
                return Err(CylonError::new(
                    Code::Invalid,
                    format!("Column index {} out of range", col_idx),
                ));
            }
            let column = batch.column(col_idx);
            let kernel = create_hash_partition_kernel(column.data_type())?;
            kernel.update_hash(column, &mut hashes)?;
        }

        Ok(Self { hashes })
    }

    /// Get hash for a row index
    pub fn hash(&self, idx: i64) -> u32 {
        self.hashes[idx as usize]
    }
}

/// Hash function for dual tables (left and right)
/// Ported from cpp/src/cylon/arrow/arrow_comparator.hpp DualTableRowIndexHash (lines 238-260)
pub struct DualTableRowIndexHash {
    /// Hash functions for [left_table, right_table]
    table_hashes: [TableRowIndexHash; 2],
}

impl DualTableRowIndexHash {
    /// Create hash function for two tables
    /// Ported from cpp/src/cylon/arrow/arrow_comparator.cpp DualTableRowIndexHash::Make
    pub fn new(left: &RecordBatch, right: &RecordBatch) -> CylonResult<Self> {
        let left_hash = TableRowIndexHash::new(left)?;
        let right_hash = TableRowIndexHash::new(right)?;

        Ok(Self {
            table_hashes: [left_hash, right_hash],
        })
    }

    /// Hash a row index (with bit 63 encoding)
    /// Ported from cpp/src/cylon/arrow/arrow_comparator.cpp line 918-920
    pub fn hash(&self, idx: i64) -> u32 {
        let table_idx = check_bit(idx);
        let row_idx = clear_bit(idx);
        self.table_hashes[table_idx].hash(row_idx)
    }
}

/// Trait for comparing rows within a single column across two tables
/// Ported from cpp/src/cylon/arrow/arrow_comparator.hpp DualArrayIndexComparator
trait DualArrayIndexComparator: Send + Sync {
    /// Compare two row indices (with bit 63 encoding)
    fn compare(&self, index1: i64, index2: i64) -> std::cmp::Ordering;

    /// Check equality of two row indices (with bit 63 encoding)
    fn equal_to(&self, index1: i64, index2: i64) -> bool;
}

/// Comparator for primitive numeric types
/// Ported from cpp/src/cylon/arrow/arrow_comparator.cpp DualNumericRowIndexComparator (lines 425-452)
struct DualPrimitiveComparator<T>
where
    T: arrow::datatypes::ArrowPrimitiveType,
{
    /// Arrays for [left_table, right_table]
    arrays: [Arc<arrow::array::PrimitiveArray<T>>; 2],
}

impl<T> DualPrimitiveComparator<T>
where
    T: arrow::datatypes::ArrowPrimitiveType,
    T::Native: PartialOrd,
{
    fn new(left: &ArrayRef, right: &ArrayRef) -> CylonResult<Self> {
        use arrow::array::PrimitiveArray;

        let left_array = left.as_any()
            .downcast_ref::<PrimitiveArray<T>>()
            .ok_or_else(|| CylonError::new(Code::Invalid, "Failed to downcast left array".to_string()))?
            .clone();

        let right_array = right.as_any()
            .downcast_ref::<PrimitiveArray<T>>()
            .ok_or_else(|| CylonError::new(Code::Invalid, "Failed to downcast right array".to_string()))?
            .clone();

        Ok(Self {
            arrays: [Arc::new(left_array), Arc::new(right_array)],
        })
    }
}

impl<T> DualArrayIndexComparator for DualPrimitiveComparator<T>
where
    T: arrow::datatypes::ArrowPrimitiveType,
    T::Native: PartialOrd,
{
    /// Ported from cpp/src/cylon/arrow/arrow_comparator.cpp line 434-437
    fn compare(&self, index1: i64, index2: i64) -> std::cmp::Ordering {
        let table1 = check_bit(index1);
        let row1 = clear_bit(index1) as usize;
        let table2 = check_bit(index2);
        let row2 = clear_bit(index2) as usize;

        let val1 = self.arrays[table1].value(row1);
        let val2 = self.arrays[table2].value(row2);

        val1.partial_cmp(&val2).unwrap_or(std::cmp::Ordering::Equal)
    }

    /// Ported from cpp/src/cylon/arrow/arrow_comparator.cpp line 445-448
    fn equal_to(&self, index1: i64, index2: i64) -> bool {
        let table1 = check_bit(index1);
        let row1 = clear_bit(index1) as usize;
        let table2 = check_bit(index2);
        let row2 = clear_bit(index2) as usize;

        self.arrays[table1].value(row1) == self.arrays[table2].value(row2)
    }
}

/// Comparator for string types
/// Ported from cpp/src/cylon/arrow/arrow_comparator.cpp DualBinaryRowIndexComparator (lines 454-479)
struct DualStringComparator {
    /// Arrays for [left_table, right_table]
    arrays: [Arc<arrow::array::StringArray>; 2],
}

impl DualStringComparator {
    fn new(left: &ArrayRef, right: &ArrayRef) -> CylonResult<Self> {
        use arrow::array::StringArray;

        let left_array = left.as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| CylonError::new(Code::Invalid, "Failed to downcast left string array".to_string()))?
            .clone();

        let right_array = right.as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| CylonError::new(Code::Invalid, "Failed to downcast right string array".to_string()))?
            .clone();

        Ok(Self {
            arrays: [Arc::new(left_array), Arc::new(right_array)],
        })
    }
}

impl DualArrayIndexComparator for DualStringComparator {
    /// Ported from cpp/src/cylon/arrow/arrow_comparator.cpp line 463-467
    fn compare(&self, index1: i64, index2: i64) -> std::cmp::Ordering {
        let table1 = check_bit(index1);
        let row1 = clear_bit(index1) as usize;
        let table2 = check_bit(index2);
        let row2 = clear_bit(index2) as usize;

        let val1 = self.arrays[table1].value(row1);
        let val2 = self.arrays[table2].value(row2);

        val1.cmp(val2)
    }

    /// Ported from cpp/src/cylon/arrow/arrow_comparator.cpp line 469-473
    fn equal_to(&self, index1: i64, index2: i64) -> bool {
        let table1 = check_bit(index1);
        let row1 = clear_bit(index1) as usize;
        let table2 = check_bit(index2);
        let row2 = clear_bit(index2) as usize;

        self.arrays[table1].value(row1) == self.arrays[table2].value(row2)
    }
}

/// Comparator for large string types
struct DualLargeStringComparator {
    /// Arrays for [left_table, right_table]
    arrays: [Arc<arrow::array::LargeStringArray>; 2],
}

impl DualLargeStringComparator {
    fn new(left: &ArrayRef, right: &ArrayRef) -> CylonResult<Self> {
        use arrow::array::LargeStringArray;

        let left_array = left.as_any()
            .downcast_ref::<LargeStringArray>()
            .ok_or_else(|| CylonError::new(Code::Invalid, "Failed to downcast left large string array".to_string()))?
            .clone();

        let right_array = right.as_any()
            .downcast_ref::<LargeStringArray>()
            .ok_or_else(|| CylonError::new(Code::Invalid, "Failed to downcast right large string array".to_string()))?
            .clone();

        Ok(Self {
            arrays: [Arc::new(left_array), Arc::new(right_array)],
        })
    }
}

impl DualArrayIndexComparator for DualLargeStringComparator {
    fn compare(&self, index1: i64, index2: i64) -> std::cmp::Ordering {
        let table1 = check_bit(index1);
        let row1 = clear_bit(index1) as usize;
        let table2 = check_bit(index2);
        let row2 = clear_bit(index2) as usize;

        let val1 = self.arrays[table1].value(row1);
        let val2 = self.arrays[table2].value(row2);

        val1.cmp(val2)
    }

    fn equal_to(&self, index1: i64, index2: i64) -> bool {
        let table1 = check_bit(index1);
        let row1 = clear_bit(index1) as usize;
        let table2 = check_bit(index2);
        let row2 = clear_bit(index2) as usize;

        self.arrays[table1].value(row1) == self.arrays[table2].value(row2)
    }
}

/// Equality comparator for dual tables
/// Ported from cpp/src/cylon/arrow/arrow_comparator.hpp DualTableRowIndexEqualTo (lines 270-299)
pub struct DualTableRowIndexEqualTo {
    /// Comparators for each column
    comparators: Vec<Box<dyn DualArrayIndexComparator>>,
}

impl DualTableRowIndexEqualTo {
    /// Create equality comparator for two tables
    /// Ported from cpp/src/cylon/arrow/arrow_comparator.cpp DualTableRowIndexEqualTo::Make (lines 922-970)
    pub fn new(left: &RecordBatch, right: &RecordBatch) -> CylonResult<Self> {
        if left.num_columns() != right.num_columns() {
            return Err(CylonError::new(
                Code::Invalid,
                "Tables must have same number of columns".to_string(),
            ));
        }

        let mut comparators: Vec<Box<dyn DualArrayIndexComparator>> = Vec::new();

        for col_idx in 0..left.num_columns() {
            let left_col = left.column(col_idx);
            let right_col = right.column(col_idx);

            if left_col.data_type() != right_col.data_type() {
                return Err(CylonError::new(
                    Code::Invalid,
                    format!("Column {} has different types", col_idx),
                ));
            }

            let comparator = Self::create_comparator(left_col, right_col)?;
            comparators.push(comparator);
        }

        Ok(Self { comparators })
    }

    /// Create a comparator for a specific data type
    fn create_comparator(
        left: &ArrayRef,
        right: &ArrayRef,
    ) -> CylonResult<Box<dyn DualArrayIndexComparator>> {
        use arrow::datatypes::*;

        match left.data_type() {
            DataType::Int8 => Ok(Box::new(DualPrimitiveComparator::<Int8Type>::new(left, right)?)),
            DataType::Int16 => Ok(Box::new(DualPrimitiveComparator::<Int16Type>::new(left, right)?)),
            DataType::Int32 => Ok(Box::new(DualPrimitiveComparator::<Int32Type>::new(left, right)?)),
            DataType::Int64 => Ok(Box::new(DualPrimitiveComparator::<Int64Type>::new(left, right)?)),
            DataType::UInt8 => Ok(Box::new(DualPrimitiveComparator::<UInt8Type>::new(left, right)?)),
            DataType::UInt16 => Ok(Box::new(DualPrimitiveComparator::<UInt16Type>::new(left, right)?)),
            DataType::UInt32 => Ok(Box::new(DualPrimitiveComparator::<UInt32Type>::new(left, right)?)),
            DataType::UInt64 => Ok(Box::new(DualPrimitiveComparator::<UInt64Type>::new(left, right)?)),
            DataType::Float32 => Ok(Box::new(DualPrimitiveComparator::<Float32Type>::new(left, right)?)),
            DataType::Float64 => Ok(Box::new(DualPrimitiveComparator::<Float64Type>::new(left, right)?)),
            DataType::Utf8 => Ok(Box::new(DualStringComparator::new(left, right)?)),
            DataType::LargeUtf8 => Ok(Box::new(DualLargeStringComparator::new(left, right)?)),
            DataType::Date32 => Ok(Box::new(DualPrimitiveComparator::<Date32Type>::new(left, right)?)),
            DataType::Date64 => Ok(Box::new(DualPrimitiveComparator::<Date64Type>::new(left, right)?)),
            DataType::Timestamp(_, _) => Ok(Box::new(DualPrimitiveComparator::<TimestampMicrosecondType>::new(left, right)?)),
            DataType::Time32(_) => Ok(Box::new(DualPrimitiveComparator::<Time32MillisecondType>::new(left, right)?)),
            DataType::Time64(_) => Ok(Box::new(DualPrimitiveComparator::<Time64MicrosecondType>::new(left, right)?)),
            _ => Err(CylonError::new(
                Code::Invalid,
                format!("Unsupported data type for comparison: {:?}", left.data_type()),
            )),
        }
    }

    /// Check if two row indices are equal (with bit 63 encoding)
    /// Ported from cpp/src/cylon/arrow/arrow_comparator.cpp DualTableRowIndexEqualTo::operator()
    pub fn equal(&self, record1: i64, record2: i64) -> bool {
        for comparator in &self.comparators {
            if !comparator.equal_to(record1, record2) {
                return false;
            }
        }
        true
    }

    /// Compare two row indices (with bit 63 encoding)
    pub fn compare(&self, record1: i64, record2: i64) -> std::cmp::Ordering {
        for comparator in &self.comparators {
            match comparator.compare(record1, record2) {
                std::cmp::Ordering::Equal => continue,
                other => return other,
            }
        }
        std::cmp::Ordering::Equal
    }
}

// -----------------------------------------------------------------------------
// Single table comparators (ported from arrow_comparator.hpp lines 59-188)
// -----------------------------------------------------------------------------

/// Trait for comparing rows within a single column of a single table
/// Ported from cpp/src/cylon/arrow/arrow_comparator.hpp ArrayIndexComparator (lines 59-88)
trait ArrayIndexComparator: Send + Sync {
    /// Compare two row indices
    fn compare(&self, index1: i64, index2: i64) -> std::cmp::Ordering;

    /// Check equality of two row indices
    fn equal_to(&self, index1: i64, index2: i64) -> bool;
}

/// Comparator for primitive types in a single table
struct PrimitiveComparator<T>
where
    T: arrow::datatypes::ArrowPrimitiveType,
{
    array: Arc<arrow::array::PrimitiveArray<T>>,
}

impl<T> PrimitiveComparator<T>
where
    T: arrow::datatypes::ArrowPrimitiveType,
    T::Native: PartialOrd,
{
    fn new(array: &ArrayRef) -> CylonResult<Self> {
        use arrow::array::PrimitiveArray;

        let arr = array.as_any()
            .downcast_ref::<PrimitiveArray<T>>()
            .ok_or_else(|| CylonError::new(Code::Invalid, "Failed to downcast array".to_string()))?
            .clone();

        Ok(Self {
            array: Arc::new(arr),
        })
    }
}

impl<T> ArrayIndexComparator for PrimitiveComparator<T>
where
    T: arrow::datatypes::ArrowPrimitiveType,
    T::Native: PartialOrd,
{
    fn compare(&self, index1: i64, index2: i64) -> std::cmp::Ordering {
        let val1 = self.array.value(index1 as usize);
        let val2 = self.array.value(index2 as usize);
        val1.partial_cmp(&val2).unwrap_or(std::cmp::Ordering::Equal)
    }

    fn equal_to(&self, index1: i64, index2: i64) -> bool {
        self.array.value(index1 as usize) == self.array.value(index2 as usize)
    }
}

/// Comparator for string types in a single table
struct StringComparator {
    array: Arc<arrow::array::StringArray>,
}

impl StringComparator {
    fn new(array: &ArrayRef) -> CylonResult<Self> {
        use arrow::array::StringArray;

        let arr = array.as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| CylonError::new(Code::Invalid, "Failed to downcast string array".to_string()))?
            .clone();

        Ok(Self {
            array: Arc::new(arr),
        })
    }
}

impl ArrayIndexComparator for StringComparator {
    fn compare(&self, index1: i64, index2: i64) -> std::cmp::Ordering {
        let val1 = self.array.value(index1 as usize);
        let val2 = self.array.value(index2 as usize);
        val1.cmp(val2)
    }

    fn equal_to(&self, index1: i64, index2: i64) -> bool {
        self.array.value(index1 as usize) == self.array.value(index2 as usize)
    }
}

/// Comparator for large string types in a single table
struct LargeStringComparator {
    array: Arc<arrow::array::LargeStringArray>,
}

impl LargeStringComparator {
    fn new(array: &ArrayRef) -> CylonResult<Self> {
        use arrow::array::LargeStringArray;

        let arr = array.as_any()
            .downcast_ref::<LargeStringArray>()
            .ok_or_else(|| CylonError::new(Code::Invalid, "Failed to downcast large string array".to_string()))?
            .clone();

        Ok(Self {
            array: Arc::new(arr),
        })
    }
}

impl ArrayIndexComparator for LargeStringComparator {
    fn compare(&self, index1: i64, index2: i64) -> std::cmp::Ordering {
        let val1 = self.array.value(index1 as usize);
        let val2 = self.array.value(index2 as usize);
        val1.cmp(val2)
    }

    fn equal_to(&self, index1: i64, index2: i64) -> bool {
        self.array.value(index1 as usize) == self.array.value(index2 as usize)
    }
}

/// Wrapper for comparator with sort direction
struct ComparatorWithDirection {
    comparator: Box<dyn ArrayIndexComparator>,
    ascending: bool,
}

/// Equality comparator for a single table
/// Ported from cpp/src/cylon/arrow/arrow_comparator.hpp TableRowIndexEqualTo (lines 162-188)
pub struct TableRowIndexEqualTo {
    /// Comparators for each column with their sort directions
    comparators: Vec<ComparatorWithDirection>,
}

impl TableRowIndexEqualTo {
    /// Create equality comparator for a table
    pub fn new(batch: &RecordBatch) -> CylonResult<Self> {
        let all_columns: Vec<usize> = (0..batch.num_columns()).collect();
        Self::new_with_columns(batch, &all_columns)
    }

    /// Create equality comparator for specific columns of a table
    /// Ported from cpp/src/cylon/arrow/arrow_comparator.cpp TableRowIndexEqualTo::Make (lines 760-785)
    pub fn new_with_columns(batch: &RecordBatch, col_indices: &[usize]) -> CylonResult<Self> {
        // Default to ascending for all columns
        let sort_directions = vec![true; col_indices.len()];
        Self::new_with_columns_and_directions(batch, col_indices, &sort_directions)
    }

    /// Create equality comparator for specific columns with sort directions
    /// Ported from cpp/src/cylon/arrow/arrow_comparator.cpp TableRowIndexEqualTo::Make (lines 760-785)
    pub fn new_with_columns_and_directions(
        batch: &RecordBatch,
        col_indices: &[usize],
        sort_directions: &[bool],
    ) -> CylonResult<Self> {
        if col_indices.len() != sort_directions.len() {
            return Err(CylonError::new(
                Code::Invalid,
                format!("col_indices length {} != sort_directions length {}",
                    col_indices.len(), sort_directions.len())
            ));
        }

        let mut comparators: Vec<ComparatorWithDirection> = Vec::new();

        for (i, &col_idx) in col_indices.iter().enumerate() {
            if col_idx >= batch.num_columns() {
                return Err(CylonError::new(
                    Code::Invalid,
                    format!("Column index {} out of range", col_idx),
                ));
            }

            let column = batch.column(col_idx);
            let comparator = Self::create_comparator(column)?;
            comparators.push(ComparatorWithDirection {
                comparator,
                ascending: sort_directions[i],
            });
        }

        Ok(Self { comparators })
    }

    /// Create a comparator for a specific data type
    fn create_comparator(column: &ArrayRef) -> CylonResult<Box<dyn ArrayIndexComparator>> {
        use arrow::datatypes::*;

        match column.data_type() {
            DataType::Int8 => Ok(Box::new(PrimitiveComparator::<Int8Type>::new(column)?)),
            DataType::Int16 => Ok(Box::new(PrimitiveComparator::<Int16Type>::new(column)?)),
            DataType::Int32 => Ok(Box::new(PrimitiveComparator::<Int32Type>::new(column)?)),
            DataType::Int64 => Ok(Box::new(PrimitiveComparator::<Int64Type>::new(column)?)),
            DataType::UInt8 => Ok(Box::new(PrimitiveComparator::<UInt8Type>::new(column)?)),
            DataType::UInt16 => Ok(Box::new(PrimitiveComparator::<UInt16Type>::new(column)?)),
            DataType::UInt32 => Ok(Box::new(PrimitiveComparator::<UInt32Type>::new(column)?)),
            DataType::UInt64 => Ok(Box::new(PrimitiveComparator::<UInt64Type>::new(column)?)),
            DataType::Float32 => Ok(Box::new(PrimitiveComparator::<Float32Type>::new(column)?)),
            DataType::Float64 => Ok(Box::new(PrimitiveComparator::<Float64Type>::new(column)?)),
            DataType::Utf8 => Ok(Box::new(StringComparator::new(column)?)),
            DataType::LargeUtf8 => Ok(Box::new(LargeStringComparator::new(column)?)),
            DataType::Date32 => Ok(Box::new(PrimitiveComparator::<Date32Type>::new(column)?)),
            DataType::Date64 => Ok(Box::new(PrimitiveComparator::<Date64Type>::new(column)?)),
            DataType::Timestamp(_, _) => Ok(Box::new(PrimitiveComparator::<TimestampMicrosecondType>::new(column)?)),
            DataType::Time32(_) => Ok(Box::new(PrimitiveComparator::<Time32MillisecondType>::new(column)?)),
            DataType::Time64(_) => Ok(Box::new(PrimitiveComparator::<Time64MicrosecondType>::new(column)?)),
            _ => Err(CylonError::new(
                Code::Invalid,
                format!("Unsupported data type for comparison: {:?}", column.data_type()),
            )),
        }
    }

    /// Check if two row indices are equal
    /// Ported from cpp/src/cylon/arrow/arrow_comparator.cpp line 806-811
    pub fn equal(&self, record1: i64, record2: i64) -> bool {
        for comp_with_dir in &self.comparators {
            if !comp_with_dir.comparator.equal_to(record1, record2) {
                return false;
            }
        }
        true
    }

    /// Compare two row indices
    /// Ported from cpp/src/cylon/arrow/arrow_comparator.cpp line 813-821
    /// Applies sort direction: if ascending=false, reverses the comparison
    pub fn compare(&self, record1: i64, record2: i64) -> std::cmp::Ordering {
        for comp_with_dir in &self.comparators {
            let ordering = comp_with_dir.comparator.compare(record1, record2);
            match ordering {
                std::cmp::Ordering::Equal => continue,
                std::cmp::Ordering::Less if comp_with_dir.ascending => return std::cmp::Ordering::Less,
                std::cmp::Ordering::Less => return std::cmp::Ordering::Greater,  // Reverse for descending
                std::cmp::Ordering::Greater if comp_with_dir.ascending => return std::cmp::Ordering::Greater,
                std::cmp::Ordering::Greater => return std::cmp::Ordering::Less,  // Reverse for descending
            }
        }
        std::cmp::Ordering::Equal
    }
}
