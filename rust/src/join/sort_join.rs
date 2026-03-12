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

//! Sort-merge join implementation
//!
//! Ported from cpp/src/cylon/join/sort_join.cpp

use std::time::Instant;
use arrow::array::{
    Array, ArrayRef, RecordBatch, UInt32Array,
    Int8Array, Int16Array, Int32Array, Int64Array,
    UInt8Array, UInt16Array, UInt64Array,
    Float32Array, Float64Array,
};
use arrow::compute::sort_to_indices;
use arrow::datatypes::DataType;
use arrow_row::{RowConverter, SortField, Rows};
use log::info;

use crate::table::Table;
use crate::join::config::{JoinConfig, JoinType};
use crate::join::utils::build_final_table;
use crate::error::{CylonResult, CylonError, Code};

/// Combine chunks in a RecordBatch if needed
/// Corresponds to C++ COMBINE_CHUNKS_RETURN_CYLON_STATUS macro (sort_join.cpp:63-65)
/// Note: RecordBatch columns are already single arrays in Rust Arrow, so this is a no-op
fn combine_chunks(batch: &RecordBatch) -> CylonResult<RecordBatch> {
    Ok(batch.clone())
}

/* SINGLE INDEX */

/// Advance through sorted indices collecting rows with equal keys
/// Corresponds to C++ advance template function (sort_join.cpp:28-50)
///
/// Deviations from C++:
/// - C++ uses UInt64Array for sorted_indices, Rust uses UInt32Array because arrow::compute::sort_to_indices
///   returns UInt32Array. This is functionally equivalent for practical table sizes.
/// - C++ uses raw pointer to key, Rust returns Option<T> for key
fn advance<T, F>(
    subset: &mut Vec<i64>,
    sorted_indices: &UInt32Array,  // C++ uses UInt64Array (sort_join.cpp:31)
    current_index: &mut i64,       // always int64_t (sort_join.cpp:32)
    data_column: &dyn Array,
    key: &mut Option<T>,
    get_value: F,
) where
    T: PartialEq + Copy,
    F: Fn(&dyn Array, usize) -> T,
{
    subset.clear();
    if *current_index == sorted_indices.len() as i64 {
        *key = None;
        return;
    }

    let data_index = sorted_indices.value(*current_index as usize) as i64;
    let key_value = get_value(data_column, data_index as usize);
    *key = Some(key_value);

    while *current_index < sorted_indices.len() as i64 {
        let idx = sorted_indices.value(*current_index as usize) as i64;
        if get_value(data_column, idx as usize) != key_value {
            break;
        }
        subset.push(idx);
        *current_index += 1;
        if *current_index == sorted_indices.len() as i64 {
            break;
        }
    }
}

/// Single-column sort join implementation
/// Corresponds to C++ do_sorted_join template function (sort_join.cpp:52-173)
fn do_sorted_join<T, F>(
    left_table: &Table,
    right_table: &Table,
    left_tab_comb: &RecordBatch,
    right_tab_comb: &RecordBatch,
    left_join_column_idx: usize,
    right_join_column_idx: usize,
    join_type: JoinType,
    left_table_suffix: &str,
    right_table_suffix: &str,
    get_value: F,
) -> CylonResult<Table>
where
    T: PartialOrd + Copy,
    F: Fn(&dyn Array, usize) -> T + Copy,
{
    // sort columns (sort_join.cpp:68-81)
    let left_join_column = left_tab_comb.column(left_join_column_idx);
    let right_join_column = right_tab_comb.column(right_join_column_idx);

    // C++ uses SortIndices which returns UInt64Array (sort_join.cpp:73-81)
    // Rust sort_to_indices returns UInt32Array - functionally equivalent
    let left_index_sorted_column = sort_to_indices(left_join_column.as_ref(), None, None)
        .map_err(|e| CylonError::new(
            Code::ExecutionError,
            format!("Failed to sort left column: {}", e),
        ))?;

    let right_index_sorted_column = sort_to_indices(right_join_column.as_ref(), None, None)
        .map_err(|e| CylonError::new(
            Code::ExecutionError,
            format!("Failed to sort right column: {}", e),
        ))?;

    // (sort_join.cpp:83-91)
    let mut left_key: Option<T> = None;
    let mut right_key: Option<T> = None;
    let mut left_subset: Vec<i64> = Vec::new();
    let mut right_subset: Vec<i64> = Vec::new();
    let mut left_current_index: i64 = 0;
    let mut right_current_index: i64 = 0;

    let mut left_indices: Vec<i64> = Vec::new();
    let mut right_indices: Vec<i64> = Vec::new();
    let init_vec_size = std::cmp::min(left_join_column.len(), right_join_column.len());
    left_indices.reserve(init_vec_size);
    right_indices.reserve(init_vec_size);

    // Initial advance (sort_join.cpp:93-97)
    advance(
        &mut left_subset, &left_index_sorted_column, &mut left_current_index,
        left_join_column.as_ref(), &mut left_key, get_value,
    );

    advance(
        &mut right_subset, &right_index_sorted_column, &mut right_current_index,
        right_join_column.as_ref(), &mut right_key, get_value,
    );

    // Main loop (sort_join.cpp:98-135)
    while !left_subset.is_empty() && !right_subset.is_empty() {
        let lk = left_key.unwrap();
        let rk = right_key.unwrap();

        if lk == rk {
            // Keys match - produce cartesian product (sort_join.cpp:100-105)
            for &left_idx in &left_subset {
                for &right_idx in &right_subset {
                    left_indices.push(left_idx);
                    right_indices.push(right_idx);
                }
            }
            // advance (sort_join.cpp:107-111)
            advance(
                &mut left_subset, &left_index_sorted_column, &mut left_current_index,
                left_join_column.as_ref(), &mut left_key, get_value,
            );
            advance(
                &mut right_subset, &right_index_sorted_column, &mut right_current_index,
                right_join_column.as_ref(), &mut right_key, get_value,
            );
        } else if lk < rk {
            // if this is a left join, include them all (sort_join.cpp:114-118)
            if join_type == JoinType::Left || join_type == JoinType::FullOuter {
                for &left_idx in &left_subset {
                    left_indices.push(left_idx);
                    right_indices.push(-1);
                }
            }
            advance(
                &mut left_subset, &left_index_sorted_column, &mut left_current_index,
                left_join_column.as_ref(), &mut left_key, get_value,
            );
        } else {
            // if this is a right join, include them all (sort_join.cpp:125-129)
            if join_type == JoinType::Right || join_type == JoinType::FullOuter {
                for &right_idx in &right_subset {
                    left_indices.push(-1);
                    right_indices.push(right_idx);
                }
            }
            advance(
                &mut right_subset, &right_index_sorted_column, &mut right_current_index,
                right_join_column.as_ref(), &mut right_key, get_value,
            );
        }
    }

    // specially handling left and right join (sort_join.cpp:138-158)
    if join_type == JoinType::Left || join_type == JoinType::FullOuter {
        while !left_subset.is_empty() {
            for &left_idx in &left_subset {
                left_indices.push(left_idx);
                right_indices.push(-1);
            }
            advance(
                &mut left_subset, &left_index_sorted_column, &mut left_current_index,
                left_join_column.as_ref(), &mut left_key, get_value,
            );
        }
    }

    if join_type == JoinType::Right || join_type == JoinType::FullOuter {
        while !right_subset.is_empty() {
            for &right_idx in &right_subset {
                left_indices.push(-1);
                right_indices.push(right_idx);
            }
            advance(
                &mut right_subset, &right_index_sorted_column, &mut right_current_index,
                right_join_column.as_ref(), &mut right_key, get_value,
            );
        }
    }

    // clear the sort columns (sort_join.cpp:160-162)
    // In Rust, we use drop() instead of .reset()
    drop(left_index_sorted_column);
    drop(right_index_sorted_column);

    // build final table (sort_join.cpp:164-171)
    build_final_table(
        &left_indices,
        &right_indices,
        left_table,
        right_table,
        left_table_suffix,
        right_table_suffix,
    )
}

/* SINGLE INDEX INPLACE */

/// Advance through sorted data directly (no index indirection)
/// Corresponds to C++ advance_inplace_array template function (sort_join.cpp:177-195)
///
/// Unlike `advance` which uses sorted indices, this function works directly on
/// sorted data for better cache locality.
fn advance_inplace_array<T, F>(
    subset: &mut Vec<i64>,
    current_index: &mut i64,      // always int64_t (sort_join.cpp:179)
    data_column: &dyn Array,
    length: i64,
    key: &mut Option<T>,
    get_value: F,
) where
    T: PartialEq + Copy,
    F: Fn(&dyn Array, usize) -> T,
{
    subset.clear();
    if *current_index == length {
        *key = None;
        return;
    }

    // (sort_join.cpp:186-187)
    let key_value = get_value(data_column, *current_index as usize);
    *key = Some(key_value);

    // (sort_join.cpp:188-194)
    while *current_index < length && get_value(data_column, *current_index as usize) == key_value {
        subset.push(*current_index);
        *current_index += 1;
        if *current_index == length {
            break;
        }
    }
}

/// Single-column inplace sort join implementation
/// Corresponds to C++ do_inplace_sorted_join template function (sort_join.cpp:197-322)
///
/// This version sorts the join columns in place and works directly on sorted data,
/// providing better cache locality for numeric types.
fn do_inplace_sorted_join<T, F>(
    left_table: &Table,
    right_table: &Table,
    left_tab_comb: &RecordBatch,
    right_tab_comb: &RecordBatch,
    left_join_column_idx: usize,
    right_join_column_idx: usize,
    join_type: JoinType,
    left_table_suffix: &str,
    right_table_suffix: &str,
    get_value: F,
) -> CylonResult<Table>
where
    T: PartialOrd + Copy,
    F: Fn(&dyn Array, usize) -> T + Copy,
{
    use crate::arrow::arrow_kernels::sort_indices_inplace;
    use crate::join::utils::build_final_table_inplace_index;

    // COMBINE_CHUNKS equivalent (sort_join.cpp:207-210)
    let left_tab_comb = combine_chunks(left_tab_comb)?;
    let right_tab_comb = combine_chunks(right_tab_comb)?;

    // sort columns (sort_join.cpp:212-216)
    let left_join_column = left_tab_comb.column(left_join_column_idx);
    let right_join_column = right_tab_comb.column(right_join_column_idx);

    // C++ uses SortIndicesInPlace which sorts in place and returns mapping (sort_join.cpp:218-226)
    // sort_indices_inplace returns (sorted_array, indices) in one operation using introsort
    let (left_join_column_sorted, left_index_sorted_column) = sort_indices_inplace(left_join_column)?;
    let (right_join_column_sorted, right_index_sorted_column) = sort_indices_inplace(right_join_column)?;

    // (sort_join.cpp:228-236)
    let mut left_key: Option<T> = None;
    let mut right_key: Option<T> = None;
    let mut left_subset: Vec<i64> = Vec::new();
    let mut right_subset: Vec<i64> = Vec::new();
    let mut left_current_index: i64 = 0;
    let mut right_current_index: i64 = 0;

    let mut left_indices: Vec<i64> = Vec::new();
    let mut right_indices: Vec<i64> = Vec::new();
    let init_vec_size = std::cmp::min(left_join_column_sorted.len(), right_join_column_sorted.len());
    left_indices.reserve(init_vec_size);
    right_indices.reserve(init_vec_size);

    // (sort_join.cpp:238-239)
    let col_length = left_join_column_sorted.len() as i64;
    let right_col_length = right_join_column_sorted.len() as i64;

    // Initial advance (sort_join.cpp:241-245)
    advance_inplace_array(
        &mut left_subset, &mut left_current_index,
        left_join_column_sorted.as_ref(), col_length, &mut left_key, get_value,
    );

    advance_inplace_array(
        &mut right_subset, &mut right_current_index,
        right_join_column_sorted.as_ref(), right_col_length, &mut right_key, get_value,
    );

    // Main loop (sort_join.cpp:246-283)
    while !left_subset.is_empty() && !right_subset.is_empty() {
        let lk = left_key.unwrap();
        let rk = right_key.unwrap();

        if lk == rk {
            // Keys match - produce cartesian product (sort_join.cpp:248-253)
            for &left_idx in &left_subset {
                for &right_idx in &right_subset {
                    left_indices.push(left_idx);
                    right_indices.push(right_idx);
                }
            }
            // advance (sort_join.cpp:255-259)
            advance_inplace_array(
                &mut left_subset, &mut left_current_index,
                left_join_column_sorted.as_ref(), col_length, &mut left_key, get_value,
            );
            advance_inplace_array(
                &mut right_subset, &mut right_current_index,
                right_join_column_sorted.as_ref(), right_col_length, &mut right_key, get_value,
            );
        } else if lk < rk {
            // if this is a left join, include them all (sort_join.cpp:262-267)
            if join_type == JoinType::Left || join_type == JoinType::FullOuter {
                for &left_idx in &left_subset {
                    left_indices.push(left_idx);
                    right_indices.push(-1);
                }
            }
            advance_inplace_array(
                &mut left_subset, &mut left_current_index,
                left_join_column_sorted.as_ref(), col_length, &mut left_key, get_value,
            );
        } else {
            // if this is a right join, include them all (sort_join.cpp:273-278)
            if join_type == JoinType::Right || join_type == JoinType::FullOuter {
                for &right_idx in &right_subset {
                    left_indices.push(-1);
                    right_indices.push(right_idx);
                }
            }
            advance_inplace_array(
                &mut right_subset, &mut right_current_index,
                right_join_column_sorted.as_ref(), right_col_length, &mut right_key, get_value,
            );
        }
    }

    // specially handling left and right join (sort_join.cpp:285-306)
    if join_type == JoinType::Left || join_type == JoinType::FullOuter {
        while !left_subset.is_empty() {
            for &left_idx in &left_subset {
                left_indices.push(left_idx);
                right_indices.push(-1);
            }
            advance_inplace_array(
                &mut left_subset, &mut left_current_index,
                left_join_column_sorted.as_ref(), col_length, &mut left_key, get_value,
            );
        }
    }

    if join_type == JoinType::Right || join_type == JoinType::FullOuter {
        while !right_subset.is_empty() {
            for &right_idx in &right_subset {
                left_indices.push(-1);
                right_indices.push(right_idx);
            }
            advance_inplace_array(
                &mut right_subset, &mut right_current_index,
                right_join_column_sorted.as_ref(), right_col_length, &mut right_key, get_value,
            );
        }
    }

    // build final table (sort_join.cpp:308-320)
    build_final_table_inplace_index(
        left_join_column_idx,
        right_join_column_idx,
        &left_indices,
        &right_indices,
        &left_index_sorted_column,
        &right_index_sorted_column,
        &left_join_column_sorted,   // Sorted column (C++ modifies in place, Rust passes separately)
        &right_join_column_sorted,  // Sorted column (C++ modifies in place, Rust passes separately)
        left_table,
        right_table,
        left_table_suffix,
        right_table_suffix,
    )
}

/* MULTI INDEX */

/// Advance for multi-column join using row comparator
/// Corresponds to C++ advance_multi_index template function (sort_join.cpp:328-361)
///
/// Deviations from C++:
/// - C++ uses DualTableRowIndexEqualTo comparator with SetBit for left/right distinction (sort_join.cpp:341-360)
/// - Rust uses separate left_rows and right_rows from arrow_row, so SetBit is not needed
/// - The side parameter is kept for structural parity with C++, but the comparison logic
///   uses the appropriate rows object directly instead of SetBit
fn advance_multi_index<const SIDE: bool>(
    rows: &Rows,
    sorted_indices: &UInt32Array,  // C++ uses UInt64Array
    subset: &mut Vec<i64>,
    current_index: &mut i64,
    key_index: &mut i64,
) {
    subset.clear();
    if *current_index == sorted_indices.len() as i64 {
        return;
    }

    let data_index = sorted_indices.value(*current_index as usize) as i64;
    *key_index = data_index;

    // C++ (sort_join.cpp:341-360):
    // if (side == Left) uses comp(data_index, *key_index)
    // else uses comp(SetBit(data_index), SetBit(*key_index))
    // In Rust, we use separate rows objects so no SetBit needed
    while *current_index < sorted_indices.len() as i64 {
        let idx = sorted_indices.value(*current_index as usize) as i64;
        // Compare rows for equality
        if rows.row(idx as usize) != rows.row(*key_index as usize) {
            break;
        }
        subset.push(idx);
        *current_index += 1;
        if *current_index == sorted_indices.len() as i64 {
            break;
        }
    }
}

/// Multi-column sort join implementation
/// Corresponds to C++ do_multi_index_sorted_join (sort_join.cpp:363-496)
fn do_multi_index_sorted_join(
    left_table: &Table,
    right_table: &Table,
    left_tab_comb: &RecordBatch,
    right_tab_comb: &RecordBatch,
    left_join_column_indices: &[usize],
    right_join_column_indices: &[usize],
    join_type: JoinType,
    left_table_suffix: &str,
    right_table_suffix: &str,
) -> CylonResult<Table> {
    // combine chunks if multiple chunks are available (sort_join.cpp:375-378)
    let t11 = Instant::now();

    // Note: combine_chunks is already done in sort_join before calling this function
    // This matches C++ lines 377-378

    let t22 = Instant::now();
    info!("CombineBeforeShuffle chunks time : {:?}", t22.duration_since(t11).as_millis());

    // create sorter and do index sort (sort_join.cpp:386)
    let t1 = Instant::now();

    // Extract join columns
    let mut left_arrays: Vec<ArrayRef> = Vec::new();
    let mut right_arrays: Vec<ArrayRef> = Vec::new();

    for &col_idx in left_join_column_indices {
        left_arrays.push(left_tab_comb.column(col_idx).clone());
    }
    for &col_idx in right_join_column_indices {
        right_arrays.push(right_tab_comb.column(col_idx).clone());
    }

    // Create row converter for sorting and comparison
    // C++ uses DualTableRowIndexEqualTo (sort_join.cpp:409-412)
    // Rust uses arrow_row for similar functionality
    let left_fields: Vec<SortField> = left_arrays
        .iter()
        .map(|arr| SortField::new(arr.data_type().clone()))
        .collect();

    let right_fields: Vec<SortField> = right_arrays
        .iter()
        .map(|arr| SortField::new(arr.data_type().clone()))
        .collect();

    let mut left_converter = RowConverter::new(left_fields)
        .map_err(|e| CylonError::new(
            Code::ExecutionError,
            format!("Failed to create left row converter: {}", e),
        ))?;

    let mut right_converter = RowConverter::new(right_fields)
        .map_err(|e| CylonError::new(
            Code::ExecutionError,
            format!("Failed to create right row converter: {}", e),
        ))?;

    let left_rows = left_converter.convert_columns(&left_arrays)
        .map_err(|e| CylonError::new(
            Code::ExecutionError,
            format!("Failed to convert left columns: {}", e),
        ))?;

    let right_rows = right_converter.convert_columns(&right_arrays)
        .map_err(|e| CylonError::new(
            Code::ExecutionError,
            format!("Failed to convert right columns: {}", e),
        ))?;

    // Sort indices for both tables
    // C++ uses SortIndicesMultiColumns (sort_join.cpp:388-392)
    let left_num_rows = left_rows.num_rows();
    let right_num_rows = right_rows.num_rows();

    let mut left_sort_indices: Vec<u32> = (0..left_num_rows as u32).collect();
    left_sort_indices.sort_by(|&a, &b| {
        left_rows.row(a as usize).cmp(&left_rows.row(b as usize))
    });
    let left_index_sorted_column = UInt32Array::from(left_sort_indices);

    let t2 = Instant::now();
    info!("Left sorting time : {:?}", t2.duration_since(t1).as_millis());

    // (sort_join.cpp:398-402)
    let mut right_sort_indices: Vec<u32> = (0..right_num_rows as u32).collect();
    right_sort_indices.sort_by(|&a, &b| {
        right_rows.row(a as usize).cmp(&right_rows.row(b as usize))
    });
    let right_index_sorted_column = UInt32Array::from(right_sort_indices);

    let t1 = Instant::now();
    info!("right sorting time : {:?}", t1.duration_since(t2).as_millis());

    // (sort_join.cpp:414-423)
    let mut left_key_index: i64 = 0;
    let mut right_key_index: i64 = 0;
    let mut left_subset: Vec<i64> = Vec::new();
    let mut right_subset: Vec<i64> = Vec::new();
    let mut left_current_index: i64 = 0;
    let mut right_current_index: i64 = 0;

    let mut left_indices: Vec<i64> = Vec::new();
    let mut right_indices: Vec<i64> = Vec::new();
    let init_vec_size = std::cmp::min(left_num_rows, right_num_rows);
    left_indices.reserve(init_vec_size);
    right_indices.reserve(init_vec_size);

    // Initial advance (sort_join.cpp:425-429)
    // C++ uses advance_multi_index<Left> and advance_multi_index<Right>
    advance_multi_index::<true>(
        &left_rows, &left_index_sorted_column, &mut left_subset,
        &mut left_current_index, &mut left_key_index,
    );

    advance_multi_index::<false>(
        &right_rows, &right_index_sorted_column, &mut right_subset,
        &mut right_current_index, &mut right_key_index,
    );

    // Main loop (sort_join.cpp:431-462)
    while !left_subset.is_empty() && !right_subset.is_empty() {
        // C++ uses multi_tab_comp->compare (sort_join.cpp:432)
        // Rust: compare the key rows directly
        let cmp = left_rows.row(left_key_index as usize)
            .cmp(&right_rows.row(right_key_index as usize));

        match cmp {
            std::cmp::Ordering::Equal => {
                // Keys match (sort_join.cpp:434-438)
                for &left_idx in &left_subset {
                    for &right_idx in &right_subset {
                        left_indices.push(left_idx);
                        right_indices.push(right_idx);
                    }
                }
                // advance (sort_join.cpp:441-444)
                advance_multi_index::<true>(
                    &left_rows, &left_index_sorted_column, &mut left_subset,
                    &mut left_current_index, &mut left_key_index,
                );
                advance_multi_index::<false>(
                    &right_rows, &right_index_sorted_column, &mut right_subset,
                    &mut right_current_index, &mut right_key_index,
                );
            }
            std::cmp::Ordering::Less => {
                // Left key is smaller (sort_join.cpp:447-452)
                if join_type == JoinType::Left || join_type == JoinType::FullOuter {
                    left_indices.extend(left_subset.iter());
                    right_indices.extend(std::iter::repeat(-1i64).take(left_subset.len()));
                }
                advance_multi_index::<true>(
                    &left_rows, &left_index_sorted_column, &mut left_subset,
                    &mut left_current_index, &mut left_key_index,
                );
            }
            std::cmp::Ordering::Greater => {
                // Right key is smaller (sort_join.cpp:455-460)
                if join_type == JoinType::Right || join_type == JoinType::FullOuter {
                    left_indices.extend(std::iter::repeat(-1i64).take(right_subset.len()));
                    right_indices.extend(right_subset.iter());
                }
                advance_multi_index::<false>(
                    &right_rows, &right_index_sorted_column, &mut right_subset,
                    &mut right_current_index, &mut right_key_index,
                );
            }
        }
    }

    // specially handling left and right join (sort_join.cpp:465-473)
    if join_type == JoinType::Left || join_type == JoinType::FullOuter {
        left_indices.extend(left_subset.iter());
        right_indices.extend(std::iter::repeat(-1i64).take(left_subset.len()));
    }

    if join_type == JoinType::Right || join_type == JoinType::FullOuter {
        left_indices.extend(std::iter::repeat(-1i64).take(right_subset.len()));
        right_indices.extend(right_subset.iter());
    }

    // clear the sort columns (sort_join.cpp:475-479)
    // In Rust, we use drop() instead of .reset()
    drop(left_index_sorted_column);
    drop(right_index_sorted_column);
    drop(left_subset);
    drop(right_subset);

    let t2 = Instant::now();
    info!("Index join time : {:?}", t2.duration_since(t1).as_millis());
    info!("Building final table with number of tuples - {}", left_indices.len());

    let t1 = Instant::now();

    // build final table (sort_join.cpp:487-490)
    let result = build_final_table(
        &left_indices,
        &right_indices,
        left_table,
        right_table,
        left_table_suffix,
        right_table_suffix,
    );

    let t2 = Instant::now();
    info!("Built final table in : {:?}", t2.duration_since(t1).as_millis());
    info!("Done and produced : {}", left_indices.len());

    result
}

/// Main sort join entry point
/// Corresponds to C++ SortJoin function (sort_join.cpp:525-763)
pub fn sort_join(
    left_table: &Table,
    right_table: &Table,
    config: &JoinConfig,
) -> CylonResult<Table> {
    let left_col_indices = config.left_column_indices();
    let right_col_indices = config.right_column_indices();

    let left_batch = left_table.batch(0).ok_or_else(|| {
        CylonError::new(Code::Invalid, "left table has no batches".to_string())
    })?;

    let right_batch = right_table.batch(0).ok_or_else(|| {
        CylonError::new(Code::Invalid, "right table has no batches".to_string())
    })?;

    // Combine chunks (sort_join.cpp:63-65)
    let left_tab_comb = combine_chunks(left_batch)?;
    let right_tab_comb = combine_chunks(right_batch)?;

    // sort joins (sort_join.cpp:534-760)
    if left_col_indices.len() == 1 && right_col_indices.len() == 1 {
        let left_join_column_idx = left_col_indices[0];
        let right_join_column_idx = right_col_indices[0];

        if left_join_column_idx >= left_tab_comb.num_columns() {
            return Err(CylonError::new(
                Code::Invalid,
                format!("left column index {} out of bounds", left_join_column_idx),
            ));
        }

        if right_join_column_idx >= right_tab_comb.num_columns() {
            return Err(CylonError::new(
                Code::Invalid,
                format!("right column index {} out of bounds", right_join_column_idx),
            ));
        }

        let left_type = left_tab_comb.column(left_join_column_idx).data_type();
        let right_type = right_tab_comb.column(right_join_column_idx).data_type();

        // (sort_join.cpp:538-541)
        if left_type != right_type {
            return Err(CylonError::new(
                Code::Invalid,
                "The join column types of two tables mismatches.".to_string(),
            ));
        }

        // Type dispatch (sort_join.cpp:543-749)
        // For numeric types, use do_inplace_sorted_join (sort_join.cpp:508-517)
        // For non-numeric types, use do_sorted_join (sort_join.cpp:518-522)
        match left_type {
            DataType::Int8 => {
                // is_number_type -> do_inplace_sorted_join
                do_inplace_sorted_join(
                    left_table, right_table,
                    &left_tab_comb, &right_tab_comb,
                    left_join_column_idx, right_join_column_idx,
                    config.join_type(),
                    config.left_table_suffix(), config.right_table_suffix(),
                    |arr, idx| arr.as_any().downcast_ref::<Int8Array>().unwrap().value(idx),
                )
            }
            DataType::Int16 => {
                // is_number_type -> do_inplace_sorted_join
                do_inplace_sorted_join(
                    left_table, right_table,
                    &left_tab_comb, &right_tab_comb,
                    left_join_column_idx, right_join_column_idx,
                    config.join_type(),
                    config.left_table_suffix(), config.right_table_suffix(),
                    |arr, idx| arr.as_any().downcast_ref::<Int16Array>().unwrap().value(idx),
                )
            }
            DataType::Int32 => {
                // is_number_type -> do_inplace_sorted_join
                do_inplace_sorted_join(
                    left_table, right_table,
                    &left_tab_comb, &right_tab_comb,
                    left_join_column_idx, right_join_column_idx,
                    config.join_type(),
                    config.left_table_suffix(), config.right_table_suffix(),
                    |arr, idx| arr.as_any().downcast_ref::<Int32Array>().unwrap().value(idx),
                )
            }
            DataType::Int64 => {
                // is_number_type -> do_inplace_sorted_join
                do_inplace_sorted_join(
                    left_table, right_table,
                    &left_tab_comb, &right_tab_comb,
                    left_join_column_idx, right_join_column_idx,
                    config.join_type(),
                    config.left_table_suffix(), config.right_table_suffix(),
                    |arr, idx| arr.as_any().downcast_ref::<Int64Array>().unwrap().value(idx),
                )
            }
            DataType::UInt8 => {
                // is_number_type -> do_inplace_sorted_join
                do_inplace_sorted_join(
                    left_table, right_table,
                    &left_tab_comb, &right_tab_comb,
                    left_join_column_idx, right_join_column_idx,
                    config.join_type(),
                    config.left_table_suffix(), config.right_table_suffix(),
                    |arr, idx| arr.as_any().downcast_ref::<UInt8Array>().unwrap().value(idx),
                )
            }
            DataType::UInt16 => {
                // is_number_type -> do_inplace_sorted_join
                do_inplace_sorted_join(
                    left_table, right_table,
                    &left_tab_comb, &right_tab_comb,
                    left_join_column_idx, right_join_column_idx,
                    config.join_type(),
                    config.left_table_suffix(), config.right_table_suffix(),
                    |arr, idx| arr.as_any().downcast_ref::<UInt16Array>().unwrap().value(idx),
                )
            }
            DataType::UInt32 => {
                // is_number_type -> do_inplace_sorted_join
                do_inplace_sorted_join(
                    left_table, right_table,
                    &left_tab_comb, &right_tab_comb,
                    left_join_column_idx, right_join_column_idx,
                    config.join_type(),
                    config.left_table_suffix(), config.right_table_suffix(),
                    |arr, idx| arr.as_any().downcast_ref::<UInt32Array>().unwrap().value(idx),
                )
            }
            DataType::UInt64 => {
                // is_number_type -> do_inplace_sorted_join
                do_inplace_sorted_join(
                    left_table, right_table,
                    &left_tab_comb, &right_tab_comb,
                    left_join_column_idx, right_join_column_idx,
                    config.join_type(),
                    config.left_table_suffix(), config.right_table_suffix(),
                    |arr, idx| arr.as_any().downcast_ref::<UInt64Array>().unwrap().value(idx),
                )
            }
            DataType::Float32 => {
                // is_number_type -> do_inplace_sorted_join
                do_inplace_sorted_join(
                    left_table, right_table,
                    &left_tab_comb, &right_tab_comb,
                    left_join_column_idx, right_join_column_idx,
                    config.join_type(),
                    config.left_table_suffix(), config.right_table_suffix(),
                    |arr, idx| {
                        let v = arr.as_any().downcast_ref::<Float32Array>().unwrap().value(idx);
                        ordered_float::OrderedFloat(v)
                    },
                )
            }
            DataType::Float64 => {
                // is_number_type -> do_inplace_sorted_join
                do_inplace_sorted_join(
                    left_table, right_table,
                    &left_tab_comb, &right_tab_comb,
                    left_join_column_idx, right_join_column_idx,
                    config.join_type(),
                    config.left_table_suffix(), config.right_table_suffix(),
                    |arr, idx| {
                        let v = arr.as_any().downcast_ref::<Float64Array>().unwrap().value(idx);
                        ordered_float::OrderedFloat(v)
                    },
                )
            }
            // String and Binary types use multi-column path due to lifetime complexity
            // with references in the do_sorted_join generic approach
            DataType::Utf8 | DataType::Binary | DataType::LargeUtf8 | DataType::LargeBinary => {
                do_multi_index_sorted_join(
                    left_table,
                    right_table,
                    &left_tab_comb,
                    &right_tab_comb,
                    left_col_indices,
                    right_col_indices,
                    config.join_type(),
                    config.left_table_suffix(),
                    config.right_table_suffix(),
                )
            }
            dt => {
                Err(CylonError::new(
                    Code::Invalid,
                    format!("Un-supported type {:?}", dt),
                ))
            }
        }
    } else {
        // Multi-column join (sort_join.cpp:751-760)
        do_multi_index_sorted_join(
            left_table,
            right_table,
            &left_tab_comb,
            &right_tab_comb,
            left_col_indices,
            right_col_indices,
            config.join_type(),
            config.left_table_suffix(),
            config.right_table_suffix(),
        )
    }
}
