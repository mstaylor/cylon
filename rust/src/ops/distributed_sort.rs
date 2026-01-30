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

//! Distributed sort using sample sort algorithm
//!
//! Ported from cpp/src/cylon/table.cpp (DistributedSort, DistributedSortRegularSampling)
//!
//! Supports two partitioning strategies:
//! - **Histogram-based** (default, matches C++ API): Uses MapToSortPartitions with
//!   configurable num_bins and num_samples for balanced partitioning via AllReduce
//! - **Split-points**: Legacy approach using gathered samples and broadcast split points

use crate::error::{CylonResult, CylonError, Code};
use crate::partition::{map_to_sort_partitions, split_by_partition, RangePartitionOptions};
use crate::table::Table;
use crate::util::arrow_utils;
use std::sync::Arc;
use arrow::array::{Array, ArrayRef, Int64Array};
use arrow::record_batch::RecordBatch;
use std::cmp::Ordering;

/// Sort options for distributed sort
/// Corresponds to C++ DistributedSort parameters (table.cpp:701-710)
#[derive(Debug, Clone)]
pub struct SortOptions {
    /// Number of samples for histogram building (default: num_bins if 0)
    /// Corresponds to C++ num_samples parameter
    pub num_samples: u64,
    /// Number of histogram bins for range partitioning (default: num_partitions * 16 if 0)
    /// Corresponds to C++ num_bins parameter
    pub num_bins: u32,
    /// Use histogram-based partitioning (true) or split-points (false)
    /// Default is true to match C++ behavior
    pub use_histogram: bool,
}

impl Default for SortOptions {
    fn default() -> Self {
        Self {
            num_samples: 0,  // 0 means use default (num_bins)
            num_bins: 0,     // 0 means use default (num_partitions * 16)
            use_histogram: true,  // Match C++ behavior
        }
    }
}

impl SortOptions {
    /// Create options with specific num_samples and num_bins (matches C++ API)
    pub fn with_histogram(num_samples: u64, num_bins: u32) -> Self {
        Self {
            num_samples,
            num_bins,
            use_histogram: true,
        }
    }

    /// Create options using split-points approach (legacy)
    pub fn with_split_points(num_samples: u64) -> Self {
        Self {
            num_samples,
            num_bins: 0,
            use_histogram: false,
        }
    }
}

/// Partition table by split points (range partitioning)
/// Corresponds to C++ GetSplitPointIndices + Split (table.cpp:564-618)
///
/// Compares each row with split points to determine which partition it belongs to.
/// This preserves sort order across partitions.
///
/// # Arguments
/// * `table` - Sorted table to partition
/// * `split_points` - Table containing split point values (world_size-1 rows)
/// * `sort_columns` - Columns to compare
/// * `sort_directions` - Sort direction for each column
/// * `num_partitions` - Number of partitions to create (typically world_size)
///
/// # Returns
/// Vector of RecordBatches, one per partition
fn partition_by_split_points(
    table: &Table,
    split_points: &Table,
    sort_columns: &[usize],
    sort_directions: &[bool],
    num_partitions: usize,
) -> CylonResult<Vec<RecordBatch>> {
    if table.rows() == 0 {
        // Return empty partitions
        let schema = table.schema()
            .ok_or_else(|| CylonError::new(Code::Invalid, "Table has no schema".to_string()))?;
        let empty_batch = RecordBatch::new_empty(schema);
        return Ok(vec![empty_batch; num_partitions]);
    }

    // For each row, determine which partition it belongs to by comparing with split points
    let num_rows = table.rows() as usize;  // i64 -> usize
    let mut partition_indices = Vec::with_capacity(num_rows);

    for row_idx in 0..num_rows {
        // Binary search to find which partition this row belongs to
        let mut partition: usize = 0;

        for split_idx in 0..(split_points.rows() as usize) {
            // Compare row with split_point[split_idx]
            if compare_rows(table, row_idx, split_points, split_idx, sort_columns, sort_directions)? == Ordering::Less {
                break;
            }
            partition = split_idx + 1;
        }

        partition_indices.push(partition);
    }

    // Group rows by partition
    let mut partition_row_indices: Vec<Vec<i64>> = vec![Vec::new(); num_partitions];
    for (row_idx, &partition) in partition_indices.iter().enumerate() {
        if partition < num_partitions {
            partition_row_indices[partition].push(row_idx as i64);
        }
    }

    // Create a batch for each partition
    let mut result = Vec::with_capacity(num_partitions);
    for indices in partition_row_indices {
        if indices.is_empty() {
            // Empty partition
            let schema = table.schema()
                .ok_or_else(|| CylonError::new(Code::Invalid, "Table has no schema".to_string()))?;
            result.push(RecordBatch::new_empty(schema));
        } else {
            // Take rows for this partition using Arrow compute (C++ pattern)
            let indices_array = Int64Array::from(indices);
            let partition_table = arrow_utils::take_rows(table, &indices_array)?;
            let batch = partition_table.batch(0)
                .ok_or_else(|| CylonError::new(Code::Invalid, "Partition table has no batches".to_string()))?
                .clone();
            result.push(batch);
        }
    }

    Ok(result)
}

/// Compare two rows from different tables
/// Returns Ordering::Less if row1 < row2, Ordering::Greater if row1 > row2, Ordering::Equal if equal
fn compare_rows(
    table1: &Table,
    row1_idx: usize,
    table2: &Table,
    row2_idx: usize,
    columns: &[usize],
    directions: &[bool],
) -> CylonResult<Ordering> {
    use arrow::array::*;

    for (i, &col_idx) in columns.iter().enumerate() {
        let batch1 = table1.batch(0)
            .ok_or_else(|| CylonError::new(Code::Invalid, "Table1 has no batches".to_string()))?;
        let batch2 = table2.batch(0)
            .ok_or_else(|| CylonError::new(Code::Invalid, "Table2 has no batches".to_string()))?;

        let array1 = batch1.column(col_idx);
        let array2 = batch2.column(col_idx);

        // Compare the values at the row indices
        let cmp = compare_array_values(array1, row1_idx, array2, row2_idx)?;

        let result = if directions[i] {
            cmp  // Ascending
        } else {
            cmp.reverse()  // Descending
        };

        if result != Ordering::Equal {
            return Ok(result);
        }
    }

    Ok(Ordering::Equal)
}

/// Compare two array values at specific indices
fn compare_array_values(
    array1: &ArrayRef,
    idx1: usize,
    array2: &ArrayRef,
    idx2: usize,
) -> CylonResult<Ordering> {
    use arrow::array::*;
    use arrow::datatypes::*;

    // Handle nulls
    if array1.is_null(idx1) && array2.is_null(idx2) {
        return Ok(Ordering::Equal);
    }
    if array1.is_null(idx1) {
        return Ok(Ordering::Less);
    }
    if array2.is_null(idx2) {
        return Ok(Ordering::Greater);
    }

    // Compare based on data type
    match array1.data_type() {
        DataType::Int32 => {
            let arr1 = array1.as_any().downcast_ref::<Int32Array>().unwrap();
            let arr2 = array2.as_any().downcast_ref::<Int32Array>().unwrap();
            Ok(arr1.value(idx1).cmp(&arr2.value(idx2)))
        }
        DataType::Int64 => {
            let arr1 = array1.as_any().downcast_ref::<Int64Array>().unwrap();
            let arr2 = array2.as_any().downcast_ref::<Int64Array>().unwrap();
            Ok(arr1.value(idx1).cmp(&arr2.value(idx2)))
        }
        DataType::Float32 => {
            let arr1 = array1.as_any().downcast_ref::<Float32Array>().unwrap();
            let arr2 = array2.as_any().downcast_ref::<Float32Array>().unwrap();
            let v1 = arr1.value(idx1);
            let v2 = arr2.value(idx2);
            Ok(v1.partial_cmp(&v2).unwrap_or(Ordering::Equal))
        }
        DataType::Float64 => {
            let arr1 = array1.as_any().downcast_ref::<Float64Array>().unwrap();
            let arr2 = array2.as_any().downcast_ref::<Float64Array>().unwrap();
            let v1 = arr1.value(idx1);
            let v2 = arr2.value(idx2);
            Ok(v1.partial_cmp(&v2).unwrap_or(Ordering::Equal))
        }
        DataType::Utf8 => {
            let arr1 = array1.as_any().downcast_ref::<StringArray>().unwrap();
            let arr2 = array2.as_any().downcast_ref::<StringArray>().unwrap();
            Ok(arr1.value(idx1).cmp(arr2.value(idx2)))
        }
        _ => {
            Err(CylonError::new(
                Code::NotImplemented,
                format!("Comparison not implemented for type: {:?}", array1.data_type())
            ))
        }
    }
}

/// Split-points based partitioning (legacy approach)
/// Uses gather/broadcast of samples to determine partition boundaries
fn distributed_sort_split_points(
    local_sorted: &Table,
    sort_columns: &[usize],
    sort_directions: &[bool],
    num_samples: u64,
    world_size: i32,
) -> CylonResult<Vec<Table>> {
    let ctx = local_sorted.get_context();

    // Sample the sorted table
    let sampling_ratio = if num_samples == 0 { 2 } else { num_samples as usize };
    let sample_count = (world_size as usize * sampling_ratio).min(local_sorted.rows() as usize);

    let sample_result = arrow_utils::sample_table_uniform(local_sorted, sample_count, Some(sort_columns))?;

    // Gather samples to root
    let comm = ctx.get_communicator()
        .ok_or_else(|| CylonError::new(Code::Invalid, "No communicator set".to_string()))?;

    let gathered_samples = comm.gather(&sample_result, 0, true, ctx.clone())?;

    // Root determines split points
    let mut split_points_opt = if ctx.get_rank() == 0 {
        let sample_refs: Vec<&Table> = gathered_samples.iter().collect();
        let merged_samples = crate::table::merge_sorted_table(
            &sample_refs,
            sort_columns,
            sort_directions
        )?;

        let num_split_points = (merged_samples.rows() as usize).min((world_size - 1) as usize);
        let split_points = arrow_utils::sample_table_uniform(&merged_samples, num_split_points, Some(sort_columns))?;
        Some(split_points)
    } else {
        None
    };

    // Broadcast split points
    comm.bcast(&mut split_points_opt, 0, ctx.clone())?;
    let split_points = split_points_opt.unwrap();

    // Partition by split points
    let partitioned_batches = partition_by_split_points(
        local_sorted,
        &split_points,
        sort_columns,
        sort_directions,
        world_size as usize
    )?;

    // Convert batches to Tables
    partitioned_batches
        .into_iter()
        .map(|batch| Table::from_record_batch(ctx.clone(), batch))
        .collect()
}

/// Distributed sort with single column
/// Corresponds to C++ DistributedSort (table.cpp:752-759)
///
/// Performs a globally distributed sort using the sample sort algorithm.
/// After completion, the data is globally sorted and distributed across all processes.
///
/// # Arguments
/// * `table` - Input table
/// * `sort_column` - Column index to sort by
/// * `ascending` - Sort direction (true = ascending, false = descending)
///
/// # Returns
/// Globally sorted table, distributed across all processes
///
/// # Example
/// ```ignore
/// let sorted = distributed_sort(&table, 0, true)?;
/// ```
pub fn distributed_sort(
    table: &Table,
    sort_column: usize,
    ascending: bool,
) -> CylonResult<Table> {
    distributed_sort_multi(table, &[sort_column], &[ascending], SortOptions::default())
}

/// Distributed sort with multiple columns
/// Corresponds to C++ DistributedSort (table.cpp:701-759)
///
/// # Algorithm
/// 1. **Local sort**: Each process sorts its local data
/// 2. **Range partition**: Partition data using histogram-based or split-points approach
/// 3. **All-to-all exchange**: Redistribute partitions so each process gets one partition
/// 4. **Merge**: Each process merges its received sorted partitions
///
/// ## Histogram-based partitioning (default, matches C++ API)
/// - Uses MapToSortPartitions with configurable num_bins and num_samples
/// - AllReduce for global min/max and histogram
/// - Better handling of skewed data distributions
///
/// ## Split-points partitioning (legacy)
/// - Gather samples to root, determine split points, broadcast
/// - Simpler but may be less balanced for skewed data
///
/// # Arguments
/// * `table` - Input table
/// * `sort_columns` - Column indices to sort by
/// * `sort_directions` - Sort direction for each column (true = ascending)
/// * `sort_options` - Sort configuration (num_samples, num_bins, use_histogram)
///
/// # Returns
/// Globally sorted table, distributed across all processes
pub fn distributed_sort_multi(
    table: &Table,
    sort_columns: &[usize],
    sort_directions: &[bool],
    sort_options: SortOptions,
) -> CylonResult<Table> {
    // Validation
    // Corresponds to C++ table.cpp:626-634
    if sort_columns.len() > table.columns() as usize {
        return Err(CylonError::new(
            Code::Invalid,
            "Number of sort columns cannot be larger than number of columns".to_string()
        ));
    }

    if sort_columns.len() != sort_directions.len() {
        return Err(CylonError::new(
            Code::Invalid,
            "Sort columns and directions must have same length".to_string()
        ));
    }

    // Currently only single-column sort supported for histogram-based partitioning
    if sort_options.use_histogram && sort_columns.len() > 1 {
        return Err(CylonError::new(
            Code::NotImplemented,
            "Histogram-based partitioning only supports single-column sort. Use SortOptions::with_split_points() for multi-column.".to_string()
        ));
    }

    let ctx = table.get_context();
    let world_size = ctx.get_world_size();
    let rank = ctx.get_rank();

    // If not distributed or world_size == 1, use local sort
    // Corresponds to C++ table.cpp:637-639
    if !ctx.is_distributed() || world_size == 1 {
        return crate::table::sort_multi(table, sort_columns, sort_directions);
    }

    // Step 1: Local sort
    // Corresponds to C++ table.cpp:641-643
    let local_sorted = crate::table::sort_multi(table, sort_columns, sort_directions)?;

    // Step 2: Range partition using chosen strategy
    let partitioned_tables = if sort_options.use_histogram {
        // Histogram-based partitioning (matches C++ MapToSortPartitions)
        // Corresponds to C++ table.cpp:706-714
        let range_options = RangePartitionOptions::new(
            sort_directions[0],  // ascending
            sort_options.num_samples,
            sort_options.num_bins,
        );

        let mapping = map_to_sort_partitions(
            &local_sorted,
            sort_columns[0],
            world_size as u32,
            &range_options,
        )?;

        split_by_partition(&local_sorted, &mapping, world_size as usize)?
    } else {
        // Split-points partitioning (legacy approach)
        distributed_sort_split_points(
            &local_sorted,
            sort_columns,
            sort_directions,
            sort_options.num_samples,
            world_size,
        )?
    };

    // Convert Tables to RecordBatches for all-to-all
    let partitioned_batches: Vec<RecordBatch> = partitioned_tables
        .iter()
        .map(|t| {
            if t.num_batches() == 0 {
                let schema = t.schema().unwrap_or_else(|| table.schema().unwrap());
                RecordBatch::new_empty(schema)
            } else {
                t.batch(0).unwrap().clone()
            }
        })
        .collect();

    // Step 7: All-to-all exchange using ArrowAllToAll
    // Corresponds to C++ all_to_all_arrow_tables_separated_arrow_table (table.cpp:109-158)
    let schema = table.schema()
        .ok_or_else(|| CylonError::new(Code::Invalid, "Table has no schema".to_string()))?;

    // Include all ranks (including self) in neighbours, matching C++ behavior
    // We don't send data to self, but we DO send/receive finish messages to/from self
    let neighbours: Vec<i32> = (0..world_size).collect();
    let edge_id = ctx.get_next_sequence();

    // Track received tables from each rank
    use std::sync::Mutex;
    let received_tables: Arc<Mutex<Vec<Option<Table>>>> = Arc::new(Mutex::new(vec![None; world_size as usize]));
    let received_tables_clone = received_tables.clone();

    // Callback to receive tables
    let arrow_callback = Box::new(move |source: i32, table: Table, _reference: i32| {
        let mut tables = received_tables_clone.lock().unwrap();
        tables[source as usize] = Some(table);
        true
    });

    // Get communicator and create channel (works with any communicator type)
    use crate::net::buffer::VecBuffer;
    use crate::arrow::arrow_all_to_all::ArrowAllToAll;

    let communicator = ctx.get_communicator()
        .ok_or_else(|| CylonError::new(Code::Invalid, "No communicator".to_string()))?;

    // Create channel using the generic create_channel() method
    // Corresponds to C++ table.cpp:109 - channel creation
    let channel = communicator.create_channel()?;

    // Create allocator
    struct SimpleAllocator;
    impl crate::net::Allocator for SimpleAllocator {
        fn allocate(&self, size: usize) -> CylonResult<Box<dyn crate::net::Buffer>> {
            Ok(Box::new(VecBuffer::new(size)))
        }
    }

    // Create ArrowAllToAll
    let mut arrow_all_to_all = ArrowAllToAll::new(
        rank,
        neighbours.clone(),
        neighbours.clone(),
        edge_id,
        arrow_callback,
        schema.clone(),
        ctx.clone(),
        channel,
        Box::new(SimpleAllocator),
    )?;

    // Insert partitions: send partition i to rank i
    // Corresponds to C++ table.cpp:133-142
    for (i, batch) in partitioned_batches.iter().enumerate() {
        if i as i32 != rank && batch.num_rows() > 0 {
            // Create table from batch and insert
            let partition_table = Table::from_record_batch(ctx.clone(), batch.clone())?;
            arrow_all_to_all.insert(partition_table, i as i32);
        }
    }

    // Keep local partition (don't send to self)
    let local_partition = if (rank as usize) < partitioned_batches.len() {
        Table::from_record_batch(ctx.clone(), partitioned_batches[rank as usize].clone())?
    } else {
        Table::from_record_batch(ctx.clone(), RecordBatch::new_empty(schema.clone()))?
    };

    // Mark as finished and complete communication
    arrow_all_to_all.finish();
    while !arrow_all_to_all.is_complete()? {
        // Progress communication
    }
    arrow_all_to_all.close();

    // Step 8: Merge received sorted partitions
    // Corresponds to C++ MergeSortedTable (table.cpp:685)
    let mut all_partitions = vec![&local_partition];
    let received = received_tables.lock().unwrap();
    let received_refs: Vec<&Table> = received.iter()
        .filter_map(|opt| opt.as_ref())
        .collect();
    all_partitions.extend(received_refs);

    if all_partitions.is_empty() || (all_partitions.len() == 1 && all_partitions[0].rows() == 0) {
        // No data, return empty table
        return Table::from_record_batch(ctx.clone(), RecordBatch::new_empty(schema));
    }

    // Merge all received partitions (they are already sorted)
    crate::table::merge_sorted_table(&all_partitions, sort_columns, sort_directions)
}
