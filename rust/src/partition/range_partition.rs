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

//! Range-based (sorted) partitioning implementation
//!
//! Ported from cpp/src/cylon/arrow/arrow_partition_kernels.cpp (RangePartitionKernel)
//! and cpp/src/cylon/partition/partition.cpp (MapToSortPartitions)
//!
//! This module provides range partitioning for distributed sort operations.
//! It uses a histogram-based approach to determine partition boundaries.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, PrimitiveArray, UInt64Array};
use arrow::datatypes::{
    DataType, Float32Type, Float64Type,
    Int8Type, Int16Type, Int32Type, Int64Type,
    UInt8Type, UInt16Type, UInt32Type, UInt64Type,
    ArrowPrimitiveType,
};

use crate::ctx::CylonContext;
use crate::error::{Code, CylonError, CylonResult};
use crate::net::comm_operations::ReduceOp;
use crate::scalar::Scalar;
use crate::table::{Table, Column};

/// Result of range partition mapping
/// Contains target partition for each row and histogram of partition sizes
#[derive(Debug)]
pub struct PartitionMapping {
    /// Target partition index for each row (0..num_partitions)
    pub target_partitions: Vec<u32>,
    /// Number of rows in each partition
    pub partition_histogram: Vec<u32>,
}

/// Options for range partitioning
#[derive(Debug, Clone)]
pub struct RangePartitionOptions {
    /// Number of samples to use for histogram building
    /// Default: num_partitions * 16 if 0
    pub num_samples: u64,
    /// Number of histogram bins
    /// Default: num_partitions * 16 if 0
    pub num_bins: u32,
    /// Sort direction (true = ascending, false = descending)
    pub ascending: bool,
}

impl Default for RangePartitionOptions {
    fn default() -> Self {
        Self {
            num_samples: 0,
            num_bins: 0,
            ascending: true,
        }
    }
}

impl RangePartitionOptions {
    pub fn new(ascending: bool, num_samples: u64, num_bins: u32) -> Self {
        Self {
            num_samples,
            num_bins,
            ascending,
        }
    }
}

/// Map rows to sort-based partitions
/// Corresponds to C++ MapToSortPartitions (partition.cpp:172-205)
///
/// Uses histogram-based binning to assign rows to partitions for distributed sort.
/// This ensures balanced partitions even with skewed data distributions.
///
/// # Algorithm
/// 1. Sample the column data
/// 2. Calculate local min/max of samples
/// 3. AllReduce to get global min/max
/// 4. Build local histogram of samples across bins
/// 5. AllReduce histogram to get global distribution
/// 6. Divide histogram into quantiles to map bins to partitions
/// 7. Assign each row to a partition based on its bin
///
/// # Arguments
/// * `table` - Input table
/// * `column_idx` - Column index to partition by
/// * `num_partitions` - Number of partitions (typically world_size)
/// * `options` - Partition options (num_samples, num_bins, ascending)
///
/// # Returns
/// PartitionMapping with target partition for each row and histogram
pub fn map_to_sort_partitions(
    table: &Table,
    column_idx: usize,
    num_partitions: u32,
    options: &RangePartitionOptions,
) -> CylonResult<PartitionMapping> {
    let ctx = table.get_context();

    // Get the column to partition by
    let column = table.column(column_idx)?;
    let num_rows = column.len();

    // Set defaults for num_bins and num_samples
    let num_bins = if options.num_bins == 0 {
        num_partitions * 16
    } else {
        options.num_bins
    };

    let num_samples = if options.num_samples == 0 {
        num_bins as u64
    } else {
        options.num_samples
    };

    // Dispatch based on data type
    match column.data_type() {
        DataType::Int8 => range_partition_impl::<Int8Type>(
            ctx, &column, num_rows, num_partitions, num_bins, num_samples, options.ascending
        ),
        DataType::Int16 => range_partition_impl::<Int16Type>(
            ctx, &column, num_rows, num_partitions, num_bins, num_samples, options.ascending
        ),
        DataType::Int32 => range_partition_impl::<Int32Type>(
            ctx, &column, num_rows, num_partitions, num_bins, num_samples, options.ascending
        ),
        DataType::Int64 => range_partition_impl::<Int64Type>(
            ctx, &column, num_rows, num_partitions, num_bins, num_samples, options.ascending
        ),
        DataType::UInt8 => range_partition_impl::<UInt8Type>(
            ctx, &column, num_rows, num_partitions, num_bins, num_samples, options.ascending
        ),
        DataType::UInt16 => range_partition_impl::<UInt16Type>(
            ctx, &column, num_rows, num_partitions, num_bins, num_samples, options.ascending
        ),
        DataType::UInt32 => range_partition_impl::<UInt32Type>(
            ctx, &column, num_rows, num_partitions, num_bins, num_samples, options.ascending
        ),
        DataType::UInt64 => range_partition_impl::<UInt64Type>(
            ctx, &column, num_rows, num_partitions, num_bins, num_samples, options.ascending
        ),
        DataType::Float32 => range_partition_impl::<Float32Type>(
            ctx, &column, num_rows, num_partitions, num_bins, num_samples, options.ascending
        ),
        DataType::Float64 => range_partition_impl::<Float64Type>(
            ctx, &column, num_rows, num_partitions, num_bins, num_samples, options.ascending
        ),
        dt => Err(CylonError::new(
            Code::NotImplemented,
            format!("Range partition not supported for data type: {:?}", dt),
        )),
    }
}

/// Convert a primitive value to f64
trait ToF64 {
    fn to_f64(self) -> f64;
}

macro_rules! impl_to_f64 {
    ($($t:ty),*) => {
        $(
            impl ToF64 for $t {
                #[inline]
                fn to_f64(self) -> f64 {
                    self as f64
                }
            }
        )*
    };
}

impl_to_f64!(i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);

/// Generic implementation of range partition for primitive types
/// Corresponds to C++ RangePartitionKernel (arrow_partition_kernels.cpp:442-607)
fn range_partition_impl<T>(
    ctx: Arc<CylonContext>,
    column: &ArrayRef,
    num_rows: usize,
    num_partitions: u32,
    num_bins: u32,
    num_samples: u64,
    ascending: bool,
) -> CylonResult<PartitionMapping>
where
    T: ArrowPrimitiveType,
    T::Native: PartialOrd + Copy + ToF64 + std::fmt::Debug,
{
    let arr = column.as_any()
        .downcast_ref::<PrimitiveArray<T>>()
        .ok_or_else(|| CylonError::new(Code::TypeError, "Failed to downcast array".to_string()))?;

    // Check for nulls - range partition doesn't support them
    if arr.null_count() > 0 {
        return Err(CylonError::new(
            Code::Invalid,
            "Range partition kernel doesn't support null values".to_string(),
        ));
    }

    if num_rows == 0 {
        return Ok(PartitionMapping {
            target_partitions: Vec::new(),
            partition_histogram: vec![0; num_partitions as usize],
        });
    }

    // Step 1: Sample the data
    let sampled_values = sample_array::<T>(arr, num_samples as usize)?;

    // Step 2-3: Calculate local min/max and AllReduce to get global min/max
    let (global_min, global_max) = get_global_min_max::<T>(&ctx, &sampled_values)?;

    if global_min >= global_max {
        // All values are the same - put everything in first partition
        return Ok(PartitionMapping {
            target_partitions: vec![0; num_rows],
            partition_histogram: {
                let mut hist = vec![0u32; num_partitions as usize];
                hist[0] = num_rows as u32;
                hist
            },
        });
    }

    let range = global_max - global_min;

    // Step 4: Build local histogram of samples
    let local_histogram = build_histogram::<T>(&sampled_values, global_min, range, num_bins)?;

    // Step 5: AllReduce histogram to get global distribution
    let global_histogram = allreduce_histogram(&ctx, &local_histogram)?;

    // Step 6: Divide histogram into quantiles to map bins to partitions
    let bin_to_partition = compute_bin_to_partition(
        &global_histogram,
        num_partitions,
        ctx.get_world_size() as u64 * num_samples,
    );

    // Step 7: Assign each row to a partition based on its bin
    let mut target_partitions = Vec::with_capacity(num_rows);
    let mut partition_histogram = vec![0u32; num_partitions as usize];

    for i in 0..num_rows {
        let val: f64 = arr.value(i).to_f64();
        let bin = get_bin_position(val, global_min, global_max, range, num_bins);

        let partition = if ascending {
            bin_to_partition[bin]
        } else {
            num_partitions - 1 - bin_to_partition[bin]
        };

        target_partitions.push(partition);
        partition_histogram[partition as usize] += 1;
    }

    Ok(PartitionMapping {
        target_partitions,
        partition_histogram,
    })
}

/// Sample values from an array
fn sample_array<T>(arr: &PrimitiveArray<T>, num_samples: usize) -> CylonResult<Vec<f64>>
where
    T: ArrowPrimitiveType,
    T::Native: ToF64,
{
    let len = arr.len();
    if len == 0 {
        return Ok(Vec::new());
    }

    let actual_samples = num_samples.min(len);
    let mut samples = Vec::with_capacity(actual_samples);

    if actual_samples == len {
        // Use all values
        for i in 0..len {
            samples.push(arr.value(i).to_f64());
        }
    } else {
        // Uniform sampling
        let step = len as f64 / actual_samples as f64;
        for i in 0..actual_samples {
            let idx = (i as f64 * step) as usize;
            samples.push(arr.value(idx).to_f64());
        }
    }

    Ok(samples)
}

/// Get global min/max via AllReduce
fn get_global_min_max<T>(
    ctx: &Arc<CylonContext>,
    samples: &[f64],
) -> CylonResult<(f64, f64)>
where
    T: ArrowPrimitiveType,
{
    // Calculate local min/max
    let (local_min, local_max) = if samples.is_empty() {
        // No samples - use extreme values so AllReduce will find correct global values
        (f64::MAX, f64::MIN)
    } else {
        let min = samples.iter().cloned().fold(f64::MAX, f64::min);
        let max = samples.iter().cloned().fold(f64::MIN, f64::max);
        (min, max)
    };

    // AllReduce to get global min/max
    if ctx.is_distributed() {
        if let Some(comm) = ctx.get_communicator() {
            let min_scalar = Scalar::float64(local_min);
            let max_scalar = Scalar::float64(local_max);

            let global_min_scalar = comm.all_reduce_scalar(&min_scalar, ReduceOp::Min)?;
            let global_max_scalar = comm.all_reduce_scalar(&max_scalar, ReduceOp::Max)?;

            // Extract values from reduced scalars
            let global_min = extract_f64_from_scalar(&global_min_scalar)?;
            let global_max = extract_f64_from_scalar(&global_max_scalar)?;

            return Ok((global_min, global_max));
        }
    }

    Ok((local_min, local_max))
}

/// Extract f64 value from Scalar
fn extract_f64_from_scalar(scalar: &Scalar) -> CylonResult<f64> {
    use arrow::array::Float64Array;

    let data = scalar.data();
    let arr = data.as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| CylonError::new(Code::TypeError, "Expected Float64Array".to_string()))?;

    Ok(arr.value(0))
}

/// Build histogram of sample values across bins
fn build_histogram<T>(
    samples: &[f64],
    min: f64,
    range: f64,
    num_bins: u32,
) -> CylonResult<Vec<u64>>
where
    T: ArrowPrimitiveType,
{
    // Histogram has num_bins + 2 entries:
    // [0] = values < min
    // [1..num_bins] = values in range
    // [num_bins+1] = values >= max
    let mut histogram = vec![0u64; (num_bins + 2) as usize];
    let max = min + range;

    for &val in samples {
        let bin = get_bin_position(val, min, max, range, num_bins);
        histogram[bin] += 1;
    }

    Ok(histogram)
}

/// Get bin position for a value
/// Returns index in [0, num_bins+1] range
#[inline]
fn get_bin_position(val: f64, min: f64, max: f64, range: f64, num_bins: u32) -> usize {
    // Corresponds to C++ get_bin_pos (arrow_partition_kernels.cpp:595-598)
    // bin 0: val < min
    // bin 1..num_bins: val in [min, max)
    // bin num_bins+1: val >= max

    if val < min {
        0
    } else if val >= max {
        (num_bins + 1) as usize
    } else {
        1 + ((val - min) * num_bins as f64 / range) as usize
    }
}

/// AllReduce histogram to get global distribution
fn allreduce_histogram(
    ctx: &Arc<CylonContext>,
    local_histogram: &[u64],
) -> CylonResult<Vec<u64>> {
    if !ctx.is_distributed() || ctx.get_world_size() == 1 {
        return Ok(local_histogram.to_vec());
    }

    if let Some(comm) = ctx.get_communicator() {
        // Create a Column from the histogram for AllReduce
        let histogram_array = UInt64Array::from(local_histogram.to_vec());
        let histogram_col = Column::new(Arc::new(histogram_array));

        let reduced_col = comm.all_reduce_column(&histogram_col, ReduceOp::Sum)?;

        // Extract values from reduced column
        let reduced_array = reduced_col.data();
        let uint64_arr = reduced_array.as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| CylonError::new(Code::TypeError, "Expected UInt64Array".to_string()))?;

        let result: Vec<u64> = (0..uint64_arr.len())
            .map(|i| uint64_arr.value(i))
            .collect();

        return Ok(result);
    }

    Ok(local_histogram.to_vec())
}

/// Compute bin to partition mapping based on quantiles
/// Corresponds to C++ RangePartitionKernel::build_bin_to_partition (arrow_partition_kernels.cpp:569-591)
fn compute_bin_to_partition(
    global_histogram: &[u64],
    num_partitions: u32,
    total_samples: u64,
) -> Vec<u32> {
    let quantile = 1.0 / num_partitions as f64;
    let mut prefix_sum = 0.0;
    let mut curr_partition = 0u32;
    let mut target_quantile = quantile;

    let mut bin_to_partition = Vec::with_capacity(global_histogram.len());

    for &count in global_histogram {
        bin_to_partition.push(curr_partition);

        let freq = count as f64 / total_samples as f64;
        prefix_sum += freq;

        if prefix_sum > target_quantile {
            if curr_partition < num_partitions - 1 {
                curr_partition += 1;
            }
            target_quantile += quantile;
        }
    }

    bin_to_partition
}

/// Split a table based on partition mapping
/// Corresponds to C++ Split (partition.cpp:67-79)
///
/// # Arguments
/// * `table` - Input table to split
/// * `mapping` - Partition mapping from map_to_sort_partitions
/// * `num_partitions` - Number of partitions
///
/// # Returns
/// Vector of Tables, one per partition
pub fn split_by_partition(
    table: &Table,
    mapping: &PartitionMapping,
    num_partitions: usize,
) -> CylonResult<Vec<Table>> {
    use arrow::compute::{take, concat_batches};
    use arrow::array::UInt64Array;
    use arrow::record_batch::RecordBatch;

    let ctx = table.get_context();
    let schema = table.schema()
        .ok_or_else(|| CylonError::new(Code::Invalid, "Table has no schema".to_string()))?;

    // Group row indices by partition
    let mut partition_indices: Vec<Vec<u64>> = vec![Vec::new(); num_partitions];
    for part_idx in 0..num_partitions {
        partition_indices[part_idx].reserve(mapping.partition_histogram[part_idx] as usize);
    }

    for (row_idx, &partition) in mapping.target_partitions.iter().enumerate() {
        partition_indices[partition as usize].push(row_idx as u64);
    }

    // Get combined batch
    let combined = if table.num_batches() == 0 {
        RecordBatch::new_empty(schema.clone())
    } else if table.num_batches() == 1 {
        table.batch(0).unwrap().clone()
    } else {
        concat_batches(&schema, table.batches())
            .map_err(|e| CylonError::new(Code::ExecutionError,
                format!("Failed to concat batches: {}", e)))?
    };

    // Create table for each partition
    let mut result = Vec::with_capacity(num_partitions);
    for indices in partition_indices {
        if indices.is_empty() {
            let empty_batch = RecordBatch::new_empty(schema.clone());
            result.push(Table::from_record_batch(ctx.clone(), empty_batch)?);
        } else {
            let indices_array = UInt64Array::from(indices);
            let mut partition_columns = Vec::new();

            for col_idx in 0..combined.num_columns() {
                let column = combined.column(col_idx);
                let taken = take(column.as_ref(), &indices_array, None)
                    .map_err(|e| CylonError::new(Code::ExecutionError,
                        format!("Failed to take rows: {}", e)))?;
                partition_columns.push(taken);
            }

            let partition_batch = RecordBatch::try_new(schema.clone(), partition_columns)
                .map_err(|e| CylonError::new(Code::ExecutionError,
                    format!("Failed to create batch: {}", e)))?;

            result.push(Table::from_record_batch(ctx.clone(), partition_batch)?);
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Int64Array;
    use arrow::datatypes::{Field, Schema};
    use arrow::record_batch::RecordBatch;

    fn create_test_table(ctx: Arc<CylonContext>, values: Vec<i64>) -> Table {
        let schema = Arc::new(Schema::new(vec![
            Field::new("col0", DataType::Int64, false),
        ]));

        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int64Array::from(values))],
        ).unwrap();

        Table::from_record_batch(ctx, batch).unwrap()
    }

    #[test]
    fn test_map_to_sort_partitions_basic() {
        let ctx = Arc::new(CylonContext::new(false));
        let table = create_test_table(ctx.clone(), vec![1, 5, 3, 7, 2, 8, 4, 6]);

        let options = RangePartitionOptions::new(true, 8, 4);
        let mapping = map_to_sort_partitions(&table, 0, 4, &options).unwrap();

        assert_eq!(mapping.target_partitions.len(), 8);
        assert_eq!(mapping.partition_histogram.len(), 4);

        // Total should equal number of rows
        let total: u32 = mapping.partition_histogram.iter().sum();
        assert_eq!(total, 8);
    }

    #[test]
    fn test_map_to_sort_partitions_ascending() {
        let ctx = Arc::new(CylonContext::new(false));
        // Values 0-99
        let values: Vec<i64> = (0..100).collect();
        let table = create_test_table(ctx.clone(), values);

        let options = RangePartitionOptions::new(true, 100, 10);
        let mapping = map_to_sort_partitions(&table, 0, 4, &options).unwrap();

        // Lower values should go to lower partitions
        // Value 0 should be in partition 0
        assert_eq!(mapping.target_partitions[0], 0);
        // Value 99 should be in partition 3
        assert_eq!(mapping.target_partitions[99], 3);
    }

    #[test]
    fn test_map_to_sort_partitions_descending() {
        let ctx = Arc::new(CylonContext::new(false));
        let values: Vec<i64> = (0..100).collect();
        let table = create_test_table(ctx.clone(), values);

        let options = RangePartitionOptions::new(false, 100, 10);
        let mapping = map_to_sort_partitions(&table, 0, 4, &options).unwrap();

        // Lower values should go to higher partitions (descending)
        // Value 0 should be in partition 3
        assert_eq!(mapping.target_partitions[0], 3);
        // Value 99 should be in partition 0
        assert_eq!(mapping.target_partitions[99], 0);
    }

    #[test]
    fn test_split_by_partition() {
        let ctx = Arc::new(CylonContext::new(false));
        let table = create_test_table(ctx.clone(), vec![1, 5, 3, 7, 2, 8, 4, 6]);

        let options = RangePartitionOptions::new(true, 8, 4);
        let mapping = map_to_sort_partitions(&table, 0, 2, &options).unwrap();
        let partitions = split_by_partition(&table, &mapping, 2).unwrap();

        assert_eq!(partitions.len(), 2);

        // Total rows should equal original
        let total_rows: i64 = partitions.iter().map(|t| t.rows()).sum();
        assert_eq!(total_rows, 8);
    }

    #[test]
    fn test_empty_table() {
        let ctx = Arc::new(CylonContext::new(false));
        let table = create_test_table(ctx.clone(), vec![]);

        let options = RangePartitionOptions::default();
        let mapping = map_to_sort_partitions(&table, 0, 4, &options).unwrap();

        assert_eq!(mapping.target_partitions.len(), 0);
        assert_eq!(mapping.partition_histogram.iter().sum::<u32>(), 0);
    }

    #[test]
    fn test_single_value() {
        let ctx = Arc::new(CylonContext::new(false));
        let table = create_test_table(ctx.clone(), vec![42; 10]);

        let options = RangePartitionOptions::default();
        let mapping = map_to_sort_partitions(&table, 0, 4, &options).unwrap();

        // All values are the same, should go to first partition
        assert!(mapping.target_partitions.iter().all(|&p| p == 0));
        assert_eq!(mapping.partition_histogram[0], 10);
    }

    #[test]
    fn test_get_bin_position() {
        // Test bin positioning
        let min = 0.0;
        let max = 100.0;
        let range = 100.0;
        let num_bins = 10;

        // Value below min -> bin 0
        assert_eq!(get_bin_position(-5.0, min, max, range, num_bins), 0);

        // Value at min -> bin 1
        assert_eq!(get_bin_position(0.0, min, max, range, num_bins), 1);

        // Value in middle -> appropriate bin
        assert_eq!(get_bin_position(50.0, min, max, range, num_bins), 6);

        // Value at max -> bin num_bins+1
        assert_eq!(get_bin_position(100.0, min, max, range, num_bins), 11);

        // Value above max -> bin num_bins+1
        assert_eq!(get_bin_position(150.0, min, max, range, num_bins), 11);
    }
}
