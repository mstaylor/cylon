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

//! MapReduce operations for distributed aggregations
//!
//! Ported from cpp/src/cylon/mapreduce/mapreduce.hpp and mapreduce.cpp

mod kernels;

pub use kernels::{SumKernel, MinKernel, MaxKernel, CountKernel, MeanKernel, VarKernel, StdDevKernel};

use std::sync::Arc;
use arrow::array::{Array, ArrayRef};
use arrow::datatypes::DataType;

use crate::error::{CylonResult, CylonError, Code};
use crate::table::Table;

/// Map each row to a unique group id
/// Corresponds to C++ MapToGroupKernel
pub struct MapToGroupKernel {
    // Memory pool is passed to methods rather than stored
}

impl MapToGroupKernel {
    /// Create a new MapToGroupKernel
    pub fn new() -> Self {
        Self {}
    }

    /// Map each row to a unique group id in the range [0, local_num_groups)
    /// Corresponds to C++ MapToGroupKernel::Map()
    pub fn map(
        &self,
        arrays: &[ArrayRef],
        local_group_ids: &mut Option<ArrayRef>,
        local_group_indices: &mut Option<ArrayRef>,
        local_num_groups: &mut i64,
    ) -> CylonResult<()> {
        use arrow::array::Int64Builder;
        use arrow_row::{RowConverter, SortField};
        use hashbrown::HashMap;

        if arrays.is_empty() {
            return Err(CylonError::new(
                Code::Invalid,
                "arrays cannot be empty".to_string()
            ));
        }

        let num_rows = arrays[0].len();

        // Check all arrays have the same length
        for arr in &arrays[1..] {
            if arr.len() != num_rows {
                return Err(CylonError::new(
                    Code::Invalid,
                    "array lengths should be the same".to_string()
                ));
            }
        }

        // If empty, return empty arrays
        if num_rows == 0 {
            let empty_ids = Int64Builder::new().finish();
            *local_group_ids = Some(Arc::new(empty_ids.clone()));
            *local_group_indices = Some(Arc::new(empty_ids));
            *local_num_groups = 0;
            return Ok(());
        }

        // Create a row converter to hash rows
        let fields: Vec<SortField> = arrays
            .iter()
            .map(|arr| SortField::new(arr.data_type().clone()))
            .collect();

        let mut converter = RowConverter::new(fields)
            .map_err(|e| CylonError::new(Code::ExecutionError, format!("Failed to create row converter: {}", e)))?;

        // Convert arrays to rows for hashing
        let rows = converter.convert_columns(arrays)
            .map_err(|e| CylonError::new(Code::ExecutionError, format!("Failed to convert columns: {}", e)))?;

        // Hash map: row -> group_id
        let mut hash_map: HashMap<Vec<u8>, i64> = HashMap::with_capacity(num_rows);

        let mut group_ids_builder = Int64Builder::with_capacity(num_rows);
        let mut group_indices_builder = Int64Builder::with_capacity(num_rows);

        let mut unique: i64 = 0;

        for i in 0..num_rows {
            let row_bytes = rows.row(i).as_ref().to_vec();

            // Try to insert, if new then assign new group id
            let group_id = *hash_map.entry(row_bytes).or_insert_with(|| {
                group_indices_builder.append_value(i as i64);
                let id = unique;
                unique += 1;
                id
            });

            group_ids_builder.append_value(group_id);
        }

        *local_group_ids = Some(Arc::new(group_ids_builder.finish()));
        *local_group_indices = Some(Arc::new(group_indices_builder.finish()));
        *local_num_groups = unique;

        Ok(())
    }

    /// Map rows from a table to unique group ids
    /// Corresponds to C++ MapToGroupKernel::Map() overload
    pub fn map_table(
        &self,
        table: &Table,
        key_cols: &[usize],
        local_group_ids: &mut Option<ArrayRef>,
        local_group_indices: &mut Option<ArrayRef>,
        local_num_groups: &mut i64,
    ) -> CylonResult<()> {
        // TODO: For now, we require single-batch tables
        // In the future, we should concat batches or handle multi-batch properly
        if table.num_batches() > 1 {
            return Err(CylonError::new(
                Code::Invalid,
                "MapToGroupKernel doesn't support multi-batch tables yet".to_string()
            ));
        }

        if table.num_batches() == 0 {
            let empty_ids = arrow::array::Int64Builder::new().finish();
            *local_group_ids = Some(Arc::new(empty_ids.clone()));
            *local_group_indices = Some(Arc::new(empty_ids));
            *local_num_groups = 0;
            return Ok(());
        }

        // Extract the key column arrays from the first (and only) batch
        let batch = table.batch(0).ok_or_else(|| {
            CylonError::new(Code::Invalid, "Unable to get batch 0".to_string())
        })?;

        let mut arrays = Vec::with_capacity(key_cols.len());
        for &col_idx in key_cols {
            arrays.push(batch.column(col_idx).clone());
        }

        self.map(&arrays, local_group_ids, local_group_indices, local_num_groups)
    }
}

impl Default for MapToGroupKernel {
    fn default() -> Self {
        Self::new()
    }
}

/// MapReduce kernel for distributed aggregations
/// Corresponds to C++ MapReduceKernel
pub trait MapReduceKernel: Send + Sync {
    /// Kernel name (e.g., "sum", "mean", "count")
    fn name(&self) -> &str;

    /// Output data type
    fn output_type(&self) -> &DataType;

    /// Intermediate data types used during reduction
    fn intermediate_types(&self) -> &[DataType];

    /// Number of intermediate arrays
    fn num_arrays(&self) -> usize {
        self.intermediate_types().len()
    }

    /// Combine value_col array locally based on the group_id
    /// Corresponds to C++ MapReduceKernel::CombineLocally()
    fn combine_locally(
        &self,
        value_col: &ArrayRef,
        local_group_ids: &ArrayRef,
        local_num_groups: i64,
        combined_results: &mut Vec<ArrayRef>,
    ) -> CylonResult<()>;

    /// Reduce combined_results vector to its finalized array vector
    /// Corresponds to C++ MapReduceKernel::ReduceShuffledResults()
    fn reduce_shuffled_results(
        &self,
        combined_results: &[ArrayRef],
        local_group_ids: &ArrayRef,
        local_group_indices: &ArrayRef,
        local_num_groups: i64,
        reduced_results: &mut Vec<ArrayRef>,
    ) -> CylonResult<()>;

    /// Create the final output array
    /// Corresponds to C++ MapReduceKernel::Finalize()
    fn finalize(
        &self,
        combined_results: &[ArrayRef],
        output: &mut Option<ArrayRef>,
    ) -> CylonResult<()>;

    /// Whether this kernel uses single-stage reduction
    /// Corresponds to C++ MapReduceKernel::single_stage_reduction()
    fn single_stage_reduction(&self) -> bool {
        false
    }
}

// TODO: Implement specific kernels:
// - SumKernel
// - MinKernel
// - MaxKernel
// - CountKernel
// - MeanKernel
// - VarKernel / StdDevKernel

/// Aggregation operation ID
/// Corresponds to C++ compute::AggregationOpId
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregationOpId {
    Sum,
    Min,
    Max,
    Count,
    Mean,
    Var,
    Stddev,
    Nunique,
    Quantile,
}

/// Create a MapReduceKernel based on data type and operation
/// Corresponds to C++ MakeMapReduceKernel()
pub fn make_mapreduce_kernel(
    data_type: &DataType,
    op: AggregationOpId,
) -> Option<Box<dyn MapReduceKernel>> {
    use arrow::datatypes::*;

    // For Var and Stddev, we need types that can be converted to f64
    // For other operations, we can use a wider range of types
    match op {
        AggregationOpId::Var | AggregationOpId::Stddev => {
            match data_type {
                DataType::UInt8 => make_kernel_for_type::<UInt8Type>(data_type.clone(), op),
                DataType::Int8 => make_kernel_for_type::<Int8Type>(data_type.clone(), op),
                DataType::UInt16 => make_kernel_for_type::<UInt16Type>(data_type.clone(), op),
                DataType::Int16 => make_kernel_for_type::<Int16Type>(data_type.clone(), op),
                DataType::UInt32 => make_kernel_for_type::<UInt32Type>(data_type.clone(), op),
                DataType::Int32 => make_kernel_for_type::<Int32Type>(data_type.clone(), op),
                DataType::Float32 => make_kernel_for_type::<Float32Type>(data_type.clone(), op),
                DataType::Float64 => make_kernel_for_type::<Float64Type>(data_type.clone(), op),
                // i64 and u64 don't implement Into<f64>, so we can't support them for Var/Stddev
                _ => None,
            }
        }
        _ => {
            match data_type {
                DataType::UInt8 => make_kernel_for_type::<UInt8Type>(data_type.clone(), op),
                DataType::Int8 => make_kernel_for_type::<Int8Type>(data_type.clone(), op),
                DataType::UInt16 => make_kernel_for_type::<UInt16Type>(data_type.clone(), op),
                DataType::Int16 => make_kernel_for_type::<Int16Type>(data_type.clone(), op),
                DataType::UInt32 => make_kernel_for_type::<UInt32Type>(data_type.clone(), op),
                DataType::Int32 => make_kernel_for_type::<Int32Type>(data_type.clone(), op),
                // i64 and u64 don't implement Into<f64>, use separate function
                DataType::UInt64 => make_kernel_for_type_no_f64::<UInt64Type>(data_type.clone(), op),
                DataType::Int64 => make_kernel_for_type_no_f64::<Int64Type>(data_type.clone(), op),
                DataType::Float32 => make_kernel_for_type::<Float32Type>(data_type.clone(), op),
                DataType::Float64 => make_kernel_for_type::<Float64Type>(data_type.clone(), op),
                DataType::Date32 => make_kernel_for_type::<Date32Type>(data_type.clone(), op),
                // Date64, Time64, Timestamp also use i64 internally, can't convert to f64
                DataType::Date64 => make_kernel_for_type_no_f64::<Date64Type>(data_type.clone(), op),
                DataType::Time32(_) => make_kernel_for_type::<Time32SecondType>(data_type.clone(), op),
                DataType::Time64(_) => make_kernel_for_type_no_f64::<Time64MicrosecondType>(data_type.clone(), op),
                DataType::Timestamp(_, _) => make_kernel_for_type_no_f64::<TimestampMicrosecondType>(data_type.clone(), op),
                _ => None,
            }
        }
    }
}

/// Helper function to create kernel for a specific Arrow type
/// This version is for types that support Into<f64> (needed for Var/Stddev)
/// Corresponds to C++ MakeMapReduceKernelImpl<T>()
fn make_kernel_for_type<T>(
    data_type: DataType,
    op: AggregationOpId,
) -> Option<Box<dyn MapReduceKernel>>
where
    T: arrow::datatypes::ArrowPrimitiveType + Send + Sync + 'static,
    T::Native: std::ops::Add<Output = T::Native>
        + std::ops::Div<Output = T::Native>
        + PartialOrd
        + Into<f64>
        + num_traits::NumCast
        + Default
        + Copy
        + Send
        + Sync,
{
    match op {
        AggregationOpId::Sum => Some(Box::new(SumKernel::<T>::new(data_type))),
        AggregationOpId::Min => Some(Box::new(MinKernel::<T>::new(data_type))),
        AggregationOpId::Max => Some(Box::new(MaxKernel::<T>::new(data_type))),
        AggregationOpId::Count => Some(Box::new(CountKernel::new())),
        AggregationOpId::Mean => Some(Box::new(MeanKernel::<T>::new(data_type))),
        AggregationOpId::Var => Some(Box::new(VarKernel::<T>::new(0))),
        AggregationOpId::Stddev => Some(Box::new(StdDevKernel::<T>::new(0))),
        AggregationOpId::Nunique => None, // Not supported in C++ yet
        AggregationOpId::Quantile => None, // Not supported in C++ yet
    }
}

/// Helper function for types that don't support Into<f64> (i64, u64, Date64, Time64, Timestamp)
/// These types can't be used with Var/Stddev kernels
fn make_kernel_for_type_no_f64<T>(
    data_type: DataType,
    op: AggregationOpId,
) -> Option<Box<dyn MapReduceKernel>>
where
    T: arrow::datatypes::ArrowPrimitiveType + Send + Sync + 'static,
    T::Native: std::ops::Add<Output = T::Native>
        + std::ops::Div<Output = T::Native>
        + PartialOrd
        + num_traits::NumCast
        + Default
        + Copy
        + Send
        + Sync,
{
    match op {
        AggregationOpId::Sum => Some(Box::new(SumKernel::<T>::new(data_type))),
        AggregationOpId::Min => Some(Box::new(MinKernel::<T>::new(data_type))),
        AggregationOpId::Max => Some(Box::new(MaxKernel::<T>::new(data_type))),
        AggregationOpId::Count => Some(Box::new(CountKernel::new())),
        AggregationOpId::Mean => Some(Box::new(MeanKernel::<T>::new(data_type))),
        // Var and Stddev not supported for these types
        AggregationOpId::Var | AggregationOpId::Stddev => None,
        AggregationOpId::Nunique => None,
        AggregationOpId::Quantile => None,
    }
}

/// Distributed hash groupby using mapreduce approach
/// Corresponds to C++ MapredHashGroupBy()
pub fn mapred_hash_groupby(
    table: &Table,
    key_cols: &[usize],
    aggs: &[(usize, AggregationOpId)],
    output: &mut Option<Table>,
) -> CylonResult<()> {
    use arrow::array::RecordBatch;
    use arrow::datatypes::{Field, Schema};

    let ctx = table.get_context();
    let world_size = ctx.get_world_size();

    // For now, only implement local aggregation (world_size == 1)
    // Distributed aggregation will be implemented after shuffle operations are ported
    if world_size > 1 {
        return Err(CylonError::new(
            Code::NotImplemented,
            format!("Distributed groupby not yet implemented (world_size={})", world_size)
        ));
    }

    // Require single batch for now
    if table.num_batches() != 1 {
        return Err(CylonError::new(
            Code::Invalid,
            format!("mapred_hash_groupby requires single-batch table, got {} batches", table.num_batches())
        ));
    }

    if table.num_batches() == 0 {
        return Err(CylonError::new(
            Code::Invalid,
            "Cannot group empty table".to_string()
        ));
    }

    let batch = table.batch(0).unwrap();

    // Map to groups
    let mapper = MapToGroupKernel::new();
    let mut group_ids = None;
    let mut group_indices = None;
    let mut num_groups = 0i64;

    mapper.map_table(table, key_cols, &mut group_ids, &mut group_indices, &mut num_groups)?;

    let group_ids = group_ids.ok_or_else(|| {
        CylonError::new(Code::ExecutionError, "Failed to generate group IDs".to_string())
    })?;
    let group_indices = group_indices.ok_or_else(|| {
        CylonError::new(Code::ExecutionError, "Failed to generate group indices".to_string())
    })?;

    // Build output schema: key columns + aggregation columns
    let mut output_fields = Vec::new();

    // Add key columns
    let schema = batch.schema();
    for &key_col in key_cols {
        output_fields.push(schema.field(key_col).clone());
    }

    // Add aggregation result columns
    for (val_col, agg_op) in aggs {
        let field = schema.field(*val_col);
        let kernel = make_mapreduce_kernel(field.data_type(), *agg_op)
            .ok_or_else(|| CylonError::new(
                Code::NotImplemented,
                format!("Unsupported aggregation {:?} for type {:?}", agg_op, field.data_type())
            ))?;

        let result_name = format!("{}_{}", field.name(), kernel.name());
        output_fields.push(Field::new(result_name, kernel.output_type().clone(), true));
    }

    let output_schema = Arc::new(Schema::new(output_fields));

    // Build output columns
    let mut output_columns = Vec::new();

    // Take key columns using group_indices
    for &key_col in key_cols {
        let key_array = batch.column(key_col);
        let taken = arrow::compute::take(key_array.as_ref(), &group_indices, None)
            .map_err(|e| CylonError::new(Code::ExecutionError, format!("Failed to take key column: {}", e)))?;
        output_columns.push(taken);
    }

    // Perform aggregations
    for (val_col, agg_op) in aggs {
        let field = schema.field(*val_col);
        let mut kernel = make_mapreduce_kernel(field.data_type(), *agg_op)
            .ok_or_else(|| CylonError::new(
                Code::NotImplemented,
                format!("Unsupported aggregation {:?} for type {:?}", agg_op, field.data_type())
            ))?;

        // Combine locally
        let val_array = batch.column(*val_col);
        let mut combined_results = Vec::new();
        kernel.combine_locally(val_array, &group_ids, num_groups, &mut combined_results)?;

        // Finalize
        let mut result = None;
        kernel.finalize(&combined_results, &mut result)?;

        let result_array = result.ok_or_else(|| {
            CylonError::new(Code::ExecutionError, "Kernel finalize returned None".to_string())
        })?;

        output_columns.push(result_array);
    }

    // Create output batch
    let output_batch = RecordBatch::try_new(output_schema, output_columns)
        .map_err(|e| CylonError::new(Code::ExecutionError, format!("Failed to create output batch: {}", e)))?;

    // Create output table
    *output = Some(Table::from_record_batch(ctx, output_batch)?);

    Ok(())
}
