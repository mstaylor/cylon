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

//! Table - main data structure for Cylon
//!
//! Ported from cpp/src/cylon/table.hpp

use std::sync::Arc;
use arrow::array::RecordBatch;
use arrow::datatypes::Schema;

use crate::ctx::CylonContext;
use crate::error::CylonResult;

pub mod column;
pub use column::Column;

/// Table provides the main API for using cylon for data processing
/// Corresponds to C++ Table class from cpp/src/cylon/table.hpp
pub struct Table {
    ctx: Arc<CylonContext>,
    // Using Arrow RecordBatch internally (similar to C++ using arrow::Table)
    batches: Vec<RecordBatch>,
    retain: bool,
}

impl Table {
    /// Create a table from Arrow RecordBatch
    pub fn from_record_batch(
        ctx: Arc<CylonContext>,
        batch: RecordBatch,
    ) -> CylonResult<Self> {
        Ok(Self {
            ctx,
            batches: vec![batch],
            retain: true,
        })
    }

    /// Create a table from multiple Arrow RecordBatches
    pub fn from_record_batches(
        ctx: Arc<CylonContext>,
        batches: Vec<RecordBatch>,
    ) -> CylonResult<Self> {
        Ok(Self {
            ctx,
            batches,
            retain: true,
        })
    }

    /// Get the number of columns
    pub fn columns(&self) -> i32 {
        if let Some(batch) = self.batches.first() {
            batch.num_columns() as i32
        } else {
            0
        }
    }

    /// Get the number of rows
    pub fn rows(&self) -> i64 {
        self.batches.iter().map(|b| b.num_rows() as i64).sum()
    }

    /// Check if table is empty
    pub fn is_empty(&self) -> bool {
        self.rows() == 0
    }

    /// Get the context
    pub fn get_context(&self) -> Arc<CylonContext> {
        self.ctx.clone()
    }

    /// Get the schema
    pub fn schema(&self) -> Option<Arc<Schema>> {
        self.batches.first().map(|b| b.schema())
    }

    /// Get column names
    pub fn column_names(&self) -> Vec<String> {
        if let Some(schema) = self.schema() {
            schema.fields().iter().map(|f| f.name().clone()).collect()
        } else {
            Vec::new()
        }
    }

    /// Set retention flag
    pub fn retain_memory(&mut self, retain: bool) {
        self.retain = retain;
    }

    /// Check if table retains memory
    pub fn is_retain(&self) -> bool {
        self.retain
    }

    /// Get the number of batches in the table
    pub fn num_batches(&self) -> usize {
        self.batches.len()
    }

    /// Get a reference to a specific batch
    pub fn batch(&self, index: usize) -> Option<&RecordBatch> {
        self.batches.get(index)
    }

    /// Get all batches
    pub fn batches(&self) -> &[RecordBatch] {
        &self.batches
    }

    /// Read a table from a CSV file
    /// Corresponds to C++ FromCSV
    pub fn from_csv(
        ctx: Arc<CylonContext>,
        path: &str,
        options: &crate::io::CsvReadOptions,
    ) -> CylonResult<Self> {
        crate::io::read_csv(ctx, path, options)
    }

    /// Read a table from a CSV file with default options
    pub fn from_csv_default(
        ctx: Arc<CylonContext>,
        path: &str,
    ) -> CylonResult<Self> {
        crate::io::read_csv(ctx, path, &crate::io::CsvReadOptions::default())
    }

    /// Write the table to a CSV file
    /// Corresponds to C++ WriteCSV
    pub fn to_csv(
        &self,
        path: &str,
        options: &crate::io::CsvWriteOptions,
    ) -> CylonResult<()> {
        crate::io::write_csv(self, path, options)
    }

    /// Write the table to a CSV file with default options
    pub fn to_csv_default(&self, path: &str) -> CylonResult<()> {
        crate::io::write_csv(self, path, &crate::io::CsvWriteOptions::default())
    }

    /// Project (select) specific columns from the table
    /// Corresponds to C++ Project function (table.cpp:1212)
    ///
    /// # Arguments
    /// * `column_indices` - Indices of columns to include in the projection
    ///
    /// # Returns
    /// A new table with only the specified columns
    pub fn project(&self, column_indices: &[usize]) -> CylonResult<Table> {
        if column_indices.is_empty() {
            return Err(crate::error::CylonError::new(
                crate::error::Code::Invalid,
                "column_indices cannot be empty".to_string(),
            ));
        }

        let mut projected_batches = Vec::new();

        for batch in &self.batches {
            // Validate indices
            for &idx in column_indices {
                if idx >= batch.num_columns() {
                    return Err(crate::error::CylonError::new(
                        crate::error::Code::Invalid,
                        format!("Column index {} out of range (table has {} columns)",
                                idx, batch.num_columns()),
                    ));
                }
            }

            // Build new schema with selected columns
            let fields: Vec<_> = column_indices.iter()
                .map(|&idx| batch.schema().field(idx).clone())
                .collect();
            let new_schema = Arc::new(Schema::new(fields));

            // Build column arrays
            let columns: Vec<_> = column_indices.iter()
                .map(|&idx| batch.column(idx).clone())
                .collect();

            // Create new batch
            let projected_batch = RecordBatch::try_new(new_schema, columns)
                .map_err(|e| crate::error::CylonError::new(
                    crate::error::Code::ExecutionError,
                    format!("Failed to create projected batch: {}", e),
                ))?;

            projected_batches.push(projected_batch);
        }

        Table::from_record_batches(self.ctx.clone(), projected_batches)
    }

    /// Project by column names instead of indices
    ///
    /// # Arguments
    /// * `column_names` - Names of columns to include in the projection
    ///
    /// # Returns
    /// A new table with only the specified columns
    pub fn project_by_names(&self, column_names: &[&str]) -> CylonResult<Table> {
        if column_names.is_empty() {
            return Err(crate::error::CylonError::new(
                crate::error::Code::Invalid,
                "column_names cannot be empty".to_string(),
            ));
        }

        // Get schema from first batch to resolve column names
        let schema = self.schema().ok_or_else(|| {
            crate::error::CylonError::new(crate::error::Code::Invalid, "Table has no schema".to_string())
        })?;

        // Resolve column names to indices
        let mut indices = Vec::new();
        for &col_name in column_names {
            match schema.index_of(col_name) {
                Ok(idx) => indices.push(idx),
                Err(_) => {
                    return Err(crate::error::CylonError::new(
                        crate::error::Code::Invalid,
                        format!("Column '{}' not found in table", col_name),
                    ));
                }
            }
        }

        self.project(&indices)
    }

    /// Slice the table to get a subset of rows
    /// Corresponds to C++ Slice function
    ///
    /// # Arguments
    /// * `offset` - Starting row index (0-based)
    /// * `length` - Number of rows to include
    ///
    /// # Returns
    /// A new table with the specified row range
    pub fn slice(&self, offset: usize, length: usize) -> CylonResult<Table> {
        use arrow::compute::concat_batches;

        let total_rows = self.rows() as usize;

        if offset >= total_rows {
            return Err(crate::error::CylonError::new(
                crate::error::Code::Invalid,
                format!("Offset {} is out of range (table has {} rows)", offset, total_rows),
            ));
        }

        // Combine all batches first if we have multiple
        let combined = if self.batches.len() > 1 {
            let schema = self.schema().ok_or_else(|| {
                crate::error::CylonError::new(crate::error::Code::Invalid, "Table has no schema".to_string())
            })?;

            concat_batches(&schema, &self.batches)
                .map_err(|e| crate::error::CylonError::new(
                    crate::error::Code::ExecutionError,
                    format!("Failed to combine batches: {}", e),
                ))?
        } else if self.batches.len() == 1 {
            self.batches[0].clone()
        } else {
            return Err(crate::error::CylonError::new(
                crate::error::Code::Invalid,
                "Table has no batches".to_string(),
            ));
        };

        // Adjust length if it exceeds table bounds
        let actual_length = length.min(total_rows - offset);

        // Slice the combined batch
        let sliced = combined.slice(offset, actual_length);

        Table::from_record_batch(self.ctx.clone(), sliced)
    }

    /// Get the first n rows of the table
    /// Corresponds to C++ Head function
    ///
    /// # Arguments
    /// * `n` - Number of rows to return
    ///
    /// # Returns
    /// A new table with the first n rows
    pub fn head(&self, n: usize) -> CylonResult<Table> {
        self.slice(0, n)
    }

    /// Get the last n rows of the table
    /// Corresponds to C++ Tail function
    ///
    /// # Arguments
    /// * `n` - Number of rows to return
    ///
    /// # Returns
    /// A new table with the last n rows
    pub fn tail(&self, n: usize) -> CylonResult<Table> {
        let total_rows = self.rows() as usize;
        if n >= total_rows {
            // Return the whole table if n >= total rows
            self.slice(0, total_rows)
        } else {
            self.slice(total_rows - n, n)
        }
    }

    /// Merge multiple tables vertically (concatenate rows)
    /// Corresponds to C++ Merge function (table.cpp:343)
    ///
    /// All tables must have the same schema
    ///
    /// # Arguments
    /// * `tables` - Vector of tables to merge with this table
    ///
    /// # Returns
    /// A new table containing all rows from all tables
    pub fn merge(&self, tables: &[&Table]) -> CylonResult<Table> {
        // Collect all batches from all tables
        let mut all_batches = self.batches.clone();

        let schema = self.schema().ok_or_else(|| {
            crate::error::CylonError::new(crate::error::Code::Invalid, "Table has no schema".to_string())
        })?;

        // Validate all tables have same schema and collect their batches
        for table in tables {
            let other_schema = table.schema().ok_or_else(|| {
                crate::error::CylonError::new(crate::error::Code::Invalid, "Table to merge has no schema".to_string())
            })?;

            if !schema.eq(&other_schema) {
                return Err(crate::error::CylonError::new(
                    crate::error::Code::Invalid,
                    "Cannot merge tables with different schemas".to_string(),
                ));
            }

            all_batches.extend(table.batches.clone());
        }

        Table::from_record_batches(self.ctx.clone(), all_batches)
    }

    /// Sort table by a single column
    /// Corresponds to C++ Sort function (table.cpp:372)
    ///
    /// # Arguments
    /// * `sort_column` - Index of column to sort by
    /// * `ascending` - true for ascending, false for descending
    ///
    /// # Returns
    /// A new sorted table
    pub fn sort(&self, sort_column: usize, ascending: bool) -> CylonResult<Table> {
        use arrow::compute::{lexsort_to_indices, take, SortColumn};

        // If table has 0 or 1 rows, no need to sort
        if self.rows() < 2 {
            return Table::from_record_batches(self.ctx.clone(), self.batches.clone());
        }

        // Combine all batches first
        let combined = if self.batches.len() > 1 {
            let schema = self.schema().ok_or_else(|| {
                crate::error::CylonError::new(crate::error::Code::Invalid, "Table has no schema".to_string())
            })?;

            arrow::compute::concat_batches(&schema, &self.batches)
                .map_err(|e| crate::error::CylonError::new(
                    crate::error::Code::ExecutionError,
                    format!("Failed to combine batches: {}", e),
                ))?
        } else if self.batches.len() == 1 {
            self.batches[0].clone()
        } else {
            return Err(crate::error::CylonError::new(
                crate::error::Code::Invalid,
                "Table has no batches".to_string(),
            ));
        };

        // Validate column index
        if sort_column >= combined.num_columns() {
            return Err(crate::error::CylonError::new(
                crate::error::Code::Invalid,
                format!("Sort column index {} out of range (table has {} columns)",
                        sort_column, combined.num_columns()),
            ));
        }

        // Create sort column specification
        let sort_cols = vec![SortColumn {
            values: combined.column(sort_column).clone(),
            options: Some(arrow::compute::SortOptions {
                descending: !ascending,
                nulls_first: false,
            }),
        }];

        // Get sorted indices
        let indices = lexsort_to_indices(&sort_cols, None)
            .map_err(|e| crate::error::CylonError::new(
                crate::error::Code::ExecutionError,
                format!("Failed to compute sort indices: {}", e),
            ))?;

        // Apply indices to all columns
        let mut sorted_columns = Vec::new();
        for i in 0..combined.num_columns() {
            let sorted_col = take(combined.column(i), &indices, None)
                .map_err(|e| crate::error::CylonError::new(
                    crate::error::Code::ExecutionError,
                    format!("Failed to reorder column {}: {}", i, e),
                ))?;
            sorted_columns.push(sorted_col);
        }

        // Create sorted batch
        let sorted_batch = RecordBatch::try_new(combined.schema(), sorted_columns)
            .map_err(|e| crate::error::CylonError::new(
                crate::error::Code::ExecutionError,
                format!("Failed to create sorted batch: {}", e),
            ))?;

        Table::from_record_batch(self.ctx.clone(), sorted_batch)
    }

    /// Sort table by multiple columns
    /// Corresponds to C++ Sort function (table.cpp:395)
    ///
    /// # Arguments
    /// * `sort_columns` - Indices of columns to sort by (in order of priority)
    /// * `sort_directions` - Sort direction for each column (true = ascending, false = descending)
    ///
    /// # Returns
    /// A new sorted table
    pub fn sort_multi(&self, sort_columns: &[usize], sort_directions: &[bool]) -> CylonResult<Table> {
        use arrow::compute::{lexsort_to_indices, take, SortColumn};

        if sort_columns.is_empty() {
            return Err(crate::error::CylonError::new(
                crate::error::Code::Invalid,
                "sort_columns cannot be empty".to_string(),
            ));
        }

        if sort_columns.len() != sort_directions.len() {
            return Err(crate::error::CylonError::new(
                crate::error::Code::Invalid,
                "sort_columns and sort_directions must have the same length".to_string(),
            ));
        }

        // Single column sort - delegate to simpler method
        if sort_columns.len() == 1 {
            return self.sort(sort_columns[0], sort_directions[0]);
        }

        // If table has 0 or 1 rows, no need to sort
        if self.rows() < 2 {
            return Table::from_record_batches(self.ctx.clone(), self.batches.clone());
        }

        // Combine all batches first
        let combined = if self.batches.len() > 1 {
            let schema = self.schema().ok_or_else(|| {
                crate::error::CylonError::new(crate::error::Code::Invalid, "Table has no schema".to_string())
            })?;

            arrow::compute::concat_batches(&schema, &self.batches)
                .map_err(|e| crate::error::CylonError::new(
                    crate::error::Code::ExecutionError,
                    format!("Failed to combine batches: {}", e),
                ))?
        } else if self.batches.len() == 1 {
            self.batches[0].clone()
        } else {
            return Err(crate::error::CylonError::new(
                crate::error::Code::Invalid,
                "Table has no batches".to_string(),
            ));
        };

        // Validate column indices and build sort column specs
        let mut sort_cols = Vec::new();
        for (&col_idx, &ascending) in sort_columns.iter().zip(sort_directions.iter()) {
            if col_idx >= combined.num_columns() {
                return Err(crate::error::CylonError::new(
                    crate::error::Code::Invalid,
                    format!("Sort column index {} out of range (table has {} columns)",
                            col_idx, combined.num_columns()),
                ));
            }

            sort_cols.push(SortColumn {
                values: combined.column(col_idx).clone(),
                options: Some(arrow::compute::SortOptions {
                    descending: !ascending,
                    nulls_first: false,
                }),
            });
        }

        // Get sorted indices
        let indices = lexsort_to_indices(&sort_cols, None)
            .map_err(|e| crate::error::CylonError::new(
                crate::error::Code::ExecutionError,
                format!("Failed to compute sort indices: {}", e),
            ))?;

        // Apply indices to all columns
        let mut sorted_columns = Vec::new();
        for i in 0..combined.num_columns() {
            let sorted_col = take(combined.column(i), &indices, None)
                .map_err(|e| crate::error::CylonError::new(
                    crate::error::Code::ExecutionError,
                    format!("Failed to reorder column {}: {}", i, e),
                ))?;
            sorted_columns.push(sorted_col);
        }

        // Create sorted batch
        let sorted_batch = RecordBatch::try_new(combined.schema(), sorted_columns)
            .map_err(|e| crate::error::CylonError::new(
                crate::error::Code::ExecutionError,
                format!("Failed to create sorted batch: {}", e),
            ))?;

        Table::from_record_batch(self.ctx.clone(), sorted_batch)
    }

    /// Filter rows based on a boolean mask array
    /// Corresponds to C++ Select function (table.cpp:892)
    ///
    /// This accepts a pre-computed boolean array indicating which rows to keep.
    ///
    /// # Arguments
    /// * `mask` - Boolean array where true means keep the row, false means discard
    ///
    /// # Returns
    /// A new table containing only the rows where mask is true
    pub fn select(&self, mask: &arrow::array::BooleanArray) -> CylonResult<Table> {
        use arrow::compute::filter_record_batch;

        // Validate mask length
        if mask.len() != self.rows() as usize {
            return Err(crate::error::CylonError::new(
                crate::error::Code::Invalid,
                format!("Mask length {} does not match table rows {}",
                        mask.len(), self.rows()),
            ));
        }

        // Filter each batch
        let mut filtered_batches = Vec::new();

        if self.batches.len() == 1 {
            // Single batch - simple case
            let filtered = filter_record_batch(&self.batches[0], mask)
                .map_err(|e| crate::error::CylonError::new(
                    crate::error::Code::ExecutionError,
                    format!("Failed to filter batch: {}", e),
                ))?;
            filtered_batches.push(filtered);
        } else {
            // Multiple batches - need to split mask accordingly
            let mut mask_offset = 0;
            for batch in &self.batches {
                let batch_len = batch.num_rows();

                // Slice the mask for this batch
                let batch_mask = mask.slice(mask_offset, batch_len);

                let filtered = filter_record_batch(batch, &batch_mask)
                    .map_err(|e| crate::error::CylonError::new(
                        crate::error::Code::ExecutionError,
                        format!("Failed to filter batch: {}", e),
                    ))?;

                if filtered.num_rows() > 0 {
                    filtered_batches.push(filtered);
                }

                mask_offset += batch_len;
            }
        }

        if filtered_batches.is_empty() {
            // Return empty table with same schema
            let schema = self.schema().ok_or_else(|| {
                crate::error::CylonError::new(crate::error::Code::Invalid,
                    "Table has no schema".to_string())
            })?;

            // Create empty batch
            let empty_columns: Vec<arrow::array::ArrayRef> = schema.fields().iter()
                .map(|field| {
                    arrow::array::new_empty_array(field.data_type())
                })
                .collect();

            let empty_batch = RecordBatch::try_new(schema, empty_columns)
                .map_err(|e| crate::error::CylonError::new(
                    crate::error::Code::ExecutionError,
                    format!("Failed to create empty batch: {}", e),
                ))?;

            Table::from_record_batch(self.ctx.clone(), empty_batch)
        } else {
            Table::from_record_batches(self.ctx.clone(), filtered_batches)
        }
    }
}

// Set operations - standalone functions (not Table methods)
// Ported from cpp/src/cylon/table.cpp lines 925-1107

/// Union of two tables (unique rows from both tables)
/// Corresponds to C++ Union function (table.cpp:925-995)
///
/// Returns unique rows from both tables combined
///
/// # Arguments
/// * `first` - First table
/// * `second` - Second table
///
/// # Returns
/// A new table containing unique rows from both tables
pub fn union(first: &Table, second: &Table) -> CylonResult<Table> {
    use hashbrown::HashSet;
    use arrow::compute::concat_batches;
    use crate::arrow::arrow_comparator::{DualTableRowIndexHash, DualTableRowIndexEqualTo, set_bit};
    use arrow::array::BooleanBuilder;
    use crate::error::{CylonError, Code};

    // Combine batches for both tables (C++ lines 932-933)
    let schema = first.schema().ok_or_else(|| {
        CylonError::new(Code::Invalid, "First table has no schema".to_string())
    })?;

    let left_batch = if first.batches.len() > 1 {
        concat_batches(&schema, &first.batches)
            .map_err(|e| CylonError::new(Code::ExecutionError,
                format!("Failed to combine left batches: {}", e)))?
    } else if first.batches.len() == 1 {
        first.batches[0].clone()
    } else {
        return Err(CylonError::new(Code::Invalid, "First table has no batches".to_string()));
    };

    let other_schema = second.schema().ok_or_else(|| {
        CylonError::new(Code::Invalid, "Second table has no schema".to_string())
    })?;

    // Verify schemas match (C++ line 935)
    if !schema.eq(&other_schema) {
        return Err(CylonError::new(Code::Invalid,
            "Tables must have the same schema for union".to_string()));
    }

    let right_batch = if second.batches.len() > 1 {
        concat_batches(&other_schema, &second.batches)
            .map_err(|e| CylonError::new(Code::ExecutionError,
                format!("Failed to combine right batches: {}", e)))?
    } else if second.batches.len() == 1 {
        second.batches[0].clone()
    } else {
        return Err(CylonError::new(Code::Invalid, "Second table has no batches".to_string()));
    };

    // Create hash and equality infrastructure (C++ lines 937-941)
    let hash = DualTableRowIndexHash::new(&left_batch, &right_batch)?;
    let equal_to = DualTableRowIndexEqualTo::new(&left_batch, &right_batch)?;

    // Create hash set with custom hash and equality (C++ lines 943-945)
    let mut rows_set = HashSet::with_capacity_and_hasher(
        (left_batch.num_rows() + right_batch.num_rows()) as usize,
        std::hash::BuildHasherDefault::<ahash::AHasher>::default(),
    );

    // Insert left table rows and track unique ones (C++ lines 952-960)
    let mut left_mask = BooleanBuilder::with_capacity(left_batch.num_rows());
    for i in 0..left_batch.num_rows() as i64 {
        let hash_val = hash.hash(i);
        // Check if this row is unique by attempting to insert
        let is_unique = if let Some(&existing_idx) = rows_set.iter().find(|&&idx| {
            hash.hash(idx) == hash_val && equal_to.equal(idx, i)
        }) {
            false // Already exists
        } else {
            rows_set.insert(i);
            true
        };
        left_mask.append_value(is_unique); // C++ line 958
    }
    let left_mask_array = left_mask.finish();

    // Filter left table (C++ lines 962-966)
    let filtered_left = arrow::compute::filter_record_batch(&left_batch, &left_mask_array)
        .map_err(|e| CylonError::new(Code::ExecutionError,
            format!("Failed to filter left table: {}", e)))?;

    // Insert right table rows with bit set and track unique ones (C++ lines 969-979)
    let mut right_mask = BooleanBuilder::with_capacity(right_batch.num_rows());
    for i in 0..right_batch.num_rows() as i64 {
        let idx = set_bit(i); // C++ line 974
        let hash_val = hash.hash(idx);
        // Check if this row is unique
        let is_unique = if let Some(&existing_idx) = rows_set.iter().find(|&&existing| {
            hash.hash(existing) == hash_val && equal_to.equal(existing, idx)
        }) {
            false // Already exists
        } else {
            rows_set.insert(idx);
            true
        };
        right_mask.append_value(is_unique); // C++ line 977
    }
    let right_mask_array = right_mask.finish();

    // Filter right table (C++ lines 981-984)
    let filtered_right = arrow::compute::filter_record_batch(&right_batch, &right_mask_array)
        .map_err(|e| CylonError::new(Code::ExecutionError,
            format!("Failed to filter right table: {}", e)))?;

    // Concatenate filtered tables (C++ lines 987-990)
    let result_batches = vec![filtered_left, filtered_right];
    let concatenated = concat_batches(&schema, &result_batches)
        .map_err(|e| CylonError::new(Code::ExecutionError,
            format!("Failed to concatenate tables: {}", e)))?;

    Table::from_record_batch(first.ctx.clone(), concatenated)
}

/// Subtract - rows in first table that are not in second table
/// Corresponds to C++ Subtract function (table.cpp:997-1049)
///
/// Returns rows from first table that don't exist in second table
///
/// # Arguments
/// * `first` - First table
/// * `second` - Second table (rows to subtract)
///
/// # Returns
/// A new table containing rows from first that are not in second
pub fn subtract(first: &Table, second: &Table) -> CylonResult<Table> {
    use hashbrown::HashSet;
    use arrow::compute::concat_batches;
    use crate::arrow::arrow_comparator::{DualTableRowIndexHash, DualTableRowIndexEqualTo, set_bit};
    use arrow::array::BooleanBuilder;
    use crate::error::{CylonError, Code};

    // Combine batches for both tables (C++ lines 1004-1005)
    let schema = first.schema().ok_or_else(|| {
        CylonError::new(Code::Invalid, "First table has no schema".to_string())
    })?;

    let left_batch = if first.batches.len() > 1 {
        concat_batches(&schema, &first.batches)
            .map_err(|e| CylonError::new(Code::ExecutionError,
                format!("Failed to combine left batches: {}", e)))?
    } else if first.batches.len() == 1 {
        first.batches[0].clone()
    } else {
        return Err(CylonError::new(Code::Invalid, "First table has no batches".to_string()));
    };

    let other_schema = second.schema().ok_or_else(|| {
        CylonError::new(Code::Invalid, "Second table has no schema".to_string())
    })?;

    // Verify schemas match (C++ line 1007)
    if !schema.eq(&other_schema) {
        return Err(CylonError::new(Code::Invalid,
            "Tables must have the same schema for subtract".to_string()));
    }

    let right_batch = if second.batches.len() > 1 {
        concat_batches(&other_schema, &second.batches)
            .map_err(|e| CylonError::new(Code::ExecutionError,
                format!("Failed to combine right batches: {}", e)))?
    } else if second.batches.len() == 1 {
        second.batches[0].clone()
    } else {
        return Err(CylonError::new(Code::Invalid, "Second table has no batches".to_string()));
    };

    // Create hash and equality infrastructure (C++ lines 1009-1013)
    let hash = DualTableRowIndexHash::new(&left_batch, &right_batch)?;
    let equal_to = DualTableRowIndexEqualTo::new(&left_batch, &right_batch)?;

    // Create hash set (C++ lines 1015-1017)
    let mut rows_set = HashSet::with_capacity_and_hasher(
        left_batch.num_rows() as usize,
        std::hash::BuildHasherDefault::<ahash::AHasher>::default(),
    );

    // Insert left table rows and create initial mask (C++ lines 1026-1029)
    let mut mask_builder = BooleanBuilder::with_capacity(left_batch.num_rows());
    for i in 0..left_batch.num_rows() as i64 {
        let hash_val = hash.hash(i);
        // Check if this row is unique
        let is_unique = if let Some(&_existing_idx) = rows_set.iter().find(|&&idx| {
            hash.hash(idx) == hash_val && equal_to.equal(idx, i)
        }) {
            false // Already exists
        } else {
            rows_set.insert(i);
            true
        };
        mask_builder.append_value(is_unique); // C++ line 1028
    }
    let mask_array = mask_builder.finish();

    // Get the mask as a vector so we can mutate it (C++ line 1033)
    let mut bitmask: Vec<bool> = (0..mask_array.len())
        .map(|i| mask_array.value(i))
        .collect();

    // Probe right rows and clear bits for matches (C++ lines 1036-1042)
    for i in 0..right_batch.num_rows() as i64 {
        let idx = set_bit(i); // C++ line 1038
        let hash_val = hash.hash(idx);

        // Find matching row in left table
        if let Some(&left_idx) = rows_set.iter().find(|&&left| {
            hash.hash(left) == hash_val && equal_to.equal(left, idx)
        }) {
            // Clear the bit for this row (C++ line 1040)
            bitmask[left_idx as usize] = false;
        }
    }

    // Reconstruct the boolean array
    let mut mask_builder = BooleanBuilder::with_capacity(bitmask.len());
    for &bit in &bitmask {
        mask_builder.append_value(bit);
    }
    let mask_array = mask_builder.finish();

    // Filter left table (C++ lines 1045-1048)
    let filtered = arrow::compute::filter_record_batch(&left_batch, &mask_array)
        .map_err(|e| CylonError::new(Code::ExecutionError,
            format!("Failed to filter table: {}", e)))?;

    Table::from_record_batch(first.ctx.clone(), filtered)
}

/// Intersect - rows that exist in both tables
/// Corresponds to C++ Intersect function (table.cpp:1051-1110)
///
/// Returns rows that exist in both tables
///
/// # Arguments
/// * `first` - First table
/// * `second` - Second table
///
/// # Returns
/// A new table containing rows that exist in both tables
pub fn intersect(first: &Table, second: &Table) -> CylonResult<Table> {
    use hashbrown::HashSet;
    use arrow::compute::concat_batches;
    use crate::arrow::arrow_comparator::{DualTableRowIndexHash, DualTableRowIndexEqualTo, set_bit};
    use arrow::array::BooleanBuilder;
    use crate::error::{CylonError, Code};

    // Combine batches for both tables (C++ lines 1060-1061)
    let schema = first.schema().ok_or_else(|| {
        CylonError::new(Code::Invalid, "First table has no schema".to_string())
    })?;

    let left_batch = if first.batches.len() > 1 {
        concat_batches(&schema, &first.batches)
            .map_err(|e| CylonError::new(Code::ExecutionError,
                format!("Failed to combine left batches: {}", e)))?
    } else if first.batches.len() == 1 {
        first.batches[0].clone()
    } else {
        return Err(CylonError::new(Code::Invalid, "First table has no batches".to_string()));
    };

    let other_schema = second.schema().ok_or_else(|| {
        CylonError::new(Code::Invalid, "Second table has no schema".to_string())
    })?;

    // Verify schemas match (C++ line 1063)
    if !schema.eq(&other_schema) {
        return Err(CylonError::new(Code::Invalid,
            "Tables must have the same schema for intersect".to_string()));
    }

    let right_batch = if second.batches.len() > 1 {
        concat_batches(&other_schema, &second.batches)
            .map_err(|e| CylonError::new(Code::ExecutionError,
                format!("Failed to combine right batches: {}", e)))?
    } else if second.batches.len() == 1 {
        second.batches[0].clone()
    } else {
        return Err(CylonError::new(Code::Invalid, "Second table has no batches".to_string()));
    };

    // Create hash and equality infrastructure (C++ lines 1065-1069)
    let hash = DualTableRowIndexHash::new(&left_batch, &right_batch)?;
    let equal_to = DualTableRowIndexEqualTo::new(&left_batch, &right_batch)?;

    // Create hash set (C++ lines 1071-1073)
    let mut rows_set = HashSet::with_capacity_and_hasher(
        left_batch.num_rows() as usize,
        std::hash::BuildHasherDefault::<ahash::AHasher>::default(),
    );

    // Insert all left table rows into set (C++ lines 1078-1080)
    for i in 0..left_batch.num_rows() as i64 {
        rows_set.insert(i);
    }

    // Create bitmask (all false initially) (C++ line 1083)
    let mut bitmask = vec![false; left_batch.num_rows()];

    // Probe right rows and set bits for matches (C++ lines 1086-1092)
    for i in 0..right_batch.num_rows() as i64 {
        let idx = set_bit(i); // C++ line 1088
        let hash_val = hash.hash(idx);

        // Find matching row in left table
        if let Some(&left_idx) = rows_set.iter().find(|&&left| {
            hash.hash(left) == hash_val && equal_to.equal(left, idx)
        }) {
            // Mark this left row as matching (C++ line 1090)
            bitmask[left_idx as usize] = true;
        }
    }

    // Convert bitmask to BooleanArray (C++ lines 1095-1098)
    let mut mask_builder = BooleanBuilder::with_capacity(bitmask.len());
    for &bit in &bitmask {
        mask_builder.append_value(bit);
    }
    let mask_array = mask_builder.finish();

    // Filter left table (C++ lines 1100-1107)
    let filtered = arrow::compute::filter_record_batch(&left_batch, &mask_array)
        .map_err(|e| CylonError::new(Code::ExecutionError,
            format!("Failed to filter table: {}", e)))?;

    Table::from_record_batch(first.ctx.clone(), filtered)
}

/// Unique - remove duplicate rows from a table
/// Corresponds to C++ Unique function (table.cpp:1306-1374)
///
/// Returns a table with only unique rows based on specified columns
///
/// # Arguments
/// * `table` - Input table
/// * `col_indices` - Column indices to use for uniqueness check
/// * `keep_first` - If true, keep first occurrence; if false, keep last occurrence
///
/// # Returns
/// A new table containing only unique rows
pub fn unique(table: &Table, col_indices: &[usize], keep_first: bool) -> CylonResult<Table> {
    use hashbrown::HashSet;
    use arrow::compute::concat_batches;
    use crate::arrow::arrow_comparator::{TableRowIndexHash, TableRowIndexEqualTo};
    use crate::error::{CylonError, Code};

    // Return empty table if input is empty (C++ lines 1370-1372)
    if table.is_empty() {
        return Table::from_record_batches(table.ctx.clone(), table.batches.clone());
    }

    // Combine batches (C++ lines 1316-1318)
    let schema = table.schema().ok_or_else(|| {
        CylonError::new(Code::Invalid, "Table has no schema".to_string())
    })?;

    let batch = if table.batches.len() > 1 {
        concat_batches(&schema, &table.batches)
            .map_err(|e| CylonError::new(Code::ExecutionError,
                format!("Failed to combine batches: {}", e)))?
    } else if table.batches.len() == 1 {
        table.batches[0].clone()
    } else {
        return Err(CylonError::new(Code::Invalid, "Table has no batches".to_string()));
    };

    // Create hash and equality infrastructure (C++ lines 1320-1324)
    let row_hash = TableRowIndexHash::new_with_columns(&batch, col_indices)?;
    let row_comp = TableRowIndexEqualTo::new_with_columns(&batch, col_indices)?;

    let num_rows = batch.num_rows();

    // Create hash set (C++ lines 1327-1328)
    let mut rows_set = HashSet::with_capacity_and_hasher(
        num_rows,
        std::hash::BuildHasherDefault::<ahash::AHasher>::default(),
    );

    // Collect unique row indices (C++ lines 1330-1349)
    let mut unique_indices = Vec::with_capacity(num_rows);

    if keep_first {
        // Keep first occurrence (C++ lines 1336-1341)
        for row in 0..num_rows as i64 {
            let hash_val = row_hash.hash(row);
            // Check if this row is unique
            let is_unique = if let Some(&_existing_idx) = rows_set.iter().find(|&&idx| {
                row_hash.hash(idx) == hash_val && row_comp.equal(idx, row)
            }) {
                false // Already exists
            } else {
                rows_set.insert(row);
                true
            };

            if is_unique {
                unique_indices.push(row); // C++ line 1339
            }
        }
    } else {
        // Keep last occurrence (C++ lines 1343-1348)
        for row in (0..num_rows as i64).rev() {
            let hash_val = row_hash.hash(row);
            // Check if this row is unique
            let is_unique = if let Some(&_existing_idx) = rows_set.iter().find(|&&idx| {
                row_hash.hash(idx) == hash_val && row_comp.equal(idx, row)
            }) {
                false // Already exists
            } else {
                rows_set.insert(row);
                true
            };

            if is_unique {
                unique_indices.push(row); // C++ line 1346
            }
        }
        // Reverse to maintain original order (since we iterated backwards)
        unique_indices.reverse();
    }

    // Take rows at the unique indices (C++ lines 1357-1359)
    use arrow::array::Int64Array;
    use arrow::compute::take;

    let indices_array = Int64Array::from(unique_indices);
    let mut taken_columns = Vec::new();

    for col_idx in 0..batch.num_columns() {
        let taken_col = take(batch.column(col_idx), &indices_array, None)
            .map_err(|e| CylonError::new(Code::ExecutionError,
                format!("Failed to take column {}: {}", col_idx, e)))?;
        taken_columns.push(taken_col);
    }

    let result_batch = RecordBatch::try_new(batch.schema(), taken_columns)
        .map_err(|e| CylonError::new(Code::ExecutionError,
            format!("Failed to create result batch: {}", e)))?;

    Table::from_record_batch(table.ctx.clone(), result_batch)
}

/// Join two tables
/// Corresponds to C++ Join function (table.cpp:819-854)
///
/// Joins two tables based on the provided configuration
///
/// # Arguments
/// * `left` - Left table
/// * `right` - Right table
/// * `config` - Join configuration (join type, columns, algorithm)
///
/// # Returns
/// A new table containing the joined result
pub fn join(
    left: &Table,
    right: &Table,
    config: &crate::join::JoinConfig,
) -> CylonResult<Table> {
    crate::join::join(left, right, config)
}

/// Sort table by a single column
/// Corresponds to C++ Sort function (table.cpp:372-387)
///
/// Performs local sort on the table. If table has chunked columns, they will be
/// merged in the output.
///
/// # Arguments
/// * `table` - Input table to sort
/// * `sort_column` - Column index to sort by
/// * `ascending` - Sort direction (true = ascending, false = descending)
///
/// # Returns
/// A new sorted table
///
/// # Example
/// ```ignore
/// // Sort by column 0 in ascending order
/// let sorted = sort(&table, 0, true)?;
/// ```
pub fn sort(
    table: &Table,
    sort_column: usize,
    ascending: bool,
) -> CylonResult<Table> {
    table.sort(sort_column, ascending)
}

/// Sort table by multiple columns with same direction
/// Corresponds to C++ Sort function (table.cpp:389-393)
///
/// # Arguments
/// * `table` - Input table to sort
/// * `sort_columns` - Column indices to sort by (in order of priority)
/// * `ascending` - Sort direction for all columns (true = ascending, false = descending)
///
/// # Returns
/// A new sorted table
pub fn sort_multi_same_direction(
    table: &Table,
    sort_columns: &[usize],
    ascending: bool,
) -> CylonResult<Table> {
    let sort_directions = vec![ascending; sort_columns.len()];
    table.sort_multi(sort_columns, &sort_directions)
}

/// Sort table by multiple columns with individual directions
/// Corresponds to C++ Sort function (table.cpp:395-418)
///
/// # Arguments
/// * `table` - Input table to sort
/// * `sort_columns` - Column indices to sort by (in order of priority)
/// * `sort_directions` - Sort direction for each column (true = ascending, false = descending)
///
/// # Returns
/// A new sorted table
///
/// # Example
/// ```ignore
/// // Sort by column 0 ascending, then column 1 descending
/// let sorted = sort_multi(&table, &[0, 1], &[true, false])?;
/// ```
pub fn sort_multi(
    table: &Table,
    sort_columns: &[usize],
    sort_directions: &[bool],
) -> CylonResult<Table> {
    table.sort_multi(sort_columns, sort_directions)
}

/// Project (select) specific columns from a table
/// Corresponds to C++ Project function (table.cpp:1212-1231)
///
/// Creates a new table containing only the specified columns.
/// This is also known as column selection or vertical slicing.
///
/// # Arguments
/// * `table` - Input table
/// * `project_columns` - Indices of columns to include in the result
///
/// # Returns
/// A new table with only the specified columns
///
/// # Example
/// ```ignore
/// // Select columns 0, 2, and 3 from the table
/// let projected = project(&table, &[0, 2, 3])?;
/// ```
pub fn project(
    table: &Table,
    project_columns: &[usize],
) -> CylonResult<Table> {
    table.project(project_columns)
}

// TODO: Port table operations from cpp/src/cylon/table.hpp:
// - FromCSV
// - WriteCSV
// - Merge
// - Join, DistributedJoin
// - DistributedUnion
// - Subtract, DistributedSubtract
// - Intersect, DistributedIntersect
// - Shuffle
// - HashPartition
// - Sort, DistributedSort
// - Select
// - Project
// - Unique, DistributedUnique
// - Slice, DistributedSlice
// - Head, Tail
// etc.