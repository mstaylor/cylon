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
}

// TODO: Port table operations from cpp/src/cylon/table.hpp:
// - FromCSV
// - WriteCSV
// - Merge
// - Join, DistributedJoin
// - Union, DistributedUnion
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