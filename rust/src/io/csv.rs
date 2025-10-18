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

//! CSV I/O operations
//!
//! Ported from cpp/src/cylon/io/csv_read_config.hpp and table.cpp

use std::sync::Arc;
use arrow::csv::{ReaderBuilder, WriterBuilder};
use arrow::datatypes::{Schema, DataType};

use crate::table::Table;
use crate::ctx::CylonContext;
use crate::error::{CylonResult, CylonError, Code};

/// CSV read options
/// Corresponds to C++ CSVReadOptions
#[derive(Clone)]
pub struct CsvReadOptions {
    /// Whether to use threading for reading (default: true)
    pub use_threads: bool,
    /// CSV delimiter (default: ',')
    pub delimiter: u8,
    /// Whether to treat first row as header (default: true)
    pub has_header: bool,
    /// Column names (if not using header row)
    pub column_names: Option<Vec<String>>,
    /// Batch size for reading (default: 8192)
    pub batch_size: usize,
    /// Columns to include (None = all columns)
    pub include_columns: Option<Vec<String>>,
    /// Whether to slice the table across workers (for distributed reading)
    pub slice: bool,
}

impl Default for CsvReadOptions {
    fn default() -> Self {
        Self {
            use_threads: true,
            delimiter: b',',
            has_header: true,
            column_names: None,
            batch_size: 8192,
            include_columns: None,
            slice: false,
        }
    }
}

impl CsvReadOptions {
    /// Create new CSV read options with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the delimiter character
    pub fn with_delimiter(mut self, delimiter: u8) -> Self {
        self.delimiter = delimiter;
        self
    }

    /// Set whether first row is header
    pub fn with_header(mut self, has_header: bool) -> Self {
        self.has_header = has_header;
        self
    }

    /// Set column names explicitly
    pub fn with_column_names(mut self, names: Vec<String>) -> Self {
        self.column_names = Some(names);
        self.has_header = false; // If we're providing names, don't use header row
        self
    }

    /// Set batch size for reading
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set columns to include (None = all columns)
    pub fn with_include_columns(mut self, columns: Vec<String>) -> Self {
        self.include_columns = Some(columns);
        self
    }

    /// Enable table slicing across workers
    pub fn with_slice(mut self, slice: bool) -> Self {
        self.slice = slice;
        self
    }
}

/// CSV write options
/// Corresponds to C++ CSVWriteOptions
#[derive(Clone)]
pub struct CsvWriteOptions {
    /// CSV delimiter (default: ',')
    pub delimiter: u8,
    /// Whether to write header row (default: true)
    pub has_header: bool,
}

impl Default for CsvWriteOptions {
    fn default() -> Self {
        Self {
            delimiter: b',',
            has_header: true,
        }
    }
}

impl CsvWriteOptions {
    /// Create new CSV write options with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the delimiter character
    pub fn with_delimiter(mut self, delimiter: u8) -> Self {
        self.delimiter = delimiter;
        self
    }

    /// Set whether to write header row
    pub fn with_header(mut self, has_header: bool) -> Self {
        self.has_header = has_header;
        self
    }
}

/// Read a CSV file into a Table
/// Corresponds to C++ FromCSV
pub fn read_csv(
    ctx: Arc<CylonContext>,
    path: &str,
    options: &CsvReadOptions,
) -> CylonResult<Table> {
    // Open the file
    let file = std::fs::File::open(path)
        .map_err(|e| CylonError::new(
            Code::IoError,
            format!("Failed to open CSV file '{}': {}", path, e),
        ))?;

    // Build Arrow CSV reader
    // Arrow CSV reader requires Format for inference
    let format = arrow::csv::reader::Format::default()
        .with_delimiter(options.delimiter)
        .with_header(options.has_header);

    let (schema, _) = format.infer_schema(&mut std::io::BufReader::new(file), Some(100))
        .map_err(|e| CylonError::new(
            Code::IoError,
            format!("Failed to infer CSV schema: {}", e),
        ))?;

    // Reopen file for reading
    let file = std::fs::File::open(path)
        .map_err(|e| CylonError::new(
            Code::IoError,
            format!("Failed to reopen CSV file '{}': {}", path, e),
        ))?;

    let reader_builder = ReaderBuilder::new(Arc::new(schema))
        .with_delimiter(options.delimiter)
        .with_batch_size(options.batch_size)
        .with_header(options.has_header);

    // Build the reader
    let mut reader = reader_builder
        .build(file)
        .map_err(|e| CylonError::new(
            Code::IoError,
            format!("Failed to create CSV reader: {}", e),
        ))?;

    // Read all batches
    let mut batches = Vec::new();
    loop {
        match reader.next() {
            Some(Ok(batch)) => {
                // Filter columns if requested
                if let Some(ref include_cols) = options.include_columns {
                    let schema = batch.schema();
                    let mut indices = Vec::new();
                    for col_name in include_cols {
                        if let Ok(idx) = schema.index_of(col_name) {
                            indices.push(idx);
                        } else {
                            return Err(CylonError::new(
                                Code::Invalid,
                                format!("Column '{}' not found in CSV", col_name),
                            ));
                        }
                    }

                    // Project to selected columns
                    let columns: Vec<_> = indices.iter()
                        .map(|&idx| batch.column(idx).clone())
                        .collect();
                    let fields: Vec<_> = indices.iter()
                        .map(|&idx| schema.field(idx).clone())
                        .collect();
                    let new_schema = Arc::new(Schema::new(fields));

                    let projected_batch = arrow::record_batch::RecordBatch::try_new(
                        new_schema,
                        columns,
                    ).map_err(|e| CylonError::new(
                        Code::ExecutionError,
                        format!("Failed to project columns: {}", e),
                    ))?;

                    batches.push(projected_batch);
                } else {
                    batches.push(batch);
                }
            }
            Some(Err(e)) => {
                return Err(CylonError::new(
                    Code::IoError,
                    format!("Error reading CSV: {}", e),
                ));
            }
            None => break,
        }
    }

    if batches.is_empty() {
        return Err(CylonError::new(
            Code::Invalid,
            "CSV file is empty".to_string(),
        ));
    }

    // Slice the table if requested and in distributed mode
    if options.slice && ctx.get_world_size() > 1 {
        // Combine all batches first
        let combined = if batches.len() > 1 {
            let schema = batches[0].schema();
            arrow::compute::concat_batches(&schema, &batches)
                .map_err(|e| CylonError::new(
                    Code::ExecutionError,
                    format!("Failed to combine batches: {}", e),
                ))?
        } else {
            batches.into_iter().next().unwrap()
        };

        // Calculate slice for this worker
        let total_rows = combined.num_rows();
        let world_size = ctx.get_world_size() as usize;
        let rank = ctx.get_rank() as usize;

        let rows_per_worker = total_rows / world_size;
        let remainder = total_rows % world_size;

        let start = if rank < remainder {
            rank * (rows_per_worker + 1)
        } else {
            remainder * (rows_per_worker + 1) + (rank - remainder) * rows_per_worker
        };

        let length = if rank < remainder {
            rows_per_worker + 1
        } else {
            rows_per_worker
        };

        // Slice the batch
        let sliced = combined.slice(start, length);
        Table::from_record_batch(ctx, sliced)
    } else if batches.len() > 1 {
        // Multiple batches - just use them as is
        Table::from_record_batches(ctx, batches)
    } else {
        // Single batch
        Table::from_record_batch(ctx, batches.into_iter().next().unwrap())
    }
}

/// Write a Table to a CSV file
/// Corresponds to C++ WriteCSV
pub fn write_csv(
    table: &Table,
    path: &str,
    options: &CsvWriteOptions,
) -> CylonResult<()> {
    // Create the output file
    let file = std::fs::File::create(path)
        .map_err(|e| CylonError::new(
            Code::IoError,
            format!("Failed to create CSV file '{}': {}", path, e),
        ))?;

    // Get the schema from the first batch
    let batch = table.batch(0).ok_or_else(|| {
        CylonError::new(Code::Invalid, "Table has no batches".to_string())
    })?;

    // Build Arrow CSV writer
    let mut writer = WriterBuilder::new()
        .with_delimiter(options.delimiter)
        .with_header(options.has_header)
        .build(file);

    // Write all batches
    for i in 0..table.num_batches() {
        let batch = table.batch(i).unwrap();
        writer.write(batch)
            .map_err(|e| CylonError::new(
                Code::IoError,
                format!("Failed to write CSV batch {}: {}", i, e),
            ))?;
    }

    Ok(())
}
