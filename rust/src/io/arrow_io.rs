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

//! Arrow I/O operations for reading and writing files
//!
//! Ported from cpp/src/cylon/io/arrow_io.cpp

use crate::ctx::CylonContext;
use crate::error::{CylonError, CylonResult, Code};
use crate::table::Table;
use crate::io::parquet_config::ParquetOptions;
use std::fs::File;
use std::sync::Arc;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;

/// Read a Parquet file into a Table
/// Corresponds to C++ ReadParquet function (arrow_io.cpp:144-171)
///
/// # Arguments
/// * `ctx` - Cylon context
/// * `path` - Path to the Parquet file to read
///
/// # Returns
/// A Table containing the data from the Parquet file
pub fn read_parquet(ctx: Arc<CylonContext>, path: &str) -> CylonResult<Table> {
    // Open the file (C++ arrow_io.cpp:147-150)
    let file = File::open(path)
        .map_err(|e| CylonError::new(
            Code::IoError,
            format!("Failed to open Parquet file {}: {}", path, e)
        ))?;

    // Create Parquet reader (C++ arrow_io.cpp:152-157)
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| CylonError::new(
            Code::IoError,
            format!("Failed to create Parquet reader for {}: {}", path, e)
        ))?;

    let mut reader = builder.build()
        .map_err(|e| CylonError::new(
            Code::IoError,
            format!("Failed to build Parquet reader for {}: {}", path, e)
        ))?;

    // Read all batches (C++ arrow_io.cpp:160-164)
    let mut batches = Vec::new();
    for batch_result in reader.by_ref() {
        let batch = batch_result.map_err(|e| CylonError::new(
            Code::IoError,
            format!("Failed to read batch from {}: {}", path, e)
        ))?;
        batches.push(batch);
    }

    if batches.is_empty() {
        return Err(CylonError::new(
            Code::IoError,
            format!("No data read from Parquet file: {}", path)
        ));
    }

    // Create table from batches (C++ arrow_io.cpp:166-169)
    // Note: C++ combines chunks if there are multiple, we handle multiple batches naturally
    Table::from_record_batches(ctx, batches)
}

/// Write a Table to a Parquet file
/// Corresponds to C++ WriteParquet function (arrow_io.cpp:174-197)
///
/// # Arguments
/// * `ctx` - Cylon context (for future use with options)
/// * `table` - Table to write
/// * `path` - Output file path
/// * `options` - Parquet writer options
///
/// # Returns
/// Ok(()) on success
pub fn write_parquet(
    _ctx: Arc<CylonContext>,
    table: &Table,
    path: &str,
    options: ParquetOptions,
) -> CylonResult<()> {
    // Get schema from table
    let schema = table.schema().ok_or_else(|| {
        CylonError::new(Code::Invalid, "Table has no schema".to_string())
    })?;

    // Create output file (C++ arrow_io.cpp:178-181)
    let file = File::create(path)
        .map_err(|e| CylonError::new(
            Code::IoError,
            format!("Failed to create Parquet file {}: {}", path, e)
        ))?;

    // Create Parquet writer with options (C++ arrow_io.cpp:183-191)
    let props = options.writer_properties();
    let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props))
        .map_err(|e| CylonError::new(
            Code::IoError,
            format!("Failed to create Parquet writer for {}: {}", path, e)
        ))?;

    // Write all batches
    for batch in table.batches() {
        writer.write(batch)
            .map_err(|e| CylonError::new(
                Code::IoError,
                format!("Failed to write batch to {}: {}", path, e)
            ))?;
    }

    // Close the writer (C++ arrow_io.cpp:196)
    writer.close()
        .map_err(|e| CylonError::new(
            Code::IoError,
            format!("Failed to close Parquet file {}: {}", path, e)
        ))?;

    Ok(())
}
