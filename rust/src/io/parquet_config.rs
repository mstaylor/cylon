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

//! Parquet configuration options
//!
//! Ported from cpp/src/cylon/io/parquet_config.hpp

use parquet::file::properties::WriterProperties;

/// Parquet writer options
/// Corresponds to C++ ParquetOptions (parquet_config.hpp)
pub struct ParquetOptions {
    /// Chunk size for writing
    chunk_size: usize,
    /// Writer properties
    writer_properties: Option<WriterProperties>,
}

impl ParquetOptions {
    /// Create default Parquet options
    /// Corresponds to C++ ParquetOptions constructor (parquet_config.cpp)
    pub fn new() -> Self {
        Self {
            chunk_size: 1024 * 1024, // 1MB default
            writer_properties: None,
        }
    }

    /// Set chunk size
    /// Corresponds to C++ SetChunkSize (parquet_config.hpp)
    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size;
        self
    }

    /// Get chunk size
    /// Corresponds to C++ GetChunkSize (parquet_config.hpp)
    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    /// Set custom writer properties
    pub fn with_writer_properties(mut self, props: WriterProperties) -> Self {
        self.writer_properties = Some(props);
        self
    }

    /// Get writer properties (or create default)
    /// Corresponds to C++ GetWriterProperties (parquet_config.hpp)
    pub fn writer_properties(&self) -> WriterProperties {
        self.writer_properties.clone().unwrap_or_else(|| {
            WriterProperties::builder().build()
        })
    }
}

impl Default for ParquetOptions {
    fn default() -> Self {
        Self::new()
    }
}
