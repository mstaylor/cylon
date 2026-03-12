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

//! GPU operation configuration

use super::ffi;

/// Configuration for GPU operations.
///
/// Controls memory limits and chunking behavior for
/// memory-efficient processing of large datasets.
///
/// For GPU-to-CPU memory spillover, use RMM's managed_memory_resource
/// instead of custom allocators. This enables CUDA Unified Memory (UVM)
/// which automatically migrates pages between GPU and CPU as needed.
#[derive(Debug, Clone)]
pub struct GpuConfig {
    pub(crate) inner: ffi::GcylonConfig,
}

impl GpuConfig {
    /// Create default configuration.
    ///
    /// Uses 80% of free GPU memory and enables CPU staging with pinned memory.
    pub fn new() -> Self {
        Self {
            inner: unsafe { ffi::gcylon_config_default() },
        }
    }

    /// Create low-memory configuration.
    ///
    /// Uses 60% of free GPU memory.
    /// Use this when operating under memory pressure.
    pub fn low_memory() -> Self {
        Self {
            inner: unsafe { ffi::gcylon_config_low_memory() },
        }
    }

    /// Set explicit GPU memory limit in bytes.
    ///
    /// Set to 0 for automatic limit based on `gpu_memory_fraction`.
    pub fn with_gpu_memory_limit(mut self, limit: usize) -> Self {
        self.inner.gpu_memory_limit = limit;
        self
    }

    /// Set GPU memory fraction (0.0 - 1.0).
    ///
    /// This fraction of free GPU memory will be used for operations.
    /// Only applies when `gpu_memory_limit` is 0.
    pub fn with_gpu_memory_fraction(mut self, fraction: f32) -> Self {
        self.inner.gpu_memory_fraction = fraction.clamp(0.1, 0.95);
        self
    }

    /// Set chunk size in bytes.
    ///
    /// Set to 0 for automatic chunk sizing based on available memory.
    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.inner.chunk_size_bytes = size;
        self
    }

    /// Set minimum rows per chunk.
    ///
    /// Prevents creating tiny inefficient chunks.
    pub fn with_min_chunk_rows(mut self, rows: usize) -> Self {
        self.inner.min_chunk_rows = rows;
        self
    }

    /// Get pointer to inner config for FFI calls.
    pub(crate) fn as_ptr(&self) -> *const ffi::GcylonConfig {
        &self.inner
    }
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self::new()
    }
}
