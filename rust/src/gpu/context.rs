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

//! GPU context management

use std::sync::Arc;
use crate::error::{CylonError, CylonResult, Code};
use super::ffi;
use super::table::GpuTable;

/// GPU context for distributed operations.
///
/// Wraps a gcylon context that manages MPI communication and GPU resources.
/// The context is reference-counted and can be shared across tables.
pub struct GpuContext {
    ptr: *mut ffi::GcylonContext,
}

impl GpuContext {
    /// Create a new GPU context with MPI.
    ///
    /// Initializes MPI if not already initialized and sets up GPU resources.
    ///
    /// # Example
    /// ```ignore
    /// let ctx = GpuContext::new_mpi()?;
    /// println!("Rank {} of {}", ctx.rank(), ctx.world_size());
    /// ```
    pub fn new_mpi() -> CylonResult<Arc<Self>> {
        let mut ptr = std::ptr::null_mut();
        let status = unsafe { ffi::gcylon_context_create_mpi(&mut ptr) };

        if status != ffi::GCYLON_OK {
            return Err(CylonError::new(
                Code::ExecutionError,
                format!("Failed to create GPU context: {}", get_last_error()),
            ));
        }

        Ok(Arc::new(Self { ptr }))
    }

    /// Get this worker's rank (0-indexed).
    pub fn rank(&self) -> i32 {
        unsafe { ffi::gcylon_context_get_rank(self.ptr) }
    }

    /// Get the world size (total number of workers).
    pub fn world_size(&self) -> i32 {
        unsafe { ffi::gcylon_context_get_world_size(self.ptr) }
    }

    /// Query current GPU memory information.
    pub fn memory_info(&self) -> CylonResult<GpuMemoryInfo> {
        let mut info = ffi::GcylonMemoryInfo {
            free_bytes: 0,
            total_bytes: 0,
            used_bytes: 0,
        };

        let status = unsafe { ffi::gcylon_get_gpu_memory_info(&mut info) };
        if status != ffi::GCYLON_OK {
            return Err(CylonError::new(
                Code::ExecutionError,
                "Failed to get GPU memory info",
            ));
        }

        Ok(GpuMemoryInfo {
            free: info.free_bytes,
            total: info.total_bytes,
            used: info.used_bytes,
        })
    }

    /// Create a table with sequential int64 data for testing.
    ///
    /// # Arguments
    /// * `num_columns` - Number of columns
    /// * `num_rows` - Number of rows
    /// * `start_value` - Starting value for sequential data
    /// * `step` - Value increment between rows
    ///
    /// # Example
    /// ```ignore
    /// let ctx = GpuContext::new_mpi()?;
    /// let table = ctx.create_sequential_table(4, 1000, 0, 1)?;
    /// ```
    pub fn create_sequential_table(
        self: &Arc<Self>,
        num_columns: i32,
        num_rows: i64,
        start_value: i64,
        step: i64,
    ) -> CylonResult<GpuTable> {
        let mut output = std::ptr::null_mut();
        let status = unsafe {
            ffi::gcylon_table_create_sequential(
                self.ptr,
                num_columns,
                num_rows,
                start_value,
                step,
                &mut output,
            )
        };

        if status != ffi::GCYLON_OK {
            return Err(CylonError::new(
                Code::ExecutionError,
                format!("Failed to create sequential table: {}", get_last_error()),
            ));
        }

        Ok(unsafe { GpuTable::from_raw(output, Arc::clone(self)) })
    }

    /// Create a table with random int64 data for testing.
    ///
    /// # Arguments
    /// * `num_columns` - Number of columns
    /// * `num_rows` - Number of rows
    /// * `seed` - Random seed for reproducibility
    ///
    /// # Example
    /// ```ignore
    /// let ctx = GpuContext::new_mpi()?;
    /// let table = ctx.create_random_table(4, 1000, 42)?;
    /// ```
    pub fn create_random_table(
        self: &Arc<Self>,
        num_columns: i32,
        num_rows: i64,
        seed: i32,
    ) -> CylonResult<GpuTable> {
        let mut output = std::ptr::null_mut();
        let status = unsafe {
            ffi::gcylon_table_create_random(self.ptr, num_columns, num_rows, seed, &mut output)
        };

        if status != ffi::GCYLON_OK {
            return Err(CylonError::new(
                Code::ExecutionError,
                format!("Failed to create random table: {}", get_last_error()),
            ));
        }

        Ok(unsafe { GpuTable::from_raw(output, Arc::clone(self)) })
    }

    /// Get the raw pointer for FFI calls.
    pub(crate) fn as_ptr(&self) -> *mut ffi::GcylonContext {
        self.ptr
    }
}

impl Drop for GpuContext {
    fn drop(&mut self) {
        unsafe { ffi::gcylon_context_free(self.ptr) };
    }
}

// Safety: GcylonContext handles are thread-safe (MPI is process-global)
unsafe impl Send for GpuContext {}
unsafe impl Sync for GpuContext {}

/// GPU memory information.
#[derive(Debug, Clone, Copy)]
pub struct GpuMemoryInfo {
    /// Free GPU memory in bytes.
    pub free: usize,
    /// Total GPU memory in bytes.
    pub total: usize,
    /// Used GPU memory in bytes.
    pub used: usize,
}

impl GpuMemoryInfo {
    /// Get memory usage as a fraction (0.0 - 1.0).
    pub fn usage_fraction(&self) -> f64 {
        if self.total > 0 {
            self.used as f64 / self.total as f64
        } else {
            0.0
        }
    }

    /// Get free memory as a fraction (0.0 - 1.0).
    pub fn free_fraction(&self) -> f64 {
        if self.total > 0 {
            self.free as f64 / self.total as f64
        } else {
            0.0
        }
    }
}

/// Get the last error message from gcylon.
fn get_last_error() -> String {
    unsafe {
        let ptr = ffi::gcylon_get_last_error();
        if ptr.is_null() {
            "Unknown error".to_string()
        } else {
            std::ffi::CStr::from_ptr(ptr)
                .to_string_lossy()
                .into_owned()
        }
    }
}
