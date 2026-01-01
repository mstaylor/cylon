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

//! GPU table operations

use std::sync::Arc;
use crate::error::{CylonError, CylonResult, Code};
use super::{ffi, GpuContext, GpuConfig};

/// A table stored on GPU with distributed operation support.
///
/// GpuTable wraps a gcylon GTable and provides methods for
/// distributed operations like shuffle, join, and gather.
pub struct GpuTable {
    ptr: *mut ffi::GcylonTable,
    ctx: Arc<GpuContext>,
}

impl GpuTable {
    /// Create a GpuTable from a raw FFI pointer.
    ///
    /// # Safety
    /// The pointer must be a valid GcylonTable pointer.
    pub(crate) unsafe fn from_raw(ptr: *mut ffi::GcylonTable, ctx: Arc<GpuContext>) -> Self {
        Self { ptr, ctx }
    }

    /// Get the number of rows in the table.
    pub fn num_rows(&self) -> i64 {
        unsafe { ffi::gcylon_table_num_rows(self.ptr) }
    }

    /// Get the number of columns in the table.
    pub fn num_columns(&self) -> i32 {
        unsafe { ffi::gcylon_table_num_columns(self.ptr) }
    }

    /// Get the GPU context associated with this table.
    pub fn context(&self) -> &Arc<GpuContext> {
        &self.ctx
    }

    /// Distributed shuffle (hash partition + all-to-all exchange).
    ///
    /// Partitions the table by hashing the specified columns and exchanges
    /// partitions with other workers so each worker has all rows for its
    /// hash range.
    ///
    /// # Arguments
    /// * `hash_columns` - Column indices to hash for partitioning
    /// * `config` - Optional configuration (None for defaults)
    ///
    /// # Example
    /// ```ignore
    /// let shuffled = table.shuffle(&[0], None)?;
    /// ```
    pub fn shuffle(&self, hash_columns: &[i32], config: Option<GpuConfig>) -> CylonResult<Self> {
        let config = config.unwrap_or_default();
        let mut output = std::ptr::null_mut();

        let status = unsafe {
            ffi::gcylon_shuffle(
                self.ptr,
                hash_columns.as_ptr(),
                hash_columns.len() as i32,
                &mut output,
                config.as_ptr(),
            )
        };

        check_status(status)?;
        Ok(Self { ptr: output, ctx: self.ctx.clone() })
    }

    /// AllGather - collect table from all workers.
    ///
    /// Each worker receives a copy of the combined table from all workers.
    ///
    /// # Arguments
    /// * `config` - Optional configuration (None for defaults)
    pub fn allgather(&self, config: Option<GpuConfig>) -> CylonResult<Self> {
        let config = config.unwrap_or_default();
        let mut output = std::ptr::null_mut();

        let status = unsafe {
            ffi::gcylon_allgather(self.ptr, &mut output, config.as_ptr())
        };

        check_status(status)?;
        Ok(Self { ptr: output, ctx: self.ctx.clone() })
    }

    /// Gather table to a single root worker.
    ///
    /// Only the root worker receives the combined table.
    ///
    /// # Arguments
    /// * `root` - Rank of the root worker
    /// * `config` - Optional configuration (None for defaults)
    pub fn gather(&self, root: i32, config: Option<GpuConfig>) -> CylonResult<Self> {
        let config = config.unwrap_or_default();
        let mut output = std::ptr::null_mut();

        let status = unsafe {
            ffi::gcylon_gather(self.ptr, root, &mut output, config.as_ptr())
        };

        check_status(status)?;
        Ok(Self { ptr: output, ctx: self.ctx.clone() })
    }

    /// Broadcast table from root to all workers.
    ///
    /// # Arguments
    /// * `root` - Rank of the root worker that has the data
    /// * `config` - Optional configuration (None for defaults)
    pub fn broadcast(&self, root: i32, config: Option<GpuConfig>) -> CylonResult<Self> {
        let config = config.unwrap_or_default();
        let mut output = std::ptr::null_mut();

        let status = unsafe {
            ffi::gcylon_broadcast(self.ptr, root, &mut output, config.as_ptr())
        };

        check_status(status)?;
        Ok(Self { ptr: output, ctx: self.ctx.clone() })
    }

    /// Distributed join with another table.
    ///
    /// Shuffles both tables by join keys, then performs local join.
    ///
    /// # Arguments
    /// * `right` - Right table to join with
    /// * `left_columns` - Column indices from left table for join
    /// * `right_columns` - Column indices from right table for join
    /// * `join_type` - Type of join (Inner, Left, Right, Outer)
    /// * `config` - Optional configuration (None for defaults)
    pub fn distributed_join(
        &self,
        right: &GpuTable,
        left_columns: &[i32],
        right_columns: &[i32],
        join_type: JoinType,
        config: Option<GpuConfig>,
    ) -> CylonResult<Self> {
        let config = config.unwrap_or_default();
        let mut output = std::ptr::null_mut();

        let status = unsafe {
            ffi::gcylon_distributed_join(
                self.ptr,
                right.ptr,
                left_columns.as_ptr(),
                left_columns.len() as i32,
                right_columns.as_ptr(),
                right_columns.len() as i32,
                join_type.into(),
                &mut output,
                config.as_ptr(),
            )
        };

        check_status(status)?;
        Ok(Self { ptr: output, ctx: self.ctx.clone() })
    }

    /// Distributed sort.
    ///
    /// # Arguments
    /// * `sort_columns` - Column indices to sort by
    /// * `ascending` - Sort order for each column (true=ascending, false=descending)
    /// * `config` - Optional configuration (None for defaults)
    pub fn distributed_sort(
        &self,
        sort_columns: &[i32],
        ascending: &[bool],
        config: Option<GpuConfig>,
    ) -> CylonResult<Self> {
        let config = config.unwrap_or_default();
        let mut output = std::ptr::null_mut();

        // Convert bool slice to i32 slice
        let asc: Vec<i32> = ascending.iter().map(|&b| if b { 1 } else { 0 }).collect();

        let status = unsafe {
            ffi::gcylon_distributed_sort(
                self.ptr,
                sort_columns.as_ptr(),
                sort_columns.len() as i32,
                asc.as_ptr(),
                &mut output,
                config.as_ptr(),
            )
        };

        check_status(status)?;
        Ok(Self { ptr: output, ctx: self.ctx.clone() })
    }

    /// Repartition table across workers.
    ///
    /// # Arguments
    /// * `rows_per_worker` - Target rows per worker (None for even distribution)
    /// * `config` - Optional configuration (None for defaults)
    pub fn repartition(
        &self,
        rows_per_worker: Option<&[i32]>,
        config: Option<GpuConfig>,
    ) -> CylonResult<Self> {
        let config = config.unwrap_or_default();
        let mut output = std::ptr::null_mut();

        let (ptr, len) = match rows_per_worker {
            Some(r) => (r.as_ptr(), r.len() as i32),
            None => (std::ptr::null(), 0),
        };

        let status = unsafe {
            ffi::gcylon_repartition(self.ptr, ptr, len, &mut output, config.as_ptr())
        };

        check_status(status)?;
        Ok(Self { ptr: output, ctx: self.ctx.clone() })
    }

    /// Get the raw FFI pointer.
    pub(crate) fn as_ptr(&self) -> *mut ffi::GcylonTable {
        self.ptr
    }
}

impl Drop for GpuTable {
    fn drop(&mut self) {
        unsafe { ffi::gcylon_table_free(self.ptr) };
    }
}

// Safety: GcylonTable handles are thread-safe
unsafe impl Send for GpuTable {}
unsafe impl Sync for GpuTable {}

/// Join type for distributed joins.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    /// Inner join - only matching rows
    Inner,
    /// Left join - all left rows, matching right rows
    Left,
    /// Right join - matching left rows, all right rows
    Right,
    /// Outer join - all rows from both tables
    Outer,
}

impl From<JoinType> for ffi::GcylonJoinType {
    fn from(jt: JoinType) -> Self {
        match jt {
            JoinType::Inner => ffi::GcylonJoinType::Inner,
            JoinType::Left => ffi::GcylonJoinType::Left,
            JoinType::Right => ffi::GcylonJoinType::Right,
            JoinType::Outer => ffi::GcylonJoinType::Outer,
        }
    }
}

/// Check FFI status and convert to Result.
fn check_status(status: ffi::GcylonStatus) -> CylonResult<()> {
    if status == ffi::GCYLON_OK {
        return Ok(());
    }

    let msg = unsafe {
        let ptr = ffi::gcylon_get_last_error();
        if ptr.is_null() {
            "Unknown error".to_string()
        } else {
            std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };

    let code = match status {
        ffi::GCYLON_OOM => Code::OutOfMemory,
        ffi::GCYLON_INVALID_ARG => Code::Invalid,
        _ => Code::ExecutionError,
    };

    Err(CylonError::new(code, msg))
}
