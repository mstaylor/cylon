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

//! GPU-accelerated operations via gcylon FFI
//!
//! This module provides Rust bindings to gcylon (GPU Cylon) for
//! accelerated distributed operations on NVIDIA GPUs.
//!
//! # Requirements
//! - NVIDIA GPU with CUDA support
//! - cuDF library installed
//! - gcylon library with C API
//!
//! # Example
//! ```ignore
//! use cylon::gpu::{GpuContext, GpuTable, GpuConfig, set_device, get_device_count};
//!
//! // Initialize GPU device
//! let num_gpus = get_device_count()?;
//! set_device(0)?;
//!
//! let ctx = GpuContext::new_mpi()?;
//! let config = GpuConfig::default();
//!
//! // Create GPU table and perform distributed shuffle
//! let shuffled = gpu_table.shuffle(&[0], Some(config))?;
//! ```

mod ffi;
mod config;
mod context;
mod table;

pub use config::GpuConfig;
pub use context::{GpuContext, GpuMemoryInfo};
pub use table::{GpuTable, JoinType};

use crate::error::{CylonError, CylonResult, Code};

/// Get the number of available CUDA devices.
pub fn get_device_count() -> CylonResult<i32> {
    let mut count = 0;
    let status = unsafe { ffi::gcylon_get_device_count(&mut count) };
    if status != ffi::GCYLON_OK {
        return Err(CylonError::new(
            Code::ExecutionError,
            "Failed to get CUDA device count",
        ));
    }
    Ok(count)
}

/// Set the current CUDA device.
///
/// This should be called before creating any GPU tables or contexts.
/// In MPI applications, typically set to `rank % num_gpus`.
pub fn set_device(device_id: i32) -> CylonResult<()> {
    let status = unsafe { ffi::gcylon_set_device(device_id) };
    if status != ffi::GCYLON_OK {
        return Err(CylonError::new(
            Code::ExecutionError,
            format!("Failed to set CUDA device {}", device_id),
        ));
    }
    Ok(())
}

/// Get the current CUDA device.
pub fn get_device() -> CylonResult<i32> {
    let mut device_id = -1;
    let status = unsafe { ffi::gcylon_get_device(&mut device_id) };
    if status != ffi::GCYLON_OK {
        return Err(CylonError::new(
            Code::ExecutionError,
            "Failed to get current CUDA device",
        ));
    }
    Ok(device_id)
}
