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

//! Libfabric error handling
//!
//! This module provides error conversion and handling for libfabric operations.

use std::ffi::CStr;
use crate::error::{CylonError, CylonResult, Code};
use super::libfabric_sys;

/// Convert a libfabric error code to a human-readable string
pub fn fi_strerror(errnum: i32) -> String {
    unsafe {
        let ptr = libfabric_sys::fi_strerror(-errnum);
        if ptr.is_null() {
            format!("Unknown libfabric error: {}", errnum)
        } else {
            CStr::from_ptr(ptr)
                .to_string_lossy()
                .into_owned()
        }
    }
}

/// Check a libfabric return value and convert to CylonResult
pub fn check_fi_error(ret: i32, context: &str) -> CylonResult<()> {
    if ret == 0 {
        Ok(())
    } else {
        Err(CylonError::new(
            Code::ExecutionError,
            format!("{}: {} (error code: {})", context, fi_strerror(ret), ret),
        ))
    }
}

/// Check a libfabric return value (ssize_t) and convert to CylonResult
pub fn check_fi_error_ssize(ret: isize, context: &str) -> CylonResult<usize> {
    if ret >= 0 {
        Ok(ret as usize)
    } else {
        let err = -(ret as i32);
        Err(CylonError::new(
            Code::ExecutionError,
            format!("{}: {} (error code: {})", context, fi_strerror(err), err),
        ))
    }
}

/// Check if the error is EAGAIN (would block, try again)
pub fn is_eagain(ret: isize) -> bool {
    ret == -(libfabric_sys::FI_EAGAIN as isize)
}

/// Libfabric-specific error type
#[derive(Debug, Clone)]
pub struct LibfabricError {
    pub code: i32,
    pub message: String,
    pub context: String,
}

impl LibfabricError {
    pub fn new(code: i32, context: &str) -> Self {
        Self {
            code,
            message: fi_strerror(code),
            context: context.to_string(),
        }
    }

    pub fn is_eagain(&self) -> bool {
        self.code == libfabric_sys::FI_EAGAIN as i32
    }
}

impl std::fmt::Display for LibfabricError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {} (error code: {})", self.context, self.message, self.code)
    }
}

impl std::error::Error for LibfabricError {}

impl From<LibfabricError> for CylonError {
    fn from(err: LibfabricError) -> Self {
        CylonError::new(Code::ExecutionError, err.to_string())
    }
}

/// Macro to check libfabric return values
#[macro_export]
macro_rules! fi_check {
    ($expr:expr, $context:expr) => {
        {
            let ret = $expr;
            if ret != 0 {
                return Err($crate::error::CylonError::new(
                    $crate::error::Code::ExecutionError,
                    format!("{}: {} (error code: {})",
                        $context,
                        $crate::net::libfabric::error::fi_strerror(ret),
                        ret),
                ));
            }
        }
    };
}

/// Macro to check libfabric ssize_t return values
#[macro_export]
macro_rules! fi_check_ssize {
    ($expr:expr, $context:expr) => {
        {
            let ret = $expr;
            if ret < 0 {
                let err = -(ret as i32);
                return Err($crate::error::CylonError::new(
                    $crate::error::Code::ExecutionError,
                    format!("{}: {} (error code: {})",
                        $context,
                        $crate::net::libfabric::error::fi_strerror(err),
                        err),
                ));
            }
            ret as usize
        }
    };
}

pub use fi_check;
pub use fi_check_ssize;
