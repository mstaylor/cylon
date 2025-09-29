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

//! Error handling for Cylon operations
//!
//! Ported from cpp/src/cylon/status.hpp and cpp/src/cylon/code.hpp

use std::fmt;

/// Error codes corresponding to the C++ Code enum from cpp/src/cylon/code.hpp
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Code {
    Ok = 0,
    OutOfMemory = 1,
    KeyError = 2,
    TypeError = 3,
    Invalid = 4,
    IoError = 5,
    CapacityError = 6,
    IndexError = 7,
    UnknownError = 9,
    NotImplemented = 10,
    SerializationError = 11,
    GpuMemoryError = 12,
    RError = 13,
    ValueError = 14,
    CodeGenError = 40,
    ExpressionValidationError = 41,
    ExecutionError = 42,
    AlreadyExists = 45,
}

impl fmt::Display for Code {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Code::Ok => write!(f, "OK"),
            Code::OutOfMemory => write!(f, "Out of memory"),
            Code::KeyError => write!(f, "Key error"),
            Code::TypeError => write!(f, "Type error"),
            Code::Invalid => write!(f, "Invalid"),
            Code::IoError => write!(f, "IO error"),
            Code::CapacityError => write!(f, "Capacity error"),
            Code::IndexError => write!(f, "Index error"),
            Code::UnknownError => write!(f, "Unknown error"),
            Code::NotImplemented => write!(f, "Not implemented"),
            Code::SerializationError => write!(f, "Serialization error"),
            Code::GpuMemoryError => write!(f, "GPU memory error"),
            Code::RError => write!(f, "R error"),
            Code::ValueError => write!(f, "Value error"),
            Code::CodeGenError => write!(f, "Code generation error"),
            Code::ExpressionValidationError => write!(f, "Expression validation error"),
            Code::ExecutionError => write!(f, "Execution error"),
            Code::AlreadyExists => write!(f, "Already exists"),
        }
    }
}

/// Main error type for Cylon operations
#[derive(thiserror::Error, Debug)]
pub enum CylonError {
    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Invalid operation: {0}")]
    Invalid(String),

    #[error("Not implemented: {0}")]
    NotImplemented(String),

    #[error("Index out of bounds: {0}")]
    IndexError(String),

    #[error("Type error: {0}")]
    TypeError(String),

    #[error("Key not found: {0}")]
    KeyError(String),

    #[error("Out of memory")]
    OutOfMemory,

    #[error("Network error: {0}")]
    Network(String),

    #[error("Communication error: {0}")]
    Communication(String),

    #[error("Generic error with code {code}: {message}")]
    Generic { code: Code, message: String },
}

impl CylonError {
    /// Create a new error with a specific code and message
    pub fn new(code: Code, message: impl Into<String>) -> Self {
        CylonError::Generic {
            code,
            message: message.into(),
        }
    }

    /// Get the error code
    pub fn code(&self) -> Code {
        match self {
            CylonError::Arrow(_) => Code::Invalid,
            CylonError::Io(_) => Code::IoError,
            CylonError::Serialization(_) => Code::SerializationError,
            CylonError::Invalid(_) => Code::Invalid,
            CylonError::NotImplemented(_) => Code::NotImplemented,
            CylonError::IndexError(_) => Code::IndexError,
            CylonError::TypeError(_) => Code::TypeError,
            CylonError::KeyError(_) => Code::KeyError,
            CylonError::OutOfMemory => Code::OutOfMemory,
            CylonError::Network(_) => Code::IoError,
            CylonError::Communication(_) => Code::IoError,
            CylonError::Generic { code, .. } => *code,
        }
    }

    /// Check if the error represents success
    pub fn is_ok(&self) -> bool {
        self.code() == Code::Ok
    }
}

/// Type alias for Results using CylonError
pub type CylonResult<T> = Result<T, CylonError>;

/// Status type matching the C++ Status class from cpp/src/cylon/status.hpp
#[derive(Debug, Clone)]
pub struct Status {
    code: Code,
    message: String,
}

impl Status {
    /// Create a new status with code and message
    pub fn new(code: Code, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    /// Create a success status
    pub fn ok() -> Self {
        Self {
            code: Code::Ok,
            message: String::new(),
        }
    }

    /// Get the status code (equivalent to get_code() in C++)
    pub fn get_code(&self) -> Code {
        self.code
    }

    /// Get the status message (equivalent to get_msg() in C++)
    pub fn get_msg(&self) -> &str {
        &self.message
    }

    /// Check if the status is OK (equivalent to is_ok() in C++)
    pub fn is_ok(&self) -> bool {
        self.code == Code::Ok
    }
}

impl From<CylonError> for Status {
    fn from(error: CylonError) -> Self {
        Self {
            code: error.code(),
            message: error.to_string(),
        }
    }
}

impl From<Status> for CylonResult<()> {
    fn from(status: Status) -> Self {
        if status.is_ok() {
            Ok(())
        } else {
            Err(CylonError::new(status.get_code(), status.get_msg()))
        }
    }
}