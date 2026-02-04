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

//! Error types for Cylon WASM operations

use std::fmt;
use wasm_bindgen::prelude::*;

/// Error codes matching native Cylon error codes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCode {
    /// Invalid input or configuration
    Invalid,
    /// Type mismatch error
    TypeError,
    /// Index out of bounds
    IndexError,
    /// Execution error
    ExecutionError,
    /// Unsupported operation
    Unsupported,
    /// Arrow error
    ArrowError,
    /// Serialization/deserialization error
    SerdeError,
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorCode::Invalid => write!(f, "Invalid"),
            ErrorCode::TypeError => write!(f, "TypeError"),
            ErrorCode::IndexError => write!(f, "IndexError"),
            ErrorCode::ExecutionError => write!(f, "ExecutionError"),
            ErrorCode::Unsupported => write!(f, "Unsupported"),
            ErrorCode::ArrowError => write!(f, "ArrowError"),
            ErrorCode::SerdeError => write!(f, "SerdeError"),
        }
    }
}

/// WASM-compatible error type
#[derive(Debug)]
pub struct WasmError {
    code: ErrorCode,
    message: String,
}

impl WasmError {
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    pub fn invalid(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::Invalid, message)
    }

    pub fn type_error(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::TypeError, message)
    }

    pub fn index_error(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::IndexError, message)
    }

    pub fn execution_error(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::ExecutionError, message)
    }

    pub fn unsupported(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::Unsupported, message)
    }

    pub fn arrow_error(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::ArrowError, message)
    }

    pub fn serde_error(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::SerdeError, message)
    }

    pub fn code(&self) -> ErrorCode {
        self.code
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for WasmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for WasmError {}

impl From<WasmError> for JsValue {
    fn from(err: WasmError) -> Self {
        JsValue::from_str(&format!("{}", err))
    }
}

impl From<arrow::error::ArrowError> for WasmError {
    fn from(err: arrow::error::ArrowError) -> Self {
        WasmError::arrow_error(err.to_string())
    }
}

impl From<serde_json::Error> for WasmError {
    fn from(err: serde_json::Error) -> Self {
        WasmError::serde_error(err.to_string())
    }
}

/// Result type for WASM operations
pub type WasmResult<T> = Result<T, WasmError>;
