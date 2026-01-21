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

//! Cylon WASM - WebAssembly build of Cylon DataFrame operations
//!
//! This crate provides WASM-compatible implementations of core Cylon operations:
//! - Join operations (inner, left, right, full outer)
//! - GroupBy with aggregations
//! - Filter operations
//! - SIMD-optimized computations
//!
//! # Architecture
//!
//! The crate uses Arrow-rs internally for data representation, providing code
//! compatibility with native Cylon while exposing a simple JSON-based API for
//! JavaScript and Python consumers.
//!
//! # Usage from JavaScript
//!
//! ```javascript
//! import init, { join_tables, groupby, filter } from 'cylon-wasm';
//!
//! await init();
//!
//! // Join two tables
//! const result = join_tables(leftData, rightData, {
//!     join_type: 'inner',
//!     left_on: ['id'],
//!     right_on: ['id']
//! });
//! ```

pub mod error;
pub mod simd;
pub mod table;
pub mod join;
pub mod groupby;
pub mod filter;
pub mod ops;
pub mod api;
pub mod imports;
pub mod distributed;

use wasm_bindgen::prelude::*;

/// Initialize the WASM module with panic hook for better error messages
#[wasm_bindgen(start)]
pub fn init() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// Get the version of cylon-wasm
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Check if SIMD is available (compile-time feature)
#[wasm_bindgen]
pub fn simd_available() -> bool {
    cfg!(feature = "simd")
}
