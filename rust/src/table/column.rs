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

//! Column type
//!
//! Ported from cpp/src/cylon/column.hpp

use arrow::array::ArrayRef;

/// Column wrapper around Arrow arrays
/// Corresponds to C++ Column class
pub struct Column {
    array: ArrayRef,
}

impl Column {
    pub fn new(array: ArrayRef) -> Self {
        Self { array }
    }

    pub fn array(&self) -> &ArrayRef {
        &self.array
    }
}

// TODO: Port full Column implementation from cpp/src/cylon/column.hpp