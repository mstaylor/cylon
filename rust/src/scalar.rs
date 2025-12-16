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

//! Scalar values
//!
//! Ported from cpp/src/cylon/scalar.hpp and cpp/src/cylon/scalar.cpp
//!
//! In arrow-rs, scalars are represented using `arrow::array::Scalar<T>` which
//! wraps a single-element array. This differs from the C++ Arrow implementation
//! that uses `arrow::Scalar`.

use std::sync::Arc;
use arrow::array::{ArrayRef, Array, Int8Array, Int16Array, Int32Array, Int64Array};
use arrow::array::{UInt8Array, UInt16Array, UInt32Array, UInt64Array};
use arrow::array::{Float32Array, Float64Array, BooleanArray};

use crate::data_types::DataType;
use crate::arrow::arrow_types::to_cylon_type;

/// Scalar wrapper around Arrow Scalar values
/// Corresponds to C++ Scalar class from cpp/src/cylon/scalar.hpp
///
/// In Rust, we use a single-element ArrayRef to store the scalar value,
/// similar to how arrow-rs represents scalars.
pub struct Scalar {
    /// The datatype of the scalar
    type_: Arc<DataType>,
    /// The scalar data stored as a single-element array
    data_: ArrayRef,
}

impl Scalar {
    /// Create a new Scalar from an Arrow ArrayRef containing a single element
    /// Corresponds to C++ Scalar::Scalar(std::shared_ptr<arrow::Scalar> data)
    pub fn new(data: ArrayRef) -> Self {
        debug_assert!(data.len() == 1, "Scalar array must have exactly one element");
        let type_ = to_cylon_type(data.data_type())
            .unwrap_or_else(|| Arc::new(DataType::new(crate::data_types::Type::MaxId)));
        Self { type_, data_: data }
    }

    /// Return the data type of the scalar
    /// Corresponds to C++ Scalar::type()
    pub fn data_type(&self) -> &Arc<DataType> {
        &self.type_
    }

    /// Return the data wrapped by scalar as an ArrayRef
    /// Corresponds to C++ Scalar::data()
    pub fn data(&self) -> &ArrayRef {
        &self.data_
    }

    /// Factory method to create a Scalar
    /// Corresponds to C++ Scalar::Make(std::shared_ptr<arrow::Scalar> data)
    pub fn make(data: ArrayRef) -> Arc<Scalar> {
        Arc::new(Scalar::new(data))
    }

    /// Check if the scalar value is null
    pub fn is_null(&self) -> bool {
        self.data_.is_null(0)
    }

    /// Check if the scalar value is valid (not null)
    pub fn is_valid(&self) -> bool {
        self.data_.is_valid(0)
    }
}

/// Helper functions for creating scalars from primitive values
impl Scalar {
    /// Create an Int8 scalar
    pub fn int8(value: i8) -> Arc<Scalar> {
        let array = Int8Array::from(vec![value]);
        Scalar::make(Arc::new(array))
    }

    /// Create an Int16 scalar
    pub fn int16(value: i16) -> Arc<Scalar> {
        let array = Int16Array::from(vec![value]);
        Scalar::make(Arc::new(array))
    }

    /// Create an Int32 scalar
    pub fn int32(value: i32) -> Arc<Scalar> {
        let array = Int32Array::from(vec![value]);
        Scalar::make(Arc::new(array))
    }

    /// Create an Int64 scalar
    pub fn int64(value: i64) -> Arc<Scalar> {
        let array = Int64Array::from(vec![value]);
        Scalar::make(Arc::new(array))
    }

    /// Create a UInt8 scalar
    pub fn uint8(value: u8) -> Arc<Scalar> {
        let array = UInt8Array::from(vec![value]);
        Scalar::make(Arc::new(array))
    }

    /// Create a UInt16 scalar
    pub fn uint16(value: u16) -> Arc<Scalar> {
        let array = UInt16Array::from(vec![value]);
        Scalar::make(Arc::new(array))
    }

    /// Create a UInt32 scalar
    pub fn uint32(value: u32) -> Arc<Scalar> {
        let array = UInt32Array::from(vec![value]);
        Scalar::make(Arc::new(array))
    }

    /// Create a UInt64 scalar
    pub fn uint64(value: u64) -> Arc<Scalar> {
        let array = UInt64Array::from(vec![value]);
        Scalar::make(Arc::new(array))
    }

    /// Create a Float32 scalar
    pub fn float32(value: f32) -> Arc<Scalar> {
        let array = Float32Array::from(vec![value]);
        Scalar::make(Arc::new(array))
    }

    /// Create a Float64 scalar
    pub fn float64(value: f64) -> Arc<Scalar> {
        let array = Float64Array::from(vec![value]);
        Scalar::make(Arc::new(array))
    }

    /// Create a Boolean scalar
    pub fn boolean(value: bool) -> Arc<Scalar> {
        let array = BooleanArray::from(vec![value]);
        Scalar::make(Arc::new(array))
    }
}
