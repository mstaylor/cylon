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
//! Ported from cpp/src/cylon/column.hpp and cpp/src/cylon/column.cpp

use std::sync::Arc;
use arrow::array::{ArrayRef, Array};

use crate::data_types::DataType;
use crate::arrow::arrow_types::to_cylon_type;

/// Column wrapper around Arrow arrays
/// Corresponds to C++ Column class from cpp/src/cylon/column.hpp
pub struct Column {
    /// The datatype of the column
    type_: Arc<DataType>,
    /// Pointer to the data array
    data_: ArrayRef,
}

impl Column {
    /// Create a new Column from an Arrow Array
    /// Corresponds to C++ Column::Column(std::shared_ptr<arrow::Array> data)
    pub fn new(data: ArrayRef) -> Self {
        let type_ = to_cylon_type(data.data_type())
            .unwrap_or_else(|| Arc::new(DataType::new(crate::data_types::Type::MaxId)));
        Self { type_, data_: data }
    }

    /// Return the data wrapped by column
    /// Corresponds to C++ Column::data()
    pub fn data(&self) -> &ArrayRef {
        &self.data_
    }

    /// Return the data type of the column
    /// Corresponds to C++ Column::type()
    pub fn data_type(&self) -> &Arc<DataType> {
        &self.type_
    }

    /// Return the length of the column
    /// Corresponds to C++ Column::length()
    pub fn length(&self) -> i64 {
        self.data_.len() as i64
    }

    /// Factory method to create a Column
    /// Corresponds to C++ Column::Make(std::shared_ptr<arrow::Array> data_)
    pub fn make(data: ArrayRef) -> Arc<Column> {
        Arc::new(Column::new(data))
    }

    /// Alias for data() to match existing code
    pub fn array(&self) -> &ArrayRef {
        &self.data_
    }
}

/// Trait for creating columns from vectors of primitive types
/// Corresponds to C++ Column::FromVector template
pub trait FromVector<T> {
    fn from_vector(data: &[T]) -> Arc<Column>;
}

impl FromVector<i8> for Column {
    fn from_vector(data: &[i8]) -> Arc<Column> {
        let array = arrow::array::Int8Array::from(data.to_vec());
        Column::make(Arc::new(array))
    }
}

impl FromVector<i16> for Column {
    fn from_vector(data: &[i16]) -> Arc<Column> {
        let array = arrow::array::Int16Array::from(data.to_vec());
        Column::make(Arc::new(array))
    }
}

impl FromVector<i32> for Column {
    fn from_vector(data: &[i32]) -> Arc<Column> {
        let array = arrow::array::Int32Array::from(data.to_vec());
        Column::make(Arc::new(array))
    }
}

impl FromVector<i64> for Column {
    fn from_vector(data: &[i64]) -> Arc<Column> {
        let array = arrow::array::Int64Array::from(data.to_vec());
        Column::make(Arc::new(array))
    }
}

impl FromVector<u8> for Column {
    fn from_vector(data: &[u8]) -> Arc<Column> {
        let array = arrow::array::UInt8Array::from(data.to_vec());
        Column::make(Arc::new(array))
    }
}

impl FromVector<u16> for Column {
    fn from_vector(data: &[u16]) -> Arc<Column> {
        let array = arrow::array::UInt16Array::from(data.to_vec());
        Column::make(Arc::new(array))
    }
}

impl FromVector<u32> for Column {
    fn from_vector(data: &[u32]) -> Arc<Column> {
        let array = arrow::array::UInt32Array::from(data.to_vec());
        Column::make(Arc::new(array))
    }
}

impl FromVector<u64> for Column {
    fn from_vector(data: &[u64]) -> Arc<Column> {
        let array = arrow::array::UInt64Array::from(data.to_vec());
        Column::make(Arc::new(array))
    }
}

impl FromVector<f32> for Column {
    fn from_vector(data: &[f32]) -> Arc<Column> {
        let array = arrow::array::Float32Array::from(data.to_vec());
        Column::make(Arc::new(array))
    }
}

impl FromVector<f64> for Column {
    fn from_vector(data: &[f64]) -> Arc<Column> {
        let array = arrow::array::Float64Array::from(data.to_vec());
        Column::make(Arc::new(array))
    }
}
