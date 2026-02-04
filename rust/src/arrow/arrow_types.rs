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

//! Arrow type conversion utilities
//!
//! Ported from cpp/src/cylon/arrow/arrow_types.hpp and arrow_types.cpp
//!
//! Provides conversion functions between Arrow data types and Cylon data types.

use std::sync::Arc;
use arrow::datatypes::DataType as ArrowDataType;
use arrow::datatypes::TimeUnit as ArrowTimeUnit;

use crate::data_types::{
    DataType, Type, TimeUnit,
    bool_type, int8, uint8, int16, uint16, int32, uint32, int64, uint64,
    half_float, float, double, string, binary, large_string, large_binary,
    date32, date64, time32, time64,
};

/// Convert Arrow time unit to Cylon time unit
/// Corresponds to C++ tarrow::ToCylonTimeUnit
pub fn to_cylon_time_unit(a_time_unit: ArrowTimeUnit) -> TimeUnit {
    match a_time_unit {
        ArrowTimeUnit::Second => TimeUnit::Second,
        ArrowTimeUnit::Millisecond => TimeUnit::Milli,
        ArrowTimeUnit::Microsecond => TimeUnit::Micro,
        ArrowTimeUnit::Nanosecond => TimeUnit::Nano,
    }
}

/// Convert Cylon time unit to Arrow time unit
/// Corresponds to C++ tarrow::ToArrowTimeUnit
pub fn to_arrow_time_unit(time_unit: TimeUnit) -> ArrowTimeUnit {
    match time_unit {
        TimeUnit::Second => ArrowTimeUnit::Second,
        TimeUnit::Milli => ArrowTimeUnit::Millisecond,
        TimeUnit::Micro => ArrowTimeUnit::Microsecond,
        TimeUnit::Nano => ArrowTimeUnit::Nanosecond,
    }
}

/// Convert Arrow data type to Cylon data type
/// Corresponds to C++ tarrow::ToCylonType
///
/// Returns None for unsupported types.
pub fn to_cylon_type(a_type: &ArrowDataType) -> Option<Arc<DataType>> {
    match a_type {
        ArrowDataType::Boolean => Some(bool_type()),
        ArrowDataType::UInt8 => Some(uint8()),
        ArrowDataType::Int8 => Some(int8()),
        ArrowDataType::UInt16 => Some(uint16()),
        ArrowDataType::Int16 => Some(int16()),
        ArrowDataType::UInt32 => Some(uint32()),
        ArrowDataType::Int32 => Some(int32()),
        ArrowDataType::UInt64 => Some(uint64()),
        ArrowDataType::Int64 => Some(int64()),
        ArrowDataType::Float16 => Some(half_float()),
        ArrowDataType::Float32 => Some(float()),
        ArrowDataType::Float64 => Some(double()),
        ArrowDataType::FixedSizeBinary(byte_width) => {
            Some(Arc::new(DataType::new(Type::FixedSizeBinary)))
        }
        ArrowDataType::Binary => Some(binary()),
        ArrowDataType::Utf8 => Some(string()),
        ArrowDataType::LargeUtf8 => Some(large_string()),
        ArrowDataType::LargeBinary => Some(large_binary()),
        ArrowDataType::Date32 => Some(date32()),
        ArrowDataType::Date64 => Some(date64()),
        ArrowDataType::Timestamp(_, _) => {
            // TimestampType is a separate struct, return base DataType
            Some(Arc::new(DataType::new(Type::Timestamp)))
        }
        ArrowDataType::Time32(_) => Some(time32()),
        ArrowDataType::Time64(_) => Some(time64()),
        ArrowDataType::Decimal128(_, _) => {
            // DecimalType is a separate struct, return base DataType
            Some(Arc::new(DataType::new(Type::Decimal)))
        }
        ArrowDataType::Decimal256(_, _) => {
            // DecimalType is a separate struct, return base DataType
            Some(Arc::new(DataType::new(Type::Decimal)))
        }
        // Unsupported types return None
        _ => None,
    }
}

/// Convert Arrow data type ID to Cylon Type enum
/// Corresponds to C++ tarrow::ToCylonTypeId
pub fn to_cylon_type_id(a_type: &ArrowDataType) -> Type {
    match a_type {
        ArrowDataType::Boolean => Type::Bool,
        ArrowDataType::UInt8 => Type::UInt8,
        ArrowDataType::Int8 => Type::Int8,
        ArrowDataType::UInt16 => Type::UInt16,
        ArrowDataType::Int16 => Type::Int16,
        ArrowDataType::UInt32 => Type::UInt32,
        ArrowDataType::Int32 => Type::Int32,
        ArrowDataType::UInt64 => Type::UInt64,
        ArrowDataType::Int64 => Type::Int64,
        ArrowDataType::Float16 => Type::HalfFloat,
        ArrowDataType::Float32 => Type::Float,
        ArrowDataType::Float64 => Type::Double,
        ArrowDataType::Utf8 => Type::String,
        ArrowDataType::Binary => Type::Binary,
        ArrowDataType::FixedSizeBinary(_) => Type::FixedSizeBinary,
        ArrowDataType::Date32 => Type::Date32,
        ArrowDataType::Date64 => Type::Date64,
        ArrowDataType::Timestamp(_, _) => Type::Timestamp,
        ArrowDataType::Time32(_) => Type::Time32,
        ArrowDataType::Time64(_) => Type::Time64,
        ArrowDataType::LargeUtf8 => Type::LargeString,
        ArrowDataType::LargeBinary => Type::LargeBinary,
        _ => Type::MaxId,
    }
}

/// Convert Cylon data type to Arrow data type
/// Corresponds to C++ tarrow::ToArrowType
pub fn to_arrow_type(cylon_type: &DataType) -> Option<ArrowDataType> {
    match cylon_type.get_type() {
        Type::Bool => Some(ArrowDataType::Boolean),
        Type::UInt8 => Some(ArrowDataType::UInt8),
        Type::Int8 => Some(ArrowDataType::Int8),
        Type::UInt16 => Some(ArrowDataType::UInt16),
        Type::Int16 => Some(ArrowDataType::Int16),
        Type::UInt32 => Some(ArrowDataType::UInt32),
        Type::Int32 => Some(ArrowDataType::Int32),
        Type::UInt64 => Some(ArrowDataType::UInt64),
        Type::Int64 => Some(ArrowDataType::Int64),
        Type::HalfFloat => Some(ArrowDataType::Float16),
        Type::Float => Some(ArrowDataType::Float32),
        Type::Double => Some(ArrowDataType::Float64),
        Type::String => Some(ArrowDataType::Utf8),
        Type::Binary => Some(ArrowDataType::Binary),
        Type::Date32 => Some(ArrowDataType::Date32),
        Type::Date64 => Some(ArrowDataType::Date64),
        Type::Time32 => Some(ArrowDataType::Time32(ArrowTimeUnit::Millisecond)),
        Type::Time64 => Some(ArrowDataType::Time64(ArrowTimeUnit::Microsecond)),
        Type::LargeString => Some(ArrowDataType::LargeUtf8),
        Type::LargeBinary => Some(ArrowDataType::LargeBinary),
        // Types that require additional parameters aren't fully supported in this direction
        Type::FixedSizeBinary | Type::Timestamp | Type::Duration | Type::Decimal |
        Type::Interval | Type::List | Type::FixedSizeList | Type::Extension | Type::MaxId => None,
    }
}
