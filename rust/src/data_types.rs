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

//! Data types for Cylon
//!
//! Ported from cpp/src/cylon/data_types.hpp
//! The types are a strip down from arrow types

use std::sync::Arc;

/// The data type enum corresponding to C++ Type::type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Type {
    /// Boolean as 1 bit, LSB bit-packed ordering
    Bool,
    /// Unsigned 8-bit little-endian integer
    UInt8,
    /// Signed 8-bit little-endian integer
    Int8,
    /// Unsigned 16-bit little-endian integer
    UInt16,
    /// Signed 16-bit little-endian integer
    Int16,
    /// Unsigned 32-bit little-endian integer
    UInt32,
    /// Signed 32-bit little-endian integer
    Int32,
    /// Unsigned 64-bit little-endian integer
    UInt64,
    /// Signed 64-bit little-endian integer
    Int64,
    /// 2-byte floating point value
    HalfFloat,
    /// 4-byte floating point value
    Float,
    /// 8-byte floating point value
    Double,
    /// UTF8 variable-length string as List<Char>
    String,
    /// Variable-length bytes (no guarantee of UTF8-ness)
    Binary,
    /// Fixed-size binary. Each value occupies the same number of bytes
    FixedSizeBinary,
    /// int32_t days since the UNIX epoch
    Date32,
    /// int64_t milliseconds since the UNIX epoch
    Date64,
    /// Exact timestamp encoded with int64 since UNIX epoch
    /// Default unit millisecond
    Timestamp,
    /// Time as signed 32-bit integer, representing either seconds or
    /// milliseconds since midnight
    Time32,
    /// Time as signed 64-bit integer, representing either microseconds or
    /// nanoseconds since midnight
    Time64,
    /// YEAR_MONTH or DAY_TIME interval in SQL style
    Interval,
    /// Precision- and scale-based decimal type. Storage type depends on the
    /// parameters.
    Decimal,
    /// A list of some logical data type
    List,
    /// Custom data type, implemented by user
    Extension,
    /// Fixed size list of some logical type
    FixedSizeList,
    /// Duration in various time units
    Duration,
    /// Like STRING, but with 64-bit offsets
    LargeString,
    /// Like BINARY, but with 64-bit offsets
    LargeBinary,
    /// For unsupported types
    MaxId,
}

/// The layout of the data type corresponding to C++ Layout::layout
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    FixedWidth = 1,
    VariableWidth = 2,
}

/// Time unit enum corresponding to C++ TimeUnit::type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeUnit {
    Second = 0,
    Milli = 1,
    Micro = 2,
    Nano = 3,
}

/// Base class for encapsulating a data type
/// Corresponds to C++ DataType class
#[derive(Debug, Clone)]
pub struct DataType {
    type_: Type,
    layout: Layout,
}

impl DataType {
    /// Create a new DataType with fixed width layout
    pub fn new(type_: Type) -> Self {
        Self {
            type_,
            layout: Layout::FixedWidth,
        }
    }

    /// Create a new DataType with specified layout
    pub fn new_with_layout(type_: Type, layout: Layout) -> Self {
        Self { type_, layout }
    }

    /// Get the type as an enum (equivalent to getType() in C++)
    pub fn get_type(&self) -> Type {
        self.type_
    }

    /// Get the data layout (equivalent to getLayout() in C++)
    pub fn get_layout(&self) -> Layout {
        self.layout
    }

    /// Makes a shared pointer for the DataType (equivalent to Make() in C++)
    pub fn make(type_: Type, layout: Layout) -> Arc<DataType> {
        Arc::new(DataType::new_with_layout(type_, layout))
    }

    /// Create with fixed width layout by default
    pub fn make_fixed_width(type_: Type) -> Arc<DataType> {
        Arc::new(DataType::new(type_))
    }
}

/// Fixed size binary type corresponding to C++ FixedSizeBinaryType
#[derive(Debug, Clone)]
pub struct FixedSizeBinaryType {
    base: DataType,
    pub byte_width: i32,
}

impl FixedSizeBinaryType {
    pub fn new(byte_width: i32) -> Self {
        Self {
            base: DataType::new(Type::FixedSizeBinary),
            byte_width,
        }
    }

    pub fn new_with_override_type(byte_width: i32, override_type: Type) -> Self {
        Self {
            base: DataType::new(override_type),
            byte_width,
        }
    }
}

/// Timestamp type corresponding to C++ TimestampType
#[derive(Debug, Clone)]
pub struct TimestampType {
    base: DataType,
    pub unit: TimeUnit,
    pub timezone: String,
}

impl TimestampType {
    pub fn new(unit: TimeUnit, timezone: String) -> Self {
        Self {
            base: DataType::new(Type::Timestamp),
            unit,
            timezone,
        }
    }
}

/// Duration type corresponding to C++ DurationType
#[derive(Debug, Clone)]
pub struct DurationType {
    base: DataType,
    pub unit: TimeUnit,
}

impl DurationType {
    pub fn new(unit: TimeUnit) -> Self {
        Self {
            base: DataType::new(Type::Duration),
            unit,
        }
    }
}

/// Decimal type corresponding to C++ DecimalType
#[derive(Debug, Clone)]
pub struct DecimalType {
    base: FixedSizeBinaryType,
    pub precision: i32,
    pub scale: i32,
}

impl DecimalType {
    pub fn new(byte_width: i32, precision: i32, scale: i32) -> Self {
        Self {
            base: FixedSizeBinaryType::new_with_override_type(byte_width, Type::Decimal),
            precision,
            scale,
        }
    }
}

// Factory functions corresponding to C++ TYPE_FACTORY macros

/// Create a boolean data type
pub fn bool_type() -> Arc<DataType> {
    DataType::make_fixed_width(Type::Bool)
}

/// Create an int8 data type
pub fn int8() -> Arc<DataType> {
    DataType::make_fixed_width(Type::Int8)
}

/// Create a uint8 data type
pub fn uint8() -> Arc<DataType> {
    DataType::make_fixed_width(Type::UInt8)
}

/// Create an int16 data type
pub fn int16() -> Arc<DataType> {
    DataType::make_fixed_width(Type::Int16)
}

/// Create a uint16 data type
pub fn uint16() -> Arc<DataType> {
    DataType::make_fixed_width(Type::UInt16)
}

/// Create an int32 data type
pub fn int32() -> Arc<DataType> {
    DataType::make_fixed_width(Type::Int32)
}

/// Create a uint32 data type
pub fn uint32() -> Arc<DataType> {
    DataType::make_fixed_width(Type::UInt32)
}

/// Create an int64 data type
pub fn int64() -> Arc<DataType> {
    DataType::make_fixed_width(Type::Int64)
}

/// Create a uint64 data type
pub fn uint64() -> Arc<DataType> {
    DataType::make_fixed_width(Type::UInt64)
}

/// Create a half float data type
pub fn half_float() -> Arc<DataType> {
    DataType::make_fixed_width(Type::HalfFloat)
}

/// Create a float data type
pub fn float() -> Arc<DataType> {
    DataType::make_fixed_width(Type::Float)
}

/// Create a double data type
pub fn double() -> Arc<DataType> {
    DataType::make_fixed_width(Type::Double)
}

/// Create a date32 data type
pub fn date32() -> Arc<DataType> {
    DataType::make_fixed_width(Type::Date32)
}

/// Create a date64 data type
pub fn date64() -> Arc<DataType> {
    DataType::make_fixed_width(Type::Date64)
}

/// Create a time32 data type
pub fn time32() -> Arc<DataType> {
    DataType::make_fixed_width(Type::Time32)
}

/// Create a time64 data type
pub fn time64() -> Arc<DataType> {
    DataType::make_fixed_width(Type::Time64)
}

/// Create an interval data type
pub fn interval() -> Arc<DataType> {
    DataType::make_fixed_width(Type::Interval)
}

/// Create a string data type (variable width)
pub fn string() -> Arc<DataType> {
    DataType::make(Type::String, Layout::VariableWidth)
}

/// Create a binary data type (variable width)
pub fn binary() -> Arc<DataType> {
    DataType::make(Type::Binary, Layout::VariableWidth)
}

/// Create a large string data type (variable width)
pub fn large_string() -> Arc<DataType> {
    DataType::make(Type::LargeString, Layout::VariableWidth)
}

/// Create a large binary data type (variable width)
pub fn large_binary() -> Arc<DataType> {
    DataType::make(Type::LargeBinary, Layout::VariableWidth)
}

/// Create a fixed size binary data type
pub fn fixed_size_binary(byte_width: i32) -> Arc<FixedSizeBinaryType> {
    Arc::new(FixedSizeBinaryType::new(byte_width))
}

/// Create a decimal data type
pub fn decimal(byte_width: i32, precision: i32, scale: i32) -> Arc<DecimalType> {
    Arc::new(DecimalType::new(byte_width, precision, scale))
}

/// Create a timestamp data type
pub fn timestamp(unit: TimeUnit, timezone: String) -> Arc<TimestampType> {
    Arc::new(TimestampType::new(unit, timezone))
}

/// Create a duration data type
pub fn duration(unit: TimeUnit) -> Arc<DurationType> {
    Arc::new(DurationType::new(unit))
}