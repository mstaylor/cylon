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

//! MPI operations and helper functions
//!
//! Ported from cpp/src/cylon/net/mpi/mpi_operations.hpp and mpi_operations.cpp

use mpi::collective::SystemOperation;
use mpi::datatype::{UncommittedUserDatatype, UserDatatype};
use mpi::raw::AsRaw;

use crate::data_types::{DataType, Type};
use crate::error::{Code, CylonError, CylonResult};
use crate::net::comm_operations::ReduceOp;

/// Convert Cylon ReduceOp to MPI operation
/// Corresponds to GetMPIOp() in cpp/src/cylon/net/mpi/mpi_operations.cpp
pub fn get_mpi_op(reduce_op: ReduceOp) -> SystemOperation {
    match reduce_op {
        ReduceOp::Sum => SystemOperation::sum(),
        ReduceOp::Min => SystemOperation::min(),
        ReduceOp::Max => SystemOperation::max(),
        ReduceOp::Prod => SystemOperation::product(),
        ReduceOp::Land => SystemOperation::logical_and(),
        ReduceOp::Lor => SystemOperation::logical_or(),
        ReduceOp::Band => SystemOperation::bitwise_and(),
        ReduceOp::Bor => SystemOperation::bitwise_or(),
    }
}

/// Get MPI datatype from Cylon DataType
/// Corresponds to GetMPIDataType() in cpp/src/cylon/net/mpi/mpi_operations.cpp
///
/// Returns None for unsupported or complex types
pub fn get_mpi_datatype_id(data_type: &DataType) -> Option<mpi::datatype::DatatypeRef<'static>> {
    match data_type.get_type() {
        Type::Bool => Some(mpi::datatype::UserDatatype::bool().as_ref()),
        Type::UInt8 => Some(mpi::datatype::UserDatatype::u8().as_ref()),
        Type::Int8 => Some(mpi::datatype::UserDatatype::i8().as_ref()),
        Type::UInt16 => Some(mpi::datatype::UserDatatype::u16().as_ref()),
        Type::Int16 => Some(mpi::datatype::UserDatatype::i16().as_ref()),
        Type::UInt32 => Some(mpi::datatype::UserDatatype::u32().as_ref()),
        Type::Int32 => Some(mpi::datatype::UserDatatype::i32().as_ref()),
        Type::UInt64 => Some(mpi::datatype::UserDatatype::u64().as_ref()),
        Type::Int64 => Some(mpi::datatype::UserDatatype::i64().as_ref()),
        Type::Float => Some(mpi::datatype::UserDatatype::f32().as_ref()),
        Type::Double => Some(mpi::datatype::UserDatatype::f64().as_ref()),
        Type::FixedSizeBinary | Type::String | Type::Binary |
        Type::LargeString | Type::LargeBinary => {
            // Treat as bytes
            Some(mpi::datatype::UserDatatype::u8().as_ref())
        }
        Type::Date32 | Type::Time32 => Some(mpi::datatype::UserDatatype::u32().as_ref()),
        Type::Date64 | Type::Timestamp | Type::Time64 => {
            Some(mpi::datatype::UserDatatype::u64().as_ref())
        }
        // Unsupported types
        Type::HalfFloat | Type::Decimal | Type::Duration | Type::Interval |
        Type::List | Type::FixedSizeList | Type::Extension | Type::MaxId => None,
    }
}

// TODO: The following require full Table/Column serialization support:
//
// 1. TableSerializer/Deserializer - Convert Arrow Tables to/from byte buffers
//    - Uses Arrow IPC format
//    - Handles buffer alignment and offsets
//    - Port from cpp/src/cylon/serialize/table_serialize.hpp
//
// 2. ColumnSerializer/Deserializer - Convert Arrow Arrays to/from byte buffers
//    - Port from cpp/src/cylon/serialize/table_serialize.hpp
//
// 3. MPI collective operations for Tables:
//    - AllGather(Table) -> Vec<Table>
//    - Gather(Table, root) -> Vec<Table>
//    - Bcast(Table, root) -> Table
//    - Port from cpp/src/cylon/net/mpi/mpi_operations.cpp
//
// 4. MPI collective operations for Columns:
//    - AllReduce(Column) -> Column
//    - Allgather(Column) -> Vec<Column>
//    - Port from MPICommunicator in cpp/src/cylon/net/mpi/mpi_communicator.cpp
//
// 5. MPI collective operations for Scalars:
//    - AllReduce(Scalar) -> Scalar
//    - Allgather(Scalar) -> Column
//    - Port from MPICommunicator in cpp/src/cylon/net/mpi/mpi_communicator.cpp
//
// These operations form the core of distributed table processing in Cylon.
// They require:
// - Arrow IPC serialization/deserialization
// - Buffer management for variable-sized data
// - Schema propagation across workers
// - Proper memory pooling