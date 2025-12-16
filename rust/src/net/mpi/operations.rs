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
//! Updated for rsmpi 0.8 API

use std::sync::Arc;
use mpi::collective::SystemOperation;
use mpi::datatype::Equivalence; // rsmpi 0.8 uses Equivalence trait
use mpi::environment::Universe;
use mpi::traits::*;

use crate::data_types::{DataType, Type};
use crate::error::{Code, CylonError, CylonResult};
use crate::net::comm_operations::ReduceOp;
use crate::net::ops::TableGatherImpl;
use crate::ctx::CylonContext;
use crate::table::Table;
use crate::net::serialize::{serialize_table, deserialize_table};

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
/// Updated for rsmpi 0.8 - uses Equivalent trait instead of UserDatatype static methods
pub fn get_mpi_datatype_id(data_type: &DataType) -> Option<mpi::datatype::DatatypeRef<'static>> {
    match data_type.get_type() {
        Type::Bool => Some(bool::equivalent_datatype()),
        Type::UInt8 => Some(u8::equivalent_datatype()),
        Type::Int8 => Some(i8::equivalent_datatype()),
        Type::UInt16 => Some(u16::equivalent_datatype()),
        Type::Int16 => Some(i16::equivalent_datatype()),
        Type::UInt32 => Some(u32::equivalent_datatype()),
        Type::Int32 => Some(i32::equivalent_datatype()),
        Type::UInt64 => Some(u64::equivalent_datatype()),
        Type::Int64 => Some(i64::equivalent_datatype()),
        Type::Float => Some(f32::equivalent_datatype()),
        Type::Double => Some(f64::equivalent_datatype()),
        Type::FixedSizeBinary | Type::String | Type::Binary |
        Type::LargeString | Type::LargeBinary => {
            // Treat as bytes
            Some(u8::equivalent_datatype())
        }
        Type::Date32 | Type::Time32 => Some(u32::equivalent_datatype()),
        Type::Date64 | Type::Timestamp | Type::Time64 => {
            Some(u64::equivalent_datatype())
        }
        // Unsupported types
        Type::HalfFloat | Type::Decimal | Type::Duration | Type::Interval |
        Type::List | Type::FixedSizeList | Type::Extension | Type::MaxId => None,
    }
}

use crate::net::ops::base_ops::{AllReduceImpl, AllGatherImpl};

/// MPI AllReduce implementation
/// Corresponds to C++ MpiAllReduceImpl from cpp/src/cylon/net/mpi/mpi_operations.hpp
///
/// This struct implements the AllReduceImpl trait for MPI.
/// It uses MPI_Allreduce for the buffer operation.
pub struct MpiAllReduceImpl {
    universe: Arc<std::sync::Mutex<Option<Universe>>>,
}

impl MpiAllReduceImpl {
    /// Create a new MpiAllReduceImpl
    ///
    /// # Arguments
    /// * `universe` - The MPI universe
    pub fn new(universe: Arc<std::sync::Mutex<Option<Universe>>>) -> Self {
        Self { universe }
    }
}

impl AllReduceImpl for MpiAllReduceImpl {
    fn allreduce_buffer(
        &self,
        send_buf: &[u8],
        rcv_buf: &mut [u8],
        count: i32,
        data_type: &DataType,
        reduce_op: ReduceOp,
    ) -> CylonResult<()> {
        let universe_lock = self.universe.lock().unwrap();
        let universe = universe_lock.as_ref().ok_or_else(|| {
            CylonError::new(Code::Invalid, "MPI not initialized".to_string())
        })?;

        let world = universe.world();

        // Validate the datatype is supported for MPI operations
        let _mpi_dtype = get_mpi_datatype_id(data_type).ok_or_else(|| {
            CylonError::new(
                Code::NotImplemented,
                format!("Unknown MPI datatype for {:?}", data_type),
            )
        })?;

        // Get MPI operation
        let mpi_op = get_mpi_op(reduce_op);

        // We need to use raw MPI calls since rsmpi's high-level API
        // doesn't support arbitrary buffer operations
        // Cast buffers to appropriate types based on data_type
        match data_type.get_type() {
            Type::Int64 => {
                let send_slice = unsafe {
                    std::slice::from_raw_parts(
                        send_buf.as_ptr() as *const i64,
                        count as usize,
                    )
                };
                let rcv_slice = unsafe {
                    std::slice::from_raw_parts_mut(
                        rcv_buf.as_mut_ptr() as *mut i64,
                        count as usize,
                    )
                };
                world.all_reduce_into(send_slice, rcv_slice, mpi_op);
            }
            Type::Int32 => {
                let send_slice = unsafe {
                    std::slice::from_raw_parts(
                        send_buf.as_ptr() as *const i32,
                        count as usize,
                    )
                };
                let rcv_slice = unsafe {
                    std::slice::from_raw_parts_mut(
                        rcv_buf.as_mut_ptr() as *mut i32,
                        count as usize,
                    )
                };
                world.all_reduce_into(send_slice, rcv_slice, mpi_op);
            }
            Type::Float => {
                let send_slice = unsafe {
                    std::slice::from_raw_parts(
                        send_buf.as_ptr() as *const f32,
                        count as usize,
                    )
                };
                let rcv_slice = unsafe {
                    std::slice::from_raw_parts_mut(
                        rcv_buf.as_mut_ptr() as *mut f32,
                        count as usize,
                    )
                };
                world.all_reduce_into(send_slice, rcv_slice, mpi_op);
            }
            Type::Double => {
                let send_slice = unsafe {
                    std::slice::from_raw_parts(
                        send_buf.as_ptr() as *const f64,
                        count as usize,
                    )
                };
                let rcv_slice = unsafe {
                    std::slice::from_raw_parts_mut(
                        rcv_buf.as_mut_ptr() as *mut f64,
                        count as usize,
                    )
                };
                world.all_reduce_into(send_slice, rcv_slice, mpi_op);
            }
            Type::Int8 => {
                let send_slice = unsafe {
                    std::slice::from_raw_parts(
                        send_buf.as_ptr() as *const i8,
                        count as usize,
                    )
                };
                let rcv_slice = unsafe {
                    std::slice::from_raw_parts_mut(
                        rcv_buf.as_mut_ptr() as *mut i8,
                        count as usize,
                    )
                };
                world.all_reduce_into(send_slice, rcv_slice, mpi_op);
            }
            Type::UInt8 => {
                let send_slice = unsafe {
                    std::slice::from_raw_parts(
                        send_buf.as_ptr() as *const u8,
                        count as usize,
                    )
                };
                let rcv_slice = unsafe {
                    std::slice::from_raw_parts_mut(
                        rcv_buf.as_mut_ptr() as *mut u8,
                        count as usize,
                    )
                };
                world.all_reduce_into(send_slice, rcv_slice, mpi_op);
            }
            Type::Int16 => {
                let send_slice = unsafe {
                    std::slice::from_raw_parts(
                        send_buf.as_ptr() as *const i16,
                        count as usize,
                    )
                };
                let rcv_slice = unsafe {
                    std::slice::from_raw_parts_mut(
                        rcv_buf.as_mut_ptr() as *mut i16,
                        count as usize,
                    )
                };
                world.all_reduce_into(send_slice, rcv_slice, mpi_op);
            }
            Type::UInt16 => {
                let send_slice = unsafe {
                    std::slice::from_raw_parts(
                        send_buf.as_ptr() as *const u16,
                        count as usize,
                    )
                };
                let rcv_slice = unsafe {
                    std::slice::from_raw_parts_mut(
                        rcv_buf.as_mut_ptr() as *mut u16,
                        count as usize,
                    )
                };
                world.all_reduce_into(send_slice, rcv_slice, mpi_op);
            }
            Type::UInt32 => {
                let send_slice = unsafe {
                    std::slice::from_raw_parts(
                        send_buf.as_ptr() as *const u32,
                        count as usize,
                    )
                };
                let rcv_slice = unsafe {
                    std::slice::from_raw_parts_mut(
                        rcv_buf.as_mut_ptr() as *mut u32,
                        count as usize,
                    )
                };
                world.all_reduce_into(send_slice, rcv_slice, mpi_op);
            }
            Type::UInt64 => {
                let send_slice = unsafe {
                    std::slice::from_raw_parts(
                        send_buf.as_ptr() as *const u64,
                        count as usize,
                    )
                };
                let rcv_slice = unsafe {
                    std::slice::from_raw_parts_mut(
                        rcv_buf.as_mut_ptr() as *mut u64,
                        count as usize,
                    )
                };
                world.all_reduce_into(send_slice, rcv_slice, mpi_op);
            }
            _ => {
                return Err(CylonError::new(
                    Code::NotImplemented,
                    format!("MPI AllReduce not implemented for type {:?}", data_type),
                ));
            }
        }

        Ok(())
    }
}

/// MPI AllGather implementation for Column/Scalar operations
/// Corresponds to C++ MpiAllgatherImpl from cpp/src/cylon/net/mpi/mpi_operations.hpp
///
/// This struct implements the AllGatherImpl trait for MPI.
/// It uses MPI_Allgather for buffer sizes and MPI_Allgatherv for data.
pub struct MpiAllgatherImpl {
    universe: Arc<std::sync::Mutex<Option<Universe>>>,
}

impl MpiAllgatherImpl {
    /// Create a new MpiAllgatherImpl
    ///
    /// # Arguments
    /// * `universe` - The MPI universe
    /// * `_world_size` - Number of processes (kept for API compatibility)
    pub fn new(universe: Arc<std::sync::Mutex<Option<Universe>>>, _world_size: i32) -> Self {
        Self {
            universe,
        }
    }
}

impl AllGatherImpl for MpiAllgatherImpl {
    fn allgather_buffer_size(
        &self,
        send_data: &[i32],
        _num_buffers: i32,
        rcv_data: &mut [i32],
    ) -> CylonResult<()> {
        let universe_lock = self.universe.lock().unwrap();
        let universe = universe_lock.as_ref().ok_or_else(|| {
            CylonError::new(Code::Invalid, "MPI not initialized".to_string())
        })?;

        let world = universe.world();

        // Use rsmpi's all_gather_into for i32 slices
        world.all_gather_into(send_data, rcv_data);

        Ok(())
    }

    fn iallgather_buffer_data(
        &mut self,
        _buf_idx: i32,
        send_data: &[u8],
        send_count: i32,
        recv_data: &mut [u8],
        recv_count: &[i32],
        displacements: &[i32],
    ) -> CylonResult<()> {
        // rsmpi doesn't have a high-level API for Iallgatherv with displacements
        // We need to use the synchronous allgatherv or raw MPI calls
        //
        // For now, we'll do a synchronous allgatherv using raw pointers
        // This matches the C++ behavior where we use MPI_Iallgatherv
        // but wait at the end in WaitAll anyway

        let universe_lock = self.universe.lock().unwrap();
        let universe = universe_lock.as_ref().ok_or_else(|| {
            CylonError::new(Code::Invalid, "MPI not initialized".to_string())
        })?;

        let world = universe.world();

        // Use raw MPI for allgatherv with variable counts and displacements
        // This requires using mpi-sys directly
        unsafe {
            let send_ptr = if send_count > 0 { send_data.as_ptr() } else { std::ptr::null() };
            let recv_ptr = if recv_data.len() > 0 { recv_data.as_mut_ptr() } else { std::ptr::null_mut() };

            let status = mpi_sys::MPI_Allgatherv(
                send_ptr as *const _,
                send_count,
                mpi_sys::RSMPI_UINT8_T,
                recv_ptr as *mut _,
                recv_count.as_ptr(),
                displacements.as_ptr(),
                mpi_sys::RSMPI_UINT8_T,
                world.as_communicator().as_raw(),
            );

            if status != mpi_sys::MPI_SUCCESS as i32 {
                return Err(CylonError::new(
                    Code::ExecutionError,
                    "MPI_Allgatherv failed".to_string(),
                ));
            }
        }

        Ok(())
    }

    fn wait_all(&mut self) -> CylonResult<()> {
        // Since we're using synchronous allgatherv in iallgather_buffer_data,
        // there's nothing to wait for
        Ok(())
    }
}