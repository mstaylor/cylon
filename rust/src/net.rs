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

//! Networking and communication components
//!
//! Ported from cpp/src/cylon/net/

use async_trait::async_trait;
use std::sync::Arc;

use crate::error::CylonResult;

pub mod buffer;
pub mod channel;
pub mod comm_config;
pub mod comm_operations;
pub mod communicator;
pub mod ops;
pub mod request;
pub mod serialize;

#[cfg(feature = "mpi")]
pub mod mpi;

#[cfg(feature = "gloo")]
pub mod gloo;

#[cfg(feature = "ucx")]
pub mod ucx;

#[cfg(feature = "ucc")]
pub mod ucc;

#[cfg(feature = "fmi")]
pub mod fmi;

// Re-exports for convenience
pub use comm_config::*;
pub use communicator::Communicator;

/// Communication type enum corresponding to C++ CommType
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommType {
    Local,
    #[cfg(feature = "mpi")]
    Mpi,
    #[cfg(feature = "gloo")]
    Gloo,
    #[cfg(feature = "ucx")]
    Ucx,
    #[cfg(feature = "ucc")]
    Ucc,
    #[cfg(feature = "fmi")]
    Fmi,
}

use crate::net::request::CylonRequest;

/// Constants for channel protocol
pub const CYLON_CHANNEL_HEADER_SIZE: usize = 8;
pub const CYLON_MSG_FIN: i32 = 1;
pub const CYLON_MSG_NOT_FIN: i32 = 0;

/// Maximum size of pending data requests
pub const MAX_PENDING: usize = 1000;

/// Buffer trait for network communication
/// Corresponds to C++ Buffer interface from cpp/src/cylon/net/buffer.hpp
pub trait Buffer: Send + Sync {
    fn get_byte_buffer(&self) -> &[u8];
    fn get_byte_buffer_mut(&mut self) -> &mut [u8];
    fn size(&self) -> usize;
}

/// Callback trait for channel send completion
/// Corresponds to C++ ChannelSendCallback from cpp/src/cylon/net/channel.hpp
pub trait ChannelSendCallback: Send + Sync {
    fn send_complete(&mut self, request: Box<CylonRequest>);
    fn send_finish_complete(&mut self, request: Box<CylonRequest>);
}

/// Callback trait for channel receive completion
/// Corresponds to C++ ChannelReceiveCallback from cpp/src/cylon/net/channel.hpp
pub trait ChannelReceiveCallback: Send + Sync {
    fn received_data(&mut self, receive_id: i32, buffer: Box<dyn Buffer>, length: usize);
    fn received_header(&mut self, receive_id: i32, finished: i32, header: Option<Vec<i32>>);
}

/// Allocator trait for buffer allocation
/// Corresponds to C++ Allocator interface
pub trait Allocator: Send + Sync {
    fn allocate(&self, size: usize) -> CylonResult<Box<dyn Buffer>>;
}

/// Channel trait for point-to-point communication with progress-based model
/// Corresponds to C++ Channel interface from cpp/src/cylon/net/channel.hpp
pub trait Channel: Send + Sync {
    /// Initialize the channel
    fn init(
        &mut self,
        edge: i32,
        receives: &[i32],
        sends: &[i32],
        rcv_callback: Box<dyn ChannelReceiveCallback>,
        send_callback: Box<dyn ChannelSendCallback>,
        allocator: Box<dyn Allocator>,
    ) -> CylonResult<()>;

    /// Send a request (-1 if not accepted, 1 if accepted)
    fn send(&mut self, request: Box<CylonRequest>) -> i32;

    /// Send finish message to target
    fn send_fin(&mut self, request: Box<CylonRequest>) -> i32;

    /// Progress pending sends (must be called repeatedly)
    fn progress_sends(&mut self);

    /// Progress pending receives (must be called repeatedly)
    fn progress_receives(&mut self);

    /// Notify that the operation is completed
    fn notify_completed(&mut self) {}

    /// Close the channel
    fn close(&mut self);
}

// The Communicator trait is now defined in communicator.rs to match C++ structure
// (cpp/src/cylon/net/communicator.hpp defines the base Communicator class,
//  cpp/src/cylon/net/mpi/mpi_communicator.cpp implements it)