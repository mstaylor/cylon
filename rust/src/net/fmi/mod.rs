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

//! FMI (Function-as-a-service Message Interface) Communication Module
//!
//! This module provides a communication layer for distributed computing that supports
//! direct TCP connections via TCPunch NAT hole punching.
//!
//! # Architecture
//!
//! The architecture follows the C++ implementation with two layers:
//!
//! ## Layer 1: Thirdparty FMI Library (ported from cpp/src/cylon/thridparty/fmi/)
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                     FMI Communicator                            │
//! │         (High-level API wrapping Channel)                       │
//! └─────────────────────────────────────────────────────────────────┘
//!                                │
//!                                ▼
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                        Channel (base)                           │
//! │    (peer_id, num_peers, comm_name, send, recv, collectives)     │
//! └─────────────────────────────────────────────────────────────────┘
//!                                │
//!                                ▼
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                      PeerToPeer Channel                         │
//! │     (send_object, recv_object, binomial tree collectives)       │
//! └─────────────────────────────────────────────────────────────────┘
//!                                │
//!                                ▼
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                      Direct Channel                             │
//! │              (TCPunch connection, socket management)            │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Layer 2: Cylon Integration (ported from cpp/src/cylon/net/fmi/)
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                  FMICommunicator (Cylon)                        │
//! │    (implements Cylon Communicator trait, wraps FMI Communicator)│
//! └─────────────────────────────────────────────────────────────────┘
//!                                │
//!                                ▼
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                   FMIChannel (Cylon)                            │
//! │      (implements Cylon Channel trait, progress-based model)     │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

// Layer 1: Thirdparty FMI library (ported from cpp/src/cylon/thridparty/fmi/)
pub mod tcpunch;
pub mod common;
pub mod channel;
pub mod peer_to_peer;
pub mod direct;
pub mod communicator;

// Layer 2: Cylon integration (ported from cpp/src/cylon/net/fmi/)
pub mod cylon_communicator;
pub mod cylon_channel;
pub mod cylon_operations;

// Re-export main types from Layer 1 (FMI library)
pub use common::*;
pub use channel::Channel as FmiChannel;
pub use communicator::Communicator as FmiCommunicator;

// Re-export Cylon integration types
pub use cylon_communicator::{FMIConfig, FMIConfigBuilder, FMICommunicator};
pub use cylon_channel::FMICylonChannel;
pub use cylon_operations::{
    FmiTableAllgatherImpl,
    FmiTableGatherImpl,
    FmiTableBcastImpl,
    FmiAllReduceImpl,
    FmiAllgatherImpl,
};
