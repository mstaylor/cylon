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
//! The architecture closely follows the C++ implementation:
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

pub mod tcpunch;
pub mod common;
pub mod channel;
pub mod peer_to_peer;
pub mod direct;
pub mod communicator;

// Re-export main types
pub use common::*;
pub use channel::Channel;
pub use communicator::Communicator;
