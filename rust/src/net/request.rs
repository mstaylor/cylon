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

//! CylonRequest for network communication
//!
//! Ported from cpp/src/cylon/net/cylon_request.hpp

/// Request structure for channel communication
/// Corresponds to C++ CylonRequest class
pub struct CylonRequest {
    pub buffer: Vec<u8>,
    pub target: i32,
    pub header: [i32; 6],
    pub header_length: usize,
}

impl CylonRequest {
    /// Create a new request with buffer and target
    pub fn new(target: i32, buffer: Vec<u8>) -> Self {
        Self {
            buffer,
            target,
            header: [0; 6],
            header_length: 0,
        }
    }

    /// Create a new request with buffer, target, and header
    pub fn new_with_header(target: i32, buffer: Vec<u8>, header: &[i32]) -> Self {
        let mut req = Self::new(target, buffer);
        let len = header.len().min(6);
        req.header[..len].copy_from_slice(&header[..len]);
        req.header_length = len;
        req
    }

    /// Create a finish request (empty buffer)
    pub fn new_finish(target: i32) -> Self {
        Self {
            buffer: Vec::new(),
            target,
            header: [0; 6],
            header_length: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}
