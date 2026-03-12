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

//! Libfabric endpoint management
//!
//! This module manages libfabric endpoints for communication.

use std::ptr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::error::{CylonError, CylonResult, Code};
use super::libfabric_sys::*;
use super::error::fi_strerror;
use super::context::FabricContext;
use super::address_vector::AddressVector;

/// Endpoint manages a libfabric endpoint for send/receive operations
pub struct Endpoint {
    /// The endpoint handle
    ep: *mut fid_ep,
    /// Reference to the fabric context
    ctx: Arc<FabricContext>,
    /// Reference to the address vector
    av: Arc<AddressVector>,
    /// Local address
    local_addr: Vec<u8>,
    /// Whether the endpoint is enabled
    enabled: AtomicBool,
    /// Whether the endpoint has been closed
    closed: AtomicBool,
}

// Safety: Endpoint manages raw pointers but ensures proper cleanup
unsafe impl Send for Endpoint {}
unsafe impl Sync for Endpoint {}

impl Endpoint {
    /// Create a new endpoint
    pub fn new(ctx: Arc<FabricContext>, av: Arc<AddressVector>) -> CylonResult<Self> {
        unsafe {
            // Create endpoint
            let mut ep: *mut fid_ep = ptr::null_mut();
            let ret = fi_endpoint(ctx.domain(), ctx.info(), &mut ep, ptr::null_mut());

            if ret != 0 {
                return Err(CylonError::new(
                    Code::ExecutionError,
                    format!("fi_endpoint failed: {} (error code: {})", fi_strerror(ret), ret),
                ));
            }

            // Bind completion queue to endpoint
            let ret = fi_ep_bind(
                ep,
                &mut (*ctx.cq()).fid,
                (FI_TRANSMIT | FI_RECV) as u64,
            );

            if ret != 0 {
                fi_close(&mut (*ep).fid);
                return Err(CylonError::new(
                    Code::ExecutionError,
                    format!("fi_ep_bind (cq) failed: {} (error code: {})", fi_strerror(ret), ret),
                ));
            }

            // Bind address vector to endpoint
            let ret = fi_ep_bind(
                ep,
                &mut (*av.av()).fid,
                0,
            );

            if ret != 0 {
                fi_close(&mut (*ep).fid);
                return Err(CylonError::new(
                    Code::ExecutionError,
                    format!("fi_ep_bind (av) failed: {} (error code: {})", fi_strerror(ret), ret),
                ));
            }

            // Enable endpoint
            let ret = fi_enable(ep);
            if ret != 0 {
                fi_close(&mut (*ep).fid);
                return Err(CylonError::new(
                    Code::ExecutionError,
                    format!("fi_enable failed: {} (error code: {})", fi_strerror(ret), ret),
                ));
            }

            // Get local address
            let mut addrlen: usize = 0;
            fi_getname(&mut (*ep).fid, ptr::null_mut(), &mut addrlen);

            let mut local_addr = vec![0u8; addrlen];
            let ret = fi_getname(&mut (*ep).fid, local_addr.as_mut_ptr() as *mut _, &mut addrlen);

            if ret != 0 {
                fi_close(&mut (*ep).fid);
                return Err(CylonError::new(
                    Code::ExecutionError,
                    format!("fi_getname failed: {} (error code: {})", fi_strerror(ret), ret),
                ));
            }

            local_addr.truncate(addrlen);

            Ok(Self {
                ep,
                ctx,
                av,
                local_addr,
                enabled: AtomicBool::new(true),
                closed: AtomicBool::new(false),
            })
        }
    }

    /// Get the endpoint handle
    pub fn ep(&self) -> *mut fid_ep {
        self.ep
    }

    /// Get the local address
    pub fn local_addr(&self) -> &[u8] {
        &self.local_addr
    }

    /// Get the local address as a hex string (for debugging)
    pub fn local_addr_hex(&self) -> String {
        self.local_addr.iter()
            .map(|b| format!("{:02x}", b))
            .collect::<Vec<_>>()
            .join("")
    }

    /// Check if the endpoint is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::SeqCst)
    }

    /// Check if the endpoint has been closed
    pub fn is_closed(&self) -> bool {
        self.closed.load(Ordering::SeqCst)
    }

    /// Send data to a remote peer (non-blocking)
    ///
    /// Returns Ok(true) if the send was posted, Ok(false) if EAGAIN
    pub fn send(
        &self,
        buf: &[u8],
        dest_addr: fi_addr_t,
        context: *mut std::ffi::c_void,
    ) -> CylonResult<bool> {
        if self.is_closed() {
            return Err(CylonError::new(
                Code::ExecutionError,
                "Endpoint is closed".to_string(),
            ));
        }

        unsafe {
            let ret = fi_send(
                self.ep,
                buf.as_ptr() as *const _,
                buf.len(),
                ptr::null_mut(), // No memory descriptor for basic sends
                dest_addr,
                context,
            );

            if ret == 0 {
                Ok(true)
            } else if ret == -(FI_EAGAIN as isize) {
                Ok(false)
            } else {
                Err(CylonError::new(
                    Code::ExecutionError,
                    format!("fi_send failed: {} (error code: {})", fi_strerror(-(ret as i32)), ret),
                ))
            }
        }
    }

    /// Send tagged data to a remote peer (non-blocking)
    pub fn tsend(
        &self,
        buf: &[u8],
        dest_addr: fi_addr_t,
        tag: u64,
        context: *mut std::ffi::c_void,
    ) -> CylonResult<bool> {
        if self.is_closed() {
            return Err(CylonError::new(
                Code::ExecutionError,
                "Endpoint is closed".to_string(),
            ));
        }

        unsafe {
            let ret = fi_tsend(
                self.ep,
                buf.as_ptr() as *const _,
                buf.len(),
                ptr::null_mut(),
                dest_addr,
                tag,
                context,
            );

            if ret == 0 {
                Ok(true)
            } else if ret == -(FI_EAGAIN as isize) {
                Ok(false)
            } else {
                Err(CylonError::new(
                    Code::ExecutionError,
                    format!("fi_tsend failed: {} (error code: {})", fi_strerror(-(ret as i32)), ret),
                ))
            }
        }
    }

    /// Post a receive buffer (non-blocking)
    ///
    /// Returns Ok(true) if the receive was posted, Ok(false) if EAGAIN
    pub fn recv(
        &self,
        buf: &mut [u8],
        src_addr: fi_addr_t,
        context: *mut std::ffi::c_void,
    ) -> CylonResult<bool> {
        if self.is_closed() {
            return Err(CylonError::new(
                Code::ExecutionError,
                "Endpoint is closed".to_string(),
            ));
        }

        unsafe {
            let ret = fi_recv(
                self.ep,
                buf.as_mut_ptr() as *mut _,
                buf.len(),
                ptr::null_mut(),
                src_addr,
                context,
            );

            if ret == 0 {
                Ok(true)
            } else if ret == -(FI_EAGAIN as isize) {
                Ok(false)
            } else {
                Err(CylonError::new(
                    Code::ExecutionError,
                    format!("fi_recv failed: {} (error code: {})", fi_strerror(-(ret as i32)), ret),
                ))
            }
        }
    }

    /// Post a tagged receive buffer (non-blocking)
    pub fn trecv(
        &self,
        buf: &mut [u8],
        src_addr: fi_addr_t,
        tag: u64,
        ignore: u64,
        context: *mut std::ffi::c_void,
    ) -> CylonResult<bool> {
        if self.is_closed() {
            return Err(CylonError::new(
                Code::ExecutionError,
                "Endpoint is closed".to_string(),
            ));
        }

        unsafe {
            let ret = fi_trecv(
                self.ep,
                buf.as_mut_ptr() as *mut _,
                buf.len(),
                ptr::null_mut(),
                src_addr,
                tag,
                ignore,
                context,
            );

            if ret == 0 {
                Ok(true)
            } else if ret == -(FI_EAGAIN as isize) {
                Ok(false)
            } else {
                Err(CylonError::new(
                    Code::ExecutionError,
                    format!("fi_trecv failed: {} (error code: {})", fi_strerror(-(ret as i32)), ret),
                ))
            }
        }
    }

    /// Close the endpoint
    pub fn close(&self) -> CylonResult<()> {
        if self.closed.swap(true, Ordering::SeqCst) {
            return Ok(()); // Already closed
        }

        unsafe {
            if !self.ep.is_null() {
                fi_close(&mut (*self.ep).fid);
            }
        }

        Ok(())
    }
}

impl Drop for Endpoint {
    fn drop(&mut self) {
        let _ = self.close();
    }
}
