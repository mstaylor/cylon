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

//! Libfabric address vector management
//!
//! This module manages address vectors for mapping remote peer addresses
//! to fi_addr_t handles used in communication operations.

use std::ptr;
use std::sync::{Arc, RwLock};
use std::sync::atomic::{AtomicBool, Ordering};
use std::collections::HashMap;

use crate::error::{CylonError, CylonResult, Code};
use super::libfabric_sys::*;
use super::error::fi_strerror;
use super::context::FabricContext;

/// AddressVector manages a libfabric address vector
///
/// The address vector maps raw addresses (obtained via OOB) to fi_addr_t
/// handles that can be used in libfabric operations.
pub struct AddressVector {
    /// The address vector handle
    av: *mut fid_av,
    /// Reference to the fabric context
    ctx: Arc<FabricContext>,
    /// Mapping from rank to fi_addr_t
    rank_to_addr: RwLock<HashMap<i32, fi_addr_t>>,
    /// Whether the AV has been closed
    closed: AtomicBool,
}

// Safety: AddressVector manages raw pointers but ensures proper cleanup
unsafe impl Send for AddressVector {}
unsafe impl Sync for AddressVector {}

impl AddressVector {
    /// Create a new address vector
    pub fn new(ctx: Arc<FabricContext>) -> CylonResult<Self> {
        unsafe {
            let mut av_attr: fi_av_attr = std::mem::zeroed();
            av_attr.type_ = FI_AV_TABLE;
            av_attr.count = ctx.config().av_size;
            av_attr.ep_per_node = 0;

            let mut av: *mut fid_av = ptr::null_mut();
            let ret = fi_av_open(ctx.domain(), &av_attr, &mut av, ptr::null_mut());

            if ret != 0 {
                return Err(CylonError::new(
                    Code::ExecutionError,
                    format!("fi_av_open failed: {} (error code: {})", fi_strerror(ret), ret),
                ));
            }

            Ok(Self {
                av,
                ctx,
                rank_to_addr: RwLock::new(HashMap::new()),
                closed: AtomicBool::new(false),
            })
        }
    }

    /// Get the address vector handle
    pub fn av(&self) -> *mut fid_av {
        self.av
    }

    /// Insert an address into the address vector
    ///
    /// Returns the fi_addr_t handle for the inserted address
    pub fn insert(&self, addr: &[u8], rank: i32) -> CylonResult<fi_addr_t> {
        if self.closed.load(Ordering::SeqCst) {
            return Err(CylonError::new(
                Code::ExecutionError,
                "Address vector is closed".to_string(),
            ));
        }

        unsafe {
            let mut fi_addr: fi_addr_t = FI_ADDR_UNSPEC;
            let ret = fi_av_insert(
                self.av,
                addr.as_ptr() as *const _,
                1,
                &mut fi_addr,
                0,
                ptr::null_mut(),
            );

            if ret != 1 {
                return Err(CylonError::new(
                    Code::ExecutionError,
                    format!("fi_av_insert failed: expected 1, got {} (error code: {})",
                        ret, if ret < 0 { fi_strerror(-(ret as i32)) } else { "success".to_string() }),
                ));
            }

            // Store the mapping
            self.rank_to_addr.write().unwrap().insert(rank, fi_addr);

            log::debug!("Inserted address for rank {} -> fi_addr {}", rank, fi_addr);

            Ok(fi_addr)
        }
    }

    /// Insert multiple addresses into the address vector
    ///
    /// Returns the fi_addr_t handles for the inserted addresses
    pub fn insert_many(&self, addrs: &[Vec<u8>], ranks: &[i32]) -> CylonResult<Vec<fi_addr_t>> {
        if addrs.len() != ranks.len() {
            return Err(CylonError::new(
                Code::Invalid,
                "Address and rank arrays must have same length".to_string(),
            ));
        }

        let mut results = Vec::with_capacity(addrs.len());
        for (addr, rank) in addrs.iter().zip(ranks.iter()) {
            results.push(self.insert(addr, *rank)?);
        }

        Ok(results)
    }

    /// Look up the fi_addr_t for a rank
    pub fn lookup(&self, rank: i32) -> Option<fi_addr_t> {
        self.rank_to_addr.read().unwrap().get(&rank).copied()
    }

    /// Get all known fi_addr_t handles
    pub fn all_addrs(&self) -> Vec<fi_addr_t> {
        self.rank_to_addr.read().unwrap().values().copied().collect()
    }

    /// Get the number of addresses in the vector
    pub fn len(&self) -> usize {
        self.rank_to_addr.read().unwrap().len()
    }

    /// Check if the address vector is empty
    pub fn is_empty(&self) -> bool {
        self.rank_to_addr.read().unwrap().is_empty()
    }

    /// Remove an address from the vector
    pub fn remove(&self, rank: i32) -> CylonResult<()> {
        if self.closed.load(Ordering::SeqCst) {
            return Ok(());
        }

        let fi_addr = self.rank_to_addr.write().unwrap().remove(&rank);

        if let Some(addr) = fi_addr {
            unsafe {
                let ret = fi_av_remove(self.av, &addr, 1, 0);
                if ret != 0 {
                    log::warn!("fi_av_remove failed for rank {}: {}", rank, fi_strerror(ret));
                }
            }
        }

        Ok(())
    }

    /// Close the address vector
    pub fn close(&self) -> CylonResult<()> {
        if self.closed.swap(true, Ordering::SeqCst) {
            return Ok(()); // Already closed
        }

        unsafe {
            if !self.av.is_null() {
                fi_close(&mut (*self.av).fid);
            }
        }

        Ok(())
    }
}

impl Drop for AddressVector {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

/// AVSet manages an address vector set for collective operations
pub struct AVSet {
    /// The AV set handle
    av_set: *mut fid_av_set,
    /// Collective address
    coll_addr: fi_addr_t,
    /// Reference to parent AV
    _av: Arc<AddressVector>,
    /// Whether the set has been closed
    closed: AtomicBool,
}

// Safety: AVSet manages raw pointers but ensures proper cleanup
unsafe impl Send for AVSet {}
unsafe impl Sync for AVSet {}

impl AVSet {
    /// Create a new AV set for collective operations
    pub fn new(av: Arc<AddressVector>, members: &[fi_addr_t]) -> CylonResult<Self> {
        unsafe {
            let mut av_set_attr: fi_av_set_attr = std::mem::zeroed();
            av_set_attr.count = members.len();
            av_set_attr.start_addr = FI_ADDR_UNSPEC;
            av_set_attr.end_addr = FI_ADDR_UNSPEC;
            av_set_attr.stride = 1;

            let mut av_set: *mut fid_av_set = ptr::null_mut();
            let ret = fi_av_set(av.av(), &av_set_attr, &mut av_set, ptr::null_mut());

            if ret != 0 {
                return Err(CylonError::new(
                    Code::ExecutionError,
                    format!("fi_av_set failed: {} (error code: {})", fi_strerror(ret), ret),
                ));
            }

            // Insert members into the set
            for &addr in members {
                let ret = fi_av_set_insert(av_set, addr);
                if ret != 0 {
                    fi_close(&mut (*av_set).fid);
                    return Err(CylonError::new(
                        Code::ExecutionError,
                        format!("fi_av_set_insert failed: {} (error code: {})", fi_strerror(ret), ret),
                    ));
                }
            }

            // Get the collective address
            let mut coll_addr: fi_addr_t = FI_ADDR_UNSPEC;
            let ret = fi_av_set_addr(av_set, &mut coll_addr);
            if ret != 0 {
                fi_close(&mut (*av_set).fid);
                return Err(CylonError::new(
                    Code::ExecutionError,
                    format!("fi_av_set_addr failed: {} (error code: {})", fi_strerror(ret), ret),
                ));
            }

            Ok(Self {
                av_set,
                coll_addr,
                _av: av,
                closed: AtomicBool::new(false),
            })
        }
    }

    /// Get the AV set handle
    pub fn av_set(&self) -> *mut fid_av_set {
        self.av_set
    }

    /// Get the collective address
    pub fn coll_addr(&self) -> fi_addr_t {
        self.coll_addr
    }

    /// Close the AV set
    pub fn close(&self) -> CylonResult<()> {
        if self.closed.swap(true, Ordering::SeqCst) {
            return Ok(()); // Already closed
        }

        unsafe {
            if !self.av_set.is_null() {
                fi_close(&mut (*self.av_set).fid);
            }
        }

        Ok(())
    }
}

impl Drop for AVSet {
    fn drop(&mut self) {
        let _ = self.close();
    }
}
