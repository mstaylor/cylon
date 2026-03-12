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

//! Libfabric context management
//!
//! This module manages the libfabric fabric, domain, and completion queue resources.

use std::ffi::CString;
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::error::{CylonError, CylonResult, Code};
use super::libfabric_sys::*;
use super::error::{check_fi_error, fi_strerror};
use super::{LibfabricConfig, EndpointType, ProgressMode};

/// FabricContext manages the core libfabric resources
///
/// This struct owns and manages:
/// - fi_info: Provider information
/// - fid_fabric: Fabric instance
/// - fid_domain: Domain instance
/// - fid_cq: Completion queue
pub struct FabricContext {
    /// Provider information
    info: *mut fi_info,
    /// Fabric instance
    fabric: *mut fid_fabric,
    /// Domain instance
    domain: *mut fid_domain,
    /// Completion queue for send/recv completions
    cq: *mut fid_cq,
    /// Whether the context has been finalized
    finalized: AtomicBool,
    /// Configuration
    config: LibfabricConfig,
    /// Provider name (for logging)
    provider_name: String,
}

// Safety: FabricContext manages raw pointers but ensures proper cleanup
unsafe impl Send for FabricContext {}
unsafe impl Sync for FabricContext {}

impl FabricContext {
    /// Create a new FabricContext with the given configuration
    pub fn new(config: &LibfabricConfig) -> CylonResult<Self> {
        unsafe {
            // Create hints for provider selection
            let hints = fi_allocinfo();
            if hints.is_null() {
                return Err(CylonError::new(
                    Code::OutOfMemory,
                    "Failed to allocate fi_info hints".to_string(),
                ));
            }

            // Set required capabilities - FI_MSG for point-to-point messaging
            // FI_COLLECTIVE is optional - will use software fallback if not available
            (*hints).caps = FI_MSG as u64;

            // Try to get collective support if available
            (*hints).mode = FI_CONTEXT as u64;

            // Set endpoint type - ep_attr is already allocated by fi_allocinfo
            if !(*hints).ep_attr.is_null() {
                (*(*hints).ep_attr).type_ = match config.endpoint_type {
                    EndpointType::ReliableDatagram => FI_EP_RDM,
                    EndpointType::Message => FI_EP_MSG,
                };
            }

            // Set domain attributes - domain_attr is already allocated by fi_allocinfo
            if !(*hints).domain_attr.is_null() {
                (*(*hints).domain_attr).av_type = FI_AV_TABLE;
            }

            // Set provider if specified
            if let Some(ref provider) = config.provider {
                let provider_cstr = CString::new(provider.as_str())
                    .map_err(|_| CylonError::new(Code::Invalid, "Invalid provider name".to_string()))?;
                (*hints).fabric_attr = fi_allocinfo() as *mut fi_fabric_attr;
                if !(*hints).fabric_attr.is_null() {
                    (*(*hints).fabric_attr).prov_name = provider_cstr.into_raw();
                }
            }

            // Get provider info
            let mut info: *mut fi_info = ptr::null_mut();
            let ret = fi_getinfo(
                FI_VERSION(1, 9),
                ptr::null(),
                ptr::null(),
                0,
                hints,
                &mut info,
            );

            // Free hints
            fi_freeinfo(hints);

            if ret != 0 {
                return Err(CylonError::new(
                    Code::ExecutionError,
                    format!("fi_getinfo failed: {} (error code: {})", fi_strerror(ret), ret),
                ));
            }

            if info.is_null() {
                return Err(CylonError::new(
                    Code::ExecutionError,
                    "No suitable libfabric provider found".to_string(),
                ));
            }

            // Get provider name for logging
            let provider_name = if !(*info).fabric_attr.is_null()
                && !(*(*info).fabric_attr).prov_name.is_null()
            {
                std::ffi::CStr::from_ptr((*(*info).fabric_attr).prov_name)
                    .to_string_lossy()
                    .into_owned()
            } else {
                "unknown".to_string()
            };

            log::info!("Selected libfabric provider: {}", provider_name);

            // Create fabric
            let mut fabric: *mut fid_fabric = ptr::null_mut();
            let ret = fi_fabric((*info).fabric_attr, &mut fabric, ptr::null_mut());
            if ret != 0 {
                fi_freeinfo(info);
                return Err(CylonError::new(
                    Code::ExecutionError,
                    format!("fi_fabric failed: {} (error code: {})", fi_strerror(ret), ret),
                ));
            }

            // Create domain
            let mut domain: *mut fid_domain = ptr::null_mut();
            let ret = fi_domain(fabric, info, &mut domain, ptr::null_mut());
            if ret != 0 {
                fi_close(&mut (*fabric).fid);
                fi_freeinfo(info);
                return Err(CylonError::new(
                    Code::ExecutionError,
                    format!("fi_domain failed: {} (error code: {})", fi_strerror(ret), ret),
                ));
            }

            // Create completion queue
            let mut cq_attr: fi_cq_attr = std::mem::zeroed();
            cq_attr.size = config.cq_size;
            cq_attr.format = FI_CQ_FORMAT_TAGGED;
            cq_attr.wait_obj = FI_WAIT_NONE; // Non-blocking

            let mut cq: *mut fid_cq = ptr::null_mut();
            let ret = fi_cq_open(domain, &mut cq_attr, &mut cq, ptr::null_mut());
            if ret != 0 {
                fi_close(&mut (*domain).fid);
                fi_close(&mut (*fabric).fid);
                fi_freeinfo(info);
                return Err(CylonError::new(
                    Code::ExecutionError,
                    format!("fi_cq_open failed: {} (error code: {})", fi_strerror(ret), ret),
                ));
            }

            Ok(Self {
                info,
                fabric,
                domain,
                cq,
                finalized: AtomicBool::new(false),
                config: config.clone(),
                provider_name,
            })
        }
    }

    /// Get the provider name
    pub fn provider_name(&self) -> &str {
        &self.provider_name
    }

    /// Get the fi_info pointer
    pub fn info(&self) -> *mut fi_info {
        self.info
    }

    /// Get the fabric pointer
    pub fn fabric(&self) -> *mut fid_fabric {
        self.fabric
    }

    /// Get the domain pointer
    pub fn domain(&self) -> *mut fid_domain {
        self.domain
    }

    /// Get the completion queue pointer
    pub fn cq(&self) -> *mut fid_cq {
        self.cq
    }

    /// Get the configuration
    pub fn config(&self) -> &LibfabricConfig {
        &self.config
    }

    /// Check if the context has been finalized
    pub fn is_finalized(&self) -> bool {
        self.finalized.load(Ordering::SeqCst)
    }

    /// Poll the completion queue for completions
    ///
    /// Returns the number of completions read, or 0 if none available
    pub fn poll_cq(&self, entries: &mut [fi_cq_tagged_entry]) -> CylonResult<usize> {
        if self.is_finalized() {
            return Ok(0);
        }

        unsafe {
            let ret = fi_cq_read(
                self.cq,
                entries.as_mut_ptr() as *mut _,
                entries.len(),
            );

            if ret >= 0 {
                Ok(ret as usize)
            } else if ret == -(FI_EAGAIN as isize) {
                Ok(0) // No completions available
            } else {
                Err(CylonError::new(
                    Code::ExecutionError,
                    format!("fi_cq_read failed: {} (error code: {})",
                        fi_strerror(-(ret as i32)), ret),
                ))
            }
        }
    }

    /// Wait for at least one completion with timeout
    ///
    /// Returns the number of completions read
    pub fn wait_cq(&self, entries: &mut [fi_cq_tagged_entry], timeout_ms: i32) -> CylonResult<usize> {
        if self.is_finalized() {
            return Ok(0);
        }

        unsafe {
            let ret = fi_cq_sread(
                self.cq,
                entries.as_mut_ptr() as *mut _,
                entries.len(),
                ptr::null(),
                timeout_ms,
            );

            if ret >= 0 {
                Ok(ret as usize)
            } else if ret == -(FI_EAGAIN as isize) {
                Ok(0) // Timeout, no completions
            } else {
                Err(CylonError::new(
                    Code::ExecutionError,
                    format!("fi_cq_sread failed: {} (error code: {})",
                        fi_strerror(-(ret as i32)), ret),
                ))
            }
        }
    }

    /// Read completion queue error details
    pub fn read_cq_error(&self) -> Option<fi_cq_err_entry> {
        unsafe {
            let mut err_entry: fi_cq_err_entry = std::mem::zeroed();
            let ret = fi_cq_readerr(self.cq, &mut err_entry, 0);
            if ret > 0 {
                Some(err_entry)
            } else {
                None
            }
        }
    }

    /// Finalize and cleanup resources
    pub fn finalize(&self) -> CylonResult<()> {
        if self.finalized.swap(true, Ordering::SeqCst) {
            return Ok(()); // Already finalized
        }

        unsafe {
            // Close in reverse order of creation
            if !self.cq.is_null() {
                fi_close(&mut (*self.cq).fid);
            }
            if !self.domain.is_null() {
                fi_close(&mut (*self.domain).fid);
            }
            if !self.fabric.is_null() {
                fi_close(&mut (*self.fabric).fid);
            }
            if !self.info.is_null() {
                fi_freeinfo(self.info);
            }
        }

        Ok(())
    }
}

impl Drop for FabricContext {
    fn drop(&mut self) {
        let _ = self.finalize();
    }
}

/// Query available providers
pub fn query_providers() -> CylonResult<Vec<String>> {
    unsafe {
        let hints = fi_allocinfo();
        if hints.is_null() {
            return Err(CylonError::new(
                Code::OutOfMemory,
                "Failed to allocate fi_info hints".to_string(),
            ));
        }

        let mut info: *mut fi_info = ptr::null_mut();
        let ret = fi_getinfo(
            FI_VERSION(1, 9),
            ptr::null(),
            ptr::null(),
            0,
            hints,
            &mut info,
        );

        fi_freeinfo(hints);

        if ret != 0 {
            return Err(CylonError::new(
                Code::ExecutionError,
                format!("fi_getinfo failed: {}", fi_strerror(ret)),
            ));
        }

        let mut providers = Vec::new();
        let mut current = info;

        while !current.is_null() {
            if !(*current).fabric_attr.is_null()
                && !(*(*current).fabric_attr).prov_name.is_null()
            {
                let name = std::ffi::CStr::from_ptr((*(*current).fabric_attr).prov_name)
                    .to_string_lossy()
                    .into_owned();
                if !providers.contains(&name) {
                    providers.push(name);
                }
            }
            current = (*current).next;
        }

        fi_freeinfo(info);

        Ok(providers)
    }
}
