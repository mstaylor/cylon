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

//! UCX operations and helper functions
//!
//! Ported from cpp/src/cylon/net/ucx/ucx_operations.hpp/cpp

use std::ptr;

use crate::error::{CylonError, CylonResult, Code};

// Import UCX FFI bindings
use super::ucx_sys::*;

/// Hold the completion status of a communication
///
/// Corresponds to C++ ucxContext from ucx_operations.hpp:29-31
#[repr(C)]
pub struct UcxContext {
    /// If completed show 1, else 0
    pub completed: i32,
}

/// Hold the data related to a communication endpoint
///
/// Corresponds to C++ ucxWorkerAddr from ucx_operations.hpp:38-41
pub struct UcxWorkerAddr {
    /// The address of the ucp worker
    pub addr: *mut ucp_address_t,
    /// The size of the worker address
    pub addr_size: usize,
}

// Note: C++ version has no destructor - caller must manually delete
// This matches the C++ implementation exactly

/// Create a UCP worker on the given UCP context
///
/// Corresponds to C++ cylon::ucx::initWorker (ucx_operations.cpp:27-76)
///
/// # Arguments
/// * `ucp_context` - The context to be passed to init the worker
/// * `ucp_worker` - Output parameter for the created UCP worker
///
/// # Returns
/// * `UcxWorkerAddr` containing the worker address and size
pub unsafe fn init_worker(
    ucp_context: ucp_context_h,
    ucp_worker: *mut ucp_worker_h,
) -> CylonResult<Box<UcxWorkerAddr>> {
    // Corresponds to C++ lines 32-34
    let mut worker_params: ucp_worker_params_t = std::mem::zeroed();
    let mut status: ucs_status_t;

    // New worker
    // Corresponds to C++ line 37: auto worker = new ucxWorkerAddr();
    let mut worker = Box::new(UcxWorkerAddr {
        addr: ptr::null_mut(),
        addr_size: 0,
    });

    // Init values to worker params
    // Corresponds to C++ line 40: memset(&workerParams, 0, sizeof(workerParams));
    // (Already done with zeroed())

    // Thread mode params
    // Corresponds to C++ lines 43-44
    worker_params.field_mask = ucp_worker_params_field_UCP_WORKER_PARAM_FIELD_THREAD_MODE as u64;
    worker_params.thread_mode = ucs_thread_mode_t_UCS_THREAD_MODE_SINGLE;

    // Create UCP worker - 1:many -> context:worker
    // Corresponds to C++ line 49: status = ucp_worker_create(ucpContext, &workerParams, ucpWorker);
    status = ucp_worker_create(ucp_context, &worker_params, ucp_worker);

    // Check status of worker
    // Corresponds to C++ lines 52-55
    if status != ucs_status_t_UCS_OK {
        let msg = std::ffi::CStr::from_ptr(ucs_status_string(status))
            .to_string_lossy()
            .into_owned();
        return Err(CylonError::new(
            Code::ExecutionError,
            format!("Failed to create a UCP worker for the given UCP context: {}", msg),
        ));
    }

    // Get worker address
    // Corresponds to C++ lines 57-59
    status = ucp_worker_get_address(
        *ucp_worker,
        &mut worker.addr,
        &mut worker.addr_size,
    );

    // Check status of worker
    // Corresponds to C++ lines 61-64
    if status != ucs_status_t_UCS_OK {
        let msg = std::ffi::CStr::from_ptr(ucs_status_string(status))
            .to_string_lossy()
            .into_owned();

        // Cleanup worker on error
        // Corresponds to C++ lines 71-73
        ucp_worker_destroy(*ucp_worker);

        return Err(CylonError::new(
            Code::ExecutionError,
            format!("Failed to get the address of the given UCP worker: {}", msg),
        ));
    }

    // Corresponds to C++ line 66: return worker;
    Ok(worker)
}

/// Initialize a default UCP context
///
/// Corresponds to C++ cylon::ucx::initContext (ucx_operations.cpp:84-117)
///
/// # Arguments
/// * `ucp_context` - Output parameter for the UCP context
/// * `config` - The configuration descriptor (can be null)
///
/// # Returns
/// * `CylonResult<()>` - Status of the context init
pub unsafe fn init_context(
    ucp_context: *mut ucp_context_h,
    config: *const ucp_config_t,
) -> CylonResult<()> {
    // UCP params - The structure defines the parameters that are used for
    // UCP library tuning during UCP library "initialization".
    // Corresponds to C++ lines 88-90
    let mut ucp_params: ucp_params_t = std::mem::zeroed();
    let status: ucs_status_t;

    // Init UCP Params
    // Corresponds to C++ line 93: std::memset(&ucpParams, 0, sizeof(ucpParams));
    // (Already done with zeroed())

    // The enumeration allows specifying which fields in ucp_params_t are present
    // Corresponds to C++ lines 97-102
    ucp_params.field_mask = (ucp_params_field_UCP_PARAM_FIELD_FEATURES
        | ucp_params_field_UCP_PARAM_FIELD_REQUEST_SIZE
        | ucp_params_field_UCP_PARAM_FIELD_REQUEST_INIT) as u64;

    // Set support for tags
    ucp_params.features = ucp_feature_UCP_FEATURE_TAG as u64;
    ucp_params.request_size = std::mem::size_of::<UcxContext>();

    // Init UCP context
    // Corresponds to C++ line 108: status = ucp_init(&ucpParams, config, ucpContext);
    // Note: In Rust we call ucp_init_version directly (C macro ucp_init expands to this)
    status = ucp_init_version(
        UCP_API_MAJOR,
        UCP_API_MINOR,
        &ucp_params,
        config,
        ucp_context,
    );

    // Check context init
    // Corresponds to C++ lines 111-114
    if status != ucs_status_t_UCS_OK {
        let msg = std::ffi::CStr::from_ptr(ucs_status_string(status))
            .to_string_lossy()
            .into_owned();
        return Err(CylonError::new(
            Code::ExecutionError,
            format!("Failed to initialize UCP context: {}", msg),
        ));
    }

    // Corresponds to C++ line 116: return Status::OK();
    Ok(())
}
