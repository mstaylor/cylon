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

//! Tests for UCX operations

#[cfg(feature = "ucx")]
mod ucx_tests {
    use std::ptr;
    use cylon::net::ucx::ucx_sys::*;
    use cylon::net::ucx::operations::*;

    #[test]
    fn test_init_context() {
        unsafe {
            let mut ucp_context: ucp_context_h = ptr::null_mut();
            let result = init_context(&mut ucp_context, ptr::null_mut());

            assert!(result.is_ok(), "Failed to initialize UCP context");
            assert!(!ucp_context.is_null(), "UCP context should not be null");

            // Cleanup
            ucp_cleanup(ucp_context);
        }
    }

    #[test]
    fn test_init_worker() {
        unsafe {
            let mut ucp_context: ucp_context_h = ptr::null_mut();
            let result = init_context(&mut ucp_context, ptr::null_mut());
            assert!(result.is_ok(), "Failed to initialize UCP context");

            let mut ucp_worker: ucp_worker_h = ptr::null_mut();
            let worker_result = init_worker(ucp_context, &mut ucp_worker);

            assert!(worker_result.is_ok(), "Failed to initialize UCP worker");

            let worker_addr = worker_result.unwrap();
            assert!(!worker_addr.addr.is_null(), "Worker address should not be null");
            assert!(worker_addr.addr_size > 0, "Worker address size should be greater than 0");

            println!("Worker address size: {}", worker_addr.addr_size);

            // Cleanup
            ucp_worker_destroy(ucp_worker);
            ucp_cleanup(ucp_context);
        }
    }

    #[test]
    fn test_manual_cleanup() {
        unsafe {
            let mut ucp_context: ucp_context_h = ptr::null_mut();
            let _ = init_context(&mut ucp_context, ptr::null_mut());

            let mut ucp_worker: ucp_worker_h = ptr::null_mut();
            let worker_result = init_worker(ucp_context, &mut ucp_worker);

            if let Ok(worker_addr) = worker_result {
                // Note: C++ version requires manual cleanup
                // Caller is responsible for cleaning up, just like in C++:
                // delete (ucpRecvWorkerAddr); // C++ line 258
                // In Rust, we just drop the Box which frees the memory
                // but does NOT release the UCP address (matching C++ behavior)
                drop(worker_addr);
            }

            // Cleanup
            ucp_worker_destroy(ucp_worker);
            ucp_cleanup(ucp_context);
        }
    }
}
