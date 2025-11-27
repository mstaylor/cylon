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
//!
//! Comprehensive tests for UCX initialization and worker management

#[cfg(feature = "ucx")]
mod ucx_tests {
    use std::ptr;
    use cylon::net::ucx::ucx_sys::*;
    use cylon::net::ucx::operations::*;

    #[test]
    fn test_init_context() {
        unsafe {
            let mut ucp_context: ucp_context_h = ptr::null_mut();
            let result = init_context(&mut ucp_context, ptr::null());

            assert!(result.is_ok(), "Failed to initialize UCP context");
            assert!(!ucp_context.is_null(), "UCP context should not be null");

            println!("✓ UCP context initialized successfully");

            // Cleanup
            ucp_cleanup(ucp_context);
        }
    }

    #[test]
    fn test_init_worker() {
        unsafe {
            let mut ucp_context: ucp_context_h = ptr::null_mut();
            let result = init_context(&mut ucp_context, ptr::null());
            assert!(result.is_ok(), "Failed to initialize UCP context");

            let mut ucp_worker: ucp_worker_h = ptr::null_mut();
            let worker_result = init_worker(ucp_context, &mut ucp_worker);

            assert!(worker_result.is_ok(), "Failed to initialize UCP worker");

            let worker_addr = worker_result.unwrap();
            assert!(!worker_addr.addr.is_null(), "Worker address should not be null");
            assert!(worker_addr.addr_size > 0, "Worker address size should be greater than 0");

            println!("✓ Worker initialized - address size: {} bytes", worker_addr.addr_size);

            // Cleanup
            drop(worker_addr);
            ucp_worker_destroy(ucp_worker);
            ucp_cleanup(ucp_context);
        }
    }

    #[test]
    fn test_multiple_workers() {
        // Test creating multiple workers on the same context
        // Corresponds to C++ ucx_communicator pattern with recv and send workers
        unsafe {
            let mut ucp_context: ucp_context_h = ptr::null_mut();
            let result = init_context(&mut ucp_context, ptr::null());
            assert!(result.is_ok());

            // Create receive worker
            let mut ucp_recv_worker: ucp_worker_h = ptr::null_mut();
            let recv_result = init_worker(ucp_context, &mut ucp_recv_worker);
            assert!(recv_result.is_ok());
            let recv_addr = recv_result.unwrap();

            // Create send worker
            let mut ucp_send_worker: ucp_worker_h = ptr::null_mut();
            let send_result = init_worker(ucp_context, &mut ucp_send_worker);
            assert!(send_result.is_ok());
            let send_addr = send_result.unwrap();

            println!("✓ Created 2 workers:");
            println!("  Recv worker address size: {} bytes", recv_addr.addr_size);
            println!("  Send worker address size: {} bytes", send_addr.addr_size);

            // Cleanup
            drop(recv_addr);
            drop(send_addr);
            ucp_worker_destroy(ucp_send_worker);
            ucp_worker_destroy(ucp_recv_worker);
            ucp_cleanup(ucp_context);
        }
    }

    #[test]
    fn test_manual_cleanup() {
        unsafe {
            let mut ucp_context: ucp_context_h = ptr::null_mut();
            let _ = init_context(&mut ucp_context, ptr::null());

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

            println!("✓ Manual cleanup test passed");

            // Cleanup
            ucp_worker_destroy(ucp_worker);
            ucp_cleanup(ucp_context);
        }
    }

    #[test]
    fn test_context_with_null_config() {
        // Test that we can initialize with null config
        unsafe {
            let mut ucp_context: ucp_context_h = ptr::null_mut();
            let result = init_context(&mut ucp_context, ptr::null());

            assert!(result.is_ok());
            assert!(!ucp_context.is_null());

            println!("✓ Context initialized with null config");

            ucp_cleanup(ucp_context);
        }
    }

    #[test]
    fn test_ucx_constants() {
        // Test that UCX constants are properly defined
        unsafe {
            // Check API version constants exist
            let _major = UCP_API_MAJOR;
            let _minor = UCP_API_MINOR;

            println!("✓ UCP API version: {}.{}", UCP_API_MAJOR, UCP_API_MINOR);

            // Check status constants
            assert!(ucs_status_t_UCS_OK == 0);

            println!("✓ UCX constants test passed");
        }
    }

    #[test]
    fn test_worker_address_properties() {
        // Test UcxWorkerAddr structure properties
        unsafe {
            let mut ucp_context: ucp_context_h = ptr::null_mut();
            init_context(&mut ucp_context, ptr::null()).expect("Context init failed");

            let mut ucp_worker: ucp_worker_h = ptr::null_mut();
            let worker_addr = init_worker(ucp_context, &mut ucp_worker)
                .expect("Worker init failed");

            // Verify address properties
            assert!(!worker_addr.addr.is_null());
            assert!(worker_addr.addr_size > 0);
            assert!(worker_addr.addr_size < 10000); // Sanity check (addresses shouldn't be huge)

            println!("✓ Worker address properties:");
            println!("  Address: {:?}", worker_addr.addr);
            println!("  Size: {} bytes", worker_addr.addr_size);

            drop(worker_addr);
            ucp_worker_destroy(ucp_worker);
            ucp_cleanup(ucp_context);
        }
    }
}

#[cfg(not(feature = "ucx"))]
mod ucx_disabled {
    #[test]
    fn ucx_feature_not_enabled() {
        println!("UCX operations tests skipped - 'ucx' feature not enabled");
        println!("Run with: cargo test --features ucx");
    }
}
