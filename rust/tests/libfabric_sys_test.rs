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

//! Tests for Libfabric FFI bindings and helper functions
//!
//! These tests verify the FFI type definitions, constants, and helper functions.
//! Most tests don't require libfabric runtime.

#[cfg(feature = "libfabric")]
mod libfabric_sys_tests {
    use cylon::net::libfabric::libfabric_sys::*;

    #[test]
    fn test_fi_version() {
        let version = FI_VERSION(1, 9);
        assert_eq!(version, (1 << 16) | 9);
        assert_eq!(version, 0x10009);

        let version_2_0 = FI_VERSION(2, 0);
        assert_eq!(version_2_0, 0x20000);

        let version_1_14 = FI_VERSION(1, 14);
        assert_eq!(version_1_14, 0x1000e);
    }

    #[test]
    fn test_fi_addr_unspec() {
        // FI_ADDR_UNSPEC should be the maximum u64 value (all bits set)
        assert_eq!(FI_ADDR_UNSPEC, !0u64);
        assert_eq!(FI_ADDR_UNSPEC, u64::MAX);
    }

    #[test]
    fn test_endpoint_type_constants() {
        // Verify endpoint type constants are distinct
        assert_ne!(FI_EP_UNSPEC, FI_EP_MSG);
        assert_ne!(FI_EP_MSG, FI_EP_DGRAM);
        assert_ne!(FI_EP_DGRAM, FI_EP_RDM);
        assert_ne!(FI_EP_RDM, FI_EP_UNSPEC);
    }

    #[test]
    fn test_av_type_constants() {
        // Verify AV type constants are distinct
        assert_ne!(FI_AV_UNSPEC, FI_AV_MAP);
        assert_ne!(FI_AV_MAP, FI_AV_TABLE);
        assert_ne!(FI_AV_TABLE, FI_AV_UNSPEC);
    }

    #[test]
    fn test_progress_mode_constants() {
        // Verify progress mode constants are distinct
        assert_ne!(FI_PROGRESS_UNSPEC, FI_PROGRESS_AUTO);
        assert_ne!(FI_PROGRESS_AUTO, FI_PROGRESS_MANUAL);
        assert_ne!(FI_PROGRESS_MANUAL, FI_PROGRESS_UNSPEC);
    }

    #[test]
    fn test_cq_format_constants() {
        // Verify CQ format constants are distinct
        assert_ne!(FI_CQ_FORMAT_UNSPEC, FI_CQ_FORMAT_CONTEXT);
        assert_ne!(FI_CQ_FORMAT_CONTEXT, FI_CQ_FORMAT_MSG);
        assert_ne!(FI_CQ_FORMAT_MSG, FI_CQ_FORMAT_DATA);
        assert_ne!(FI_CQ_FORMAT_DATA, FI_CQ_FORMAT_TAGGED);
    }

    #[test]
    fn test_wait_obj_constants() {
        // Verify wait object constants are distinct
        assert_ne!(FI_WAIT_NONE, FI_WAIT_UNSPEC);
        assert_ne!(FI_WAIT_UNSPEC, FI_WAIT_SET);
        assert_ne!(FI_WAIT_SET, FI_WAIT_FD);
        assert_ne!(FI_WAIT_FD, FI_WAIT_YIELD);
    }

    #[test]
    fn test_atomic_op_constants() {
        // Verify atomic operation constants are distinct
        assert_ne!(FI_MIN, FI_MAX);
        assert_ne!(FI_MAX, FI_SUM);
        assert_ne!(FI_SUM, FI_PROD);
        assert_ne!(FI_PROD, FI_LOR);
        assert_ne!(FI_LOR, FI_LAND);
        assert_ne!(FI_LAND, FI_BOR);
        assert_ne!(FI_BOR, FI_BAND);
    }

    #[test]
    fn test_datatype_constants() {
        // Verify signed integer types are distinct
        assert_ne!(FI_INT8, FI_INT16);
        assert_ne!(FI_INT16, FI_INT32);
        assert_ne!(FI_INT32, FI_INT64);

        // Verify unsigned integer types are distinct
        assert_ne!(FI_UINT8, FI_UINT16);
        assert_ne!(FI_UINT16, FI_UINT32);
        assert_ne!(FI_UINT32, FI_UINT64);

        // Verify floating point types are distinct
        assert_ne!(FI_FLOAT, FI_DOUBLE);

        // Verify signed vs unsigned are distinct
        assert_ne!(FI_INT8, FI_UINT8);
        assert_ne!(FI_INT16, FI_UINT16);
        assert_ne!(FI_INT32, FI_UINT32);
        assert_ne!(FI_INT64, FI_UINT64);
    }

    #[test]
    fn test_is_fi_eagain() {
        let eagain = -(FI_EAGAIN as isize);
        assert!(is_fi_eagain(eagain));

        // Test that other values return false
        assert!(!is_fi_eagain(0));
        assert!(!is_fi_eagain(-1));
        assert!(!is_fi_eagain(1));
    }

    #[test]
    fn test_is_fi_success() {
        assert!(is_fi_success(FI_SUCCESS as i32));
        assert!(is_fi_success(0));

        // Test that other values return false
        assert!(!is_fi_success(-1));
        assert!(!is_fi_success(1));
        assert!(!is_fi_success(-(FI_EAGAIN as i32)));
    }

    #[test]
    fn test_fi_context_default() {
        let ctx = fi_context::default();
        // All pointers should be null by default
        for ptr in ctx.internal.iter() {
            assert!(ptr.is_null());
        }
    }

    #[test]
    fn test_fi_context2_default() {
        let ctx = fi_context2::default();
        // All pointers should be null by default
        for ptr in ctx.internal.iter() {
            assert!(ptr.is_null());
        }
    }

    #[test]
    fn test_fi_context_size() {
        // fi_context should have 4 pointers
        let ctx = fi_context::default();
        assert_eq!(ctx.internal.len(), 4);
    }

    #[test]
    fn test_fi_context2_size() {
        // fi_context2 should have 8 pointers
        let ctx = fi_context2::default();
        assert_eq!(ctx.internal.len(), 8);
    }

    /// Test that FI_VERSION produces expected bit layout
    #[test]
    fn test_fi_version_bit_layout() {
        // Major version in upper 16 bits, minor in lower 16 bits
        let v = FI_VERSION(1, 9);
        let major = (v >> 16) & 0xFFFF;
        let minor = v & 0xFFFF;
        assert_eq!(major, 1);
        assert_eq!(minor, 9);
    }

    /// Test that fi_allocinfo returns a pointer (requires libfabric runtime)
    #[test]
    #[ignore] // Requires libfabric to be installed
    fn test_fi_allocinfo() {
        unsafe {
            let info = fi_allocinfo();
            assert!(!info.is_null(), "fi_allocinfo returned null");
            // Clean up
            fi_freeinfo(info);
        }
    }

    /// Test fi_close with null pointer
    #[test]
    fn test_fi_close_null() {
        unsafe {
            let result = fi_close(std::ptr::null_mut());
            // Should return -FI_EINVAL for null pointer
            assert!(result < 0, "fi_close should fail with null pointer");
        }
    }

    /// Test fi_cq_read with null pointer
    #[test]
    fn test_fi_cq_read_null() {
        unsafe {
            let mut buf: [u8; 64] = [0; 64];
            let result = fi_cq_read(
                std::ptr::null_mut(),
                buf.as_mut_ptr() as *mut _,
                1,
            );
            // Should return error for null CQ
            assert!(result < 0, "fi_cq_read should fail with null CQ");
        }
    }

    /// Test fi_send with null pointer
    #[test]
    fn test_fi_send_null() {
        unsafe {
            let data: [u8; 8] = [0; 8];
            let result = fi_send(
                std::ptr::null_mut(),
                data.as_ptr() as *const _,
                data.len(),
                std::ptr::null_mut(),
                0,
                std::ptr::null_mut(),
            );
            // Should return error for null endpoint
            assert!(result < 0, "fi_send should fail with null endpoint");
        }
    }

    /// Test fi_recv with null pointer
    #[test]
    fn test_fi_recv_null() {
        unsafe {
            let mut buf: [u8; 8] = [0; 8];
            let result = fi_recv(
                std::ptr::null_mut(),
                buf.as_mut_ptr() as *mut _,
                buf.len(),
                std::ptr::null_mut(),
                0,
                std::ptr::null_mut(),
            );
            // Should return error for null endpoint
            assert!(result < 0, "fi_recv should fail with null endpoint");
        }
    }

    /// Test fi_barrier with null pointer
    #[test]
    fn test_fi_barrier_null() {
        unsafe {
            let result = fi_barrier(
                std::ptr::null_mut(),
                0,
                std::ptr::null_mut(),
            );
            // Should return error for null endpoint
            assert!(result < 0, "fi_barrier should fail with null endpoint");
        }
    }

    /// Test that fi_strerror returns valid strings (requires libfabric runtime)
    #[test]
    #[ignore] // Requires libfabric to be installed
    fn test_fi_strerror() {
        use cylon::net::libfabric::error::fi_strerror;

        // Test with known error codes
        let eagain_str = fi_strerror(FI_EAGAIN as i32);
        assert!(!eagain_str.is_empty(), "fi_strerror should return non-empty string");
        println!("FI_EAGAIN: {}", eagain_str);

        let einval_str = fi_strerror(FI_EINVAL as i32);
        assert!(!einval_str.is_empty());
        println!("FI_EINVAL: {}", einval_str);
    }

    /// Test query_providers function (requires libfabric runtime)
    #[test]
    #[ignore] // Requires libfabric to be installed
    fn test_query_providers() {
        use cylon::net::libfabric::query_providers;

        let result = query_providers();
        match result {
            Ok(providers) => {
                println!("Found {} providers:", providers.len());
                for p in &providers {
                    println!("  - {}", p);
                }
                // Should have at least one provider (even if just sockets)
                assert!(!providers.is_empty());
            }
            Err(e) => {
                panic!("query_providers failed: {:?}", e);
            }
        }
    }
}

#[cfg(not(feature = "libfabric"))]
mod libfabric_disabled {
    #[test]
    fn libfabric_feature_not_enabled() {
        println!("Libfabric sys tests skipped - feature not enabled");
    }
}
