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

//! Host Import Declarations
//!
//! This module declares external functions that the host (Python/Node.js/browser)
//! must provide. These enable WASM to orchestrate distributed operations while
//! the host handles actual communication (FMI/MPI/UCX/etc).
//!
//! # Architecture (Approach B: WASM with Host Imports)
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                         WASM Module                             │
//! │  ┌──────────────────────────────────────────────────────────┐  │
//! │  │  Distributed Operations (orchestration logic)            │  │
//! │  │  - distributed_join()                                    │  │
//! │  │  - distributed_union()                                   │  │
//! │  │  - distributed_intersect()                               │  │
//! │  └─────────────────────┬────────────────────────────────────┘  │
//! │                        │ calls                                  │
//! │  ┌─────────────────────▼────────────────────────────────────┐  │
//! │  │  Host Imports (extern "C")                               │  │
//! │  │  - host_all_to_all()                                     │  │
//! │  │  - host_broadcast()                                      │  │
//! │  │  - host_barrier()                                        │  │
//! │  │  - host_get_rank() / host_get_world_size()              │  │
//! │  └─────────────────────┬────────────────────────────────────┘  │
//! └────────────────────────┼────────────────────────────────────────┘
//!                          │ provided by
//! ┌────────────────────────▼────────────────────────────────────────┐
//! │                    Host Runtime                                  │
//! │  Python (wasmtime) / Node.js / Browser                          │
//! │  - Implements FMI, MPI, UCX, or Redis-based communication       │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Memory Protocol
//!
//! Data transfer uses a pointer+length protocol:
//! 1. WASM allocates memory for input data
//! 2. WASM writes data to that memory
//! 3. WASM calls host import with ptr+len
//! 4. Host reads from WASM memory, performs operation
//! 5. Host allocates WASM memory for result, writes result
//! 6. Host returns result ptr+len (packed as i64: high=ptr, low=len)
//! 7. WASM reads result and frees memory

use std::alloc::{alloc, dealloc, Layout};

// =============================================================================
// Memory Management (exported for host use)
// =============================================================================

/// Allocate memory in WASM - callable by host to prepare result buffers
#[no_mangle]
pub extern "C" fn wasm_alloc(size: usize) -> *mut u8 {
    if size == 0 {
        return std::ptr::null_mut();
    }
    let layout = Layout::from_size_align(size, 8).unwrap();
    unsafe { alloc(layout) }
}

/// Free memory in WASM - callable by host to clean up
#[no_mangle]
pub extern "C" fn wasm_free(ptr: *mut u8, size: usize) {
    if ptr.is_null() || size == 0 {
        return;
    }
    let layout = Layout::from_size_align(size, 8).unwrap();
    unsafe { dealloc(ptr, layout) }
}

// =============================================================================
// Host Import Declarations
// =============================================================================

#[link(wasm_import_module = "cylon_host")]
extern "C" {
    /// Get this worker's rank (0 to world_size-1)
    pub fn host_get_rank() -> i32;

    /// Get total number of workers
    pub fn host_get_world_size() -> i32;

    /// Barrier synchronization - all workers wait until all reach this point
    pub fn host_barrier();

    /// Broadcast data from root to all workers
    ///
    /// Args:
    ///   data_ptr: Pointer to data buffer (only valid on root)
    ///   data_len: Length of data
    ///   root: Rank of the broadcasting worker
    ///   result_ptr_out: Output pointer for result buffer
    ///   result_len_out: Output pointer for result length
    ///
    /// The host allocates result buffer using wasm_alloc and writes the
    /// received data there. Returns 0 on success, non-zero on error.
    pub fn host_broadcast(
        data_ptr: *const u8,
        data_len: usize,
        root: i32,
        result_ptr_out: *mut *mut u8,
        result_len_out: *mut usize,
    ) -> i32;

    /// All-to-all exchange of data
    ///
    /// This is the key primitive for distributed shuffles:
    /// - Each worker sends different data to each other worker
    /// - Each worker receives data from all other workers
    ///
    /// Args:
    ///   partitions_ptr: Pointer to array of (ptr, len) pairs, one per destination
    ///   num_partitions: Number of partitions (must equal world_size)
    ///   results_ptr_out: Output pointer for array of (ptr, len) pairs received
    ///   num_results_out: Output pointer for number of results
    ///
    /// Partition data format: [ptr0, len0, ptr1, len1, ...]
    /// Each partition is Arrow IPC bytes destined for worker i.
    /// Result data format: same, but received from each worker.
    ///
    /// The host allocates result buffers using wasm_alloc.
    /// Returns 0 on success, non-zero on error.
    pub fn host_all_to_all(
        partitions_ptr: *const usize,
        num_partitions: usize,
        results_ptr_out: *mut *mut usize,
        num_results_out: *mut usize,
    ) -> i32;

    /// Gather data from all workers to root
    ///
    /// Args:
    ///   data_ptr: Pointer to this worker's data
    ///   data_len: Length of data
    ///   root: Rank of the gathering worker
    ///   results_ptr_out: Output pointer for array of (ptr, len) pairs (only valid on root)
    ///   num_results_out: Output pointer for number of results
    ///
    /// The host allocates result buffers using wasm_alloc.
    /// Returns 0 on success, non-zero on error.
    pub fn host_gather(
        data_ptr: *const u8,
        data_len: usize,
        root: i32,
        results_ptr_out: *mut *mut usize,
        num_results_out: *mut usize,
    ) -> i32;

    /// Scatter data from root to all workers
    ///
    /// Args:
    ///   partitions_ptr: Pointer to array of (ptr, len) pairs (only valid on root)
    ///   num_partitions: Number of partitions
    ///   root: Rank of the scattering worker
    ///   result_ptr_out: Output pointer for this worker's result
    ///   result_len_out: Output pointer for result length
    ///
    /// The host allocates result buffer using wasm_alloc.
    /// Returns 0 on success, non-zero on error.
    pub fn host_scatter(
        partitions_ptr: *const usize,
        num_partitions: usize,
        root: i32,
        result_ptr_out: *mut *mut u8,
        result_len_out: *mut usize,
    ) -> i32;

    /// All-gather - each worker contributes data, all receive all data
    ///
    /// Args:
    ///   data_ptr: Pointer to this worker's data
    ///   data_len: Length of data
    ///   results_ptr_out: Output pointer for array of (ptr, len) pairs
    ///   num_results_out: Output pointer for number of results
    ///
    /// The host allocates result buffers using wasm_alloc.
    /// Returns 0 on success, non-zero on error.
    pub fn host_all_gather(
        data_ptr: *const u8,
        data_len: usize,
        results_ptr_out: *mut *mut usize,
        num_results_out: *mut usize,
    ) -> i32;
}

// =============================================================================
// Safe Wrappers
// =============================================================================

/// Get current worker's rank
pub fn get_rank() -> i32 {
    unsafe { host_get_rank() }
}

/// Get total number of workers
pub fn get_world_size() -> i32 {
    unsafe { host_get_world_size() }
}

/// Barrier synchronization
pub fn barrier() {
    unsafe { host_barrier() }
}

/// Broadcast bytes from root to all workers
pub fn broadcast(data: &[u8], root: i32) -> Result<Vec<u8>, String> {
    let mut result_ptr: *mut u8 = std::ptr::null_mut();
    let mut result_len: usize = 0;

    let rank = get_rank();
    let (data_ptr, data_len) = if rank == root {
        (data.as_ptr(), data.len())
    } else {
        (std::ptr::null(), 0)
    };

    let err = unsafe {
        host_broadcast(
            data_ptr,
            data_len,
            root,
            &mut result_ptr,
            &mut result_len,
        )
    };

    if err != 0 {
        return Err(format!("broadcast failed with error code {}", err));
    }

    if result_ptr.is_null() || result_len == 0 {
        return Ok(Vec::new());
    }

    // Copy result and free
    let result = unsafe {
        let slice = std::slice::from_raw_parts(result_ptr, result_len);
        slice.to_vec()
    };
    wasm_free(result_ptr, result_len);

    Ok(result)
}

/// All-to-all exchange of byte arrays
///
/// partitions: One byte array per destination worker
/// Returns: One byte array from each source worker
pub fn all_to_all(partitions: Vec<Vec<u8>>) -> Result<Vec<Vec<u8>>, String> {
    let world_size = get_world_size() as usize;
    if partitions.len() != world_size {
        return Err(format!(
            "all_to_all requires {} partitions, got {}",
            world_size,
            partitions.len()
        ));
    }

    // Build array of (ptr, len) pairs
    let mut partition_info: Vec<usize> = Vec::with_capacity(world_size * 2);
    for p in &partitions {
        partition_info.push(p.as_ptr() as usize);
        partition_info.push(p.len());
    }

    let mut results_ptr: *mut usize = std::ptr::null_mut();
    let mut num_results: usize = 0;

    let err = unsafe {
        host_all_to_all(
            partition_info.as_ptr(),
            world_size,
            &mut results_ptr,
            &mut num_results,
        )
    };

    if err != 0 {
        return Err(format!("all_to_all failed with error code {}", err));
    }

    // Parse results
    let mut results = Vec::with_capacity(num_results);
    if !results_ptr.is_null() && num_results > 0 {
        let result_info = unsafe {
            std::slice::from_raw_parts(results_ptr, num_results * 2)
        };

        for i in 0..num_results {
            let ptr = result_info[i * 2] as *const u8;
            let len = result_info[i * 2 + 1];

            if !ptr.is_null() && len > 0 {
                let data = unsafe {
                    std::slice::from_raw_parts(ptr, len).to_vec()
                };
                // Free individual result buffer
                wasm_free(ptr as *mut u8, len);
                results.push(data);
            } else {
                results.push(Vec::new());
            }
        }

        // Free the info array itself
        wasm_free(results_ptr as *mut u8, num_results * 2 * std::mem::size_of::<usize>());
    }

    Ok(results)
}

/// Gather byte arrays from all workers to root
pub fn gather(data: &[u8], root: i32) -> Result<Option<Vec<Vec<u8>>>, String> {
    let mut results_ptr: *mut usize = std::ptr::null_mut();
    let mut num_results: usize = 0;

    let err = unsafe {
        host_gather(
            data.as_ptr(),
            data.len(),
            root,
            &mut results_ptr,
            &mut num_results,
        )
    };

    if err != 0 {
        return Err(format!("gather failed with error code {}", err));
    }

    let rank = get_rank();
    if rank != root {
        return Ok(None);
    }

    // Parse results on root
    let mut results = Vec::with_capacity(num_results);
    if !results_ptr.is_null() && num_results > 0 {
        let result_info = unsafe {
            std::slice::from_raw_parts(results_ptr, num_results * 2)
        };

        for i in 0..num_results {
            let ptr = result_info[i * 2] as *const u8;
            let len = result_info[i * 2 + 1];

            if !ptr.is_null() && len > 0 {
                let data = unsafe {
                    std::slice::from_raw_parts(ptr, len).to_vec()
                };
                wasm_free(ptr as *mut u8, len);
                results.push(data);
            } else {
                results.push(Vec::new());
            }
        }

        wasm_free(results_ptr as *mut u8, num_results * 2 * std::mem::size_of::<usize>());
    }

    Ok(Some(results))
}

/// Scatter byte arrays from root to all workers
pub fn scatter(partitions: Option<Vec<Vec<u8>>>, root: i32) -> Result<Vec<u8>, String> {
    let mut result_ptr: *mut u8 = std::ptr::null_mut();
    let mut result_len: usize = 0;

    let rank = get_rank();

    let (partitions_ptr, num_partitions) = if rank == root {
        let parts = partitions.ok_or("scatter: root must provide partitions")?;
        let mut partition_info: Vec<usize> = Vec::with_capacity(parts.len() * 2);
        for p in &parts {
            partition_info.push(p.as_ptr() as usize);
            partition_info.push(p.len());
        }
        // Need to keep parts alive, so leak temporarily
        let ptr = partition_info.as_ptr();
        let len = parts.len();
        std::mem::forget(partition_info);
        (ptr, len)
    } else {
        (std::ptr::null(), 0)
    };

    let err = unsafe {
        host_scatter(
            partitions_ptr,
            num_partitions,
            root,
            &mut result_ptr,
            &mut result_len,
        )
    };

    if err != 0 {
        return Err(format!("scatter failed with error code {}", err));
    }

    if result_ptr.is_null() {
        return Ok(Vec::new());
    }

    let result = unsafe {
        std::slice::from_raw_parts(result_ptr, result_len).to_vec()
    };
    wasm_free(result_ptr, result_len);

    Ok(result)
}

/// All-gather byte arrays
pub fn all_gather(data: &[u8]) -> Result<Vec<Vec<u8>>, String> {
    let mut results_ptr: *mut usize = std::ptr::null_mut();
    let mut num_results: usize = 0;

    let err = unsafe {
        host_all_gather(
            data.as_ptr(),
            data.len(),
            &mut results_ptr,
            &mut num_results,
        )
    };

    if err != 0 {
        return Err(format!("all_gather failed with error code {}", err));
    }

    let mut results = Vec::with_capacity(num_results);
    if !results_ptr.is_null() && num_results > 0 {
        let result_info = unsafe {
            std::slice::from_raw_parts(results_ptr, num_results * 2)
        };

        for i in 0..num_results {
            let ptr = result_info[i * 2] as *const u8;
            let len = result_info[i * 2 + 1];

            if !ptr.is_null() && len > 0 {
                let data = unsafe {
                    std::slice::from_raw_parts(ptr, len).to_vec()
                };
                wasm_free(ptr as *mut u8, len);
                results.push(data);
            } else {
                results.push(Vec::new());
            }
        }

        wasm_free(results_ptr as *mut u8, num_results * 2 * std::mem::size_of::<usize>());
    }

    Ok(results)
}