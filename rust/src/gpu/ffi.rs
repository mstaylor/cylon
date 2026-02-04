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

//! Raw FFI bindings to gcylon C API
//!
//! These bindings correspond to cpp/src/gcylon/c_api/gcylon_c.h

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

use std::os::raw::c_char;

pub type GcylonStatus = i32;
pub const GCYLON_OK: GcylonStatus = 0;
pub const GCYLON_ERROR: GcylonStatus = -1;
pub const GCYLON_OOM: GcylonStatus = -2;
pub const GCYLON_INVALID_ARG: GcylonStatus = -3;

/// Opaque context handle
#[repr(C)]
pub struct GcylonContext {
    _private: [u8; 0],
}

/// Opaque table handle
#[repr(C)]
pub struct GcylonTable {
    _private: [u8; 0],
}

/// Configuration for GPU operations
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GcylonConfig {
    pub gpu_memory_limit: usize,
    pub gpu_memory_fraction: f32,
    pub chunk_size_bytes: usize,
    pub min_chunk_rows: usize,
    pub allow_cpu_staging: i32,
    pub use_pinned_memory: i32,
}

/// GPU memory information
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GcylonMemoryInfo {
    pub free_bytes: usize,
    pub total_bytes: usize,
    pub used_bytes: usize,
}

/// Join type enumeration
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GcylonJoinType {
    Inner = 0,
    Left = 1,
    Right = 2,
    Outer = 3,
}

#[link(name = "gcylon")]
extern "C" {
    // Configuration
    pub fn gcylon_config_default() -> GcylonConfig;
    pub fn gcylon_config_low_memory() -> GcylonConfig;

    // Context
    pub fn gcylon_context_create_mpi(ctx: *mut *mut GcylonContext) -> GcylonStatus;
    pub fn gcylon_context_free(ctx: *mut GcylonContext);
    pub fn gcylon_context_get_rank(ctx: *mut GcylonContext) -> i32;
    pub fn gcylon_context_get_world_size(ctx: *mut GcylonContext) -> i32;

    // GPU Device Management
    pub fn gcylon_get_device_count(count: *mut i32) -> GcylonStatus;
    pub fn gcylon_set_device(device_id: i32) -> GcylonStatus;
    pub fn gcylon_get_device(device_id: *mut i32) -> GcylonStatus;

    // Memory
    pub fn gcylon_get_gpu_memory_info(info: *mut GcylonMemoryInfo) -> GcylonStatus;

    // Table
    pub fn gcylon_table_num_rows(table: *mut GcylonTable) -> i64;
    pub fn gcylon_table_num_columns(table: *mut GcylonTable) -> i32;
    pub fn gcylon_table_free(table: *mut GcylonTable);

    // Table creation for testing
    pub fn gcylon_table_create_sequential(
        ctx: *mut GcylonContext,
        num_columns: i32,
        num_rows: i64,
        start_value: i64,
        step: i64,
        output: *mut *mut GcylonTable,
    ) -> GcylonStatus;

    pub fn gcylon_table_create_random(
        ctx: *mut GcylonContext,
        num_columns: i32,
        num_rows: i64,
        seed: i32,
        output: *mut *mut GcylonTable,
    ) -> GcylonStatus;

    // Operations
    pub fn gcylon_shuffle(
        input: *mut GcylonTable,
        hash_columns: *const i32,
        num_hash_columns: i32,
        output: *mut *mut GcylonTable,
        config: *const GcylonConfig,
    ) -> GcylonStatus;

    pub fn gcylon_allgather(
        input: *mut GcylonTable,
        output: *mut *mut GcylonTable,
        config: *const GcylonConfig,
    ) -> GcylonStatus;

    pub fn gcylon_gather(
        input: *mut GcylonTable,
        root: i32,
        output: *mut *mut GcylonTable,
        config: *const GcylonConfig,
    ) -> GcylonStatus;

    pub fn gcylon_broadcast(
        input: *mut GcylonTable,
        root: i32,
        output: *mut *mut GcylonTable,
        config: *const GcylonConfig,
    ) -> GcylonStatus;

    pub fn gcylon_distributed_join(
        left: *mut GcylonTable,
        right: *mut GcylonTable,
        left_columns: *const i32,
        num_left_columns: i32,
        right_columns: *const i32,
        num_right_columns: i32,
        join_type: GcylonJoinType,
        output: *mut *mut GcylonTable,
        config: *const GcylonConfig,
    ) -> GcylonStatus;

    pub fn gcylon_distributed_sort(
        input: *mut GcylonTable,
        sort_columns: *const i32,
        num_sort_columns: i32,
        ascending: *const i32,
        output: *mut *mut GcylonTable,
        config: *const GcylonConfig,
    ) -> GcylonStatus;

    pub fn gcylon_repartition(
        input: *mut GcylonTable,
        rows_per_worker: *const i32,
        num_workers: i32,
        output: *mut *mut GcylonTable,
        config: *const GcylonConfig,
    ) -> GcylonStatus;

    // Error handling
    pub fn gcylon_status_string(status: GcylonStatus) -> *const c_char;
    pub fn gcylon_get_last_error() -> *const c_char;
}
