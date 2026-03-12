/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file gcylon_c.h
 * @brief C API wrapper for gcylon GPU-accelerated distributed operations.
 *
 * This header provides a C-compatible interface to the gcylon C++ library,
 * enabling FFI integration with other languages like Rust, Python, etc.
 *
 * The C API is a thin wrapper around the existing C++ implementation.
 * All actual logic is implemented in the C++ layer (gtable_api.hpp/cpp).
 */

#ifndef GCYLON_C_API_H
#define GCYLON_C_API_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Types - Opaque handles that wrap C++ types
// ============================================================================

/** Opaque context handle (wraps cylon::CylonContext) */
typedef struct GcylonContext GcylonContext;

/** Opaque table handle (wraps gcylon::GTable) */
typedef struct GcylonTable GcylonTable;

/** Status codes */
typedef int32_t GcylonStatus;

#define GCYLON_OK            0
#define GCYLON_ERROR        -1
#define GCYLON_OOM          -2
#define GCYLON_INVALID_ARG  -3

/**
 * Configuration for GPU operations.
 * Maps to gcylon::GcylonConfig in C++.
 */
typedef struct {
    size_t gpu_memory_limit;
    float gpu_memory_fraction;
    size_t chunk_size_bytes;
    size_t min_chunk_rows;
} GcylonConfig;

/** Join type enumeration */
typedef enum {
    GCYLON_JOIN_INNER = 0,
    GCYLON_JOIN_LEFT = 1,
    GCYLON_JOIN_RIGHT = 2,
    GCYLON_JOIN_OUTER = 3
} GcylonJoinType;

/** GPU memory information */
typedef struct {
    size_t free_bytes;
    size_t total_bytes;
    size_t used_bytes;
} GcylonMemoryInfo;

// ============================================================================
// Configuration (thin wrappers)
// ============================================================================

GcylonConfig gcylon_config_default(void);
GcylonConfig gcylon_config_low_memory(void);

// ============================================================================
// Context Management (wrappers around CylonContext)
// ============================================================================

GcylonStatus gcylon_context_create_mpi(GcylonContext** ctx);
void gcylon_context_free(GcylonContext* ctx);
int32_t gcylon_context_get_rank(GcylonContext* ctx);
int32_t gcylon_context_get_world_size(GcylonContext* ctx);

// ============================================================================
// GPU Device Management
// ============================================================================

/** Get the number of available CUDA devices */
GcylonStatus gcylon_get_device_count(int32_t* count);

/** Set the current CUDA device */
GcylonStatus gcylon_set_device(int32_t device_id);

/** Get the current CUDA device */
GcylonStatus gcylon_get_device(int32_t* device_id);

// ============================================================================
// Memory Info (wrapper around memory_utils.hpp)
// ============================================================================

GcylonStatus gcylon_get_gpu_memory_info(GcylonMemoryInfo* info);

// ============================================================================
// Table Management (wrappers around GTable)
// ============================================================================

int64_t gcylon_table_num_rows(GcylonTable* table);
int32_t gcylon_table_num_columns(GcylonTable* table);
void gcylon_table_free(GcylonTable* table);

/**
 * Create a table with sequential int64 data for testing.
 *
 * @param ctx Context handle
 * @param num_columns Number of columns
 * @param num_rows Number of rows
 * @param start_value Starting value for sequential data
 * @param step Value increment between rows
 * @param output Output table handle
 * @return Status code
 */
GcylonStatus gcylon_table_create_sequential(
    GcylonContext* ctx,
    int32_t num_columns,
    int64_t num_rows,
    int64_t start_value,
    int64_t step,
    GcylonTable** output
);

/**
 * Create a table with random int64 data for testing.
 *
 * @param ctx Context handle
 * @param num_columns Number of columns
 * @param num_rows Number of rows
 * @param seed Random seed
 * @param output Output table handle
 * @return Status code
 */
GcylonStatus gcylon_table_create_random(
    GcylonContext* ctx,
    int32_t num_columns,
    int64_t num_rows,
    int32_t seed,
    GcylonTable** output
);

// ============================================================================
// Distributed Operations (wrappers around gtable_api.hpp functions)
// ============================================================================

/** Wrapper for gcylon::SmartShuffle */
GcylonStatus gcylon_shuffle(
    GcylonTable* input,
    const int32_t* hash_columns,
    int32_t num_hash_columns,
    GcylonTable** output,
    const GcylonConfig* config
);

/** Wrapper for gcylon::SmartAllGather */
GcylonStatus gcylon_allgather(
    GcylonTable* input,
    GcylonTable** output,
    const GcylonConfig* config
);

/** Wrapper for gcylon::Gather */
GcylonStatus gcylon_gather(
    GcylonTable* input,
    int32_t root,
    GcylonTable** output,
    const GcylonConfig* config
);

/** Wrapper for gcylon::Broadcast */
GcylonStatus gcylon_broadcast(
    GcylonTable* input,
    int32_t root,
    GcylonTable** output,
    const GcylonConfig* config
);

/** Wrapper for gcylon::DistributedJoin */
GcylonStatus gcylon_distributed_join(
    GcylonTable* left,
    GcylonTable* right,
    const int32_t* left_columns,
    int32_t num_left_columns,
    const int32_t* right_columns,
    int32_t num_right_columns,
    GcylonJoinType join_type,
    GcylonTable** output,
    const GcylonConfig* config
);

/** Wrapper for gcylon::DistributedSort */
GcylonStatus gcylon_distributed_sort(
    GcylonTable* input,
    const int32_t* sort_columns,
    int32_t num_sort_columns,
    const int32_t* ascending,
    GcylonTable** output,
    const GcylonConfig* config
);

/** Wrapper for gcylon::Repartition */
GcylonStatus gcylon_repartition(
    GcylonTable* input,
    const int32_t* rows_per_worker,
    int32_t num_workers,
    GcylonTable** output,
    const GcylonConfig* config
);

// ============================================================================
// Error Handling
// ============================================================================

const char* gcylon_status_string(GcylonStatus status);
const char* gcylon_get_last_error(void);

#ifdef __cplusplus
}
#endif

#endif // GCYLON_C_API_H
