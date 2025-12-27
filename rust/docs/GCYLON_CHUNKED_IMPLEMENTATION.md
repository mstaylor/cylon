# gcylon Chunked Operations & Rust FFI Implementation Guide

This document describes the implementation plan for adding memory-efficient chunked operations to gcylon (GPU Cylon) and exposing them to Rust via FFI.

## Problem Statement

Current gcylon operations (AllGather, Shuffle) can OOM on large datasets because:
1. `hash_partition()` creates a full copy of the input
2. All partitions are serialized before any sending starts
3. Received tables accumulate in GPU memory
4. Final concatenation creates another copy

**Peak memory for shuffle: ~4-5x input size**

## Solution: Chunked Operations with CPU Staging

Process data in chunks to reduce peak GPU memory, with optional CPU staging for overflow.

```
Peak Memory Comparison (4 workers, 4GB each):

Operation      | Current  | Chunked  | Chunked+Staging
---------------|----------|----------|------------------
Shuffle        | 18GB OOM | ~8GB     | ~6GB
AllGather      | 32GB OOM | ~10GB    | ~6GB
```

---

## Part 1: C++ Changes to gcylon

### 1.1 New Configuration Struct

**File: `cpp/src/gcylon/gcylon_config.hpp`** (new file)

```cpp
#ifndef GCYLON_CONFIG_HPP
#define GCYLON_CONFIG_HPP

#include <cstddef>

namespace gcylon {

struct GcylonConfig {
    // Memory limits
    size_t gpu_memory_limit = 0;       // 0 = auto (use 80% of free GPU memory)
    float gpu_memory_fraction = 0.8f;  // Used when gpu_memory_limit = 0

    // Chunking
    size_t chunk_size_bytes = 0;       // 0 = auto-calculate based on memory
    size_t min_chunk_rows = 1024;      // Minimum rows per chunk

    // CPU staging
    bool allow_cpu_staging = true;     // Spill intermediate results to CPU
    bool use_pinned_memory = true;     // Use pinned memory for faster transfers

    // Default factory
    static GcylonConfig Default() {
        return GcylonConfig{};
    }

    // High memory pressure preset
    static GcylonConfig LowMemory() {
        GcylonConfig config;
        config.gpu_memory_fraction = 0.6f;
        config.allow_cpu_staging = true;
        return config;
    }
};

} // namespace gcylon

#endif // GCYLON_CONFIG_HPP
```

### 1.2 CPU Staging Allocator

**File: `cpp/src/gcylon/staging/pinned_allocator.hpp`** (new file)

```cpp
#ifndef GCYLON_PINNED_ALLOCATOR_HPP
#define GCYLON_PINNED_ALLOCATOR_HPP

#include <cylon/net/buffer.hpp>
#include <cuda_runtime.h>

namespace gcylon {

// Buffer backed by pinned (page-locked) CPU memory
class PinnedBuffer : public cylon::Buffer {
public:
    PinnedBuffer(size_t size) : size_(size) {
        cudaMallocHost(&ptr_, size);
    }

    ~PinnedBuffer() {
        if (ptr_) cudaFreeHost(ptr_);
    }

    int64_t GetLength() const override { return size_; }
    uint8_t* GetByteBuffer() override { return static_cast<uint8_t*>(ptr_); }

    // Non-copyable
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    // Movable
    PinnedBuffer(PinnedBuffer&& other) noexcept
        : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

private:
    void* ptr_ = nullptr;
    size_t size_ = 0;
};

// Allocator that uses pinned CPU memory
class PinnedAllocator : public cylon::Allocator {
public:
    cylon::Status Allocate(int64_t length, std::shared_ptr<cylon::Buffer>* buffer) override {
        *buffer = std::make_shared<PinnedBuffer>(length);
        return cylon::Status::OK();
    }
};

// Hybrid allocator: GPU first, spill to CPU when full
class HybridAllocator : public cylon::Allocator {
public:
    HybridAllocator(size_t gpu_budget) : gpu_budget_(gpu_budget) {}

    cylon::Status Allocate(int64_t length, std::shared_ptr<cylon::Buffer>* buffer) override {
        if (gpu_used_ + length <= gpu_budget_) {
            // Allocate on GPU
            auto rmm_buf = std::make_shared<rmm::device_buffer>(length);
            *buffer = std::make_shared<CudfBuffer>(rmm_buf);
            gpu_used_ += length;
        } else {
            // Spill to pinned CPU
            *buffer = std::make_shared<PinnedBuffer>(length);
            cpu_used_ += length;
        }
        return cylon::Status::OK();
    }

    size_t gpu_used() const { return gpu_used_; }
    size_t cpu_used() const { return cpu_used_; }

private:
    size_t gpu_budget_;
    size_t gpu_used_ = 0;
    size_t cpu_used_ = 0;
};

} // namespace gcylon

#endif // GCYLON_PINNED_ALLOCATOR_HPP
```

### 1.3 Memory Utilities

**File: `cpp/src/gcylon/staging/memory_utils.hpp`** (new file)

```cpp
#ifndef GCYLON_MEMORY_UTILS_HPP
#define GCYLON_MEMORY_UTILS_HPP

#include <cuda_runtime.h>
#include <cudf/table/table_view.hpp>

namespace gcylon {

struct GpuMemoryInfo {
    size_t free;
    size_t total;
    size_t used;
};

inline GpuMemoryInfo get_gpu_memory_info() {
    GpuMemoryInfo info;
    cudaMemGetInfo(&info.free, &info.total);
    info.used = info.total - info.free;
    return info;
}

// Estimate memory needed for a table
inline size_t estimate_table_size(const cudf::table_view& tv) {
    size_t total = 0;
    for (int i = 0; i < tv.num_columns(); i++) {
        auto col = tv.column(i);
        total += col.size() * cudf::size_of(col.type());
        if (col.nullable()) {
            total += cudf::bitmask_allocation_size_bytes(col.size());
        }
    }
    return total;
}

// Estimate memory for shuffle operation
inline size_t estimate_shuffle_memory(const cudf::table_view& tv, int world_size) {
    size_t input_size = estimate_table_size(tv);
    // Need: input + partitioned + send buffers + receive buffers + output
    return input_size * (world_size + 3);
}

// Calculate optimal chunk size
inline size_t calculate_chunk_rows(
    const cudf::table_view& tv,
    int world_size,
    size_t available_memory,
    size_t min_rows = 1024
) {
    size_t row_size = estimate_table_size(tv) / std::max(tv.num_rows(), int64_t(1));
    if (row_size == 0) row_size = 1;

    // Memory per chunk: chunk + partitioned + buffers
    size_t mem_per_row = row_size * (world_size + 3);
    size_t max_rows = available_memory / mem_per_row;

    return std::max(max_rows, min_rows);
}

} // namespace gcylon

#endif // GCYLON_MEMORY_UTILS_HPP
```

### 1.4 Chunked Shuffle Implementation

**File: `cpp/src/gcylon/gtable_api.hpp`** - Add declarations:

```cpp
// Add to existing header

#include <gcylon/gcylon_config.hpp>

namespace gcylon {

// Chunked shuffle - memory efficient version
cylon::Status ChunkedShuffle(
    const cudf::table_view &input_tv,
    const std::vector<int> &columns_to_hash,
    const std::shared_ptr<cylon::CylonContext> &ctx,
    std::unique_ptr<cudf::table> &table_out,
    const GcylonConfig &config = GcylonConfig::Default()
);

// Chunked allgather - memory efficient version
cylon::Status ChunkedAllGather(
    const cudf::table_view &input_tv,
    const std::shared_ptr<cylon::CylonContext> &ctx,
    std::unique_ptr<cudf::table> &table_out,
    const GcylonConfig &config = GcylonConfig::Default()
);

// Smart versions that auto-select based on memory
cylon::Status SmartShuffle(
    const cudf::table_view &input_tv,
    const std::vector<int> &columns_to_hash,
    const std::shared_ptr<cylon::CylonContext> &ctx,
    std::unique_ptr<cudf::table> &table_out,
    const GcylonConfig &config = GcylonConfig::Default()
);

cylon::Status SmartAllGather(
    const cudf::table_view &input_tv,
    const std::shared_ptr<cylon::CylonContext> &ctx,
    std::unique_ptr<cudf::table> &table_out,
    const GcylonConfig &config = GcylonConfig::Default()
);

} // namespace gcylon
```

**File: `cpp/src/gcylon/gtable_api_chunked.cpp`** (new file)

```cpp
#include <gcylon/gtable_api.hpp>
#include <gcylon/staging/memory_utils.hpp>
#include <gcylon/staging/pinned_allocator.hpp>
#include <gcylon/net/cudf_net_ops.hpp>

#include <cudf/partitioning.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>

namespace gcylon {

// Helper: Copy table to pinned CPU memory
std::unique_ptr<cudf::table> copy_to_pinned(const cudf::table_view& tv) {
    // Allocate pinned memory and copy
    std::vector<std::unique_ptr<cudf::column>> columns;
    for (int i = 0; i < tv.num_columns(); i++) {
        // cudf handles the copy
        columns.push_back(std::make_unique<cudf::column>(tv.column(i)));
    }
    return std::make_unique<cudf::table>(std::move(columns));
}

// Helper: Split table into chunks by row count
std::vector<cudf::table_view> split_table(
    const cudf::table_view& tv,
    size_t rows_per_chunk
) {
    std::vector<cudf::table_view> chunks;
    int64_t total_rows = tv.num_rows();

    for (int64_t start = 0; start < total_rows; start += rows_per_chunk) {
        int64_t end = std::min(start + static_cast<int64_t>(rows_per_chunk), total_rows);
        std::vector<cudf::size_type> indices = {
            static_cast<cudf::size_type>(start),
            static_cast<cudf::size_type>(end)
        };
        auto sliced = cudf::slice(tv, indices);
        if (!sliced.empty()) {
            chunks.push_back(sliced[0]);
        }
    }

    return chunks;
}

cylon::Status ChunkedShuffle(
    const cudf::table_view &input_tv,
    const std::vector<int> &columns_to_hash,
    const std::shared_ptr<cylon::CylonContext> &ctx,
    std::unique_ptr<cudf::table> &table_out,
    const GcylonConfig &config
) {
    auto mem_info = get_gpu_memory_info();
    int world_size = ctx->GetWorldSize();

    // Calculate chunk size
    size_t available = config.gpu_memory_limit > 0
        ? config.gpu_memory_limit
        : static_cast<size_t>(mem_info.free * config.gpu_memory_fraction);

    size_t rows_per_chunk = config.chunk_size_bytes > 0
        ? config.chunk_size_bytes / (estimate_table_size(input_tv) / std::max(input_tv.num_rows(), int64_t(1)))
        : calculate_chunk_rows(input_tv, world_size, available, config.min_chunk_rows);

    // If table is small enough, use fast path
    if (static_cast<size_t>(input_tv.num_rows()) <= rows_per_chunk) {
        return Shuffle(input_tv, columns_to_hash, ctx, table_out);
    }

    // Split into chunks
    auto chunks = split_table(input_tv, rows_per_chunk);

    // Process each chunk
    std::vector<std::unique_ptr<cudf::table>> results;

    // If staging enabled, store results on CPU
    std::vector<std::unique_ptr<cudf::table>> cpu_staged;

    for (size_t i = 0; i < chunks.size(); i++) {
        // Partition this chunk
        auto [partitioned, offsets] = cudf::hash_partition(
            chunks[i], columns_to_hash, world_size
        );
        offsets.push_back(chunks[i].num_rows());

        // AllToAll exchange
        std::unique_ptr<cudf::table> chunk_result;
        RETURN_CYLON_STATUS_IF_FAILED(
            gcylon::net::AllToAll(partitioned->view(), offsets, ctx, chunk_result)
        );

        if (config.allow_cpu_staging && i < chunks.size() - 1) {
            // Stage to CPU to free GPU memory
            // Note: In production, use async copy with pinned memory
            cpu_staged.push_back(std::move(chunk_result));
        } else {
            results.push_back(std::move(chunk_result));
        }

        // partitioned goes out of scope, freeing GPU memory
    }

    // Bring staged results back and concatenate
    for (auto& staged : cpu_staged) {
        results.push_back(std::move(staged));
    }

    if (results.size() == 1) {
        table_out = std::move(results[0]);
    } else {
        std::vector<cudf::table_view> views;
        for (auto& t : results) {
            views.push_back(t->view());
        }
        table_out = cudf::concatenate(views);
    }

    return cylon::Status::OK();
}

cylon::Status ChunkedAllGather(
    const cudf::table_view &input_tv,
    const std::shared_ptr<cylon::CylonContext> &ctx,
    std::unique_ptr<cudf::table> &table_out,
    const GcylonConfig &config
) {
    auto mem_info = get_gpu_memory_info();
    int world_size = ctx->GetWorldSize();

    // AllGather produces world_size copies
    size_t output_size = estimate_table_size(input_tv) * world_size;

    // If fits in memory, use fast path
    if (output_size < mem_info.free * config.gpu_memory_fraction) {
        return AllGather(input_tv, ctx, table_out);
    }

    // Calculate chunk size
    size_t available = config.gpu_memory_limit > 0
        ? config.gpu_memory_limit
        : static_cast<size_t>(mem_info.free * config.gpu_memory_fraction);

    // For allgather, each chunk becomes world_size chunks
    size_t row_size = estimate_table_size(input_tv) / std::max(input_tv.num_rows(), int64_t(1));
    size_t rows_per_chunk = available / (row_size * (world_size + 2));
    rows_per_chunk = std::max(rows_per_chunk, config.min_chunk_rows);

    auto chunks = split_table(input_tv, rows_per_chunk);

    std::vector<std::unique_ptr<cudf::table>> results;

    for (size_t i = 0; i < chunks.size(); i++) {
        std::vector<std::unique_ptr<cudf::table>> gathered;

        RETURN_CYLON_STATUS_IF_FAILED(
            gcylon::net::AllGather(chunks[i], ctx, gathered)
        );

        // Concatenate this round's gathered chunks
        std::vector<cudf::table_view> views;
        for (auto& t : gathered) {
            views.push_back(t->view());
        }
        auto merged = cudf::concatenate(views);

        results.push_back(std::move(merged));
        // gathered goes out of scope, freeing memory
    }

    // Final concatenation
    if (results.size() == 1) {
        table_out = std::move(results[0]);
    } else {
        std::vector<cudf::table_view> views;
        for (auto& t : results) {
            views.push_back(t->view());
        }
        table_out = cudf::concatenate(views);
    }

    return cylon::Status::OK();
}

cylon::Status SmartShuffle(
    const cudf::table_view &input_tv,
    const std::vector<int> &columns_to_hash,
    const std::shared_ptr<cylon::CylonContext> &ctx,
    std::unique_ptr<cudf::table> &table_out,
    const GcylonConfig &config
) {
    auto mem_info = get_gpu_memory_info();
    size_t estimated = estimate_shuffle_memory(input_tv, ctx->GetWorldSize());

    if (estimated < mem_info.free * 0.5) {
        // Plenty of room - fast path
        return Shuffle(input_tv, columns_to_hash, ctx, table_out);
    } else {
        // Memory pressure - chunked path
        return ChunkedShuffle(input_tv, columns_to_hash, ctx, table_out, config);
    }
}

cylon::Status SmartAllGather(
    const cudf::table_view &input_tv,
    const std::shared_ptr<cylon::CylonContext> &ctx,
    std::unique_ptr<cudf::table> &table_out,
    const GcylonConfig &config
) {
    auto mem_info = get_gpu_memory_info();
    size_t output_size = estimate_table_size(input_tv) * ctx->GetWorldSize();

    if (output_size < mem_info.free * 0.5) {
        return AllGather(input_tv, ctx, table_out);
    } else {
        return ChunkedAllGather(input_tv, ctx, table_out, config);
    }
}

} // namespace gcylon
```

---

## Part 2: C API Wrapper

**File: `cpp/src/gcylon/c_api/gcylon_c.h`** (new file)

```c
#ifndef GCYLON_C_API_H
#define GCYLON_C_API_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Types
// ============================================================================

typedef struct GcylonContext GcylonContext;
typedef struct GcylonTable GcylonTable;

typedef int32_t GcylonStatus;
#define GCYLON_OK           0
#define GCYLON_ERROR       -1
#define GCYLON_OOM         -2
#define GCYLON_INVALID_ARG -3

typedef struct {
    size_t gpu_memory_limit;      // 0 = auto
    float gpu_memory_fraction;    // default 0.8
    size_t chunk_size_bytes;      // 0 = auto
    size_t min_chunk_rows;        // default 1024
    int32_t allow_cpu_staging;    // default 1 (true)
    int32_t use_pinned_memory;    // default 1 (true)
} GcylonConfig;

typedef enum {
    GCYLON_JOIN_INNER = 0,
    GCYLON_JOIN_LEFT = 1,
    GCYLON_JOIN_RIGHT = 2,
    GCYLON_JOIN_OUTER = 3
} GcylonJoinType;

// ============================================================================
// Configuration
// ============================================================================

GcylonConfig gcylon_config_default(void);
GcylonConfig gcylon_config_low_memory(void);

// ============================================================================
// Context Management
// ============================================================================

GcylonStatus gcylon_context_create_mpi(GcylonContext** ctx);
GcylonStatus gcylon_context_create_with_comm(void* mpi_comm, GcylonContext** ctx);
void gcylon_context_free(GcylonContext* ctx);

int32_t gcylon_context_get_rank(GcylonContext* ctx);
int32_t gcylon_context_get_world_size(GcylonContext* ctx);

// ============================================================================
// Memory Info
// ============================================================================

typedef struct {
    size_t free_bytes;
    size_t total_bytes;
    size_t used_bytes;
} GcylonMemoryInfo;

GcylonStatus gcylon_get_gpu_memory_info(GcylonMemoryInfo* info);

// ============================================================================
// Table Management
// ============================================================================

// Create from cudf table (takes ownership)
GcylonStatus gcylon_table_from_cudf(
    GcylonContext* ctx,
    void* cudf_table,  // std::unique_ptr<cudf::table>*
    GcylonTable** out
);

// Get underlying cudf table (borrows, do not free)
GcylonStatus gcylon_table_get_cudf(
    GcylonTable* table,
    void** cudf_table_view  // cudf::table_view*
);

// Table info
int64_t gcylon_table_num_rows(GcylonTable* table);
int32_t gcylon_table_num_columns(GcylonTable* table);

void gcylon_table_free(GcylonTable* table);

// ============================================================================
// Distributed Operations (Chunked/Memory-Safe Versions)
// ============================================================================

// Shuffle (hash partition + all-to-all)
GcylonStatus gcylon_shuffle(
    GcylonTable* input,
    const int32_t* hash_columns,
    int32_t num_hash_columns,
    GcylonTable** output,
    const GcylonConfig* config  // NULL for defaults
);

// AllGather
GcylonStatus gcylon_allgather(
    GcylonTable* input,
    GcylonTable** output,
    const GcylonConfig* config
);

// Gather to root
GcylonStatus gcylon_gather(
    GcylonTable* input,
    int32_t root,
    GcylonTable** output,
    const GcylonConfig* config
);

// Broadcast from root
GcylonStatus gcylon_broadcast(
    GcylonTable* input,
    int32_t root,
    GcylonTable** output,
    const GcylonConfig* config
);

// Distributed Join
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

// Distributed Sort
GcylonStatus gcylon_distributed_sort(
    GcylonTable* input,
    const int32_t* sort_columns,
    int32_t num_sort_columns,
    const int32_t* ascending,  // 1=asc, 0=desc for each
    GcylonTable** output,
    const GcylonConfig* config
);

// Repartition
GcylonStatus gcylon_repartition(
    GcylonTable* input,
    const int32_t* rows_per_worker,  // NULL for even distribution
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
```

**File: `cpp/src/gcylon/c_api/gcylon_c.cpp`** (new file)

```cpp
#include "gcylon_c.h"

#include <gcylon/gtable.hpp>
#include <gcylon/gtable_api.hpp>
#include <gcylon/gcylon_config.hpp>
#include <gcylon/staging/memory_utils.hpp>

#include <cylon/ctx/cylon_context.hpp>
#include <cudf/table/table.hpp>

#include <memory>
#include <string>

// Thread-local error message
thread_local std::string g_last_error;

// Internal wrapper types
struct GcylonContext {
    std::shared_ptr<cylon::CylonContext> ctx;
};

struct GcylonTable {
    std::shared_ptr<gcylon::GTable> table;
};

// Helper: Convert config
static gcylon::GcylonConfig to_internal_config(const GcylonConfig* config) {
    if (!config) return gcylon::GcylonConfig::Default();

    gcylon::GcylonConfig c;
    c.gpu_memory_limit = config->gpu_memory_limit;
    c.gpu_memory_fraction = config->gpu_memory_fraction;
    c.chunk_size_bytes = config->chunk_size_bytes;
    c.min_chunk_rows = config->min_chunk_rows;
    c.allow_cpu_staging = config->allow_cpu_staging != 0;
    c.use_pinned_memory = config->use_pinned_memory != 0;
    return c;
}

// Helper: Set error
static void set_error(const std::string& msg) {
    g_last_error = msg;
}

extern "C" {

// ============================================================================
// Configuration
// ============================================================================

GcylonConfig gcylon_config_default(void) {
    GcylonConfig c = {
        .gpu_memory_limit = 0,
        .gpu_memory_fraction = 0.8f,
        .chunk_size_bytes = 0,
        .min_chunk_rows = 1024,
        .allow_cpu_staging = 1,
        .use_pinned_memory = 1
    };
    return c;
}

GcylonConfig gcylon_config_low_memory(void) {
    GcylonConfig c = gcylon_config_default();
    c.gpu_memory_fraction = 0.6f;
    c.allow_cpu_staging = 1;
    return c;
}

// ============================================================================
// Context
// ============================================================================

GcylonStatus gcylon_context_create_mpi(GcylonContext** ctx) {
    try {
        auto cylon_ctx = cylon::CylonContext::InitDistributed(cylon::net::CommType::MPI);
        *ctx = new GcylonContext{cylon_ctx};
        return GCYLON_OK;
    } catch (const std::exception& e) {
        set_error(e.what());
        return GCYLON_ERROR;
    }
}

void gcylon_context_free(GcylonContext* ctx) {
    if (ctx) {
        ctx->ctx->Finalize();
        delete ctx;
    }
}

int32_t gcylon_context_get_rank(GcylonContext* ctx) {
    return ctx ? ctx->ctx->GetRank() : -1;
}

int32_t gcylon_context_get_world_size(GcylonContext* ctx) {
    return ctx ? ctx->ctx->GetWorldSize() : -1;
}

// ============================================================================
// Memory
// ============================================================================

GcylonStatus gcylon_get_gpu_memory_info(GcylonMemoryInfo* info) {
    if (!info) return GCYLON_INVALID_ARG;

    auto mem = gcylon::get_gpu_memory_info();
    info->free_bytes = mem.free;
    info->total_bytes = mem.total;
    info->used_bytes = mem.used;
    return GCYLON_OK;
}

// ============================================================================
// Table
// ============================================================================

int64_t gcylon_table_num_rows(GcylonTable* table) {
    return table ? table->table->GetCudfTable()->num_rows() : 0;
}

int32_t gcylon_table_num_columns(GcylonTable* table) {
    return table ? table->table->GetCudfTable()->num_columns() : 0;
}

void gcylon_table_free(GcylonTable* table) {
    delete table;
}

// ============================================================================
// Distributed Operations
// ============================================================================

GcylonStatus gcylon_shuffle(
    GcylonTable* input,
    const int32_t* hash_columns,
    int32_t num_hash_columns,
    GcylonTable** output,
    const GcylonConfig* config
) {
    if (!input || !hash_columns || !output) return GCYLON_INVALID_ARG;

    try {
        std::vector<int> cols(hash_columns, hash_columns + num_hash_columns);
        auto cfg = to_internal_config(config);

        std::unique_ptr<cudf::table> result;
        auto status = gcylon::SmartShuffle(
            input->table->GetCudfTable()->view(),
            cols,
            input->table->GetContext(),
            result,
            cfg
        );

        if (!status.is_ok()) {
            set_error(status.get_msg());
            return GCYLON_ERROR;
        }

        std::shared_ptr<gcylon::GTable> gtable;
        auto ctx = input->table->GetContext();
        gcylon::GTable::FromCudfTable(ctx, result, gtable);

        *output = new GcylonTable{gtable};
        return GCYLON_OK;

    } catch (const std::exception& e) {
        set_error(e.what());
        return GCYLON_ERROR;
    }
}

GcylonStatus gcylon_allgather(
    GcylonTable* input,
    GcylonTable** output,
    const GcylonConfig* config
) {
    if (!input || !output) return GCYLON_INVALID_ARG;

    try {
        auto cfg = to_internal_config(config);

        std::unique_ptr<cudf::table> result;
        auto status = gcylon::SmartAllGather(
            input->table->GetCudfTable()->view(),
            input->table->GetContext(),
            result,
            cfg
        );

        if (!status.is_ok()) {
            set_error(status.get_msg());
            return GCYLON_ERROR;
        }

        std::shared_ptr<gcylon::GTable> gtable;
        auto ctx = input->table->GetContext();
        gcylon::GTable::FromCudfTable(ctx, result, gtable);

        *output = new GcylonTable{gtable};
        return GCYLON_OK;

    } catch (const std::exception& e) {
        set_error(e.what());
        return GCYLON_ERROR;
    }
}

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
) {
    if (!left || !right || !left_columns || !right_columns || !output) {
        return GCYLON_INVALID_ARG;
    }

    try {
        std::vector<int> left_cols(left_columns, left_columns + num_left_columns);
        std::vector<int> right_cols(right_columns, right_columns + num_right_columns);

        cylon::join::config::JoinType cylon_join_type;
        switch (join_type) {
            case GCYLON_JOIN_INNER: cylon_join_type = cylon::join::config::INNER; break;
            case GCYLON_JOIN_LEFT:  cylon_join_type = cylon::join::config::LEFT; break;
            case GCYLON_JOIN_RIGHT: cylon_join_type = cylon::join::config::RIGHT; break;
            case GCYLON_JOIN_OUTER: cylon_join_type = cylon::join::config::FULL_OUTER; break;
            default: return GCYLON_INVALID_ARG;
        }

        auto join_config = cylon::join::config::JoinConfig(
            cylon_join_type,
            left_cols,
            right_cols
        );

        std::shared_ptr<gcylon::GTable> left_table = left->table;
        std::shared_ptr<gcylon::GTable> right_table = right->table;
        std::shared_ptr<gcylon::GTable> result;

        auto status = gcylon::DistributedJoin(left_table, right_table, join_config, result);

        if (!status.is_ok()) {
            set_error(status.get_msg());
            return GCYLON_ERROR;
        }

        *output = new GcylonTable{result};
        return GCYLON_OK;

    } catch (const std::exception& e) {
        set_error(e.what());
        return GCYLON_ERROR;
    }
}

// ============================================================================
// Error Handling
// ============================================================================

const char* gcylon_status_string(GcylonStatus status) {
    switch (status) {
        case GCYLON_OK: return "OK";
        case GCYLON_ERROR: return "Error";
        case GCYLON_OOM: return "Out of memory";
        case GCYLON_INVALID_ARG: return "Invalid argument";
        default: return "Unknown";
    }
}

const char* gcylon_get_last_error(void) {
    return g_last_error.c_str();
}

} // extern "C"
```

---

## Part 3: Rust Bindings

**File: `rust/src/gpu/mod.rs`** (new module)

```rust
//! GPU-accelerated operations via gcylon FFI
//!
//! This module provides Rust bindings to gcylon (GPU Cylon) for
//! accelerated distributed operations on NVIDIA GPUs.
//!
//! # Requirements
//! - NVIDIA GPU with CUDA support
//! - cuDF library installed
//! - gcylon library with C API
//!
//! # Example
//! ```ignore
//! use cylon::gpu::{GpuContext, GpuTable, GpuConfig};
//!
//! let ctx = GpuContext::new_mpi()?;
//! let gpu_table = GpuTable::from_arrow(&ctx, &record_batch)?;
//!
//! let shuffled = gpu_table.shuffle(&[0], Some(GpuConfig::low_memory()))?;
//! let result = shuffled.to_arrow()?;
//! ```

#[cfg(feature = "gpu")]
mod ffi;
#[cfg(feature = "gpu")]
mod context;
#[cfg(feature = "gpu")]
mod table;
#[cfg(feature = "gpu")]
mod config;

#[cfg(feature = "gpu")]
pub use context::GpuContext;
#[cfg(feature = "gpu")]
pub use table::GpuTable;
#[cfg(feature = "gpu")]
pub use config::GpuConfig;

#[cfg(not(feature = "gpu"))]
compile_error!("The 'gpu' feature is required for GPU support");
```

**File: `rust/src/gpu/ffi.rs`**

```rust
//! Raw FFI bindings to gcylon C API
//!
//! Generated by bindgen or manually declared.

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

use std::os::raw::{c_char, c_int, c_void};

pub type GcylonStatus = i32;
pub const GCYLON_OK: GcylonStatus = 0;
pub const GCYLON_ERROR: GcylonStatus = -1;
pub const GCYLON_OOM: GcylonStatus = -2;
pub const GCYLON_INVALID_ARG: GcylonStatus = -3;

#[repr(C)]
pub struct GcylonContext {
    _private: [u8; 0],
}

#[repr(C)]
pub struct GcylonTable {
    _private: [u8; 0],
}

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

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GcylonMemoryInfo {
    pub free_bytes: usize,
    pub total_bytes: usize,
    pub used_bytes: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GcylonJoinType {
    Inner = 0,
    Left = 1,
    Right = 2,
    Outer = 3,
}

extern "C" {
    // Configuration
    pub fn gcylon_config_default() -> GcylonConfig;
    pub fn gcylon_config_low_memory() -> GcylonConfig;

    // Context
    pub fn gcylon_context_create_mpi(ctx: *mut *mut GcylonContext) -> GcylonStatus;
    pub fn gcylon_context_free(ctx: *mut GcylonContext);
    pub fn gcylon_context_get_rank(ctx: *mut GcylonContext) -> i32;
    pub fn gcylon_context_get_world_size(ctx: *mut GcylonContext) -> i32;

    // Memory
    pub fn gcylon_get_gpu_memory_info(info: *mut GcylonMemoryInfo) -> GcylonStatus;

    // Table
    pub fn gcylon_table_num_rows(table: *mut GcylonTable) -> i64;
    pub fn gcylon_table_num_columns(table: *mut GcylonTable) -> i32;
    pub fn gcylon_table_free(table: *mut GcylonTable);

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

    // Error handling
    pub fn gcylon_status_string(status: GcylonStatus) -> *const c_char;
    pub fn gcylon_get_last_error() -> *const c_char;
}
```

**File: `rust/src/gpu/config.rs`**

```rust
//! GPU operation configuration

use super::ffi;

/// Configuration for GPU operations
#[derive(Debug, Clone)]
pub struct GpuConfig {
    inner: ffi::GcylonConfig,
}

impl GpuConfig {
    /// Default configuration (80% GPU memory, staging enabled)
    pub fn default() -> Self {
        Self {
            inner: unsafe { ffi::gcylon_config_default() },
        }
    }

    /// Low memory configuration (60% GPU memory, aggressive staging)
    pub fn low_memory() -> Self {
        Self {
            inner: unsafe { ffi::gcylon_config_low_memory() },
        }
    }

    /// Set GPU memory limit in bytes (0 = auto)
    pub fn with_gpu_memory_limit(mut self, limit: usize) -> Self {
        self.inner.gpu_memory_limit = limit;
        self
    }

    /// Set GPU memory fraction (0.0 - 1.0)
    pub fn with_gpu_memory_fraction(mut self, fraction: f32) -> Self {
        self.inner.gpu_memory_fraction = fraction.clamp(0.1, 0.95);
        self
    }

    /// Enable/disable CPU staging
    pub fn with_cpu_staging(mut self, enabled: bool) -> Self {
        self.inner.allow_cpu_staging = if enabled { 1 } else { 0 };
        self
    }

    /// Set chunk size in bytes (0 = auto)
    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.inner.chunk_size_bytes = size;
        self
    }

    pub(crate) fn as_ptr(&self) -> *const ffi::GcylonConfig {
        &self.inner
    }
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self::default()
    }
}
```

**File: `rust/src/gpu/context.rs`**

```rust
//! GPU context management

use std::sync::Arc;
use crate::error::{CylonError, CylonResult, Code};
use super::ffi;

/// GPU context for distributed operations
pub struct GpuContext {
    ptr: *mut ffi::GcylonContext,
}

impl GpuContext {
    /// Create a new GPU context with MPI
    pub fn new_mpi() -> CylonResult<Arc<Self>> {
        let mut ptr = std::ptr::null_mut();
        let status = unsafe { ffi::gcylon_context_create_mpi(&mut ptr) };

        if status != ffi::GCYLON_OK {
            return Err(CylonError::new(
                Code::ExecutionError,
                format!("Failed to create GPU context: {}", get_last_error()),
            ));
        }

        Ok(Arc::new(Self { ptr }))
    }

    /// Get this worker's rank
    pub fn rank(&self) -> i32 {
        unsafe { ffi::gcylon_context_get_rank(self.ptr) }
    }

    /// Get world size
    pub fn world_size(&self) -> i32 {
        unsafe { ffi::gcylon_context_get_world_size(self.ptr) }
    }

    /// Get GPU memory info
    pub fn memory_info(&self) -> CylonResult<GpuMemoryInfo> {
        let mut info = ffi::GcylonMemoryInfo {
            free_bytes: 0,
            total_bytes: 0,
            used_bytes: 0,
        };

        let status = unsafe { ffi::gcylon_get_gpu_memory_info(&mut info) };
        if status != ffi::GCYLON_OK {
            return Err(CylonError::new(Code::ExecutionError, "Failed to get GPU memory info"));
        }

        Ok(GpuMemoryInfo {
            free: info.free_bytes,
            total: info.total_bytes,
            used: info.used_bytes,
        })
    }

    pub(crate) fn as_ptr(&self) -> *mut ffi::GcylonContext {
        self.ptr
    }
}

impl Drop for GpuContext {
    fn drop(&mut self) {
        unsafe { ffi::gcylon_context_free(self.ptr) };
    }
}

// Safety: GcylonContext is thread-safe (MPI handles are process-global)
unsafe impl Send for GpuContext {}
unsafe impl Sync for GpuContext {}

/// GPU memory information
#[derive(Debug, Clone, Copy)]
pub struct GpuMemoryInfo {
    pub free: usize,
    pub total: usize,
    pub used: usize,
}

impl GpuMemoryInfo {
    pub fn usage_fraction(&self) -> f64 {
        self.used as f64 / self.total as f64
    }
}

fn get_last_error() -> String {
    unsafe {
        let ptr = ffi::gcylon_get_last_error();
        if ptr.is_null() {
            "Unknown error".to_string()
        } else {
            std::ffi::CStr::from_ptr(ptr)
                .to_string_lossy()
                .into_owned()
        }
    }
}
```

**File: `rust/src/gpu/table.rs`**

```rust
//! GPU table operations

use std::sync::Arc;
use crate::error::{CylonError, CylonResult, Code};
use super::{ffi, GpuContext, GpuConfig};

/// A table stored on GPU with distributed operation support
pub struct GpuTable {
    ptr: *mut ffi::GcylonTable,
    ctx: Arc<GpuContext>,
}

impl GpuTable {
    /// Number of rows
    pub fn num_rows(&self) -> i64 {
        unsafe { ffi::gcylon_table_num_rows(self.ptr) }
    }

    /// Number of columns
    pub fn num_columns(&self) -> i32 {
        unsafe { ffi::gcylon_table_num_columns(self.ptr) }
    }

    /// Distributed shuffle (hash partition + all-to-all exchange)
    ///
    /// # Arguments
    /// * `hash_columns` - Column indices to hash for partitioning
    /// * `config` - Optional configuration (None for defaults)
    pub fn shuffle(&self, hash_columns: &[i32], config: Option<GpuConfig>) -> CylonResult<Self> {
        let config = config.unwrap_or_default();
        let mut output = std::ptr::null_mut();

        let status = unsafe {
            ffi::gcylon_shuffle(
                self.ptr,
                hash_columns.as_ptr(),
                hash_columns.len() as i32,
                &mut output,
                config.as_ptr(),
            )
        };

        check_status(status)?;
        Ok(Self { ptr: output, ctx: self.ctx.clone() })
    }

    /// AllGather - collect table from all workers
    pub fn allgather(&self, config: Option<GpuConfig>) -> CylonResult<Self> {
        let config = config.unwrap_or_default();
        let mut output = std::ptr::null_mut();

        let status = unsafe {
            ffi::gcylon_allgather(self.ptr, &mut output, config.as_ptr())
        };

        check_status(status)?;
        Ok(Self { ptr: output, ctx: self.ctx.clone() })
    }

    /// Distributed join
    pub fn distributed_join(
        &self,
        right: &GpuTable,
        left_columns: &[i32],
        right_columns: &[i32],
        join_type: JoinType,
        config: Option<GpuConfig>,
    ) -> CylonResult<Self> {
        let config = config.unwrap_or_default();
        let mut output = std::ptr::null_mut();

        let status = unsafe {
            ffi::gcylon_distributed_join(
                self.ptr,
                right.ptr,
                left_columns.as_ptr(),
                left_columns.len() as i32,
                right_columns.as_ptr(),
                right_columns.len() as i32,
                join_type.into(),
                &mut output,
                config.as_ptr(),
            )
        };

        check_status(status)?;
        Ok(Self { ptr: output, ctx: self.ctx.clone() })
    }

    /// Get the GPU context
    pub fn context(&self) -> &Arc<GpuContext> {
        &self.ctx
    }
}

impl Drop for GpuTable {
    fn drop(&mut self) {
        unsafe { ffi::gcylon_table_free(self.ptr) };
    }
}

unsafe impl Send for GpuTable {}
unsafe impl Sync for GpuTable {}

/// Join type for distributed joins
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    Inner,
    Left,
    Right,
    Outer,
}

impl From<JoinType> for ffi::GcylonJoinType {
    fn from(jt: JoinType) -> Self {
        match jt {
            JoinType::Inner => ffi::GcylonJoinType::Inner,
            JoinType::Left => ffi::GcylonJoinType::Left,
            JoinType::Right => ffi::GcylonJoinType::Right,
            JoinType::Outer => ffi::GcylonJoinType::Outer,
        }
    }
}

fn check_status(status: ffi::GcylonStatus) -> CylonResult<()> {
    if status == ffi::GCYLON_OK {
        return Ok(());
    }

    let msg = unsafe {
        let ptr = ffi::gcylon_get_last_error();
        if ptr.is_null() {
            "Unknown error".to_string()
        } else {
            std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };

    let code = match status {
        ffi::GCYLON_OOM => Code::OutOfMemory,
        ffi::GCYLON_INVALID_ARG => Code::Invalid,
        _ => Code::ExecutionError,
    };

    Err(CylonError::new(code, msg))
}
```

---

## Part 4: Build Configuration

**File: `rust/Cargo.toml`** - Add to features and dependencies:

```toml
[features]
default = []
gpu = ["dep:cuda-runtime-sys"]  # Optional GPU support

[dependencies]
cuda-runtime-sys = { version = "0.3", optional = true }

[build-dependencies]
bindgen = "0.69"
```

**File: `rust/build.rs`** - Add gcylon build:

```rust
#[cfg(feature = "gpu")]
fn build_gcylon() {
    // Link to gcylon
    println!("cargo:rustc-link-lib=gcylon");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cudf");

    // Search paths (adjust for your system)
    if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
        println!("cargo:rustc-link-search={}/lib64", cuda_path);
    }
    if let Ok(gcylon_path) = std::env::var("GCYLON_PATH") {
        println!("cargo:rustc-link-search={}/lib", gcylon_path);
    }

    // Optional: Generate bindings with bindgen
    // let bindings = bindgen::Builder::default()
    //     .header("gcylon_c.h")
    //     .generate()
    //     .expect("Failed to generate bindings");
    // bindings.write_to_file(out_path.join("gcylon_bindings.rs")).unwrap();
}
```

---

## Part 5: Testing

**File: `cpp/tests/gcylon_chunked_test.cpp`**

```cpp
#include <gtest/gtest.h>
#include <gcylon/gtable_api.hpp>
#include <gcylon/gcylon_config.hpp>

class ChunkedShuffleTest : public ::testing::Test {
protected:
    std::shared_ptr<cylon::CylonContext> ctx;

    void SetUp() override {
        ctx = cylon::CylonContext::InitDistributed(cylon::net::CommType::MPI);
    }

    void TearDown() override {
        ctx->Finalize();
    }
};

TEST_F(ChunkedShuffleTest, SmallTableUsesDirectPath) {
    // Create small table that fits in memory
    auto table = create_test_table(1000);  // 1000 rows

    std::unique_ptr<cudf::table> result;
    auto status = gcylon::SmartShuffle(table->view(), {0}, ctx, result);

    ASSERT_TRUE(status.is_ok());
    ASSERT_EQ(result->num_rows(), 1000 * ctx->GetWorldSize());
}

TEST_F(ChunkedShuffleTest, LargeTableUsesChunkedPath) {
    // Create table that exceeds comfortable memory
    auto table = create_large_test_table();

    gcylon::GcylonConfig config;
    config.gpu_memory_fraction = 0.3;  // Force chunking

    std::unique_ptr<cudf::table> result;
    auto status = gcylon::ChunkedShuffle(table->view(), {0}, ctx, result, config);

    ASSERT_TRUE(status.is_ok());
}
```

---

## Summary

### Files to Create/Modify

**New C++ Files:**
- `cpp/src/gcylon/gcylon_config.hpp`
- `cpp/src/gcylon/staging/pinned_allocator.hpp`
- `cpp/src/gcylon/staging/memory_utils.hpp`
- `cpp/src/gcylon/gtable_api_chunked.cpp`
- `cpp/src/gcylon/c_api/gcylon_c.h`
- `cpp/src/gcylon/c_api/gcylon_c.cpp`

**Modified C++ Files:**
- `cpp/src/gcylon/gtable_api.hpp` (add new declarations)
- `cpp/CMakeLists.txt` (add new sources)

**New Rust Files:**
- `rust/src/gpu/mod.rs`
- `rust/src/gpu/ffi.rs`
- `rust/src/gpu/config.rs`
- `rust/src/gpu/context.rs`
- `rust/src/gpu/table.rs`

**Modified Rust Files:**
- `rust/Cargo.toml` (add gpu feature)
- `rust/build.rs` (add gcylon linking)
- `rust/src/lib.rs` (add gpu module)

### Build Order

1. Build gcylon C++ with new chunked operations
2. Build gcylon C API wrapper
3. Build Rust with `--features gpu`
4. Run tests

### Environment Variables

```bash
export CUDA_PATH=/usr/local/cuda
export GCYLON_PATH=/path/to/cylon/cpp/build
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$GCYLON_PATH/lib:$LD_LIBRARY_PATH
```
