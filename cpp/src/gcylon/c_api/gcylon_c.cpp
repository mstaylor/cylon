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
 * C API implementation - thin wrapper around existing C++ code.
 * All operations delegate to gtable_api.hpp functions.
 */

#include "gcylon_c.h"

#include <gcylon/gtable.hpp>
#include <gcylon/gtable_api.hpp>
#include <gcylon/gcylon_config.hpp>
#include <gcylon/staging/memory_utils.hpp>
#include <gcylon/utils/construct.hpp>

#include <cylon/ctx/cylon_context.hpp>
#include <cylon/join/join_config.hpp>
#include <cylon/net/mpi/mpi_communicator.hpp>
#include <cudf/table/table.hpp>
#include <cuda_runtime.h>

#include <memory>
#include <string>
#include <vector>

// Thread-local error message
thread_local std::string g_last_error;

// Opaque wrapper types
struct GcylonContext {
    std::shared_ptr<cylon::CylonContext> ctx;
};

struct GcylonTable {
    std::shared_ptr<gcylon::GTable> table;
};

// Helper: C config to C++ config
static gcylon::GcylonConfig to_cpp_config(const GcylonConfig* config) {
    if (!config) return gcylon::GcylonConfig::Default();
    gcylon::GcylonConfig c;
    c.gpu_memory_limit = config->gpu_memory_limit;
    c.gpu_memory_fraction = config->gpu_memory_fraction;
    c.chunk_size_bytes = config->chunk_size_bytes;
    c.min_chunk_rows = config->min_chunk_rows;
    return c;
}

static void set_error(const std::string& msg) { g_last_error = msg; }

static GcylonStatus to_c_status(const cylon::Status& s) {
    if (s.is_ok()) return GCYLON_OK;
    g_last_error = s.get_msg();
    return s.get_code() == cylon::Code::OutOfMemory ? GCYLON_OOM : GCYLON_ERROR;
}

extern "C" {

// Configuration
GcylonConfig gcylon_config_default(void) {
    auto c = gcylon::GcylonConfig::Default();
    return {c.gpu_memory_limit, c.gpu_memory_fraction, c.chunk_size_bytes,
            c.min_chunk_rows};
}

GcylonConfig gcylon_config_low_memory(void) {
    auto c = gcylon::GcylonConfig::LowMemory();
    return {c.gpu_memory_limit, c.gpu_memory_fraction, c.chunk_size_bytes,
            c.min_chunk_rows};
}

// Context
GcylonStatus gcylon_context_create_mpi(GcylonContext** ctx) {
    if (!ctx) return GCYLON_INVALID_ARG;
    try {
        auto config = std::make_shared<cylon::net::MPIConfig>();
        std::shared_ptr<cylon::CylonContext> cylon_ctx;
        auto s = cylon::CylonContext::InitDistributed(config, &cylon_ctx);
        if (!s.is_ok()) { set_error(s.get_msg()); return GCYLON_ERROR; }
        *ctx = new GcylonContext{cylon_ctx};
        return GCYLON_OK;
    } catch (const std::exception& e) { set_error(e.what()); return GCYLON_ERROR; }
}

void gcylon_context_free(GcylonContext* ctx) {
    if (ctx) { if (ctx->ctx) ctx->ctx->Finalize(); delete ctx; }
}

int32_t gcylon_context_get_rank(GcylonContext* ctx) {
    return ctx && ctx->ctx ? ctx->ctx->GetRank() : -1;
}

int32_t gcylon_context_get_world_size(GcylonContext* ctx) {
    return ctx && ctx->ctx ? ctx->ctx->GetWorldSize() : -1;
}

// GPU Device Management
GcylonStatus gcylon_get_device_count(int32_t* count) {
    if (!count) return GCYLON_INVALID_ARG;
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        set_error(cudaGetErrorString(err));
        return GCYLON_ERROR;
    }
    *count = device_count;
    return GCYLON_OK;
}

GcylonStatus gcylon_set_device(int32_t device_id) {
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        set_error(cudaGetErrorString(err));
        return GCYLON_ERROR;
    }
    return GCYLON_OK;
}

GcylonStatus gcylon_get_device(int32_t* device_id) {
    if (!device_id) return GCYLON_INVALID_ARG;
    int dev = -1;
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) {
        set_error(cudaGetErrorString(err));
        return GCYLON_ERROR;
    }
    *device_id = dev;
    return GCYLON_OK;
}

// Memory
GcylonStatus gcylon_get_gpu_memory_info(GcylonMemoryInfo* info) {
    if (!info) return GCYLON_INVALID_ARG;
    auto m = gcylon::get_gpu_memory_info();
    info->free_bytes = m.free; info->total_bytes = m.total; info->used_bytes = m.used;
    return GCYLON_OK;
}

// Table
int64_t gcylon_table_num_rows(GcylonTable* t) {
    return t && t->table ? t->table->GetCudfTable()->num_rows() : 0;
}

int32_t gcylon_table_num_columns(GcylonTable* t) {
    return t && t->table ? t->table->GetCudfTable()->num_columns() : 0;
}

void gcylon_table_free(GcylonTable* t) { delete t; }

GcylonStatus gcylon_table_create_sequential(GcylonContext* ctx, int32_t ncols, int64_t nrows,
    int64_t start, int64_t step, GcylonTable** output) {
    if (!ctx || !output || ncols <= 0 || nrows <= 0) return GCYLON_INVALID_ARG;
    try {
        auto tbl = constructTable(ncols, nrows, start, static_cast<int>(step), false);
        if (!tbl) { set_error("Failed to construct table"); return GCYLON_ERROR; }
        std::shared_ptr<gcylon::GTable> gt;
        auto s = gcylon::GTable::FromCudfTable(ctx->ctx, tbl, gt);
        if (!s.is_ok()) return to_c_status(s);
        *output = new GcylonTable{gt};
        return GCYLON_OK;
    } catch (const std::exception& e) { set_error(e.what()); return GCYLON_ERROR; }
}

GcylonStatus gcylon_table_create_random(GcylonContext* ctx, int32_t ncols, int64_t nrows,
    int32_t seed, GcylonTable** output) {
    if (!ctx || !output || ncols <= 0 || nrows <= 0) return GCYLON_INVALID_ARG;
    try {
        auto tbl = constructRandomDataTable(ncols, nrows, seed);
        if (!tbl) { set_error("Failed to construct random table"); return GCYLON_ERROR; }
        std::shared_ptr<gcylon::GTable> gt;
        auto s = gcylon::GTable::FromCudfTable(ctx->ctx, tbl, gt);
        if (!s.is_ok()) return to_c_status(s);
        *output = new GcylonTable{gt};
        return GCYLON_OK;
    } catch (const std::exception& e) { set_error(e.what()); return GCYLON_ERROR; }
}

// Operations - all delegate to existing C++ functions

GcylonStatus gcylon_shuffle(GcylonTable* input, const int32_t* cols, int32_t ncols,
                            GcylonTable** output, const GcylonConfig* config) {
    if (!input || !cols || !output) return GCYLON_INVALID_ARG;
    try {
        std::vector<int> hash_cols(cols, cols + ncols);
        std::unique_ptr<cudf::table> result;
        auto ctx = input->table->GetContext();
        auto s = gcylon::SmartShuffle(input->table->GetCudfTable()->view(), hash_cols,
                                      ctx, result, to_cpp_config(config));
        if (!s.is_ok()) return to_c_status(s);
        std::shared_ptr<gcylon::GTable> gt;
        s = gcylon::GTable::FromCudfTable(ctx, result, gt);
        if (!s.is_ok()) return to_c_status(s);
        *output = new GcylonTable{gt};
        return GCYLON_OK;
    } catch (const std::exception& e) { set_error(e.what()); return GCYLON_ERROR; }
}

GcylonStatus gcylon_allgather(GcylonTable* input, GcylonTable** output, const GcylonConfig* config) {
    if (!input || !output) return GCYLON_INVALID_ARG;
    try {
        std::unique_ptr<cudf::table> result;
        auto ctx = input->table->GetContext();
        auto s = gcylon::SmartAllGather(input->table->GetCudfTable()->view(),
                                        ctx, result, to_cpp_config(config));
        if (!s.is_ok()) return to_c_status(s);
        std::shared_ptr<gcylon::GTable> gt;
        s = gcylon::GTable::FromCudfTable(ctx, result, gt);
        if (!s.is_ok()) return to_c_status(s);
        *output = new GcylonTable{gt};
        return GCYLON_OK;
    } catch (const std::exception& e) { set_error(e.what()); return GCYLON_ERROR; }
}

GcylonStatus gcylon_gather(GcylonTable* input, int32_t root, GcylonTable** output,
                           const GcylonConfig* /* config */) {
    if (!input || !output) return GCYLON_INVALID_ARG;
    try {
        std::unique_ptr<cudf::table> result;
        auto ctx = input->table->GetContext();
        auto s = gcylon::Gather(input->table->GetCudfTable()->view(), ctx, result, root);
        if (!s.is_ok()) return to_c_status(s);
        std::shared_ptr<gcylon::GTable> gt;
        s = gcylon::GTable::FromCudfTable(ctx, result, gt);
        if (!s.is_ok()) return to_c_status(s);
        *output = new GcylonTable{gt};
        return GCYLON_OK;
    } catch (const std::exception& e) { set_error(e.what()); return GCYLON_ERROR; }
}

GcylonStatus gcylon_broadcast(GcylonTable* input, int32_t root, GcylonTable** output,
                              const GcylonConfig* /* config */) {
    if (!input || !output) return GCYLON_INVALID_ARG;
    try {
        std::unique_ptr<cudf::table> result;
        auto ctx = input->table->GetContext();
        auto s = gcylon::Broadcast(input->table->GetCudfTable()->view(), root, ctx, result);
        if (!s.is_ok()) return to_c_status(s);
        std::shared_ptr<gcylon::GTable> gt;
        s = gcylon::GTable::FromCudfTable(ctx, result, gt);
        if (!s.is_ok()) return to_c_status(s);
        *output = new GcylonTable{gt};
        return GCYLON_OK;
    } catch (const std::exception& e) { set_error(e.what()); return GCYLON_ERROR; }
}

GcylonStatus gcylon_distributed_join(GcylonTable* left, GcylonTable* right,
    const int32_t* lcols, int32_t nlcols, const int32_t* rcols, int32_t nrcols,
    GcylonJoinType jtype, GcylonTable** output, const GcylonConfig* /* config */) {
    if (!left || !right || !lcols || !rcols || !output) return GCYLON_INVALID_ARG;
    try {
        std::vector<int> left_cols(lcols, lcols + nlcols);
        std::vector<int> right_cols(rcols, rcols + nrcols);
        cylon::join::config::JoinType cjt;
        switch (jtype) {
            case GCYLON_JOIN_INNER: cjt = cylon::join::config::INNER; break;
            case GCYLON_JOIN_LEFT:  cjt = cylon::join::config::LEFT; break;
            case GCYLON_JOIN_RIGHT: cjt = cylon::join::config::RIGHT; break;
            case GCYLON_JOIN_OUTER: cjt = cylon::join::config::FULL_OUTER; break;
            default: return GCYLON_INVALID_ARG;
        }
        auto jc = cylon::join::config::JoinConfig(cjt, left_cols, right_cols,
            cylon::join::config::HASH, "", "");
        std::shared_ptr<gcylon::GTable> result;
        auto s = gcylon::DistributedJoin(left->table, right->table, jc, result);
        if (!s.is_ok()) return to_c_status(s);
        *output = new GcylonTable{result};
        return GCYLON_OK;
    } catch (const std::exception& e) { set_error(e.what()); return GCYLON_ERROR; }
}

GcylonStatus gcylon_distributed_sort(GcylonTable* input, const int32_t* cols, int32_t ncols,
    const int32_t* asc, GcylonTable** output, const GcylonConfig* /* config */) {
    if (!input || !cols || !output) return GCYLON_INVALID_ARG;
    try {
        std::vector<int32_t> sort_cols(cols, cols + ncols);
        std::vector<cudf::order> orders;
        for (int32_t i = 0; i < ncols; i++)
            orders.push_back(asc && asc[i] ? cudf::order::ASCENDING : cudf::order::DESCENDING);
        std::unique_ptr<cudf::table> result;
        auto ctx = input->table->GetContext();
        auto s = gcylon::DistributedSort(input->table->GetCudfTable()->view(), sort_cols, orders,
                                         ctx, result);
        if (!s.is_ok()) return to_c_status(s);
        std::shared_ptr<gcylon::GTable> gt;
        s = gcylon::GTable::FromCudfTable(ctx, result, gt);
        if (!s.is_ok()) return to_c_status(s);
        *output = new GcylonTable{gt};
        return GCYLON_OK;
    } catch (const std::exception& e) { set_error(e.what()); return GCYLON_ERROR; }
}

GcylonStatus gcylon_repartition(GcylonTable* input, const int32_t* rows_per_worker,
    int32_t nworkers, GcylonTable** output, const GcylonConfig* /* config */) {
    if (!input || !output) return GCYLON_INVALID_ARG;
    try {
        std::vector<int32_t> pm;
        if (rows_per_worker && nworkers > 0) pm.assign(rows_per_worker, rows_per_worker + nworkers);
        std::unique_ptr<cudf::table> result;
        auto ctx = input->table->GetContext();
        auto s = gcylon::Repartition(input->table->GetCudfTable()->view(), ctx, result, pm);
        if (!s.is_ok()) return to_c_status(s);
        std::shared_ptr<gcylon::GTable> gt;
        s = gcylon::GTable::FromCudfTable(ctx, result, gt);
        if (!s.is_ok()) return to_c_status(s);
        *output = new GcylonTable{gt};
        return GCYLON_OK;
    } catch (const std::exception& e) { set_error(e.what()); return GCYLON_ERROR; }
}

// Error handling
const char* gcylon_status_string(GcylonStatus s) {
    switch (s) {
        case GCYLON_OK: return "OK";
        case GCYLON_ERROR: return "Error";
        case GCYLON_OOM: return "Out of memory";
        case GCYLON_INVALID_ARG: return "Invalid argument";
        default: return "Unknown";
    }
}

const char* gcylon_get_last_error(void) { return g_last_error.c_str(); }

} // extern "C"
