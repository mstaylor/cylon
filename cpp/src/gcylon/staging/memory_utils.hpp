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

#ifndef GCYLON_MEMORY_UTILS_HPP
#define GCYLON_MEMORY_UTILS_HPP

#include <cuda_runtime.h>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/bit.hpp>
#include <algorithm>
#include <cstddef>

namespace gcylon {

/**
 * GPU memory information structure.
 */
struct GpuMemoryInfo {
    size_t free;   // Free memory in bytes
    size_t total;  // Total memory in bytes
    size_t used;   // Used memory in bytes

    float usage_fraction() const {
        return total > 0 ? static_cast<float>(used) / static_cast<float>(total) : 0.0f;
    }
};

/**
 * Query current GPU memory state.
 *
 * @return GpuMemoryInfo structure with current memory statistics
 */
inline GpuMemoryInfo get_gpu_memory_info() {
    GpuMemoryInfo info;
    cudaError_t err = cudaMemGetInfo(&info.free, &info.total);
    if (err != cudaSuccess) {
        info.free = 0;
        info.total = 0;
    }
    info.used = info.total - info.free;
    return info;
}

/**
 * Estimate memory needed for a cudf table.
 *
 * This provides a rough estimate based on column types and row counts.
 * The actual memory usage may be higher due to alignment and metadata.
 *
 * @param tv Table view to estimate
 * @return Estimated size in bytes
 */
inline size_t estimate_table_size(const cudf::table_view& tv) {
    if (tv.num_columns() == 0 || tv.num_rows() == 0) {
        return 0;
    }

    size_t total = 0;
    for (int i = 0; i < tv.num_columns(); i++) {
        auto col = tv.column(i);
        // Data size based on type width
        total += static_cast<size_t>(col.size()) * cudf::size_of(col.type());
        // Null bitmask if present
        if (col.nullable()) {
            total += cudf::bitmask_allocation_size_bytes(col.size());
        }
    }
    return total;
}

/**
 * Estimate peak memory for a shuffle operation.
 *
 * Shuffle requires:
 * - Input table (already allocated)
 * - Hash partition output (copy of input, partitioned)
 * - Serialization buffers for each partition
 * - Receive buffers for incoming data
 * - Final concatenated output
 *
 * @param tv Input table view
 * @param world_size Number of workers in the cluster
 * @return Estimated peak memory in bytes
 */
inline size_t estimate_shuffle_memory(const cudf::table_view& tv, int world_size) {
    size_t input_size = estimate_table_size(tv);
    // Conservative estimate: input + partitioned + send buffers + receive buffers + output
    // Factor of (world_size + 3) accounts for partitioned data plus buffers
    return input_size * static_cast<size_t>(world_size + 3);
}

/**
 * Estimate peak memory for an AllGather operation.
 *
 * AllGather produces world_size copies of the input, so output is:
 * input_size * world_size
 *
 * @param tv Input table view
 * @param world_size Number of workers in the cluster
 * @return Estimated peak memory in bytes
 */
inline size_t estimate_allgather_memory(const cudf::table_view& tv, int world_size) {
    size_t input_size = estimate_table_size(tv);
    // Output is world_size times the input, plus intermediate buffers
    return input_size * static_cast<size_t>(world_size + 2);
}

/**
 * Calculate optimal chunk size in rows for memory-efficient processing.
 *
 * Given available memory and operation requirements, calculates how many
 * rows can be processed per chunk without exceeding memory limits.
 *
 * @param tv Input table view (for row size estimation)
 * @param world_size Number of workers in the cluster
 * @param available_memory Available GPU memory for the operation
 * @param min_rows Minimum rows per chunk (to avoid tiny inefficient chunks)
 * @return Recommended rows per chunk
 */
inline size_t calculate_chunk_rows(
    const cudf::table_view& tv,
    int world_size,
    size_t available_memory,
    size_t min_rows = 1024
) {
    int64_t num_rows = tv.num_rows();
    if (num_rows <= 0) {
        return min_rows;
    }

    size_t table_size = estimate_table_size(tv);
    size_t row_size = table_size / static_cast<size_t>(num_rows);
    if (row_size == 0) {
        row_size = 1;
    }

    // Memory per chunk: chunk_data + partitioned + buffers
    // Multiply by (world_size + 3) for shuffle overhead
    size_t mem_per_row = row_size * static_cast<size_t>(world_size + 3);
    if (mem_per_row == 0) {
        return static_cast<size_t>(num_rows);
    }

    size_t max_rows = available_memory / mem_per_row;

    // Clamp to [min_rows, num_rows]
    return std::max(min_rows, std::min(max_rows, static_cast<size_t>(num_rows)));
}

/**
 * Check if an operation would fit in available GPU memory.
 *
 * @param estimated_memory Estimated memory requirement in bytes
 * @param safety_fraction Fraction of free memory to consider "safe" (default 0.8)
 * @return true if operation should fit, false if chunking is recommended
 */
inline bool fits_in_gpu_memory(size_t estimated_memory, float safety_fraction = 0.8f) {
    auto mem_info = get_gpu_memory_info();
    size_t safe_memory = static_cast<size_t>(static_cast<float>(mem_info.free) * safety_fraction);
    return estimated_memory <= safe_memory;
}

/**
 * Calculate available memory for an operation given configuration.
 *
 * @param config_limit Explicit limit (0 = use fraction)
 * @param config_fraction Fraction of free memory to use
 * @return Available memory in bytes
 */
inline size_t get_available_memory(size_t config_limit, float config_fraction) {
    if (config_limit > 0) {
        return config_limit;
    }
    auto mem_info = get_gpu_memory_info();
    return static_cast<size_t>(static_cast<float>(mem_info.free) * config_fraction);
}

} // namespace gcylon

#endif // GCYLON_MEMORY_UTILS_HPP
