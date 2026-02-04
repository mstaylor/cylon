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

#ifndef GCYLON_CONFIG_HPP
#define GCYLON_CONFIG_HPP

#include <cstddef>

namespace gcylon {

/**
 * Configuration for memory-efficient chunked operations in gcylon.
 *
 * These settings control how gcylon handles large datasets that may
 * exceed available GPU memory. By processing data in chunks and
 * optionally staging intermediate results to CPU memory, gcylon can
 * handle datasets larger than available GPU memory.
 */
struct GcylonConfig {
    // Memory limits
    size_t gpu_memory_limit = 0;       // 0 = auto (use gpu_memory_fraction of free GPU memory)
    float gpu_memory_fraction = 0.8f;  // Used when gpu_memory_limit = 0

    // Chunking
    size_t chunk_size_bytes = 0;       // 0 = auto-calculate based on memory
    size_t min_chunk_rows = 1024;      // Minimum rows per chunk

    // CPU staging
    bool allow_cpu_staging = true;     // Spill intermediate results to CPU
    bool use_pinned_memory = true;     // Use pinned memory for faster transfers

    /**
     * Create default configuration.
     * Uses 80% of free GPU memory, enables CPU staging with pinned memory.
     */
    static GcylonConfig Default() {
        return GcylonConfig{};
    }

    /**
     * Create configuration optimized for high memory pressure scenarios.
     * Uses only 60% of free GPU memory and enables aggressive CPU staging.
     */
    static GcylonConfig LowMemory() {
        GcylonConfig config;
        config.gpu_memory_fraction = 0.6f;
        config.allow_cpu_staging = true;
        return config;
    }

    /**
     * Create configuration that disables chunking.
     * Use only when you're certain the data fits in GPU memory.
     */
    static GcylonConfig NoChunking() {
        GcylonConfig config;
        config.gpu_memory_fraction = 1.0f;
        config.allow_cpu_staging = false;
        return config;
    }
};

} // namespace gcylon

#endif // GCYLON_CONFIG_HPP
