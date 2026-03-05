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
 * exceed available GPU memory. By processing data in chunks, gcylon
 * can perform distributed operations (Shuffle, AllGather) on datasets
 * that would otherwise cause GPU OOM errors.
 *
 * For GPU-to-CPU memory spillover (beyond network chunking), use
 * RMM's managed_memory_resource instead of custom allocators:
 *
 *   auto mr = rmm::mr::managed_memory_resource{};
 *   rmm::mr::set_current_device_resource(&mr);
 *
 * This enables CUDA Unified Memory (UVM) which automatically migrates
 * pages between GPU and CPU as needed.
 */
struct GcylonConfig {
    // Memory limits
    size_t gpu_memory_limit = 0;       // 0 = auto (use gpu_memory_fraction of free GPU memory)
    float gpu_memory_fraction = 0.8f;  // Used when gpu_memory_limit = 0

    // Chunking
    size_t chunk_size_bytes = 0;       // 0 = auto-calculate based on memory
    size_t min_chunk_rows = 1024;      // Minimum rows per chunk

    /**
     * Create default configuration.
     * Uses 80% of free GPU memory with auto chunk sizing.
     */
    static GcylonConfig Default() {
        return GcylonConfig{};
    }

    /**
     * Create configuration optimized for high memory pressure scenarios.
     * Uses only 60% of free GPU memory.
     */
    static GcylonConfig LowMemory() {
        GcylonConfig config;
        config.gpu_memory_fraction = 0.6f;
        return config;
    }
};

} // namespace gcylon

#endif // GCYLON_CONFIG_HPP
