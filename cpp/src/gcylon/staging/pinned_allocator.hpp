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

#ifndef GCYLON_PINNED_ALLOCATOR_HPP
#define GCYLON_PINNED_ALLOCATOR_HPP

#include <cylon/net/buffer.hpp>
#include <gcylon/cudf_buffer.hpp>
#include <cuda_runtime.h>
#include <rmm/device_buffer.hpp>

namespace gcylon {

/**
 * Buffer backed by pinned (page-locked) CPU memory.
 *
 * Pinned memory allows faster GPU<->CPU transfers because:
 * 1. The OS cannot page it out
 * 2. CUDA can use DMA for transfers
 * 3. Async transfers are possible
 */
class PinnedBuffer : public cylon::Buffer {
public:
    explicit PinnedBuffer(size_t size) : ptr_(nullptr), size_(size) {
        cudaError_t err = cudaMallocHost(&ptr_, size);
        if (err != cudaSuccess) {
            ptr_ = nullptr;
            size_ = 0;
        }
    }

    ~PinnedBuffer() override {
        if (ptr_) {
            cudaFreeHost(ptr_);
            ptr_ = nullptr;
        }
    }

    int64_t GetLength() const override {
        return static_cast<int64_t>(size_);
    }

    uint8_t* GetByteBuffer() override {
        return static_cast<uint8_t*>(ptr_);
    }

    bool is_valid() const {
        return ptr_ != nullptr;
    }

    // Non-copyable
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    // Movable
    PinnedBuffer(PinnedBuffer&& other) noexcept
        : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    PinnedBuffer& operator=(PinnedBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_) {
                cudaFreeHost(ptr_);
            }
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

private:
    void* ptr_;
    size_t size_;
};

/**
 * Allocator that uses pinned CPU memory.
 *
 * Use this allocator when you need to stage data between GPU and CPU
 * with optimal transfer performance.
 */
class PinnedAllocator : public cylon::Allocator {
public:
    cylon::Status Allocate(int64_t length, std::shared_ptr<cylon::Buffer>* buffer) override {
        auto pinned = std::make_shared<PinnedBuffer>(static_cast<size_t>(length));
        if (!pinned->is_valid()) {
            return cylon::Status(cylon::Code::OutOfMemory,
                "Failed to allocate pinned memory of size " + std::to_string(length));
        }
        *buffer = pinned;
        return cylon::Status::OK();
    }
};

/**
 * Hybrid allocator: GPU first, spill to CPU when GPU budget is exhausted.
 *
 * This allocator tracks a GPU memory budget and allocates on GPU when
 * space is available. When the GPU budget is exceeded, it falls back
 * to pinned CPU memory for optimal transfer performance.
 */
class HybridAllocator : public cylon::Allocator {
public:
    explicit HybridAllocator(size_t gpu_budget)
        : gpu_budget_(gpu_budget), gpu_used_(0), cpu_used_(0) {}

    cylon::Status Allocate(int64_t length, std::shared_ptr<cylon::Buffer>* buffer) override {
        size_t len = static_cast<size_t>(length);

        if (gpu_used_ + len <= gpu_budget_) {
            // Allocate on GPU
            try {
                auto rmm_buf = std::make_shared<rmm::device_buffer>(len, rmm::cuda_stream_default);
                *buffer = std::make_shared<CudfBuffer>(rmm_buf);
                gpu_used_ += len;
                return cylon::Status::OK();
            } catch (const std::exception& e) {
                // Fall through to CPU allocation
            }
        }

        // Spill to pinned CPU memory
        auto pinned = std::make_shared<PinnedBuffer>(len);
        if (!pinned->is_valid()) {
            return cylon::Status(cylon::Code::OutOfMemory,
                "Failed to allocate memory (GPU and pinned both failed)");
        }
        *buffer = pinned;
        cpu_used_ += len;
        return cylon::Status::OK();
    }

    size_t gpu_used() const { return gpu_used_; }
    size_t cpu_used() const { return cpu_used_; }
    size_t gpu_budget() const { return gpu_budget_; }

    void reset_tracking() {
        gpu_used_ = 0;
        cpu_used_ = 0;
    }

private:
    size_t gpu_budget_;
    size_t gpu_used_;
    size_t cpu_used_;
};

} // namespace gcylon

#endif // GCYLON_PINNED_ALLOCATOR_HPP
