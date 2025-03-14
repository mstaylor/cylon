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

#include "fmi_operations.hpp"
#include "cylon/util/macros.hpp"


namespace cylon {
    namespace net {
        template<typename T>
        FMI::Utils::Function<T> get_function(cylon::net::ReduceOp reduce_op) {
            switch (reduce_op) {
                case cylon::net::SUM: {
                    return FMI::Utils::Function<T>([](T a, T b) { return a + b; }, true, true);
                }
                case cylon::net::MIN: {
                    return FMI::Utils::Function<T>([](T a, T b) { return std::min(a, b); }, true, true);
                }
                case cylon::net::MAX: {
                    return FMI::Utils::Function<T>([](T a, T b) { return std::max(a, b); }, true, true);
                }
                case cylon::net::PROD: {
                    return FMI::Utils::Function<T>([](T a, T b) { return a * b; }, true, true);
                }
                default:
                    return nullptr;
            }
        }

        void FmiTableAllgatherImpl::Init(int num_buffers) {
            CYLON_UNUSED(num_buffers);
        }

        Status FmiTableAllgatherImpl::AllgatherBufferSizes(const int32_t *send_data, int num_buffers,
                                                           int32_t *rcv_data) const {
            return Status();
        }

        Status FmiTableAllgatherImpl::IallgatherBufferData(int buf_idx, const uint8_t *send_data, int32_t send_count,
                                                           uint8_t *recv_data, const std::vector<int32_t> &recv_count,
                                                           const std::vector<int32_t> &displacements) {
            return Status();
        }

        Status FmiTableAllgatherImpl::WaitAll(int num_buffers) {
            return Status();
        }

        void FmiTableGatherImpl::Init(int num_buffers) {

        }

        Status FmiTableGatherImpl::GatherBufferSizes(const int32_t *send_data, int num_buffers, int32_t *rcv_data,
                                                     int gather_root) const {
            return Status();
        }

        Status FmiTableGatherImpl::IgatherBufferData(int buf_idx, const uint8_t *send_data, int32_t send_count,
                                                     uint8_t *recv_data, const std::vector<int32_t> &recv_count,
                                                     const std::vector<int32_t> &displacements, int gather_root) {
            return Status();
        }

        Status FmiTableGatherImpl::WaitAll(int num_buffers) {
            return Status();
        }

        void FmiTableBcastImpl::Init(int32_t num_buffers) {

        }

        Status FmiTableBcastImpl::BcastBufferSizes(int32_t *buffer, int32_t count, int32_t bcast_root) const {
            return Status();
        }

        Status FmiTableBcastImpl::BcastBufferData(uint8_t *buf_data, int32_t send_count, int32_t bcast_root) const {
            return Status();
        }

        Status FmiTableBcastImpl::IbcastBufferData(int32_t buf_idx, uint8_t *buf_data, int32_t send_count,
                                                   int32_t bcast_root) {
            return Status();
        }

        Status FmiTableBcastImpl::WaitAll(int32_t num_buffers) {
            return Status();
        }

        Status FmiAllReduceImpl::AllReduceBuffer(const void *send_buf, void *rcv_buf, int count,
                                                 const std::shared_ptr<DataType> &data_type, ReduceOp reduce_op) const {
            return Status();
        }

        Status
        FmiAllgatherImpl::AllgatherBufferSize(const int32_t *send_data, int32_t num_buffers, int32_t *rcv_data) const {
            return Status();
        }

        Status FmiAllgatherImpl::IallgatherBufferData(int32_t buf_idx, const uint8_t *send_data, int32_t send_count,
                                                      uint8_t *recv_data, const std::vector<int32_t> &recv_count,
                                                      const std::vector<int32_t> &displacements) {
            return Status();
        }

        Status FmiAllgatherImpl::WaitAll() {
            return Status();
        }
    }
}