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
template<typename T>
FMI::Utils::Function<T> get_function(cylon::net::ReduceOp reduce_op) {
    switch (reduce_op) {
        case cylon::net::SUM: {
            return FMI::Utils::Function<T> ([](T a, T b) {return a + b;}, true, true);
        }
        case cylon::net::MIN: {
            return FMI::Utils::Function<T>([] (T a, T b) {return std::min(a, b);}, true, true);
        }
        case cylon::net::MAX: {
            return FMI::Utils::Function<T>([] (T a, T b) {return std::max(a, b);}, true, true);
        }
        case cylon::net::PROD: {
            return FMI::Utils::Function<T>([] (T a, T b) {return a * b;}, true, true);
        }
        default: return nullptr;
    }
}

cylon::fmi::FmiAllReduceImpl::FmiAllReduceImpl(const FMI::Communicator &comm) : comm_(comm) {}

cylon::Status cylon::fmi::FmiAllReduceImpl::AllReduceBuffer(const void *send_buf, void *rcv_buf, int count,
                                                            const std::shared_ptr<DataType> &data_type,
                                                            cylon::net::ReduceOp reduce_op) const {
    return cylon::Status();
}
