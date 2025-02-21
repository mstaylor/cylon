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

#ifndef CYLON_FMI_OPERATIONS_HPP
#define CYLON_FMI_OPERATIONS_HPP

#include <cylon/net/comm_operations.hpp>
#include <cylon/thridparty/fmi/utils/Function.hpp>
#include "cylon/status.hpp"
#include "cylon/net/ops/base_ops.hpp"
#include <cylon/thridparty/fmi/Communicator.hpp>

namespace cylon {
    namespace fmi {

        template<typename T>
        FMI::Utils::Function<T> get_function(cylon::net::ReduceOp reduce_op);

        class FmiAllReduceImpl : public net::AllReduceImpl {
        public:
            explicit FmiAllReduceImpl(const FMI::Communicator &comm);

            Status AllReduceBuffer(const void *send_buf,
                                   void *rcv_buf,
                                   int count,
                                   const std::shared_ptr<DataType> &data_type,
                                   net::ReduceOp reduce_op) const override;

        private:
            FMI::Communicator comm_;
        };

    }
}

#endif //CYLON_FMI_OPERATIONS_HPP
