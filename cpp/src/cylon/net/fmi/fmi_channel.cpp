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


#include <iostream>
#include <utility>
#include <cylon/status.hpp>
#include <cmath>
#include <algorithm>
#include <glog/logging.h>

#include "fmi_channel.hpp"
#include <cylon/util/macros.hpp>

namespace cylon {

    void FMIChannel::init(int edge, const std::vector<int> &receives, const std::vector<int> &sendIds,
                                      ChannelReceiveCallback *rcv, ChannelSendCallback *send, Allocator *alloc) {

    }

    int FMIChannel::send(std::shared_ptr<CylonRequest> request) {
        return 0;
    }

    int FMIChannel::sendFin(std::shared_ptr<CylonRequest> request) {
        return 0;
    }

    void FMIChannel::progressSends() {

    }

    void FMIChannel::progressReceives() {

    }

    void FMIChannel::close() {

    }

    Status FMIChannel::FMI_Irecv(void *buffer, size_t count, int source,
                                                    std::shared_ptr<FMI::Communicator> *comm_ptr_) {
        return Status();
    }

    Status FMIChannel::FMIIsend(const void *buffer, size_t count,
                                                   std::shared_ptr<FMI::Communicator> *comm_ptr_) const {
        return Status();
    }

    void FMIChannel::sendFinishHeader(const std::pair<const int, PendingSend *> &x) const {

    }

    void FMIChannel::sendHeader(const std::pair<const int, PendingSend *> &x) const {

    }
}
