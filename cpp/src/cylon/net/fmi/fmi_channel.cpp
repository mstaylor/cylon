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

    namespace fmi {

        FMIChannel::FMIChannel(const FMI::Communicator *com) :
                rank(com->getPeerId()),
                worldSize(com->getNumPeers()) {}


        void FMIChannel::init(int ed,
                              const std::vector<int> &receives,
                              const std::vector<int> &sendIds,
                              ChannelReceiveCallback *rcv,
                              ChannelSendCallback *send_fn,
                              Allocator *alloc) {
            // Storing the parameters given by the Cylon Channel class
            edge = ed;
            rcv_fn = rcv;
            send_comp_fn = send_fn;
            allocator = alloc;

            // Get the number of receives and sends to be used in iterations
            int numReci = (int) receives.size();
            int numSends = (int) sendIds.size();
            // Int variable used when iterating
            int sIndx;

            // Iterate and set the receives
            for (sIndx = 0; sIndx < numReci; sIndx++) {
                // Rank of the node receiving from
                int recvRank = receives.at(sIndx);
                // Init a new pending receive for the request
                auto *buf = new PendingReceive();
                buf->receiveId = recvRank;
                // Add to pendingReceive object to pendingReceives map
                pendingReceives.insert(std::pair<int, PendingReceive *>(recvRank, buf));
                // Receive for the initial header buffer
                // Init context
                buf->context = new fmi::fmiContext();
                buf->context->completed = 0;
                // UCX receive
                FMI_Irecv(buf->headerBuf,
                          CYLON_CHANNEL_HEADER_SIZE * sizeof(int),
                          recvRank,
                          buf->context);
                // Init status of the receive
                buf->status = RECEIVE_LENGTH_POSTED;
            }


            // Iterate and set the sends
            for (sIndx = 0; sIndx < numSends; sIndx++) {
                // Rank of the node sending to
                int sendRank = sendIds.at(sIndx);
                // Init a new pending send for the request
                sends[sendRank] = new PendingSend();
            }
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
                                     fmi::fmiContext* ctx) {
            return Status();
        }

        Status FMIChannel::FMIIsend(const void *buffer, size_t count,
                                    fmi::fmiContext* ctx) const {
            return Status();
        }

        void FMIChannel::sendFinishHeader(const std::pair<const int, PendingSend *> &x) const {

        }

        void FMIChannel::sendHeader(const std::pair<const int, PendingSend *> &x) const {

        }

    }

}
