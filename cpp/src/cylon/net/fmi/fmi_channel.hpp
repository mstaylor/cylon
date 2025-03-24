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

#ifndef CYLON_FMI_CHANNEL_HPP
#define CYLON_FMI_CHANNEL_HPP

#include <unordered_map>
#include <queue>

#include "cylon/net/channel.hpp"
#include "cylon/thridparty/fmi/Communicator.hpp"
#include "fmi_operations.hpp"

namespace cylon {

    namespace fmi {


        enum FMISendStatus {
            SEND_INIT = 0,
            SEND_LENGTH_POSTED = 1,
            SEND_POSTED = 2,
            SEND_FINISH = 3,
            SEND_DONE = 4
        };

        enum FMIReceiveStatus {
            RECEIVE_INIT = 0,
            RECEIVE_LENGTH_POSTED = 1,
            RECEIVE_POSTED = 2,
            RECEIVED_FIN = 3
        };

        /**
        * Keep track about the length buffer to receive the length first
        */
        struct PendingSend {
            //  we allow upto 8 ints for the header
            int headerBuf[CYLON_CHANNEL_HEADER_SIZE]{};
            // segments of data to be sent
            std::queue<std::shared_ptr<CylonRequest>> pendingData{};

            FMISendStatus status = SEND_INIT;

            // the current send, if it is a actual send
            std::shared_ptr<CylonRequest> currentSend{};

            // UCX context - For tracking the progress of the message
            FMI::Utils::fmiContext *context;

        };

        struct PendingReceive {
            // we allow upto 8 integer header
            int headerBuf[CYLON_CHANNEL_HEADER_SIZE]{};
            int receiveId{};
            // Buffers are untyped: they simply denote a physical memory
            // area regardless of its intended meaning or interpretation.
            std::shared_ptr<Buffer> data{};
            int length{};
            FMIReceiveStatus status = RECEIVE_INIT;
            // FMI context - For tracking the progress of the message
            FMI::Utils::fmiContext *context;
        };


        class FMIChannel : public Channel {

            /**
            * Link the necessary parameters associated with the communicator to the channel
            * @param [in] com - The UCX communicator that created the channel
            * @return
            */
            explicit FMIChannel(const std::shared_ptr<FMI::Communicator> *com);


            /**
            * Initialize the channel
            *
            * @param receives receive from these ranks
            */
            void init(int edge,
                      const std::vector<int> &receives,
                      const std::vector<int> &sendIds,
                      ChannelReceiveCallback *rcv,
                      ChannelSendCallback *send,
                      Allocator *alloc) override;

            /**
            * Send the message to the target.
            *
            * @param request the request
            * @return true if accepted
            */
            int send(std::shared_ptr<CylonRequest> request) override;

            /**
            * Send the message to the target.
            *
            * @param request the request
            * @return true if accepted
            */
            int sendFin(std::shared_ptr<CylonRequest> request) override;

            /**
             * This method, will send the messages, It will first send a message with length and then
             */
            void progressSends() override;

            /**
             * Progress the pending receivers
             */
            void progressReceives() override;

            void close() override;

        private:
            int edge;
            // keep track of the length buffers for each receiver
            std::unordered_map<int, PendingSend *> sends;
            // keep track of the posted receives
            std::unordered_map<int, PendingReceive *> pendingReceives;
            // we got finish requests
            std::unordered_map<int, std::shared_ptr<CylonRequest>> finishRequests;
            // receive callback function
            ChannelReceiveCallback *rcv_fn;
            // send complete callback function
            ChannelSendCallback *send_comp_fn;
            // allocator
            Allocator *allocator;
            // mpi rank
            int rank;
            // mpi world size
            int worldSize;

            const std::shared_ptr<FMI::Communicator> *communicator;


            /**
             * UCX Receive
             * Modeled after the IRECV function of MPI
             * @param [out] buffer - Pointer to the output buffer
             * @param [in] count - Size of the receiving data
             * @param [in] sender - MPI id of the sender
             * @param [out] ctx - ucx::ucxContext object, used for tracking the progress of the request
             * @return Cylon Status
             */
            template<typename T>
            Status FMI_Irecv(FMI::Comm::Data<T> &buf,
                             int sender,
                             FMI::Utils::fmiContext* ctx);

            /**
             * UCX Send
             * Modeled after the ISEND function of MPI
             * @param [out] buffer - Pointer to the buffer to send
             * @param [in] count - Size of the receiving data
             * @param [in] ep - Endpoint to send the data to
             * @param [out] request - UCX Context object
             *                        Used for tracking the progress of the request
             * @return Cylon Status
             */
            template<typename T>
            Status FMI_Isend(FMI::Comm::Data<T> &buf,
                             int source,
                             FMI::Utils::fmiContext* request) const;

            /**
             * Send finish request
             * @param x the target, pendingSend pair
             */
            void sendFinishHeader(const std::pair<const int, PendingSend *> &x) const;

            /**
             * Send the length
             * @param x the target, pendingSend pair
             */
            void sendHeader(const std::pair<const int, PendingSend *> &x) const;

        };
    }


}
#endif //CYLON_FMI_CHANNEL_HPP
