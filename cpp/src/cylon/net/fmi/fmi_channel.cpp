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
#include "cylon/thridparty/fmi/Data.hpp"
#include <cylon/util/macros.hpp>

namespace cylon {

    namespace fmi {

        inline const char* NbxStatusToString(FMI::Utils::NbxStatus status) {
            switch (status) {
                case FMI::Utils::NbxStatus::SUCCESS: return "SUCCESS";
                case FMI::Utils::NbxStatus::RECEIVE_FAILED: return "RECEIVE_FAILED";
                case FMI::Utils::NbxStatus::SEND_FAILED: return "SEND_FAILED";
                case FMI::Utils::NbxStatus::DUMMY_SEND_FAILED: return "DUMMY_SEND_FAILED";
                case FMI::Utils::NbxStatus::CONNECTION_CLOSED_BY_PEER: return "CONNECTION_CLOSED_BY_PEER";
                case FMI::Utils::NbxStatus::SOCKET_CREATE_FAILED: return "SOCKET_CREATE_FAILED";
                case FMI::Utils::NbxStatus::TCP_NODELAY_FAILED: return "TCP_NODELAY_FAILED";
                case FMI::Utils::NbxStatus::FCNTL_GET_FAILED: return "FCNTL_GET_FAILED";
                case FMI::Utils::NbxStatus::FCNTL_SET_FAILED: return "FCNTL_SET_FAILED";
                case FMI::Utils::NbxStatus::ADD_EVENT_FAILED: return "ADD_EVENT_FAILED";
                case FMI::Utils::NbxStatus::EPOLL_WAIT_FAILED: return "EPOLL_WAIT_FAILED";
                case FMI::Utils::NbxStatus::SOCKET_PAIR_FAILED: return "SOCKET_PAIR_FAILED";
                case FMI::Utils::NbxStatus::SOCKET_SET_SO_RCVTIMEO_FAILED: return "SOCKET_SET_SO_RCVTIMEO_FAILED";
                case FMI::Utils::NbxStatus::SOCKET_SET_SO_SNDTIMEO_FAILED: return "SOCKET_SET_SO_SNDTIMEO_FAILED";
                case FMI::Utils::NbxStatus::SOCKET_SET_TCP_NODELAY_FAILED: return "SOCKET_SET_TCP_NODELAY_FAILED";
                case FMI::Utils::NbxStatus::SOCKET_SET_NONBLOCKING_FAILED: return "SOCKET_SET_NONBLOCKING_FAILED";
                case FMI::Utils::NbxStatus::NBX_TIMOUTOUT: return "NBX_TIMOUTOUT";
                default: return "UNKNOWN_STATUS";
            }
        }

        void recvHandler(FMI::Utils::NbxStatus status, const std::string &str, FMI::Utils::fmiContext *ctx) {
            LOG(INFO) << "recvHandler status: " << NbxStatusToString(status) << " msg: " << str;
            if (ctx != nullptr) {
                ctx->completed = 1;
            }
        }

        void sendHandler(FMI::Utils::NbxStatus status, const std::string & str, FMI::Utils::fmiContext *ctx) {
            LOG(INFO) << "sendHandler status: " << NbxStatusToString(status) << " msg: " << str;
            if (ctx != nullptr) {
                ctx->completed = 1;
            }
        }

        template<typename T>
        Status FMIChannel::FMI_Irecv(FMI::Comm::Data<T> &buf,
                                     int sender,
                                     FMI::Utils::fmiContext *ctx) {

            // Init completed
            if (ctx != nullptr) {
                ctx->completed = 0;
            }

            communicator->recv(buf, sender, ctx, mode_,recvHandler);

            return Status::OK();
        }

        template<typename T>
        Status FMIChannel::FMI_Isend(FMI::Comm::Data<T> &buf,
                                     int source,
                                     FMI::Utils::fmiContext *ctx) const {
            // Init completed
            if (ctx != nullptr) {
                ctx->completed = 0;
            }

            communicator->send(buf, source, ctx, mode_, sendHandler);

            return Status::OK();
        }

        FMIChannel::FMIChannel(std::shared_ptr<FMI::Communicator> com,
                               FMI::Utils::Mode mode):
                rank(com->getPeerId()),
                worldSize(com->getNumPeers()),
                communicator(com), mode_(mode) {}


        void FMIChannel::init(int ed,
                              const std::vector<int> &receives,
                              const std::vector<int> &sendIds,
                              ChannelReceiveCallback *rcv,
                              ChannelSendCallback *send_fn,
                              Allocator *alloc) {
            if (mode_ == FMI::Utils::BLOCKING) {
                rcv_fn = rcv;
                send_comp_fn = send_fn;
                allocator = alloc;

                for (int recvRank: receives) {
                    if (recvRank == rank) continue;
                    auto *buf = new PendingReceive();
                    buf->receiveId = recvRank;
                    //pendingReceives[recvRank] = buf;
                    pendingReceives.insert(std::pair<int, PendingReceive *>(recvRank, buf));
                    buf->context = new FMI::Utils::fmiContext;
                    buf->context->completed = 1;
                    buf->status = RECEIVE_INIT;
                }

                for (int target: sendIds) {
                    sends[target] = new PendingSend();
                }
            } else {
                // Storing the parameters given by the Cylon Channel class
                rcv_fn = rcv;
                send_comp_fn = send_fn;
                allocator = alloc;

                // Get the number of receives and sends to be used in iterations
                int numReci = (int) receives.size();
                //int numSends = (int) sendIds.size();
                // Int variable used when iterating
                int sIndx;

                // Iterate and set the receives
                for (sIndx = 0; sIndx < numReci; sIndx++) {

                    // Rank of the node receiving from
                    int recvRank = receives.at(sIndx);

                    if (rank == recvRank) {
                        continue;//FMI does not support local receives, so process during sends
                    }
                    // Init a new pending receive for the request
                    auto *buf = new PendingReceive();
                    buf->receiveId = recvRank;
                    // Add to pendingReceive object to pendingReceives map
                    pendingReceives.insert(std::pair<int, PendingReceive *>(recvRank, buf));
                    // Receive for the initial header buffer
                    // Init context
                    buf->context = new FMI::Utils::fmiContext;
                    buf->context->completed = 0;

                    auto send_data_byte_size = CYLON_CHANNEL_HEADER_SIZE * sizeof(int);
                    auto send_void_ptr = const_cast<void *>(static_cast<const void *>(buf->headerBuf));
                    FMI::Comm::Data<void *> send_void_data(send_void_ptr,
                                                           send_data_byte_size,
                                                           FMI::Comm::noop_deleter);
                    // FMI receive
                    FMI_Irecv(send_void_data,
                              recvRank,
                              buf->context);
                    // Init status of the receive
                    buf->status = RECEIVE_LENGTH_POSTED;
                }

                for (int target: sendIds) {
                    sends.emplace(target, new PendingSend());
                }
            }
        }

        int FMIChannel::send(std::shared_ptr<CylonRequest> request) {

            // Loads the pending send from sends
            PendingSend *ps = sends[request->target];
            if (ps->pendingData.size() > MAX_PENDING) {
                return -1;
            }
            // pendingData is a queue that has TXRequests
            ps->pendingData.push(request);
            return 1;
        }

        int FMIChannel::sendFin(std::shared_ptr<CylonRequest> request) {
            // Checks if the finished request is alreay in finished req
            // If so, give error
            if (finishRequests.find(request->target) != finishRequests.end()) {
                return -1;
            }

            // Add finished req to map
            finishRequests.insert(std::pair<int, std::shared_ptr<CylonRequest>>(request->target, request));
            return 1;
        }

        void FMIChannel::sendHeaderLocal(PendingSend  *pend_send) {

            auto &r = *pend_send->pendingData.front();
            //DLOG(INFO) << rank << " sendHeaderLocal";
            assert(r.headerLength <= 6);
            rcv_fn->receivedHeader(rank, CYLON_MSG_NOT_FIN, r.header, r.headerLength);
            pend_send->status = SEND_LENGTH_POSTED;
        }

        void FMIChannel::sendFinishHeaderLocal(PendingSend * pend_send) {
            rcv_fn->receivedHeader(rank, CYLON_MSG_FIN, nullptr, 0);
            pend_send->status = SEND_FINISH;
        }

        void FMIChannel::progressSendsLocal(PendingSend *pend_send) {
            if (pend_send->status == SEND_LENGTH_POSTED) {
                // now post the actual send
                // we set to the current send and pop it
                pend_send->currentSend = pend_send->pendingData.front();
                const auto &r = *pend_send->currentSend;
                std::shared_ptr<Buffer> data_buf;
                const auto &stat = allocator->Allocate(r.length, &data_buf);
                if (!stat.is_ok()) {
                    LOG(FATAL) << "Failed to allocate buffer with length " << r.length;
                }
                std::memcpy(data_buf->GetByteBuffer(), r.buffer, r.length);
                //DLOG(INFO) << "REC_DATA_LOCAL";
                rcv_fn->receivedData(rank, std::move(data_buf), r.length);

                pend_send->pendingData.pop();
                //pend_send->request = {};
                pend_send->status = SEND_POSTED;
            } else if (pend_send->status == SEND_INIT) {
                //pend_send.request = {};
                // now post the actual send
                if (!pend_send->pendingData.empty()) {
                    sendHeaderLocal(pend_send);
                } else if (finishRequests.find(rank) != finishRequests.end()) {
                    // if there are finish requests lets send them
                    sendFinishHeaderLocal(pend_send);
                }
            } else if (pend_send->status == SEND_POSTED) {
                //pend_send.request = {};
                // if there are more data to post, post the length buffer now
                if (!pend_send->pendingData.empty()) {
                    sendHeaderLocal(pend_send);
                    // we need to notify about the send completion
                    send_comp_fn->sendComplete(std::move(pend_send->currentSend));
                } else {
                    // we need to notify about the send completion
                    send_comp_fn->sendComplete(std::move(pend_send->currentSend));
                    // now check weather finish request is there
                    if (finishRequests.find(rank) != finishRequests.end()) {
                        sendFinishHeaderLocal(pend_send);
                    } else {
                        pend_send->status = SEND_INIT;
                    }
                }
            } else if (pend_send->status == SEND_FINISH) {
                // we are going to send complete
                send_comp_fn->sendFinishComplete(finishRequests[rank]);
                pend_send->status = SEND_DONE;
            } else if (pend_send->status != SEND_DONE) {
                // throw an exception and log
                LOG(FATAL) << "At an un-expected state " << pend_send->status;
            }
        }

        void FMIChannel::progressSendTo(int peer_id) {
            // Role-based ordering: Only send first if rank < peer_id
            //if (rank >= peer_id) return;

            PendingSend* ps = sends[peer_id];

            if (peer_id == rank) {
                progressSendsLocal(ps);
                return;
            }

            if (ps->status == SEND_LENGTH_POSTED) {
                // If completed
                if (ps->context->completed == 1) {
                    delete ps->context;

                    // Post the actual send

                    // Send the message
                    ps->context = new FMI::Utils::fmiContext();
                    ps->context->completed = 0;
                    auto r = ps->pendingData.front();

                    FMI::Comm::Data<void *> data(const_cast<void *>(r->buffer), r->length, FMI::Comm::noop_deleter);
                    FMI_Isend(data, r->target, ps->context);  // blocking send

                    ps->currentSend = r;
                    ps->pendingData.pop();
                    ps->status = SEND_POSTED;
                }

            } else if (ps->status == SEND_INIT) {
                if (!ps->pendingData.empty()) {
                    sendHeader({peer_id, ps});
                } else if (finishRequests.count(peer_id)) {
                    sendFinishHeader({peer_id, ps});
                }

            } else if (ps->status == SEND_POSTED) {
                if (ps->context->completed == 1) {
                    if (!ps->pendingData.empty()) {
                        sendHeader({peer_id, ps});
                        send_comp_fn->sendComplete(ps->currentSend);
                        ps->currentSend = {};
                    } else {
                        // If pending data is empty
                        // Notify about send completion
                        send_comp_fn->sendComplete(ps->currentSend);
                        ps->currentSend = {};

                        // Check if request is in finish
                        if (finishRequests.find(peer_id) != finishRequests.end()) {
                            sendFinishHeader({peer_id, ps});
                        } else {
                            // If req is not in finish then re-init
                            ps->status = SEND_INIT;
                        }
                    }
                }

            } else if (ps->status == SEND_FINISH) {
                std::shared_ptr<CylonRequest> finReq = finishRequests[peer_id];
                send_comp_fn->sendFinishComplete(finReq);
                ps->status = SEND_DONE;
                /*send_comp_fn->sendFinishComplete(finishRequests[peer_id]);
                ps->status = SEND_DONE;*/
            } else if (ps->status != SEND_DONE) {
                // If an unknown state
                // Throw an exception and log
                LOG(FATAL) << "At an un-expected state " << ps->status;
            }
        }

        void FMIChannel::progressSends() {

            if (mode_ == FMI::Utils::BLOCKING) {
                for (auto& [peer_id, send_state] : sends) {
                    if (rank < peer_id) {
                        progressSendTo(peer_id);
                    }
                }
                for (auto& [peer_id, send_state] : sends) {
                    if (rank >= peer_id) {
                        progressSendTo(peer_id);
                    }
                }
            } else {
                communicator->communicator_event_progress(FMI::Utils::Operation::SEND);

                // Iterate through the sends
                for (auto x: sends) {

                    int dest = x.first;

                    auto pend_send = x.second;


                    if (dest == rank) { // if local, short-circuit sends
                        progressSendsLocal(pend_send);
                        continue;
                    }

                    // If currently in the length posted stage of the send
                    if (x.second->status == SEND_LENGTH_POSTED) {
                        // If completed
                        if (x.second->context->completed == 1) {
                            // Destroy context object
                            //  NOTE can't use ucp_request_release here cuz we actually init our own UCX context here
                            delete x.second->context;

                            // Post the actual send
                            std::shared_ptr<CylonRequest> r = x.second->pendingData.front();
                            // Send the message
                            x.second->context = new FMI::Utils::fmiContext();
                            x.second->context->completed = 0;

                            auto send_data_byte_size = r->length;
                            auto send_void_ptr = const_cast<void *>(static_cast<const void *>(r->buffer));
                            FMI::Comm::Data<void *> send_void_data(send_void_ptr,
                                                                   send_data_byte_size,
                                                                   FMI::Comm::noop_deleter);


                            FMI_Isend(send_void_data,
                                      r->target,
                                      x.second->context);

                            // Update status
                            x.second->status = SEND_POSTED;

                            // We set to the current send and pop it
                            x.second->pendingData.pop();
                            // The update the current send in the queue of sends
                            x.second->currentSend = r;
                        }
                    } else if (x.second->status == SEND_INIT) {
                        // Send header if no pending data
                        if (!x.second->pendingData.empty()) {
                            sendHeader(x);
                        }
                    } else if (x.second->status == SEND_POSTED) {
                        // If completed
                        if (x.second->context->completed == 1) {
                            // If there are more data to post, post the length buffer now
                            if (!x.second->pendingData.empty()) {
                                // If the pending data is not empty
                                sendHeader(x);
                                // We need to notify about the send completion
                                send_comp_fn->sendComplete(x.second->currentSend);
                                x.second->currentSend = {};
                            } else {
                                // If pending data is empty
                                // Notify about send completion
                                send_comp_fn->sendComplete(x.second->currentSend);
                                x.second->currentSend = {};

                                // Check if request is in finish
                                if (finishRequests.find(x.first) != finishRequests.end()) {
                                    sendFinishHeader(x);
                                } else {
                                    // If req is not in finish then re-init
                                    x.second->status = SEND_INIT;
                                }
                            }
                        }
                    } else if (x.second->status == SEND_FINISH) {
                        if (x.second->context->completed == 1) {
                            // We are going to send complete
                            std::shared_ptr<CylonRequest> finReq = finishRequests[x.first];
                            send_comp_fn->sendFinishComplete(finReq);
                            x.second->status = SEND_DONE;
                        }
                    } else if (x.second->status != SEND_DONE) {
                        // If an unknown state
                        // Throw an exception and log
                        LOG(FATAL) << "At an un-expected state " << x.second->status;
                    }
                }
            }

        }


        void FMIChannel::progressReceiveFrom(int peer_id) {
            // Role-based ordering: Only receive first if rank >= peer_id
            //if (rank < peer_id) return;

            PendingReceive* recv = pendingReceives[peer_id];

            if (peer_id == rank) return;

            if (recv->status == RECEIVE_INIT) {

                FMI::Comm::Data<void *> header_buf(recv->headerBuf,
                                                   CYLON_CHANNEL_HEADER_SIZE * sizeof(int),
                                                   FMI::Comm::noop_deleter);
                FMI_Irecv(header_buf, peer_id, recv->context);  // blocking recv

                recv->status = RECEIVE_LENGTH_POSTED;

            } else if (recv->status == RECEIVE_LENGTH_POSTED) {
                if (recv->context->completed == 1) {


                    int length = recv->headerBuf[0];
                    int finFlag = recv->headerBuf[1];

                    if (finFlag == CYLON_MSG_FIN) {
                        recv->status = RECEIVED_FIN;
                        rcv_fn->receivedHeader(peer_id, finFlag, nullptr, 0);
                        return;
                    }

                    // Reset context
                    delete recv->context;
                    recv->context = new FMI::Utils::fmiContext;
                    recv->context->completed = 0;

                    Status stat = allocator->Allocate(length, &recv->data);

                    if (!stat.is_ok()) {
                        LOG(FATAL) << "Failed to allocate buffer with length " << length;
                    }

                    recv->length = length;

                    FMI::Comm::Data<void *> payload(recv->data->GetByteBuffer(), length, FMI::Comm::noop_deleter);

                    LOG(INFO) << "process receives RECEIVE_LENGTH_POSTED - bytebuff address: "
                              << static_cast<void *>(recv->data->GetByteBuffer())
                              << ", data.buf.get(): " << payload.get();

                    FMI_Irecv(payload, peer_id, recv->context);  // blocking recv

                    recv->status = RECEIVE_POSTED;
                    int *header = new int[6];
                    std::memcpy(header, &recv->headerBuf[2], 6 * sizeof(int));
                    rcv_fn->receivedHeader(peer_id, finFlag, header, 6);
                }
            } else if (recv->status == RECEIVE_POSTED) {
                if (recv->context->completed == 1) {
                    std::fill_n(recv->headerBuf, CYLON_CHANNEL_HEADER_SIZE, 0);

                    // Reset the context
                    delete recv->context;
                    recv->context = new FMI::Utils::fmiContext;
                    recv->context->completed = 0;

                    auto send_data_byte_size = CYLON_CHANNEL_HEADER_SIZE * sizeof(int);
                    auto send_void_ptr = const_cast<void *>(static_cast<const void *>(recv->headerBuf));

                    FMI::Comm::Data<void *> send_void_data(send_void_ptr,
                                                           send_data_byte_size,
                                                           FMI::Comm::noop_deleter);

                    LOG(INFO) << "process receives RECEIVE_POSTED - headerBuf address: "
                              << static_cast<void *>(recv->headerBuf)
                              << ", data.buf.get(): " << send_void_data.get();

                    // UCX receive
                    FMI_Irecv(send_void_data,
                              peer_id,
                              recv->context);
                    // Set state
                    recv->status = RECEIVE_LENGTH_POSTED;
                    // Call the back end
                    rcv_fn->receivedData(peer_id, recv->data, recv->length);
                }
            } else if (recv->status != RECEIVED_FIN) {
                LOG(FATAL) << "At an un-expected state " << recv->status;
            }
        }

        void FMIChannel::progressReceives() {

            if (mode_ == FMI::Utils::BLOCKING) {
                for (auto& [peer_id, recv_state] : pendingReceives) {
                    if (rank < peer_id) {
                        progressReceiveFrom(peer_id);
                    }
                }
                for (auto& [peer_id, recv_state] : pendingReceives) {
                    if (rank >= peer_id) {
                        progressReceiveFrom(peer_id);
                    }
                }
            } else {

                communicator->communicator_event_progress(FMI::Utils::Operation::RECEIVE);

                // Iterate through the pending receives
                for (auto x: pendingReceives) {
                    // Check if the buffer is posted

                    assert(x.first != rank);

                    if (x.second->status == RECEIVE_LENGTH_POSTED) {
                        // If completed request is completed
                        if (x.second->context->completed == 1) {
                            // Get data from the header
                            // read the length from the header
                            int length = x.second->headerBuf[0];
                            int finFlag = x.second->headerBuf[1];

                            // Check weather we are at the end
                            if (finFlag != CYLON_MSG_FIN) {
                                // If not at the end

                                // Malloc a buffer
                                Status stat = allocator->Allocate(length, &x.second->data);
                                if (!stat.is_ok()) {
                                    LOG(FATAL) << "Failed to allocate buffer with length " << length;
                                }

                                // Set the length
                                x.second->length = length;

                                // Reset context
                                delete x.second->context;
                                x.second->context = new FMI::Utils::fmiContext;
                                x.second->context->completed = 0;

                                // FMI receive
                                auto send_void_ptr = const_cast<void *>(static_cast<const void *>(x.second->data->GetByteBuffer()));
                                FMI::Comm::Data<void *> send_void_data(send_void_ptr,
                                                                       length,
                                                                       FMI::Comm::noop_deleter);

                                LOG(INFO) << "process receives RECEIVE_LENGTH_POSTED - bytebuff address: "
                                          << static_cast<void *>(x.second->data->GetByteBuffer())
                                          << ", data.buf.get(): " << send_void_data.get();

                                FMI_Irecv(send_void_data, x.first, x.second->context);
                                // Set the flag to true so we can identify later which buffers are posted
                                x.second->status = RECEIVE_POSTED;

                                // copy the count - 2 to the buffer
                                int *header = nullptr;
                                header = new int[6];
                                memcpy(header, &(x.second->headerBuf[2]), 6 * sizeof(int));

                                // Notify the receiver that the destination received the header
                                rcv_fn->receivedHeader(x.first, finFlag, header, 6);
                            } else {
                                // We are not expecting to receive any more
                                x.second->status = RECEIVED_FIN;
                                // Notify the receiver
                                rcv_fn->receivedHeader(x.first, finFlag, nullptr, 0);
                            }
                        }
                    } else if (x.second->status == RECEIVE_POSTED) {
                        // if request completed
                        if (x.second->context->completed == 1) {
                            // Fill header buffer
                            std::fill_n(x.second->headerBuf, CYLON_CHANNEL_HEADER_SIZE, 0);

                            // Reset the context
                            delete x.second->context;
                            x.second->context = new FMI::Utils::fmiContext;
                            x.second->context->completed = 0;

                            auto send_data_byte_size = CYLON_CHANNEL_HEADER_SIZE * sizeof(int);
                            auto send_void_ptr = const_cast<void *>(static_cast<const void *>(x.second->headerBuf));

                            FMI::Comm::Data<void *> send_void_data(send_void_ptr,
                                                                   send_data_byte_size,
                                                                   FMI::Comm::noop_deleter);

                            LOG(INFO) << "process receives RECEIVE_POSTED - headerBuf address: "
                                      << static_cast<void *>(x.second->headerBuf)
                                      << ", data.buf.get(): " << send_void_data.get();

                            // UCX receive
                            FMI_Irecv(send_void_data,
                                      x.first,
                                      x.second->context);
                            // Set state
                            x.second->status = RECEIVE_LENGTH_POSTED;
                            // Call the back end
                            rcv_fn->receivedData(x.first, x.second->data, x.second->length);
                        }
                    } else if (x.second->status != RECEIVED_FIN) {
                        LOG(FATAL) << "At an un-expected state " << x.second->status;
                    }
                }
            }
        }

        void FMIChannel::close() {
            for (auto &pendingReceive: pendingReceives) {
                delete (pendingReceive.second->context);
                delete (pendingReceive.second);
            }
            pendingReceives.clear();

            // Clear the sends
            for (auto &s: sends) {
                delete (s.second->context);
                delete (s.second);
            }
            sends.clear();
        }

        void FMIChannel::sendFinishHeader(int target, cylon::fmi::PendingSend *ps) {
            ps->headerBuf[0] = 0;
            ps->headerBuf[1] = CYLON_MSG_FIN;
            FMI::Comm::Data<void *> fin_msg(ps->headerBuf,
                                            8 * sizeof(int),
                                            FMI::Comm::noop_deleter);
            FMI_Isend(fin_msg, target, nullptr);
            ps->status = SEND_FINISH;
        }

        void FMIChannel::sendFinishHeader(const std::pair<const int, PendingSend *> &x) const {
            // for the last header we always send only the first 2 integers
            x.second->headerBuf[0] = 0;
            x.second->headerBuf[1] = CYLON_MSG_FIN;
            delete x.second->context;
            x.second->context = new FMI::Utils::fmiContext;
            x.second->context->completed = 0;

            auto send_data_byte_size = 8 * sizeof(int);
            auto send_void_ptr = const_cast<void *>(static_cast<const void *>(x.second->headerBuf));
            FMI::Comm::Data<void *> send_void_data(send_void_ptr,
                                                   send_data_byte_size,
                                                   FMI::Comm::noop_deleter);

            LOG(INFO) << "sendFinishHeader - bytebuff address: " << static_cast<void *>(x.second->headerBuf)
                      << ", data.buf.get(): " << send_void_data.get();

            FMI_Isend(send_void_data,
                      x.first,
                      x.second->context);
            x.second->status = SEND_FINISH;
        }

        void FMIChannel::sendHeader(int target, cylon::fmi::PendingSend *ps) {
            std::shared_ptr<CylonRequest> r = ps->pendingData.front();
            ps->headerBuf[0] = r->length;
            ps->headerBuf[1] = 0;
            // Copy data from CylonRequest header to the PendingSend header
            if (r->headerLength > 0) {
                std::memcpy(&ps->headerBuf[2], &(r->header[0]), r->headerLength * sizeof(int));
            }

            FMI::Comm::Data<void *> header_msg(ps->headerBuf,
                                               (2 + r->headerLength) * sizeof(int),
                                               FMI::Comm::noop_deleter);

            LOG(INFO) << "Sendheader - bytebuff address: " << static_cast<void *>(ps->headerBuf)
                      << ", data.buf.get(): " << header_msg.get();

            FMI_Isend(header_msg, target, nullptr);
            ps->status = SEND_LENGTH_POSTED;
        }

        void FMIChannel::sendHeader(const std::pair<const int, PendingSend *> &x) const {
            // Get the request
            std::shared_ptr<CylonRequest> r = x.second->pendingData.front();
            // Put the length to the buffer
            x.second->headerBuf[0] = r->length;
            x.second->headerBuf[1] = 0;

            // Copy data from CylonRequest header to the PendingSend header
            if (r->headerLength > 0) {
                memcpy(&(x.second->headerBuf[2]),
                       &(r->header[0]),
                       r->headerLength * sizeof(int));
            }
            delete x.second->context;
            // UCX send of the header
            x.second->context = new FMI::Utils::fmiContext();
            x.second->context->completed = 0;

            auto send_data_byte_size = (2 + r->headerLength) * sizeof(int);
            auto send_void_ptr = const_cast<void *>(static_cast<const void *>(x.second->headerBuf));
            FMI::Comm::Data<void *> send_void_data(send_void_ptr,
                                                   send_data_byte_size,
                                                   FMI::Comm::noop_deleter);

            LOG(INFO) << "Sendheader - bytebuff address: " << static_cast<void *>(x.second->headerBuf)
                      << ", data.buf.get(): " << send_void_data.get();


            FMI_Isend(send_void_data,
                      r->target,
                      x.second->context);

            // Update status
            x.second->status = SEND_LENGTH_POSTED;
        }



    }

}
