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

#include "libfabric_channel.hpp"
#include <glog/logging.h>
#include <cstring>
#include <cylon/util/macros.hpp>

namespace cylon {
namespace libfabric {

LibfabricChannel::LibfabricChannel(struct fid_ep *ep,
                                   struct fid_cq *cq,
                                   const std::vector<fi_addr_t> &peer_addrs,
                                   int rank, int world_size,
                                   const std::string &redis_host,
                                   int redis_port,
                                   const std::string &session_id)
    : ep_(ep), cq_(cq), peer_addrs_(peer_addrs),
      rank_(rank), worldSize_(world_size),
      redis_host_(redis_host), redis_port_(redis_port),
      session_id_(session_id) {}

void LibfabricChannel::init(int edge,
                            const std::vector<int> &receives,
                            const std::vector<int> &sendIds,
                            ChannelReceiveCallback *rcv,
                            ChannelSendCallback *send,
                            Allocator *alloc) {
    CYLON_UNUSED(edge);
    rcv_fn_ = rcv;
    send_comp_fn_ = send;
    allocator_ = alloc;

    // Initialize pending sends
    for (int target : sendIds) {
        auto *ps = new PendingSend();
        ps->target_addr = peer_addrs_[target];
        sends_[target] = ps;
    }

    // Initialize pending receives
    for (int source : receives) {
        auto *pr = new PendingReceive();
        pr->receiveId = source;
        pr->source_addr = peer_addrs_[source];
        pendingReceives_[source] = pr;
    }
}

int LibfabricChannel::send(std::shared_ptr<CylonRequest> request) {
    int target = request->target;
    auto it = sends_.find(target);
    if (it == sends_.end()) {
        return -1;
    }

    auto *ps = it->second;
    if (ps->pendingData.size() >= MAX_PENDING) {
        return -1;
    }

    ps->pendingData.push(std::move(request));
    return 1;
}

int LibfabricChannel::sendFin(std::shared_ptr<CylonRequest> request) {
    int target = request->target;
    finishRequests_[target] = std::move(request);
    return 1;
}

Status LibfabricChannel::postSend(const void *buf, size_t len, fi_addr_t dest) {
    ssize_t ret;
    while (true) {
        ret = fi_send(ep_, buf, len, nullptr, dest, nullptr);
        if (ret == 0) break;
        if (ret == -FI_EAGAIN) {
            drainCQ();
            continue;
        }
        return {Code::ExecutionError,
                "fi_send failed: " + std::string(fi_strerror(static_cast<int>(-ret)))};
    }
    // Wait for send completion
    while (!drainCQ()) {
        // spin-poll
    }
    return Status::OK();
}

Status LibfabricChannel::postRecv(void *buf, size_t len, fi_addr_t src) {
    ssize_t ret;
    while (true) {
        ret = fi_recv(ep_, buf, len, nullptr, src, nullptr);
        if (ret == 0) break;
        if (ret == -FI_EAGAIN) {
            drainCQ();
            continue;
        }
        return {Code::ExecutionError,
                "fi_recv failed: " + std::string(fi_strerror(static_cast<int>(-ret)))};
    }
    // Wait for recv completion
    while (!drainCQ()) {
        // spin-poll
    }
    return Status::OK();
}

bool LibfabricChannel::drainCQ() {
    struct fi_cq_data_entry entry;
    ssize_t ret = fi_cq_read(cq_, &entry, 1);
    if (ret > 0) {
        return true;
    }
    if (ret != -FI_EAGAIN) {
        struct fi_cq_err_entry err_entry;
        fi_cq_readerr(cq_, &err_entry, 0);
        LOG(ERROR) << "CQ error: " << fi_cq_strerror(cq_, err_entry.prov_errno,
                                                       err_entry.err_data, nullptr, 0);
    }
    return false;
}

void LibfabricChannel::sendHeader(int target, PendingSend *ps) {
    auto status = postSend(ps->headerBuf, CYLON_CHANNEL_HEADER_SIZE * sizeof(int),
                           ps->target_addr);
    if (!status.is_ok()) {
        LOG(ERROR) << "Failed to send header to " << target << ": " << status.get_msg();
    }
}

void LibfabricChannel::sendFinishHeader(int target, PendingSend *ps) {
    int finHeader[CYLON_CHANNEL_HEADER_SIZE] = {};
    finHeader[0] = CYLON_MSG_FIN;
    auto status = postSend(finHeader, CYLON_CHANNEL_HEADER_SIZE * sizeof(int),
                           ps->target_addr);
    if (!status.is_ok()) {
        LOG(ERROR) << "Failed to send fin header to " << target << ": " << status.get_msg();
    }
}

void LibfabricChannel::progressSends() {
    for (auto &pair : sends_) {
        int target = pair.first;
        PendingSend *ps = pair.second;

        switch (ps->status) {
            case LF_SEND_INIT: {
                // Check if we have data to send or a finish request
                if (!ps->pendingData.empty()) {
                    ps->currentSend = ps->pendingData.front();
                    ps->pendingData.pop();

                    // Build header: [not_fin, length, ...]
                    ps->headerBuf[0] = CYLON_MSG_NOT_FIN;
                    ps->headerBuf[1] = ps->currentSend->length;

                    sendHeader(target, ps);
                    ps->status = LF_SEND_LENGTH_POSTED;
                } else if (finishRequests_.find(target) != finishRequests_.end()) {
                    sendFinishHeader(target, ps);
                    ps->status = LF_SEND_FINISH;
                }
                break;
            }
            case LF_SEND_LENGTH_POSTED: {
                // Header sent, now send data
                auto status = postSend(ps->currentSend->buffer,
                                       ps->currentSend->length,
                                       ps->target_addr);
                if (status.is_ok()) {
                    ps->status = LF_SEND_POSTED;
                } else {
                    LOG(ERROR) << "Data send failed to " << target;
                }
                break;
            }
            case LF_SEND_POSTED: {
                // Data sent, notify completion
                send_comp_fn_->sendComplete(ps->currentSend);
                ps->currentSend = nullptr;
                ps->status = LF_SEND_INIT;
                break;
            }
            case LF_SEND_FINISH: {
                // Finish header sent
                auto it = finishRequests_.find(target);
                if (it != finishRequests_.end()) {
                    send_comp_fn_->sendFinishComplete(it->second);
                    finishRequests_.erase(it);
                }
                ps->status = LF_SEND_DONE;
                break;
            }
            case LF_SEND_DONE:
                break;
        }
    }
}

void LibfabricChannel::progressReceives() {
    for (auto &pair : pendingReceives_) {
        PendingReceive *pr = pair.second;

        switch (pr->status) {
            case LF_RECEIVE_INIT: {
                // Post receive for header
                auto status = postRecv(pr->headerBuf,
                                       CYLON_CHANNEL_HEADER_SIZE * sizeof(int),
                                       pr->source_addr);
                if (status.is_ok()) {
                    pr->status = LF_RECEIVE_LENGTH_POSTED;
                }
                break;
            }
            case LF_RECEIVE_LENGTH_POSTED: {
                // Header received, check if finish
                int finished = pr->headerBuf[0];
                if (finished == CYLON_MSG_FIN) {
                    rcv_fn_->receivedHeader(pr->receiveId, 1, pr->headerBuf,
                                            CYLON_CHANNEL_HEADER_SIZE);
                    pr->status = LF_RECEIVED_FIN;
                } else {
                    // Get data length from header
                    pr->length = pr->headerBuf[1];

                    // Allocate buffer
                    std::shared_ptr<Buffer> buf;
                    allocator_->Allocate(pr->length, &buf);
                    pr->data = buf;

                    // Post receive for data
                    auto status = postRecv(
                        const_cast<uint8_t *>(pr->data->GetByteBuffer()),
                        pr->length, pr->source_addr);
                    if (status.is_ok()) {
                        pr->status = LF_RECEIVE_POSTED;
                    }
                }
                break;
            }
            case LF_RECEIVE_POSTED: {
                // Data received
                rcv_fn_->receivedData(pr->receiveId, pr->data, pr->length);
                pr->data = nullptr;
                pr->status = LF_RECEIVE_INIT;
                break;
            }
            case LF_RECEIVED_FIN:
                break;
        }
    }
}

void LibfabricChannel::close() {
    for (auto &pair : sends_) {
        delete pair.second;
    }
    sends_.clear();

    for (auto &pair : pendingReceives_) {
        delete pair.second;
    }
    pendingReceives_.clear();
    finishRequests_.clear();
}

} // namespace libfabric
} // namespace cylon