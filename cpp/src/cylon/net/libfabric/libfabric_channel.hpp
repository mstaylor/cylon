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

#ifndef CYLON_LIBFABRIC_CHANNEL_HPP
#define CYLON_LIBFABRIC_CHANNEL_HPP

#include <unordered_map>
#include <queue>
#include <vector>
#include <string>
#include <memory>

#include "cylon/net/channel.hpp"

#include <rdma/fabric.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_cm.h>

namespace cylon {
namespace libfabric {

enum LibfabricSendStatus {
    LF_SEND_INIT = 0,
    LF_SEND_LENGTH_POSTED = 1,
    LF_SEND_POSTED = 2,
    LF_SEND_FINISH = 3,
    LF_SEND_DONE = 4
};

enum LibfabricReceiveStatus {
    LF_RECEIVE_INIT = 0,
    LF_RECEIVE_LENGTH_POSTED = 1,
    LF_RECEIVE_POSTED = 2,
    LF_RECEIVED_FIN = 3
};

struct PendingSend {
    int headerBuf[CYLON_CHANNEL_HEADER_SIZE]{};
    std::queue<std::shared_ptr<CylonRequest>> pendingData{};
    LibfabricSendStatus status = LF_SEND_INIT;
    std::shared_ptr<CylonRequest> currentSend{};
    fi_addr_t target_addr;
};

struct PendingReceive {
    int headerBuf[CYLON_CHANNEL_HEADER_SIZE]{};
    int receiveId{};
    std::shared_ptr<Buffer> data{};
    int length{};
    LibfabricReceiveStatus status = LF_RECEIVE_INIT;
    fi_addr_t source_addr;
};

/// Libfabric-based channel for point-to-point data transfer.
///
/// Uses fi_send/fi_recv with CQ polling for progress.
/// Follows the same PendingSend/PendingReceive pattern as FMI/MPI channels.
class LibfabricChannel : public Channel {
public:
    LibfabricChannel(struct fid_ep *ep,
                     struct fid_cq *cq,
                     const std::vector<fi_addr_t> &peer_addrs,
                     int rank, int world_size,
                     const std::string &redis_host,
                     int redis_port,
                     const std::string &session_id);

    void init(int edge,
              const std::vector<int> &receives,
              const std::vector<int> &sendIds,
              ChannelReceiveCallback *rcv,
              ChannelSendCallback *send,
              Allocator *alloc) override;

    int send(std::shared_ptr<CylonRequest> request) override;

    int sendFin(std::shared_ptr<CylonRequest> request) override;

    void progressSends() override;

    void progressReceives() override;

    void close() override;

private:
    /// Post a fi_send and wait for CQ completion
    Status postSend(const void *buf, size_t len, fi_addr_t dest);

    /// Post a fi_recv and wait for CQ completion
    Status postRecv(void *buf, size_t len, fi_addr_t src);

    /// Drain one CQ entry (non-blocking)
    bool drainCQ();

    /// Send header to peer
    void sendHeader(int target, PendingSend *ps);

    /// Send finish header to peer
    void sendFinishHeader(int target, PendingSend *ps);

    std::unordered_map<int, PendingSend *> sends_;
    std::unordered_map<int, PendingReceive *> pendingReceives_;
    std::unordered_map<int, std::shared_ptr<CylonRequest>> finishRequests_;

    ChannelReceiveCallback *rcv_fn_ = nullptr;
    ChannelSendCallback *send_comp_fn_ = nullptr;
    Allocator *allocator_ = nullptr;

    struct fid_ep *ep_;
    struct fid_cq *cq_;
    std::vector<fi_addr_t> peer_addrs_;
    int rank_;
    int worldSize_;
    std::string redis_host_;
    int redis_port_;
    std::string session_id_;
};

} // namespace libfabric
} // namespace cylon

#endif // CYLON_LIBFABRIC_CHANNEL_HPP