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

#ifndef CYLON_LIBFABRIC_COMMUNICATOR_HPP
#define CYLON_LIBFABRIC_COMMUNICATOR_HPP

#include <cylon/net/comm_config.hpp>
#include <cylon/net/communicator.hpp>

#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_tagged.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_collective.h>
#include <rdma/fi_eq.h>

#include <vector>
#include <string>
#include <memory>

namespace cylon {
namespace net {

class LibfabricCommunicator;

/// Configuration for Libfabric communicator.
///
/// Uses Redis for out-of-band address exchange, same as UCX Redis OOB.
/// Requires CYLON_SESSION_ID environment variable for key isolation.
class LibfabricConfig : public CommConfig {
public:
    LibfabricConfig(int world_size,
                    std::string redis_host, int redis_port,
                    std::string session_id,
                    int key_ttl = 3600,
                    std::string provider = "");

    CommType Type() override;

    ~LibfabricConfig() override;

    static std::shared_ptr<LibfabricConfig> Make(
        int world_size,
        std::string redis_host, int redis_port,
        std::string session_id,
        int key_ttl = 3600,
        std::string provider = "");

    int getWorldSize() const;
    const std::string &getRedisHost() const;
    int getRedisPort() const;
    const std::string &getSessionId() const;
    int getKeyTtl() const;
    const std::string &getProvider() const;

private:
    friend LibfabricCommunicator;
    int world_size_;
    std::string redis_host_;
    int redis_port_;
    std::string session_id_;
    int key_ttl_;
    std::string provider_;  // e.g. "efa", "verbs", "tcp", "" for auto
};

/// Libfabric communicator implementation.
///
/// Uses libfabric C API for high-performance fabric communication.
/// Initialization follows the Rust implementation:
/// 1. Connect to Redis, get rank via atomic increment
/// 2. fi_getinfo() → fi_fabric() → fi_domain() → create CQ, AV, EP
/// 3. Exchange addresses via Redis OOB allgather
/// 4. Insert peer addresses into AV
/// 5. Create AV set + fi_join_collective() for native collectives
///
/// Collective operations use native fi_barrier/fi_allreduce/fi_allgather/
/// fi_broadcast/fi_gather when supported by the provider. Falls back to
/// point-to-point fi_send/fi_recv with binomial tree algorithms when
/// hardware collectives are not available.
class LibfabricCommunicator : public Communicator {
public:
    LibfabricCommunicator(MemoryPool *pool, int32_t rank, int32_t world_size,
                          struct fi_info *fi_info,
                          struct fid_fabric *fabric,
                          struct fid_domain *domain,
                          struct fid_av *av,
                          struct fid_ep *ep,
                          struct fid_cq *cq,
                          std::vector<fi_addr_t> peer_addrs,
                          struct fid_av_set *av_set,
                          struct fid_mc *mc,
                          fi_addr_t coll_addr,
                          bool hw_coll_supported,
                          const std::string &redis_host,
                          int redis_port,
                          const std::string &session_id,
                          int key_ttl);

    ~LibfabricCommunicator() override;

    CommType GetCommType() const override;

    std::unique_ptr<Channel> CreateChannel() const override;

    void Finalize() override;

    void Barrier() override;

    Status AllGather(const std::shared_ptr<Table> &table,
                     std::vector<std::shared_ptr<Table>> *out) const override;

    Status Gather(const std::shared_ptr<Table> &table, int gather_root,
                  bool gather_from_root,
                  std::vector<std::shared_ptr<Table>> *out) const override;

    Status Bcast(std::shared_ptr<Table> *table, int bcast_root,
                 const std::shared_ptr<CylonContext> &ctx) const override;

    Status AllReduce(const std::shared_ptr<Column> &values,
                     net::ReduceOp reduce_op,
                     std::shared_ptr<Column> *output) const override;

    Status AllReduce(const std::shared_ptr<Scalar> &value,
                     net::ReduceOp reduce_op,
                     std::shared_ptr<Scalar> *output) const override;

    Status Allgather(const std::shared_ptr<Column> &values,
                     std::vector<std::shared_ptr<Column>> *output) const override;

    Status Allgather(const std::shared_ptr<Scalar> &value,
                     std::shared_ptr<Column> *output) const override;

    static Status Make(const std::shared_ptr<CommConfig> &config,
                       MemoryPool *pool,
                       std::shared_ptr<Communicator> *out);

    // Accessors for channel and operations
    struct fid_ep *getEndpoint() const { return ep_; }
    struct fid_cq *getCompletionQueue() const { return cq_; }
    const std::vector<fi_addr_t> &getPeerAddrs() const { return peer_addrs_; }
    fi_addr_t getCollAddr() const { return coll_addr_; }
    bool isHwCollSupported() const { return hw_coll_supported_; }

private:
    /// Wait for a completion queue entry (blocking spin-poll)
    Status waitCompletion() const;

    /// Send raw bytes to a peer (point-to-point fallback)
    Status sendBytes(const void *buf, size_t len, int target) const;

    /// Receive raw bytes from a peer (point-to-point fallback)
    Status recvBytes(void *buf, size_t len, int source) const;

    /// Software barrier using fi_send/fi_recv (binomial tree)
    Status swBarrier() const;

    /// Software allgather using fi_send/fi_recv (ring)
    Status swAllgatherBytes(const void *send_buf, void *recv_buf,
                            size_t count) const;

    struct fi_info *fi_info_ = nullptr;
    struct fid_fabric *fabric_ = nullptr;
    struct fid_domain *domain_ = nullptr;
    struct fid_av *av_ = nullptr;
    struct fid_ep *ep_ = nullptr;
    struct fid_cq *cq_ = nullptr;
    std::vector<fi_addr_t> peer_addrs_;

    // Collective infrastructure
    struct fid_av_set *av_set_ = nullptr;
    struct fid_mc *mc_ = nullptr;
    fi_addr_t coll_addr_ = FI_ADDR_UNSPEC;
    bool hw_coll_supported_ = false;

    std::string redis_host_;
    int redis_port_;
    std::string session_id_;
    int key_ttl_;
};

} // namespace net
} // namespace cylon

#endif // CYLON_LIBFABRIC_COMMUNICATOR_HPP