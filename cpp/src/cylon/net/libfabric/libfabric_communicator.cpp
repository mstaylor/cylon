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

#include "libfabric_communicator.hpp"
#include "libfabric_channel.hpp"
#include "libfabric_operations.hpp"
#include "libfabric_redis_oob.hpp"

#include <cylon/util/macros.hpp>

#include <rdma/fi_collective.h>

#include <glog/logging.h>
#include <cstring>
#include <algorithm>

namespace cylon {
namespace net {

// ---------------------------------------------------------------------------
// LibfabricConfig
// ---------------------------------------------------------------------------

LibfabricConfig::LibfabricConfig(int world_size,
                                 std::string redis_host, int redis_port,
                                 std::string session_id,
                                 int key_ttl,
                                 std::string provider)
    : world_size_(world_size),
      redis_host_(std::move(redis_host)),
      redis_port_(redis_port),
      session_id_(std::move(session_id)),
      key_ttl_(key_ttl),
      provider_(std::move(provider)) {}

CommType LibfabricConfig::Type() {
    return LIBFABRIC;
}

LibfabricConfig::~LibfabricConfig() = default;

std::shared_ptr<LibfabricConfig> LibfabricConfig::Make(
    int world_size,
    std::string redis_host, int redis_port,
    std::string session_id,
    int key_ttl,
    std::string provider) {
    return std::make_shared<LibfabricConfig>(world_size,
                                             std::move(redis_host), redis_port,
                                             std::move(session_id),
                                             key_ttl, std::move(provider));
}

int LibfabricConfig::getWorldSize() const { return world_size_; }
const std::string &LibfabricConfig::getRedisHost() const { return redis_host_; }
int LibfabricConfig::getRedisPort() const { return redis_port_; }
const std::string &LibfabricConfig::getSessionId() const { return session_id_; }
int LibfabricConfig::getKeyTtl() const { return key_ttl_; }
const std::string &LibfabricConfig::getProvider() const { return provider_; }

// ---------------------------------------------------------------------------
// LibfabricCommunicator
// ---------------------------------------------------------------------------

LibfabricCommunicator::LibfabricCommunicator(
    MemoryPool *pool, int32_t rank, int32_t world_size,
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
    int key_ttl)
    : Communicator(pool, rank, world_size),
      fi_info_(fi_info),
      fabric_(fabric),
      domain_(domain),
      av_(av),
      ep_(ep),
      cq_(cq),
      peer_addrs_(std::move(peer_addrs)),
      av_set_(av_set),
      mc_(mc),
      coll_addr_(coll_addr),
      hw_coll_supported_(hw_coll_supported),
      redis_host_(redis_host),
      redis_port_(redis_port),
      session_id_(session_id),
      key_ttl_(key_ttl) {}

LibfabricCommunicator::~LibfabricCommunicator() {
    if (!finalized) {
        Finalize();
    }
}

CommType LibfabricCommunicator::GetCommType() const {
    return LIBFABRIC;
}

std::unique_ptr<Channel> LibfabricCommunicator::CreateChannel() const {
    return std::make_unique<::cylon::libfabric::LibfabricChannel>(
        ep_, cq_, peer_addrs_, rank, world_size,
        redis_host_, redis_port_, session_id_);
}

void LibfabricCommunicator::Finalize() {
    if (finalized) return;

    // Close collective infrastructure first (depends on EP and AV)
    if (mc_) {
        fi_close(&mc_->fid);
        mc_ = nullptr;
    }
    if (av_set_) {
        fi_close(&av_set_->fid);
        av_set_ = nullptr;
    }

    if (ep_) {
        fi_close(&ep_->fid);
        ep_ = nullptr;
    }
    if (cq_) {
        fi_close(&cq_->fid);
        cq_ = nullptr;
    }
    if (av_) {
        fi_close(&av_->fid);
        av_ = nullptr;
    }
    if (domain_) {
        fi_close(&domain_->fid);
        domain_ = nullptr;
    }
    if (fabric_) {
        fi_close(&fabric_->fid);
        fabric_ = nullptr;
    }
    if (fi_info_) {
        fi_freeinfo(fi_info_);
        fi_info_ = nullptr;
    }

    // Best-effort Redis key cleanup
#ifdef BUILD_CYLON_REDIS
    try {
        auto oob = libfabric::LibfabricRedisOOB::Make(
            world_size, redis_host_, redis_port_, session_id_, key_ttl_);
        oob->Finalize();
    } catch (...) {}
#endif

    finalized = true;
    LOG(INFO) << "LibfabricCommunicator finalized (rank " << rank << ")";
}

void LibfabricCommunicator::Barrier() {
    Status status;
    if (hw_coll_supported_) {
        // Try native fi_barrier
        ssize_t ret;
        while (true) {
            ret = fi_barrier(ep_, coll_addr_, nullptr);
            if (ret == 0) break;
            if (ret == -FI_EAGAIN) {
                struct fi_cq_data_entry entry;
                fi_cq_read(cq_, &entry, 1);
                continue;
            }
            LOG(WARNING) << "fi_barrier failed (" << fi_strerror(-ret)
                         << "), falling back to software barrier";
            status = swBarrier();
            if (!status.is_ok()) {
                LOG(ERROR) << "Software barrier failed: " << status.get_msg();
            }
            return;
        }
        // Wait for barrier completion
        status = waitCompletion();
    } else {
        status = swBarrier();
    }
    if (!status.is_ok()) {
        LOG(ERROR) << "Libfabric barrier failed: " << status.get_msg();
    }
}

// --- Collective operations using LibfabricOperations ---

Status LibfabricCommunicator::AllGather(
    const std::shared_ptr<Table> &table,
    std::vector<std::shared_ptr<Table>> *out) const {
    ::cylon::libfabric::LibfabricTableAllgatherImpl impl(ep_, cq_, peer_addrs_, rank, world_size);
    return impl.Execute(table, out);
}

Status LibfabricCommunicator::Gather(
    const std::shared_ptr<Table> &table, int gather_root,
    bool gather_from_root,
    std::vector<std::shared_ptr<Table>> *out) const {
    ::cylon::libfabric::LibfabricTableGatherImpl impl(ep_, cq_, peer_addrs_, rank, world_size);
    return impl.Execute(table, gather_root, gather_from_root, out);
}

Status LibfabricCommunicator::Bcast(
    std::shared_ptr<Table> *table, int bcast_root,
    const std::shared_ptr<CylonContext> &ctx) const {
    ::cylon::libfabric::LibfabricTableBcastImpl impl(ep_, cq_, peer_addrs_, rank, world_size);
    return impl.Execute(table, bcast_root, ctx);
}

Status LibfabricCommunicator::AllReduce(
    const std::shared_ptr<Column> &values,
    net::ReduceOp reduce_op,
    std::shared_ptr<Column> *output) const {
    ::cylon::libfabric::LibfabricAllReduceImpl impl(ep_, cq_, peer_addrs_, rank, world_size);
    return impl.Execute(values, reduce_op, output, pool);
}

Status LibfabricCommunicator::AllReduce(
    const std::shared_ptr<Scalar> &value,
    net::ReduceOp reduce_op,
    std::shared_ptr<Scalar> *output) const {
    ::cylon::libfabric::LibfabricAllReduceImpl impl(ep_, cq_, peer_addrs_, rank, world_size);
    return impl.Execute(value, reduce_op, output, pool);
}

Status LibfabricCommunicator::Allgather(
    const std::shared_ptr<Column> &values,
    std::vector<std::shared_ptr<Column>> *output) const {
    ::cylon::libfabric::LibfabricAllgatherImpl impl(ep_, cq_, peer_addrs_, rank, world_size);
    return impl.Execute(values, world_size, output, pool);
}

Status LibfabricCommunicator::Allgather(
    const std::shared_ptr<Scalar> &value,
    std::shared_ptr<Column> *output) const {
    ::cylon::libfabric::LibfabricAllgatherImpl impl(ep_, cq_, peer_addrs_, rank, world_size);
    return impl.Execute(value, world_size, output, pool);
}

// --- Internal helpers ---

Status LibfabricCommunicator::waitCompletion() const {
    struct fi_cq_data_entry entry;
    ssize_t ret;
    while (true) {
        ret = fi_cq_read(cq_, &entry, 1);
        if (ret > 0) {
            return Status::OK();
        }
        if (ret != -FI_EAGAIN) {
            struct fi_cq_err_entry err_entry;
            fi_cq_readerr(cq_, &err_entry, 0);
            return {Code::ExecutionError,
                    "CQ error: " + std::string(fi_cq_strerror(cq_, err_entry.prov_errno,
                                                               err_entry.err_data, nullptr, 0))};
        }
        // Spin-poll (no yield — single-threaded per CLAUDE.md)
    }
}

Status LibfabricCommunicator::sendBytes(const void *buf, size_t len, int target) const {
    ssize_t ret;
    while (true) {
        ret = fi_send(ep_, buf, len, nullptr, peer_addrs_[target], nullptr);
        if (ret == 0) break;
        if (ret == -FI_EAGAIN) {
            // Drain CQ to make progress
            struct fi_cq_data_entry entry;
            fi_cq_read(cq_, &entry, 1);
            continue;
        }
        return {Code::ExecutionError,
                "fi_send failed: " + std::string(fi_strerror(-ret))};
    }
    return waitCompletion();
}

Status LibfabricCommunicator::recvBytes(void *buf, size_t len, int source) const {
    ssize_t ret;
    while (true) {
        ret = fi_recv(ep_, buf, len, nullptr, peer_addrs_[source], nullptr);
        if (ret == 0) break;
        if (ret == -FI_EAGAIN) {
            struct fi_cq_data_entry entry;
            fi_cq_read(cq_, &entry, 1);
            continue;
        }
        return {Code::ExecutionError,
                "fi_recv failed: " + std::string(fi_strerror(-ret))};
    }
    return waitCompletion();
}

Status LibfabricCommunicator::swBarrier() const {
    // Binomial tree barrier (software fallback)
    uint8_t dummy = 0;
    int mask = 1;
    while (mask < world_size) {
        int partner = rank ^ mask;
        if (partner < world_size) {
            if (rank < partner) {
                RETURN_CYLON_STATUS_IF_FAILED(sendBytes(&dummy, 1, partner));
                RETURN_CYLON_STATUS_IF_FAILED(recvBytes(&dummy, 1, partner));
            } else {
                RETURN_CYLON_STATUS_IF_FAILED(recvBytes(&dummy, 1, partner));
                RETURN_CYLON_STATUS_IF_FAILED(sendBytes(&dummy, 1, partner));
            }
        }
        mask <<= 1;
    }
    return Status::OK();
}

Status LibfabricCommunicator::swAllgatherBytes(const void *send_buf, void *recv_buf,
                                               size_t count) const {
    auto *recv = static_cast<uint8_t *>(recv_buf);
    // Copy own data
    std::memcpy(recv + rank * count, send_buf, count);

    // Ring allgather
    int send_rank = rank;
    int recv_rank = (rank - 1 + world_size) % world_size;
    for (int step = 0; step < world_size - 1; step++) {
        int dest = (rank + 1) % world_size;
        int src = (rank - 1 + world_size) % world_size;

        RETURN_CYLON_STATUS_IF_FAILED(
            sendBytes(recv + send_rank * count, count, dest));
        RETURN_CYLON_STATUS_IF_FAILED(
            recvBytes(recv + recv_rank * count, count, src));

        send_rank = recv_rank;
        recv_rank = (recv_rank - 1 + world_size) % world_size;
    }
    return Status::OK();
}

// --- Static factory ---

Status LibfabricCommunicator::Make(const std::shared_ptr<CommConfig> &config,
                                   MemoryPool *pool,
                                   std::shared_ptr<Communicator> *out) {
#ifndef BUILD_CYLON_REDIS
    return {Code::NotImplemented,
            "Libfabric communicator requires Redis (BUILD_CYLON_REDIS). "
            "Rebuild with -DCYLON_USE_REDIS=1"};
#else
    auto lf_config = std::static_pointer_cast<LibfabricConfig>(config);

    // 1. Create Redis OOB and get rank
    auto oob = libfabric::LibfabricRedisOOB::Make(
        lf_config->getWorldSize(),
        lf_config->getRedisHost(),
        lf_config->getRedisPort(),
        lf_config->getSessionId(),
        lf_config->getKeyTtl());

    int world_size, rank;
    RETURN_CYLON_STATUS_IF_FAILED(oob->getWorldSizeAndRank(world_size, rank));

    LOG(INFO) << "Libfabric: rank " << rank << " of " << world_size;

    // 2. Create libfabric hints
    struct fi_info *hints = fi_allocinfo();
    if (!hints) {
        return {Code::OutOfMemory, "Failed to allocate fi_info hints"};
    }

    hints->caps = FI_MSG | FI_COLLECTIVE;
    hints->mode = FI_CONTEXT;
    if (hints->ep_attr) {
        hints->ep_attr->type = FI_EP_RDM;
    }
    if (hints->domain_attr) {
        hints->domain_attr->av_type = FI_AV_TABLE;
    }

    // Set provider if specified
    if (!lf_config->getProvider().empty()) {
        hints->fabric_attr->prov_name = strdup(lf_config->getProvider().c_str());
    }

    // 3. Get provider info
    struct fi_info *fi_info = nullptr;
    int ret = fi_getinfo(FI_VERSION(1, 9), nullptr, nullptr, 0, hints, &fi_info);
    fi_freeinfo(hints);

    if (ret != 0 || !fi_info) {
        return {Code::ExecutionError,
                "fi_getinfo failed: " + std::string(fi_strerror(-ret))};
    }

    const char *prov_name = fi_info->fabric_attr && fi_info->fabric_attr->prov_name
                            ? fi_info->fabric_attr->prov_name : "unknown";
    LOG(INFO) << "Libfabric provider: " << prov_name;

    // 4. Create fabric
    struct fid_fabric *fabric = nullptr;
    ret = fi_fabric(fi_info->fabric_attr, &fabric, nullptr);
    if (ret != 0) {
        fi_freeinfo(fi_info);
        return {Code::ExecutionError,
                "fi_fabric failed: " + std::string(fi_strerror(-ret))};
    }

    // 5. Create domain
    struct fid_domain *domain = nullptr;
    ret = fi_domain(fabric, fi_info, &domain, nullptr);
    if (ret != 0) {
        fi_close(&fabric->fid);
        fi_freeinfo(fi_info);
        return {Code::ExecutionError,
                "fi_domain failed: " + std::string(fi_strerror(-ret))};
    }

    // 6. Create completion queue
    struct fi_cq_attr cq_attr = {};
    cq_attr.size = 128;
    cq_attr.format = FI_CQ_FORMAT_DATA;

    struct fid_cq *cq = nullptr;
    ret = fi_cq_open(domain, &cq_attr, &cq, nullptr);
    if (ret != 0) {
        fi_close(&domain->fid);
        fi_close(&fabric->fid);
        fi_freeinfo(fi_info);
        return {Code::ExecutionError,
                "fi_cq_open failed: " + std::string(fi_strerror(-ret))};
    }

    // 7. Create address vector
    struct fi_av_attr av_attr = {};
    av_attr.type = FI_AV_TABLE;
    av_attr.count = world_size;

    struct fid_av *av = nullptr;
    ret = fi_av_open(domain, &av_attr, &av, nullptr);
    if (ret != 0) {
        fi_close(&cq->fid);
        fi_close(&domain->fid);
        fi_close(&fabric->fid);
        fi_freeinfo(fi_info);
        return {Code::ExecutionError,
                "fi_av_open failed: " + std::string(fi_strerror(-ret))};
    }

    // 8. Create endpoint
    struct fid_ep *ep = nullptr;
    ret = fi_endpoint(domain, fi_info, &ep, nullptr);
    if (ret != 0) {
        fi_close(&av->fid);
        fi_close(&cq->fid);
        fi_close(&domain->fid);
        fi_close(&fabric->fid);
        fi_freeinfo(fi_info);
        return {Code::ExecutionError,
                "fi_endpoint failed: " + std::string(fi_strerror(-ret))};
    }

    // Bind CQ to EP
    ret = fi_ep_bind(ep, &cq->fid, FI_TRANSMIT | FI_RECV);
    if (ret != 0) {
        fi_close(&ep->fid);
        fi_close(&av->fid);
        fi_close(&cq->fid);
        fi_close(&domain->fid);
        fi_close(&fabric->fid);
        fi_freeinfo(fi_info);
        return {Code::ExecutionError,
                "fi_ep_bind (cq) failed: " + std::string(fi_strerror(-ret))};
    }

    // Bind AV to EP
    ret = fi_ep_bind(ep, &av->fid, 0);
    if (ret != 0) {
        fi_close(&ep->fid);
        fi_close(&av->fid);
        fi_close(&cq->fid);
        fi_close(&domain->fid);
        fi_close(&fabric->fid);
        fi_freeinfo(fi_info);
        return {Code::ExecutionError,
                "fi_ep_bind (av) failed: " + std::string(fi_strerror(-ret))};
    }

    // Enable endpoint
    ret = fi_enable(ep);
    if (ret != 0) {
        fi_close(&ep->fid);
        fi_close(&av->fid);
        fi_close(&cq->fid);
        fi_close(&domain->fid);
        fi_close(&fabric->fid);
        fi_freeinfo(fi_info);
        return {Code::ExecutionError,
                "fi_enable failed: " + std::string(fi_strerror(-ret))};
    }

    // 9. Get local address
    size_t addrlen = 0;
    fi_getname(&ep->fid, nullptr, &addrlen);

    std::vector<uint8_t> local_addr(addrlen);
    ret = fi_getname(&ep->fid, local_addr.data(), &addrlen);
    if (ret != 0) {
        fi_close(&ep->fid);
        fi_close(&av->fid);
        fi_close(&cq->fid);
        fi_close(&domain->fid);
        fi_close(&fabric->fid);
        fi_freeinfo(fi_info);
        return {Code::ExecutionError,
                "fi_getname failed: " + std::string(fi_strerror(-ret))};
    }

    LOG(INFO) << "Libfabric local address size: " << addrlen << " bytes";

    // 10. OOB allgather addresses
    std::vector<uint8_t> all_addrs(addrlen * world_size);
    RETURN_CYLON_STATUS_IF_FAILED(
        oob->OOBAllgather(local_addr.data(), all_addrs.data(), addrlen,
                          addrlen * world_size));

    // 11. Insert peer addresses into AV
    std::vector<fi_addr_t> peer_addrs(world_size);
    ret = fi_av_insert(av, all_addrs.data(), world_size, peer_addrs.data(), 0, nullptr);
    if (ret != world_size) {
        fi_close(&ep->fid);
        fi_close(&av->fid);
        fi_close(&cq->fid);
        fi_close(&domain->fid);
        fi_close(&fabric->fid);
        fi_freeinfo(fi_info);
        return {Code::ExecutionError,
                "fi_av_insert returned " + std::to_string(ret)
                + ", expected " + std::to_string(world_size)};
    }

    LOG(INFO) << "Inserted " << world_size << " addresses into AV";

    // 12. OOB barrier before proceeding
    RETURN_CYLON_STATUS_IF_FAILED(oob->Barrier("init"));

    // 13. Try to create AV set + join collective for hardware collectives
    struct fid_av_set *av_set = nullptr;
    struct fid_mc *mc = nullptr;
    fi_addr_t coll_addr = FI_ADDR_UNSPEC;
    bool hw_coll_supported = false;

    {
        struct fi_av_set_attr av_set_attr = {};
        av_set_attr.count = world_size;
        av_set_attr.start_addr = peer_addrs[0];
        av_set_attr.end_addr = peer_addrs[world_size - 1];
        av_set_attr.stride = 1;

        ret = fi_av_set(av, &av_set_attr, &av_set, nullptr);
        if (ret == 0 && av_set) {
            // Insert all peer addresses into the AV set
            bool av_set_ok = true;
            for (int i = 0; i < world_size; i++) {
                ret = fi_av_set_insert(av_set, peer_addrs[i]);
                if (ret != 0) {
                    LOG(WARNING) << "fi_av_set_insert failed for rank " << i
                                 << ": " << fi_strerror(-ret);
                    av_set_ok = false;
                    break;
                }
            }

            if (av_set_ok) {
                // Get collective address from AV set
                ret = fi_av_set_addr(av_set, &coll_addr);
                if (ret != 0) {
                    LOG(WARNING) << "fi_av_set_addr failed: " << fi_strerror(-ret);
                    coll_addr = FI_ADDR_UNSPEC;
                }
            }

            if (coll_addr != FI_ADDR_UNSPEC) {
                // Try to join collective group
                ret = fi_join_collective(ep, coll_addr, av_set, 0, &mc, nullptr);
                if (ret == 0) {
                    hw_coll_supported = true;
                    LOG(INFO) << "Hardware collectives enabled for " << world_size << " workers";
                } else {
                    LOG(WARNING) << "fi_join_collective failed (" << fi_strerror(-ret)
                                 << "), using software collectives";
                    fi_close(&av_set->fid);
                    av_set = nullptr;
                    coll_addr = FI_ADDR_UNSPEC;
                }
            } else {
                // AV set creation or address lookup failed
                if (av_set) {
                    fi_close(&av_set->fid);
                    av_set = nullptr;
                }
            }
        } else {
            LOG(WARNING) << "fi_av_set not supported (" << fi_strerror(-ret)
                         << "), using software collectives";
            av_set = nullptr;
        }
    }

    // Create communicator
    *out = std::make_shared<LibfabricCommunicator>(
        pool, rank, world_size,
        fi_info, fabric, domain, av, ep, cq,
        std::move(peer_addrs),
        av_set, mc, coll_addr, hw_coll_supported,
        lf_config->getRedisHost(),
        lf_config->getRedisPort(),
        lf_config->getSessionId(),
        lf_config->getKeyTtl());

    LOG(INFO) << "LibfabricCommunicator initialized (rank " << rank << "/" << world_size
              << ", hw_coll=" << hw_coll_supported << ")";
    return Status::OK();
#endif
}

} // namespace net
} // namespace cylon