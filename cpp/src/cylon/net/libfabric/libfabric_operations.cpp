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

#include "libfabric_operations.hpp"
#include <cylon/util/macros.hpp>

#include <rdma/fi_collective.h>
#include <rdma/fi_cm.h>

#include <glog/logging.h>
#include <cstring>
#include <algorithm>

namespace cylon {
namespace libfabric {

// ---------------------------------------------------------------------------
// CQ helper: blocking wait for one completion
// ---------------------------------------------------------------------------

static Status waitCQ(struct fid_cq *cq) {
    struct fi_cq_data_entry entry;
    while (true) {
        ssize_t ret = fi_cq_read(cq, &entry, 1);
        if (ret > 0) return Status::OK();
        if (ret != -FI_EAGAIN) {
            struct fi_cq_err_entry err;
            fi_cq_readerr(cq, &err, 0);
            return {Code::ExecutionError,
                    "CQ error: " + std::string(fi_cq_strerror(cq, err.prov_errno,
                                                               err.err_data, nullptr, 0))};
        }
    }
}

// ---------------------------------------------------------------------------
// fi_op / fi_datatype mapping
// ---------------------------------------------------------------------------

static enum fi_op to_fi_op(net::ReduceOp op) {
    switch (op) {
        case net::SUM:  return FI_SUM;
        case net::MIN:  return FI_MIN;
        case net::MAX:  return FI_MAX;
        case net::PROD: return FI_PROD;
        case net::LAND: return FI_LAND;
        case net::LOR:  return FI_LOR;
        case net::BAND: return FI_BAND;
        case net::BOR:  return FI_BOR;
        default:        return FI_SUM;
    }
}

static enum fi_datatype to_fi_datatype(const std::shared_ptr<DataType> &dt) {
    switch (dt->getType()) {
        case Type::UINT8:      return FI_UINT8;
        case Type::INT8:       return FI_INT8;
        case Type::UINT16:     return FI_UINT16;
        case Type::INT16:      return FI_INT16;
        case Type::UINT32:     return FI_UINT32;
        case Type::INT32:      return FI_INT32;
        case Type::UINT64:     return FI_UINT64;
        case Type::INT64:      return FI_INT64;
        case Type::FLOAT:      return FI_FLOAT;
        case Type::DOUBLE:     return FI_DOUBLE;
        case Type::DATE32:
        case Type::TIME32:     return FI_UINT32;
        case Type::DATE64:
        case Type::TIMESTAMP:
        case Type::TIME64:     return FI_UINT64;
        default:               return FI_UINT8;
    }
}

static size_t datatype_size(const std::shared_ptr<DataType> &dt) {
    switch (dt->getType()) {
        case Type::UINT8:
        case Type::INT8:       return 1;
        case Type::UINT16:
        case Type::INT16:      return 2;
        case Type::UINT32:
        case Type::INT32:
        case Type::FLOAT:
        case Type::DATE32:
        case Type::TIME32:     return 4;
        case Type::UINT64:
        case Type::INT64:
        case Type::DOUBLE:
        case Type::DATE64:
        case Type::TIMESTAMP:
        case Type::TIME64:     return 8;
        default:               return 1;
    }
}

// ---------------------------------------------------------------------------
// Point-to-point helpers (blocking send/recv + CQ wait)
// ---------------------------------------------------------------------------

Status lf_send(struct fid_ep *ep, struct fid_cq *cq,
               const void *buf, size_t len, fi_addr_t dest) {
    ssize_t ret;
    while (true) {
        ret = fi_send(ep, buf, len, nullptr, dest, nullptr);
        if (ret == 0) break;
        if (ret == -FI_EAGAIN) {
            struct fi_cq_data_entry entry;
            fi_cq_read(cq, &entry, 1);
            continue;
        }
        return {Code::ExecutionError,
                "fi_send failed: " + std::string(fi_strerror(-ret))};
    }
    return waitCQ(cq);
}

Status lf_recv(struct fid_ep *ep, struct fid_cq *cq,
               void *buf, size_t len, fi_addr_t src) {
    ssize_t ret;
    while (true) {
        ret = fi_recv(ep, buf, len, nullptr, src, nullptr);
        if (ret == 0) break;
        if (ret == -FI_EAGAIN) {
            struct fi_cq_data_entry entry;
            fi_cq_read(cq, &entry, 1);
            continue;
        }
        return {Code::ExecutionError,
                "fi_recv failed: " + std::string(fi_strerror(-ret))};
    }
    return waitCQ(cq);
}

// ---------------------------------------------------------------------------
// Software collective algorithms (point-to-point fallback)
// ---------------------------------------------------------------------------

/// Ring allgather
Status lf_allgather(struct fid_ep *ep, struct fid_cq *cq,
                    const std::vector<fi_addr_t> &addrs,
                    int rank, int world_size,
                    const void *send_buf, void *recv_buf, size_t count) {
    auto *recv = static_cast<uint8_t *>(recv_buf);
    std::memcpy(recv + rank * count, send_buf, count);

    int send_rank = rank;
    int recv_rank = (rank - 1 + world_size) % world_size;
    for (int step = 0; step < world_size - 1; step++) {
        int dest = (rank + 1) % world_size;
        int src = (rank - 1 + world_size) % world_size;

        RETURN_CYLON_STATUS_IF_FAILED(
            lf_send(ep, cq, recv + send_rank * count, count, addrs[dest]));
        RETURN_CYLON_STATUS_IF_FAILED(
            lf_recv(ep, cq, recv + recv_rank * count, count, addrs[src]));

        send_rank = recv_rank;
        recv_rank = (recv_rank - 1 + world_size) % world_size;
    }
    return Status::OK();
}

/// Variable-length allgather
Status lf_allgatherv(struct fid_ep *ep, struct fid_cq *cq,
                     const std::vector<fi_addr_t> &addrs,
                     int rank, int world_size,
                     const void *send_buf, size_t send_count,
                     void *recv_buf,
                     const std::vector<int32_t> &recv_counts,
                     const std::vector<int32_t> &displacements) {
    auto *recv = static_cast<uint8_t *>(recv_buf);

    // Copy own data
    std::memcpy(recv + displacements[rank], send_buf, send_count);

    // Exchange with each peer: send own data, receive theirs
    for (int i = 1; i < world_size; i++) {
        int dest = (rank + i) % world_size;
        int src = (rank - i + world_size) % world_size;

        RETURN_CYLON_STATUS_IF_FAILED(
            lf_send(ep, cq, send_buf, send_count, addrs[dest]));
        RETURN_CYLON_STATUS_IF_FAILED(
            lf_recv(ep, cq, recv + displacements[src], recv_counts[src], addrs[src]));
    }
    return Status::OK();
}

/// Gather at root
Status lf_gather(struct fid_ep *ep, struct fid_cq *cq,
                 const std::vector<fi_addr_t> &addrs,
                 int rank, int world_size,
                 const void *send_buf, size_t send_count,
                 void *recv_buf, size_t recv_count_each,
                 int root) {
    if (rank == root) {
        auto *recv = static_cast<uint8_t *>(recv_buf);
        std::memcpy(recv + root * recv_count_each, send_buf, send_count);
        for (int i = 0; i < world_size; i++) {
            if (i != root) {
                RETURN_CYLON_STATUS_IF_FAILED(
                    lf_recv(ep, cq, recv + i * recv_count_each, recv_count_each, addrs[i]));
            }
        }
    } else {
        RETURN_CYLON_STATUS_IF_FAILED(
            lf_send(ep, cq, send_buf, send_count, addrs[root]));
    }
    return Status::OK();
}

/// Variable-length gather at root
Status lf_gatherv(struct fid_ep *ep, struct fid_cq *cq,
                  const std::vector<fi_addr_t> &addrs,
                  int rank, int world_size,
                  const void *send_buf, size_t send_count,
                  void *recv_buf,
                  const std::vector<int32_t> &recv_counts,
                  const std::vector<int32_t> &displacements,
                  int root) {
    if (rank == root) {
        auto *recv = static_cast<uint8_t *>(recv_buf);
        std::memcpy(recv + displacements[root], send_buf, send_count);
        for (int i = 0; i < world_size; i++) {
            if (i != root) {
                RETURN_CYLON_STATUS_IF_FAILED(
                    lf_recv(ep, cq, recv + displacements[i], recv_counts[i], addrs[i]));
            }
        }
    } else {
        RETURN_CYLON_STATUS_IF_FAILED(
            lf_send(ep, cq, send_buf, send_count, addrs[root]));
    }
    return Status::OK();
}

/// Broadcast from root (binomial tree)
Status lf_bcast(struct fid_ep *ep, struct fid_cq *cq,
                const std::vector<fi_addr_t> &addrs,
                int rank, int world_size,
                void *buf, size_t count, int root) {
    // Shift ranks so root is rank 0, then use standard binomial tree
    int relative_rank = (rank - root + world_size) % world_size;

    // Receive phase: find parent in binomial tree
    int mask = 1;
    while (mask < world_size) {
        if (relative_rank & mask) {
            int parent = (relative_rank & ~mask);
            int abs_parent = (parent + root) % world_size;
            RETURN_CYLON_STATUS_IF_FAILED(
                lf_recv(ep, cq, buf, count, addrs[abs_parent]));
            break;
        }
        mask <<= 1;
    }

    // Send phase: send to children in binomial tree
    mask >>= 1;
    while (mask > 0) {
        int child = relative_rank | mask;
        if (child < world_size) {
            int abs_child = (child + root) % world_size;
            RETURN_CYLON_STATUS_IF_FAILED(
                lf_send(ep, cq, buf, count, addrs[abs_child]));
        }
        mask >>= 1;
    }
    return Status::OK();
}

/// Software allreduce (reduce to root 0, then broadcast)
static Status lf_sw_allreduce(struct fid_ep *ep, struct fid_cq *cq,
                               const std::vector<fi_addr_t> &addrs,
                               int rank, int world_size,
                               const void *send_buf, void *recv_buf,
                               int count,
                               const std::shared_ptr<DataType> &data_type,
                               net::ReduceOp reduce_op) {
    size_t elem_size = datatype_size(data_type);
    size_t total_bytes = count * elem_size;

    // Copy send to recv
    std::memcpy(recv_buf, send_buf, total_bytes);

    // Reduce to rank 0 using binomial tree
    std::vector<uint8_t> tmp(total_bytes);
    int mask = 1;
    while (mask < world_size) {
        int partner = rank ^ mask;
        if (partner < world_size) {
            if (rank < partner) {
                RETURN_CYLON_STATUS_IF_FAILED(
                    lf_recv(ep, cq, tmp.data(), total_bytes, addrs[partner]));
                // Apply reduce operation element-wise
                auto *dst = static_cast<uint8_t *>(recv_buf);
                for (int i = 0; i < count; i++) {
                    switch (data_type->getType()) {
#define REDUCE_CASE(TYPE, CTYPE)                                              \
    case Type::TYPE: {                                                        \
        auto *d = reinterpret_cast<CTYPE *>(dst + i * elem_size);            \
        auto *s = reinterpret_cast<const CTYPE *>(tmp.data() + i * elem_size);\
        switch (reduce_op) {                                                  \
            case net::SUM:  *d = *d + *s; break;                             \
            case net::MIN:  *d = std::min(*d, *s); break;                    \
            case net::MAX:  *d = std::max(*d, *s); break;                    \
            case net::PROD: *d = *d * *s; break;                             \
            default: break;                                                   \
        }                                                                     \
        break;                                                                \
    }
                        REDUCE_CASE(UINT8, uint8_t)
                        REDUCE_CASE(INT8, int8_t)
                        REDUCE_CASE(UINT16, uint16_t)
                        REDUCE_CASE(INT16, int16_t)
                        REDUCE_CASE(UINT32, uint32_t)
                        REDUCE_CASE(INT32, int32_t)
                        REDUCE_CASE(UINT64, uint64_t)
                        REDUCE_CASE(INT64, int64_t)
                        REDUCE_CASE(FLOAT, float)
                        REDUCE_CASE(DOUBLE, double)
#undef REDUCE_CASE
                        default: break;
                    }
                }
            } else {
                RETURN_CYLON_STATUS_IF_FAILED(
                    lf_send(ep, cq, recv_buf, total_bytes, addrs[partner]));
            }
        }
        mask <<= 1;
    }

    // Broadcast result from rank 0
    RETURN_CYLON_STATUS_IF_FAILED(
        lf_bcast(ep, cq, addrs, rank, world_size, recv_buf, total_bytes, 0));

    return Status::OK();
}

// ---------------------------------------------------------------------------
// LibfabricTableAllgatherImpl
// ---------------------------------------------------------------------------

void LibfabricTableAllgatherImpl::Init(int num_buffers) {
    CYLON_UNUSED(num_buffers);
}

Status LibfabricTableAllgatherImpl::AllgatherBufferSizes(
    const int32_t *send_data, int num_buffers, int32_t *rcv_data) const {
    size_t bytes = num_buffers * sizeof(int32_t);
    return lf_allgather(ep_, cq_, addrs_, rank_, world_size_,
                        send_data, rcv_data, bytes);
}

Status LibfabricTableAllgatherImpl::IallgatherBufferData(
    int buf_idx, const uint8_t *send_data, int32_t send_count,
    uint8_t *recv_data, const std::vector<int32_t> &recv_count,
    const std::vector<int32_t> &displacements) {
    CYLON_UNUSED(buf_idx);
    return lf_allgatherv(ep_, cq_, addrs_, rank_, world_size_,
                         send_data, send_count,
                         recv_data, recv_count, displacements);
}

Status LibfabricTableAllgatherImpl::WaitAll(int num_buffers) {
    CYLON_UNUSED(num_buffers);
    // All operations are blocking, nothing to wait for
    return Status::OK();
}

// ---------------------------------------------------------------------------
// LibfabricTableGatherImpl
// ---------------------------------------------------------------------------

void LibfabricTableGatherImpl::Init(int num_buffers) {
    CYLON_UNUSED(num_buffers);
}

Status LibfabricTableGatherImpl::GatherBufferSizes(
    const int32_t *send_data, int num_buffers, int32_t *rcv_data,
    int gather_root) const {
    size_t bytes = num_buffers * sizeof(int32_t);
    return lf_gather(ep_, cq_, addrs_, rank_, world_size_,
                     send_data, bytes, rcv_data, bytes, gather_root);
}

Status LibfabricTableGatherImpl::IgatherBufferData(
    int buf_idx, const uint8_t *send_data, int32_t send_count,
    uint8_t *recv_data, const std::vector<int32_t> &recv_count,
    const std::vector<int32_t> &displacements,
    int gather_root) {
    CYLON_UNUSED(buf_idx);
    return lf_gatherv(ep_, cq_, addrs_, rank_, world_size_,
                      send_data, send_count,
                      recv_data, recv_count, displacements, gather_root);
}

Status LibfabricTableGatherImpl::WaitAll(int num_buffers) {
    CYLON_UNUSED(num_buffers);
    return Status::OK();
}

// ---------------------------------------------------------------------------
// LibfabricTableBcastImpl
// ---------------------------------------------------------------------------

void LibfabricTableBcastImpl::Init(int32_t num_buffers) {
    CYLON_UNUSED(num_buffers);
}

Status LibfabricTableBcastImpl::BcastBufferSizes(
    int32_t *buffer, int32_t count, int32_t bcast_root) const {
    return lf_bcast(ep_, cq_, addrs_, rank_, world_size_,
                    buffer, count * sizeof(int32_t), bcast_root);
}

Status LibfabricTableBcastImpl::BcastBufferData(
    uint8_t *buf_data, int32_t send_count, int32_t bcast_root) const {
    return lf_bcast(ep_, cq_, addrs_, rank_, world_size_,
                    buf_data, send_count, bcast_root);
}

Status LibfabricTableBcastImpl::IbcastBufferData(
    int32_t buf_idx, uint8_t *buf_data, int32_t send_count,
    int32_t bcast_root) {
    CYLON_UNUSED(buf_idx);
    return lf_bcast(ep_, cq_, addrs_, rank_, world_size_,
                    buf_data, send_count, bcast_root);
}

Status LibfabricTableBcastImpl::WaitAll(int32_t num_buffers) {
    CYLON_UNUSED(num_buffers);
    return Status::OK();
}

// ---------------------------------------------------------------------------
// LibfabricAllReduceImpl
// ---------------------------------------------------------------------------

Status LibfabricAllReduceImpl::AllReduceBuffer(
    const void *send_buf, void *rcv_buf, int count,
    const std::shared_ptr<DataType> &data_type,
    net::ReduceOp reduce_op) const {

    // Use software allreduce (point-to-point fallback)
    // Native fi_allreduce is used at the communicator level for simple
    // scalar/column allreduce when hw_coll_supported is true. For the
    // table-level operation (which goes through this impl), we always
    // use the software path since the base_ops Execute() framework
    // already handles buffer serialization.
    return lf_sw_allreduce(ep_, cq_, addrs_, rank_, world_size_,
                           send_buf, rcv_buf, count, data_type, reduce_op);
}

// ---------------------------------------------------------------------------
// LibfabricAllgatherImpl
// ---------------------------------------------------------------------------

Status LibfabricAllgatherImpl::AllgatherBufferSize(
    const int32_t *send_data, int32_t num_buffers, int32_t *rcv_data) const {
    size_t bytes = num_buffers * sizeof(int32_t);
    return lf_allgather(ep_, cq_, addrs_, rank_, world_size_,
                        send_data, rcv_data, bytes);
}

Status LibfabricAllgatherImpl::IallgatherBufferData(
    int32_t buf_idx, const uint8_t *send_data, int32_t send_count,
    uint8_t *recv_data, const std::vector<int32_t> &recv_count,
    const std::vector<int32_t> &displacements) {
    CYLON_UNUSED(buf_idx);
    return lf_allgatherv(ep_, cq_, addrs_, rank_, world_size_,
                         send_data, send_count,
                         recv_data, recv_count, displacements);
}

Status LibfabricAllgatherImpl::WaitAll() {
    return Status::OK();
}

} // namespace libfabric
} // namespace cylon