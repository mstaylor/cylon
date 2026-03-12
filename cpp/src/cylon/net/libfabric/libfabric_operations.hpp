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

#ifndef CYLON_LIBFABRIC_OPERATIONS_HPP
#define CYLON_LIBFABRIC_OPERATIONS_HPP

#include <cylon/net/comm_operations.hpp>
#include <cylon/net/ops/base_ops.hpp>
#include <cylon/status.hpp>

#include <rdma/fabric.h>
#include <rdma/fi_endpoint.h>

#include <vector>
#include <cstdint>

namespace cylon {
namespace libfabric {

/// Helper for blocking fi_send + CQ wait
Status lf_send(struct fid_ep *ep, struct fid_cq *cq,
               const void *buf, size_t len, fi_addr_t dest);

/// Helper for blocking fi_recv + CQ wait
Status lf_recv(struct fid_ep *ep, struct fid_cq *cq,
               void *buf, size_t len, fi_addr_t src);

/// Ring allgather using fi_send/fi_recv
Status lf_allgather(struct fid_ep *ep, struct fid_cq *cq,
                    const std::vector<fi_addr_t> &addrs,
                    int rank, int world_size,
                    const void *send_buf, void *recv_buf, size_t count);

/// Allgatherv using fi_send/fi_recv (variable-length allgather)
Status lf_allgatherv(struct fid_ep *ep, struct fid_cq *cq,
                     const std::vector<fi_addr_t> &addrs,
                     int rank, int world_size,
                     const void *send_buf, size_t send_count,
                     void *recv_buf,
                     const std::vector<int32_t> &recv_counts,
                     const std::vector<int32_t> &displacements);

/// Gather at root using fi_send/fi_recv
Status lf_gather(struct fid_ep *ep, struct fid_cq *cq,
                 const std::vector<fi_addr_t> &addrs,
                 int rank, int world_size,
                 const void *send_buf, size_t send_count,
                 void *recv_buf, size_t recv_count_each,
                 int root);

/// Gatherv at root (variable-length gather)
Status lf_gatherv(struct fid_ep *ep, struct fid_cq *cq,
                  const std::vector<fi_addr_t> &addrs,
                  int rank, int world_size,
                  const void *send_buf, size_t send_count,
                  void *recv_buf,
                  const std::vector<int32_t> &recv_counts,
                  const std::vector<int32_t> &displacements,
                  int root);

/// Broadcast from root using binomial tree
Status lf_bcast(struct fid_ep *ep, struct fid_cq *cq,
                const std::vector<fi_addr_t> &addrs,
                int rank, int world_size,
                void *buf, size_t count, int root);

class LibfabricTableAllgatherImpl : public net::TableAllgatherImpl {
public:
    LibfabricTableAllgatherImpl(struct fid_ep *ep, struct fid_cq *cq,
                                const std::vector<fi_addr_t> &addrs,
                                int rank, int world_size)
        : ep_(ep), cq_(cq), addrs_(addrs), rank_(rank), world_size_(world_size) {}

    void Init(int num_buffers) override;

    Status AllgatherBufferSizes(const int32_t *send_data,
                                int num_buffers,
                                int32_t *rcv_data) const override;

    Status IallgatherBufferData(int buf_idx,
                                const uint8_t *send_data,
                                int32_t send_count,
                                uint8_t *recv_data,
                                const std::vector<int32_t> &recv_count,
                                const std::vector<int32_t> &displacements) override;

    Status WaitAll(int num_buffers) override;

private:
    struct fid_ep *ep_;
    struct fid_cq *cq_;
    std::vector<fi_addr_t> addrs_;
    int rank_;
    int world_size_;
};

class LibfabricTableGatherImpl : public net::TableGatherImpl {
public:
    LibfabricTableGatherImpl(struct fid_ep *ep, struct fid_cq *cq,
                             const std::vector<fi_addr_t> &addrs,
                             int rank, int world_size)
        : ep_(ep), cq_(cq), addrs_(addrs), rank_(rank), world_size_(world_size) {}

    void Init(int num_buffers) override;

    Status GatherBufferSizes(const int32_t *send_data,
                             int num_buffers,
                             int32_t *rcv_data,
                             int gather_root) const override;

    Status IgatherBufferData(int buf_idx,
                             const uint8_t *send_data,
                             int32_t send_count,
                             uint8_t *recv_data,
                             const std::vector<int32_t> &recv_count,
                             const std::vector<int32_t> &displacements,
                             int gather_root) override;

    Status WaitAll(int num_buffers) override;

private:
    struct fid_ep *ep_;
    struct fid_cq *cq_;
    std::vector<fi_addr_t> addrs_;
    int rank_;
    int world_size_;
};

class LibfabricTableBcastImpl : public net::TableBcastImpl {
public:
    LibfabricTableBcastImpl(struct fid_ep *ep, struct fid_cq *cq,
                            const std::vector<fi_addr_t> &addrs,
                            int rank, int world_size)
        : ep_(ep), cq_(cq), addrs_(addrs), rank_(rank), world_size_(world_size) {}

    void Init(int32_t num_buffers) override;

    Status BcastBufferSizes(int32_t *buffer, int32_t count, int32_t bcast_root) const override;

    Status BcastBufferData(uint8_t *buf_data, int32_t send_count, int32_t bcast_root) const override;

    Status IbcastBufferData(int32_t buf_idx,
                            uint8_t *buf_data,
                            int32_t send_count,
                            int32_t bcast_root) override;

    Status WaitAll(int32_t num_buffers) override;

private:
    struct fid_ep *ep_;
    struct fid_cq *cq_;
    std::vector<fi_addr_t> addrs_;
    int rank_;
    int world_size_;
};

class LibfabricAllReduceImpl : public net::AllReduceImpl {
public:
    LibfabricAllReduceImpl(struct fid_ep *ep, struct fid_cq *cq,
                           const std::vector<fi_addr_t> &addrs,
                           int rank, int world_size)
        : ep_(ep), cq_(cq), addrs_(addrs), rank_(rank), world_size_(world_size) {}

    Status AllReduceBuffer(const void *send_buf, void *rcv_buf, int count,
                           const std::shared_ptr<DataType> &data_type,
                           net::ReduceOp reduce_op) const override;

private:
    struct fid_ep *ep_;
    struct fid_cq *cq_;
    std::vector<fi_addr_t> addrs_;
    int rank_;
    int world_size_;
};

class LibfabricAllgatherImpl : public net::AllGatherImpl {
public:
    LibfabricAllgatherImpl(struct fid_ep *ep, struct fid_cq *cq,
                           const std::vector<fi_addr_t> &addrs,
                           int rank, int world_size)
        : ep_(ep), cq_(cq), addrs_(addrs), rank_(rank), world_size_(world_size) {}

    Status AllgatherBufferSize(const int32_t *send_data,
                               int32_t num_buffers,
                               int32_t *rcv_data) const override;

    Status IallgatherBufferData(int32_t buf_idx,
                                const uint8_t *send_data,
                                int32_t send_count,
                                uint8_t *recv_data,
                                const std::vector<int32_t> &recv_count,
                                const std::vector<int32_t> &displacements) override;

    Status WaitAll() override;

private:
    struct fid_ep *ep_;
    struct fid_cq *cq_;
    std::vector<fi_addr_t> addrs_;
    int rank_;
    int world_size_;
};

} // namespace libfabric
} // namespace cylon

#endif // CYLON_LIBFABRIC_OPERATIONS_HPP