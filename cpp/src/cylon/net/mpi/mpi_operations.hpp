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

#ifndef CYLON_CPP_SRC_CYLON_NET_MPI_MPI_OPERATIONS_HPP_
#define CYLON_CPP_SRC_CYLON_NET_MPI_MPI_OPERATIONS_HPP_

#include <memory>
#include <arrow/buffer.h>

#include <cylon/net/serialize.hpp>
#include <cylon/ctx/cylon_context.hpp>
#include <cylon/net/comm_operations.hpp>
#include <cylon/net/ops/base_ops.hpp>
#include <cylon/net/mpi/mpi_type_traits.hpp>
#include <mpi.h>

namespace cylon {
namespace mpi {

MPI_Comm GetMpiComm(const std::shared_ptr<CylonContext> &ctx);

MPI_Op GetMPIOp(cylon::net::ReduceOp reduce_op);

MPI_Datatype GetMPIDataType(const std::shared_ptr<DataType> &data_type);

class MpiAllReduceImpl : public net::AllReduceImpl {
 public:
  explicit MpiAllReduceImpl(const MPI_Comm &comm);

  Status AllReduceBuffer(const void *send_buf,
                         void *rcv_buf,
                         int count,
                         const std::shared_ptr<DataType> &data_type,
                         net::ReduceOp reduce_op) const override;

 private:
  MPI_Comm comm_;
};

cylon::Status AllReduce(const std::shared_ptr<CylonContext> &ctx,
                        const void *send_buf,
                        void *rcv_buf,
                        int count,
                        const std::shared_ptr<DataType> &data_type,
                        cylon::net::ReduceOp reduce_op);

/**
 * check whether the invoking worker is the root
 * @param root
 * @param ctx
 * @return
 */
inline bool AmIRoot(int root, const std::shared_ptr<cylon::CylonContext> &ctx) {
  return root == ctx->GetRank();
}

class MpiTableGatherImpl : public net::TableGatherImpl {
 public:
  explicit MpiTableGatherImpl(MPI_Comm comm) : TableGatherImpl(),
                                               comm_(comm),
                                               requests_({}),
                                               statuses_({}) {}
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
  MPI_Comm comm_;
  std::vector<MPI_Request> requests_;
  std::vector<MPI_Status> statuses_;
};

/**
 * Perform MPI Gather on a table
 * @param serializer TableSerializer to serialize a table
 * @param gather_root MPI rank of the gather root worker
 * @param gather_from_root whether the table will be gathered from the root also
 * @param allocator Allocator to allocate the buffer for received data
 * @param all_buffer_sizes all received buffer size (significant at gather root only)
 * @param received_buffers all received buffers (significant at gather root only)
 * @param displacements displacements in each buffer (significant at gather root only)
 * @param ctx CylonContext object
 * @return
 */
cylon::Status Gather(const std::shared_ptr<cylon::TableSerializer>& serializer,
                     int gather_root,
                     bool gather_from_root,
                     const std::shared_ptr<cylon::Allocator>& allocator,
                     std::vector<int32_t> & all_buffer_sizes,
                     std::vector<std::shared_ptr<cylon::Buffer>> & received_buffers,
                     std::vector<std::vector<int32_t>> & displacements,
                     const std::shared_ptr<cylon::CylonContext> &ctx
);

/**
 * Perform gather on an Arrow Buffer
 * all received buffers are put into "buffers"
 * @param buf buffer to send
 * @param gather_root root of the gather operation
 * @param ctx CylonContext object
 * @buffers arrow buffers received at the gather root.
 * @return
 */
cylon::Status GatherArrowBuffer(const std::shared_ptr<arrow::Buffer> &buf,
                                int gather_root,
                                const std::shared_ptr<cylon::CylonContext> &ctx,
                                std::vector<std::shared_ptr<arrow::Buffer>> &buffers
);

class MpiTableAllgatherImpl : public net::TableAllgatherImpl {
 public:
  explicit MpiTableAllgatherImpl(MPI_Comm comm);

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
  MPI_Comm comm_;
  std::vector<MPI_Request> requests_;
  std::vector<MPI_Status> statuses_;
};

/**
 * Perform MPI AllGather on a distributed table
 * Assuming all workers have a table,
 * all tables will be replicated on all workers as a single table
 * @param serializer TableSerializer to serialize a table
 * @param allocator Allocator to allocate the buffer for received data
 * @param all_buffer_sizes all received buffer size
 * @param received_buffers all received buffers
 * @param displacements displacements in each buffer
 * @param ctx CylonContext object
 * @return
 */
cylon::Status AllGather(const std::shared_ptr<cylon::TableSerializer> &serializer,
                        const std::shared_ptr<cylon::Allocator> &allocator,
                        std::vector<int32_t> &all_buffer_sizes,
                        std::vector<std::shared_ptr<cylon::Buffer>> &received_buffers,
                        std::vector<std::vector<int32_t>> &displacements,
                        const std::shared_ptr<cylon::CylonContext> &ctx
);

/**
 * Perform allgather on an Arrow Buffer
 * all received buffers are put into "buffers"
 * @param buf buffer to send
 * @param ctx CylonContext object
 * @param buffers received buffers
 * @return
 */
cylon::Status AllGatherArrowBuffer(const std::shared_ptr<arrow::Buffer> &buf,
                                   const std::shared_ptr<cylon::CylonContext> &ctx,
                                   std::vector<std::shared_ptr<arrow::Buffer>> &buffers
);

class MpiTableBcastImpl : public net::TableBcastImpl {
 public:
  explicit MpiTableBcastImpl(MPI_Comm comm);

  void Init(int32_t num_buffers) override;

  Status BcastBufferSizes(int32_t *buffer, int32_t count, int32_t bcast_root) const override;

  Status BcastBufferData(uint8_t *buf_data, int32_t send_count, int32_t bcast_root) const override;

  Status IbcastBufferData(int32_t buf_idx,
                          uint8_t *buf_data,
                          int32_t send_count,
                          int32_t bcast_root) override;

  Status WaitAll(int32_t num_buffers) override;

 private:
  MPI_Comm comm_;
  std::vector<MPI_Request> requests_;
  std::vector<MPI_Status> statuses_;
};

/**
 * Perform MPI broadcast on a table
 * @param serializer TableSerializer to serialize a table (significant at broadcast root only)
 * @param bcast_root MPI rank of the broadcaster worker, (significant at all workers)
 * @param allocator Allocator to allocate the received buffers (significant at all receiver workers)
 * @param received_buffers all received buffers (significant at all receiver workers)
 * @param data_types data types of the table column (significant at all receiver workers)
 * @param ctx CylonContext object (significant at all workers)
 * @return
 */
cylon::Status Bcast(const std::shared_ptr<cylon::TableSerializer>& serializer,
                    int bcast_root,
                    const std::shared_ptr<cylon::Allocator>& allocator,
                    std::vector<std::shared_ptr<cylon::Buffer>> &received_buffers,
                    std::vector<int32_t> &data_types,
                    const std::shared_ptr<cylon::CylonContext>& ctx
);

/**
 * Perform broadcast on an Arrow Buffer
 * @param buf buffer to broadcast if it is the root and the received buffer otherwise
 * @param bcast_root MPI rank of the broadcaster worker, (significant at all workers)
 * @param ctx CylonContext object
 * @return
 */
cylon::Status BcastArrowBuffer(std::shared_ptr<arrow::Buffer> &buf,
                               int bcast_root,
                               const std::shared_ptr<cylon::CylonContext> &ctx
);

/**
 * All gather a vector of primitives from each worker
 * Each vector has to have the same number of elements
 * @param send_data
 * @param number_of_workers
 * @param received_data
 * @return
 */
template<typename T>
cylon::Status AllGather(const std::shared_ptr<cylon::CylonContext> &ctx,
                        const std::vector<T> &send_data,
                        int number_of_workers,
                        std::vector<T> &received_data) {
  auto comm = GetMpiComm(ctx);

  received_data.resize(number_of_workers * send_data.size());
  auto dt = MPICTypeTraits<T>::MPIDataType();

  auto status = MPI_Allgather(send_data.data(),
                              send_data.size(),
                              dt,
                              received_data.data(),
                              send_data.size(),
                              dt,
                              comm);
  if (status != MPI_SUCCESS) {
    return {cylon::Code::ExecutionError, "MPI_Allgather failed!"};
  }

  return cylon::Status::OK();
}

class MpiAllgatherImpl : public net::AllGatherImpl {
 public:
  explicit MpiAllgatherImpl(MPI_Comm comm);

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
  MPI_Comm comm_;
  std::array<MPI_Request, 3> requests_;
  std::array<MPI_Status, 3> statuses_;
};

}
}
#endif //CYLON_CPP_SRC_CYLON_NET_MPI_MPI_OPERATIONS_HPP_
