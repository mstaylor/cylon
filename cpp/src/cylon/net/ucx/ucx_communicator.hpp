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

#ifndef CYLON_SRC_CYLON_COMM_UCXCOMMUNICATOR_H_
#define CYLON_SRC_CYLON_COMM_UCXCOMMUNICATOR_H_

#include <cylon/net/comm_config.hpp>
#include <cylon/net/communicator.hpp>
#include <cylon/net/ucx/ucx_operations.hpp>
#include <cylon/net/ucx/ucx_ucc_oob_contexts.hpp>

#include "cylon/util/macros.hpp"
#include "sw/redis++/redis++.h"

#ifdef BUILD_CYLON_UCC
#include <ucc/api/ucc.h>
#endif

namespace cylon {
namespace net {
class UCXConfig : public CommConfig {
  CommType Type() override;

 public:
  explicit UCXConfig(std::shared_ptr<UCXOOBContext> oobContext);

  static std::shared_ptr<UCXConfig> Make(
      std::shared_ptr<UCXOOBContext> oobContext);

  void setOOBContext(std::shared_ptr<UCXOOBContext> oobContext);

  std::shared_ptr<UCXOOBContext> getOOBContext();

 private:
  std::shared_ptr<UCXOOBContext> oobContext;
};

class UCCConfig : public CommConfig {
  CommType Type() override;

 public:
  explicit UCCConfig(std::shared_ptr<UCCOOBContext> oobContext);
  static std::shared_ptr<UCCConfig> Make(
      std::shared_ptr<UCCOOBContext> &oobContext);
  void setOOBContext(std::shared_ptr<UCCOOBContext> oobContext);
  std::shared_ptr<UCCOOBContext> getOOBContext();

 private:
  std::shared_ptr<UCCOOBContext> oobContext;
};

class UCXCommunicator : public Communicator {
 public:
  explicit UCXCommunicator(MemoryPool *pool);

  ~UCXCommunicator() override = default;

  std::unique_ptr<Channel> CreateChannel() const override;
  int GetRank() const override;
  int GetWorldSize() const override;
  void Finalize() override;
  void Barrier() override;
  CommType GetCommType() const override;

  Status AllGather(const std::shared_ptr<Table> &table,
                   std::vector<std::shared_ptr<Table>> *out) const override;
  Status Gather(const std::shared_ptr<Table> &table, int gather_root,
                bool gather_from_root,
                std::vector<std::shared_ptr<Table>> *out) const override;
  Status Bcast(std::shared_ptr<Table> *table, int bcast_root,
               const std::shared_ptr<CylonContext> &ctx) const override;
  Status AllReduce(const std::shared_ptr<Column> &column,
                   net::ReduceOp reduce_op,
                   std::shared_ptr<Column> *output) const override;
  Status AllReduce(const std::shared_ptr<Scalar> &values,
                   net::ReduceOp reduce_op,
                   std::shared_ptr<Scalar> *output) const override;
  Status Allgather(const std::shared_ptr<Column> &values,
                   std::vector<std::shared_ptr<Column>> *output) const override;
  Status Allgather(const std::shared_ptr<Scalar> &value,
                   std::shared_ptr<Column> *output) const override;

  static Status Make(const std::shared_ptr<CommConfig> &config,
                     MemoryPool *pool, std::shared_ptr<Communicator> *out);

  static Status MakeWithMPI(const std::shared_ptr<CommConfig> &config,
                            MemoryPool *pool,
                            std::shared_ptr<Communicator> *out);

  static Status MakeWithRedis(const std::shared_ptr<CommConfig> &config,
                              MemoryPool *pool,
                              std::shared_ptr<Communicator> *out);

  // # UCX specific attributes - These need to be passed to the channels created
  // from the communicator The worker for receiving
  ucp_worker_h ucpRecvWorker{};
  // The worker for sending
  ucp_worker_h ucpSendWorker{};
  // Endpoint Map
  std::unordered_map<int, ucp_ep_h> endPointMap;
  // UCP Context - Holds a UCP communication instance's global information.
  ucp_context_h ucpContext{};

  std::shared_ptr<UCXOOBContext> oobContext;
};

#ifdef BUILD_CYLON_UCC
class UCXUCCCommunicator : public Communicator {
 public:
  explicit UCXUCCCommunicator(std::shared_ptr<Communicator> ucx_comm,
                              std::shared_ptr<UCCOOBContext> &oobContext);

  static Status Make(const std::shared_ptr<CommConfig> &config,
                     MemoryPool *pool, std::shared_ptr<Communicator> *out);

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
  Status Allgather(const std::shared_ptr<Column> &values,
                   std::vector<std::shared_ptr<Column>> *output) const override;
  Status AllReduce(const std::shared_ptr<Scalar> &value,
                   net::ReduceOp reduce_op,
                   std::shared_ptr<Scalar> *output) const override;
  Status Allgather(const std::shared_ptr<Scalar> &value,
                   std::shared_ptr<Column> *output) const override;

  ucc_team_h uccTeam{};
  ucc_context_h uccContext{};
  std::shared_ptr<Communicator> ucx_comm_;
  std::shared_ptr<UCCOOBContext> oobContext;
};
#endif
}  // namespace net
}  // namespace cylon
#endif  // CYLON_SRC_CYLON_COMM_UCXCOMMUNICATOR_H_
