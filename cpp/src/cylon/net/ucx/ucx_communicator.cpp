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

#include <memory>
#include <glog/logging.h>

#include "cylon/net/communicator.hpp"
#include "cylon/net/ucx/ucx_communicator.hpp"
#include "cylon/net/ucx/ucx_channel.hpp"
#include "cylon/util/macros.hpp"

#ifdef BUILD_CYLON_UCC
#include "cylon/net/ucc/ucc_operations.hpp"
#endif

namespace cylon {
namespace net {

void mpi_check_and_finalize() {
  int mpi_finalized;
  MPI_Finalized(&mpi_finalized);
  if (!mpi_finalized) {
    MPI_Finalize();
  }
}

CommType UCCConfig::Type() {
  return CommType::UCC;
}

UCCConfig::UCCConfig(std::shared_ptr<UCCOOBContext> oob_context): oobContext(oob_context) {}

std::shared_ptr<UCCConfig> UCCConfig::Make(
    std::shared_ptr<UCCOOBContext> &oob_context) {
  return std::make_shared<UCCConfig>(oob_context);
}

std::unique_ptr<Channel> UCXCommunicator::CreateChannel() const {
  return std::make_unique<UCXChannel>(this);
}

int UCXCommunicator::GetRank() const {
  return this->rank;
}
int UCXCommunicator::GetWorldSize() const {
  return this->world_size;
}

Status UCXCommunicator::AllGather(const std::shared_ptr<Table> &table,
                                  std::vector<std::shared_ptr<Table>> *out) const {
  CYLON_UNUSED(table);
  CYLON_UNUSED(out);
  return {Code::NotImplemented, "All gather not implemented for ucx"};
}

Status UCXCommunicator::Gather(const std::shared_ptr<Table> &table,
                               int gather_root,
                               bool gather_from_root,
                               std::vector<std::shared_ptr<Table>> *out) const {
  CYLON_UNUSED(table);
  CYLON_UNUSED(gather_root);
  CYLON_UNUSED(gather_from_root);
  CYLON_UNUSED(out);
  return {Code::NotImplemented, "All gather not implemented for ucx"};
}

Status UCXCommunicator::Bcast(std::shared_ptr<Table> *table, int bcast_root,
                              const std::shared_ptr<CylonContext> &ctx) const {
  CYLON_UNUSED(table);
  CYLON_UNUSED(bcast_root);
  CYLON_UNUSED(ctx);
  return {Code::NotImplemented, "Bcast not implemented for ucx"};
}

Status UCXCommunicator::AllReduce(const std::shared_ptr<Column> &column,
                                  net::ReduceOp reduce_op,
                                  std::shared_ptr<Column> *output) const {
  CYLON_UNUSED(column);
  CYLON_UNUSED(reduce_op);
  CYLON_UNUSED(output);
  return {Code::NotImplemented, "Allreduce not implemented for ucx"};
}

UCXCommunicator::UCXCommunicator(MemoryPool *pool)
    : Communicator(pool, -1, -1) {}

Status UCXCommunicator::AllReduce(const std::shared_ptr<Scalar> &values,
                                  net::ReduceOp reduce_op,
                                  std::shared_ptr<Scalar> *output) const {
  CYLON_UNUSED(values);
  CYLON_UNUSED(reduce_op);
  CYLON_UNUSED(output);
  return {Code::NotImplemented, "Allreduce not implemented for ucx"};
}

Status UCXCommunicator::Allgather(const std::shared_ptr<Column> &values,
                                  std::vector<std::shared_ptr<Column>> *output) const {
  CYLON_UNUSED(values);
  CYLON_UNUSED(output);
  return {Code::NotImplemented, "Allgather not implemented for ucx"};
}

Status UCXCommunicator::Allgather(const std::shared_ptr<Scalar> &value,
                                  std::shared_ptr<Column> *output) const {
  CYLON_UNUSED(value);
  CYLON_UNUSED(output);
  return {Code::NotImplemented, "Allgather not implemented for ucx"};
}

Status UCXCommunicator::MakeWithMPI(const std::shared_ptr<CommConfig> &config, MemoryPool *pool,
                   std::shared_ptr<Communicator> *out) {
  CYLON_UNUSED(config);
  *out = std::make_shared<UCXCommunicator>(pool);
  auto &comm = static_cast<UCXCommunicator &>(**out);
  // Check init functions
  int initialized;
  // Int variable used when iterating
  int sIndx;
  // Address of the UCP Worker for receiving
  cylon::ucx::ucxWorkerAddr *ucpRecvWorkerAddr;
  // Address of the UCP Worker for sending
  cylon::ucx::ucxWorkerAddr *ucpSendWorkerAddr;

  // Status check when creating end-points
  ucs_status_t ucxStatus;
  // Variable to hold the current ucp address
  ucp_address_t *address;

  // MPI init
  MPI_Initialized(&initialized);
  if (!initialized) {
    RETURN_CYLON_STATUS_IF_MPI_FAILED(MPI_Init(nullptr, nullptr));
  }

  // Get the rank for checking send to self, and initializations
  MPI_Comm_rank(MPI_COMM_WORLD, &comm.rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm.world_size);

  int rank = comm.rank, world_size = comm.world_size;

  // Init context
  RETURN_CYLON_STATUS_IF_FAILED(
      cylon::ucx::initContext(&comm.ucpContext, nullptr));

  // Init recv worker and get address
  ucpRecvWorkerAddr =
      cylon::ucx::initWorker(comm.ucpContext, &comm.ucpRecvWorker);
  // Init send worker
  ucpSendWorkerAddr =
      cylon::ucx::initWorker(comm.ucpContext, &comm.ucpSendWorker);

  //  Gather all worker addresses
  // All addresses buffer for allGather
  auto allAddresses =
      std::make_unique<uint8_t[]>(ucpRecvWorkerAddr->addrSize * world_size);
  RETURN_CYLON_STATUS_IF_MPI_FAILED(MPI_Allgather(
      ucpRecvWorkerAddr->addr, (int)ucpRecvWorkerAddr->addrSize, MPI_BYTE,
      allAddresses.get(), (int)ucpRecvWorkerAddr->addrSize, MPI_BYTE,
      MPI_COMM_WORLD));

  // Iterate and set the sends
  comm.endPointMap.reserve(world_size);
  for (sIndx = 0; sIndx < world_size; sIndx++) {
    ucp_ep_params_t epParams;
    ucp_ep_h ep;

    // If not self, then check if the worker address has been received.
    //  If self,then assign local worker
    if (rank != sIndx) {
      address = reinterpret_cast<ucp_address_t *>(
          allAddresses.get() + sIndx * ucpRecvWorkerAddr->addrSize);
    } else {
      address = ucpRecvWorkerAddr->addr;
    }

    // Set params for the endpoint
    epParams.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS |
                          UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
    epParams.address = address;
    epParams.err_mode = UCP_ERR_HANDLING_MODE_NONE;

    // Create an endpoint
    ucxStatus = ucp_ep_create(comm.ucpSendWorker, &epParams, &ep);

    comm.endPointMap[sIndx] = ep;
    // Check if the endpoint was created properly
    if (ucxStatus != UCS_OK) {
      LOG(FATAL) << "Error when creating the endpoint.";
      return {Code::ExecutionError,
              "Error when creating the endpoint: " +
                  std::string(ucs_status_string(ucxStatus))};
    }
  }

  // Cleanup
  delete (ucpRecvWorkerAddr);
  delete (ucpSendWorkerAddr);

  return Status::OK();
}

Status UCXCommunicator::MakeWithRedis(const std::shared_ptr<CommConfig> &config,
                                      MemoryPool *pool,
                                      std::shared_ptr<Communicator> *out) {
  CYLON_UNUSED(config);
  *out = std::make_shared<UCXCommunicator>(pool);
  auto &comm = static_cast<UCXCommunicator &>(**out);

  const auto &ucx_config = std::static_pointer_cast<UCCConfig>(config);
  const auto &oob_context = std::static_pointer_cast<UCCRedisOOBContext>(ucx_config->oobContext);
  auto redis = oob_context->getRedis();

  // Int variable used when iterating
  int sIndx;
  // Address of the UCP Worker for receiving
  cylon::ucx::ucxWorkerAddr *ucpRecvWorkerAddr;
  // Address of the UCP Worker for sending
  cylon::ucx::ucxWorkerAddr *ucpSendWorkerAddr;

  // Status check when creating end-points
  ucs_status_t ucxStatus;
  // Variable to hold the current ucp address
  ucp_address_t *address;

  int world_size = oob_context->getWorldSize();
  comm.world_size = world_size;

  int num_cur_processes = redis->incr("num_cur_processes");

  comm.rank = num_cur_processes - 1;
  int rank = comm.rank;

  // Init context
  RETURN_CYLON_STATUS_IF_FAILED(
      cylon::ucx::initContext(&comm.ucpContext, nullptr));

  // Init recv worker and get address
  ucpRecvWorkerAddr =
      cylon::ucx::initWorker(comm.ucpContext, &comm.ucpRecvWorker);
  // Init send worker
  ucpSendWorkerAddr =
      cylon::ucx::initWorker(comm.ucpContext, &comm.ucpSendWorker);

  // Gather all worker addresses
  // All addresses buffer for allGather
  auto addr_str = std::string(
      (char *)ucpRecvWorkerAddr->addr,
      ((char *)ucpRecvWorkerAddr->addr) + ucpRecvWorkerAddr->addrSize);

  redis->hset("ucp_worker_addr_mp", std::to_string(rank), addr_str);
  std::vector<int> v(world_size, 0);
  redis->lpush("ucx_helper" + std::to_string(rank), v.begin(),
                    v.end());

  auto allAddresses =
      std::make_unique<char[]>(ucpRecvWorkerAddr->addrSize * world_size);

  for (int i = 0; i < world_size; i++) {
    if (i == rank) continue;
    auto helperName = "ucx_helper" + std::to_string(i);

    auto val = redis->hget("ucp_worker_addr_mp", std::to_string(i));
    while (!val) {
      redis->blpop(helperName);
      val = redis->hget("ucp_worker_addr_mp", std::to_string(i));
    }

    memcpy(allAddresses.get() + i * ucpRecvWorkerAddr->addrSize,
           val.value().data(), ucpRecvWorkerAddr->addrSize);
  }

  // // Iterate and set the sends
  comm.endPointMap.reserve(world_size);
  for (sIndx = 0; sIndx < world_size; sIndx++) {
    ucp_ep_params_t epParams;
    ucp_ep_h ep;

    // If not self, then check if the worker address has been received.
    //  If self,then assign local worker
    if (rank != sIndx) {
      address = reinterpret_cast<ucp_address_t *>(
          allAddresses.get() + sIndx * ucpRecvWorkerAddr->addrSize);
    } else {
      address = ucpRecvWorkerAddr->addr;
    }

    // Set params for the endpoint
    epParams.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS |
                          UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
    epParams.address = address;
    epParams.err_mode = UCP_ERR_HANDLING_MODE_NONE;

    // Create an endpoint
    ucxStatus = ucp_ep_create(comm.ucpSendWorker, &epParams, &ep);

    comm.endPointMap[sIndx] = ep;
    // Check if the endpoint was created properly
    if (ucxStatus != UCS_OK) {
      LOG(FATAL) << "Error when creating the endpoint.";
      return {Code::ExecutionError,
              "Error when creating the endpoint: " +
                  std::string(ucs_status_string(ucxStatus))};
    }
  }

  // Cleanup
  delete (ucpRecvWorkerAddr);
  delete (ucpSendWorkerAddr);

  return Status::OK();
}

Status UCXCommunicator::Make(const std::shared_ptr<CommConfig> &config, MemoryPool *pool,
                             std::shared_ptr<Communicator> *out) {
  const auto& ucc_config = std::static_pointer_cast<UCXConfig>(config);
  auto oob_context = ucc_config->getOOBContext();

  *out = std::make_shared<UCXCommunicator>(pool);
  auto &comm = static_cast<UCXCommunicator &>(**out);
  comm.oobContext = oob_context;

  // Int variable used when iterating
  int sIndx;
  // Address of the UCP Worker for receiving
  cylon::ucx::ucxWorkerAddr *ucpRecvWorkerAddr;
  // Address of the UCP Worker for sending
  cylon::ucx::ucxWorkerAddr *ucpSendWorkerAddr;

  // Status check when creating end-points
  ucs_status_t ucxStatus;
  // Variable to hold the current ucp address
  ucp_address_t *address;

  RETURN_CYLON_STATUS_IF_FAILED(oob_context->InitOOB());

  // Get the rank for checking send to self, and initializations
  RETURN_CYLON_STATUS_IF_FAILED(oob_context->getWorldSizeAndRank(comm.world_size, comm.rank));

  int rank = comm.rank, world_size = comm.world_size;

  // Init context
  RETURN_CYLON_STATUS_IF_FAILED(
      cylon::ucx::initContext(&comm.ucpContext, nullptr));

  // Init recv worker and get address
  ucpRecvWorkerAddr =
      cylon::ucx::initWorker(comm.ucpContext, &comm.ucpRecvWorker);
  // Init send worker
  ucpSendWorkerAddr =
      cylon::ucx::initWorker(comm.ucpContext, &comm.ucpSendWorker);

  //  Gather all worker addresses
  // All addresses buffer for allGather
  auto allAddresses =
      std::make_unique<uint8_t[]>(ucpRecvWorkerAddr->addrSize * world_size);
  // RETURN_CYLON_STATUS_IF_MPI_FAILED(MPI_Allgather(
  //     ucpRecvWorkerAddr->addr, (int)ucpRecvWorkerAddr->addrSize, MPI_BYTE,
  //     allAddresses.get(), (int)ucpRecvWorkerAddr->addrSize, MPI_BYTE,
  //     MPI_COMM_WORLD));


  RETURN_CYLON_STATUS_IF_FAILED(oob_context->OOBAllgather(
      (uint8_t *)ucpRecvWorkerAddr->addr, allAddresses.get(),
      (int)ucpRecvWorkerAddr->addrSize, (int)ucpRecvWorkerAddr->addrSize));

  // Iterate and set the sends
  comm.endPointMap.reserve(world_size);
  for (sIndx = 0; sIndx < world_size; sIndx++) {
    ucp_ep_params_t epParams;
    ucp_ep_h ep;

    // If not self, then check if the worker address has been received.
    //  If self,then assign local worker
    if (rank != sIndx) {
      address = reinterpret_cast<ucp_address_t *>(
          allAddresses.get() + sIndx * ucpRecvWorkerAddr->addrSize);
    } else {
      address = ucpRecvWorkerAddr->addr;
    }

    // Set params for the endpoint
    epParams.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS |
                          UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
    epParams.address = address;
    epParams.err_mode = UCP_ERR_HANDLING_MODE_NONE;

    // Create an endpoint
    ucxStatus = ucp_ep_create(comm.ucpSendWorker, &epParams, &ep);

    comm.endPointMap[sIndx] = ep;
    // Check if the endpoint was created properly
    if (ucxStatus != UCS_OK) {
      LOG(FATAL) << "Error when creating the endpoint.";
      return {Code::ExecutionError,
              "Error when creating the endpoint: " +
                  std::string(ucs_status_string(ucxStatus))};
    }
  }

  // Cleanup
  delete (ucpRecvWorkerAddr);
  delete (ucpSendWorkerAddr);

  return Status::OK();
}

void UCXCommunicator::Finalize() {
  this->oobContext->Finalize();
}

void UCXCommunicator::Barrier() {
  MPI_Barrier(MPI_COMM_WORLD);
}

CommType UCXCommunicator::GetCommType() const {
  return UCX;
}

ucc_status_t UCCRedisOOBContext::oob_allgather(void *sbuf, void *rbuf,
                                               size_t msglen, void *coll_info,
                                               void **req) {
  auto oob_allgather_func = [](void *sbuf, void *rbuf, size_t msglen,
                               void *coll_info, void **req) {
    return UCC_OK;
  };
  int world_size = ((UCCRedisOOBContext *)coll_info)->world_size;
  int rank = ((UCCRedisOOBContext *)coll_info)->rank;
  int num_comm = ((UCCRedisOOBContext *)coll_info)->num_oob_allgather;
  ((UCCRedisOOBContext *)coll_info)->num_oob_allgather++;

  auto &redis = ((UCCRedisOOBContext *)coll_info)->redis;
  *req = rbuf;
  std::string s((char *)sbuf, ((char *)sbuf) + msglen);

  redis->hset("ucc_oob_mp" + std::to_string(num_comm), std::to_string(rank), s);
  redis->lpush(
      "ucc_helper" + std::to_string(num_comm) + ":" + std::to_string(rank),
      "0");

  for (int i = 0; i < world_size; i++) {
    if (i == rank) {
      memcpy(rbuf + i * msglen, s.data(), msglen);
    } else {
      auto helperName =
          "ucc_helper" + std::to_string(num_comm) + ":" + std::to_string(i);

      // val = redis.hget("ucp_worker_addr_mp", std::to_string(i));
      sw::redis::OptionalString val;
      do {
        redis->brpoplpush(helperName, helperName, 0);
        val = redis->hget("ucc_oob_mp" + std::to_string(num_comm),
                          std::to_string(i));
      } while (!val);

      memcpy(rbuf + i * msglen, val.value().data(), msglen);
    }
  }

  // maybe need to do some cleanups here

  return UCC_OK;
}

UCCRedisOOBContext::UCCRedisOOBContext(
    int ws, std::shared_ptr<sw::redis::Redis> &rds): world_size(ws), redis(rds) {}

ucc_status_t UCCRedisOOBContext::oob_allgather_test(void *req) {
  CYLON_UNUSED(req);
  return UCC_OK;
}

ucc_status_t UCCRedisOOBContext::oob_allgather_free(void *req) {
  CYLON_UNUSED(req);
  return UCC_OK;
}

OOBType UCCRedisOOBContext::Type() {
  return OOBType::OOB_REDIS;
}

std::shared_ptr<sw::redis::Redis> UCCRedisOOBContext::getRedis() {
  return this->redis;
}

int UCCRedisOOBContext::getWorldSize() {
  return world_size;
}

void UCCRedisOOBContext::setRank(int rk) {
  rank = rk;
}

int UCCRedisOOBContext::getRank() {
  return rank;
}

#ifdef BUILD_CYLON_UCC
UCXUCCCommunicator::UCXUCCCommunicator(
    std::shared_ptr<Communicator> ucx_comm,
    std::shared_ptr<UCCOOBContext> &oob_context)
    : Communicator(ucx_comm->GetMemoryPool(), ucx_comm->GetRank(),
                   ucx_comm->GetWorldSize()),
      ucx_comm_(std::move(ucx_comm)),
      oobContext(oob_context) {}

Status UCXUCCCommunicator::Make(const std::shared_ptr<CommConfig> &config,
                                MemoryPool *pool, std::shared_ptr<Communicator> *out) {
  std::shared_ptr<Communicator> ucx_comm;
  auto ucc_config = std::static_pointer_cast<UCCConfig>(config);

  auto ucx_config = std::make_shared<UCXConfig>(ucc_config->oobContext->makeUCXOOBContext());

  UCXCommunicator::Make(ucx_config, pool, &ucx_comm);

  *out = std::make_shared<UCXUCCCommunicator>(std::move(ucx_comm),
                                              ucc_config->oobContext);

  auto &comm = *std::static_pointer_cast<UCXUCCCommunicator>(*out);
  comm.oobContext = ucc_config->oobContext;
  comm.oobContext->InitOOB(comm.GetRank());
  // auto redis = sw::redis::Redis("tcp://127.0.0.1:6379");

  // initialize UCC team and context
  ucc_context_params_t ctx_params;
  ucc_team_params_t team_params;
  ucc_context_config_h ctx_config;
  ucc_status_t status;
  ucc_lib_h lib;
  ucc_lib_config_h lib_config;

  // init ucc lib
  ucc_lib_params_t lib_params = {.mask = UCC_LIB_PARAM_FIELD_THREAD_MODE,
      .thread_mode = UCC_THREAD_SINGLE,
      .coll_types = {},
      .reduction_types = {},
      .sync_type = {}};

  RETURN_CYLON_STATUS_IF_UCC_FAILED(ucc_lib_config_read(nullptr, nullptr, &lib_config));
  RETURN_CYLON_STATUS_IF_UCC_FAILED(ucc_init(&lib_params, lib_config, &lib));
  ucc_lib_config_release(lib_config);

  // init ucc context
  ctx_params.mask = UCC_CONTEXT_PARAM_FIELD_OOB;

  // ctx_params.oob.allgather = ucc_config->oobContext->oob_allgather;
  ctx_params.oob.allgather = [](void *sbuf, void *rbuf, size_t msglen,
                                    void *coll_info, void **req) {
    return UCC_OK;
  };

  if(ucc_config->oobContext->Type() == OOBType::OOB_REDIS) {
    ctx_params.oob.allgather = team_params.oob.allgather =
        UCCRedisOOBContext::oob_allgather;
    ctx_params.oob.req_test = team_params.oob.req_test =
        UCCRedisOOBContext::oob_allgather_test;
    ctx_params.oob.req_free = team_params.oob.req_free =
        UCCRedisOOBContext::oob_allgather_free;
  } else if(ucc_config->oobContext->Type() == OOBType::OOB_MPI) {
    ctx_params.oob.allgather = team_params.oob.allgather =
        UCCMPIOOBContext::oob_allgather;
    ctx_params.oob.req_test = team_params.oob.req_test =
        UCCMPIOOBContext::oob_allgather_test;
    ctx_params.oob.req_free = team_params.oob.req_free =
        UCCMPIOOBContext::oob_allgather_free;
  } else {
    return {Code::NotImplemented, "UCC OOB communication type not supported."};
  }

  ctx_params.oob.coll_info = ucc_config->oobContext->getCollInfo();

  ctx_params.oob.n_oob_eps = static_cast<uint32_t>(comm.GetWorldSize());
  ctx_params.oob.oob_ep = static_cast<uint32_t>(comm.GetRank());

  RETURN_CYLON_STATUS_IF_UCC_FAILED(ucc_context_config_read(lib, nullptr, &ctx_config));

  RETURN_CYLON_STATUS_IF_UCC_FAILED(ucc_context_create(lib, &ctx_params, ctx_config,
                                                       &comm.uccContext));

  ucc_context_config_release(ctx_config);

  // init ucc team
  team_params.mask = UCC_TEAM_PARAM_FIELD_OOB;

  team_params.oob.coll_info = ucc_config->oobContext->getCollInfo();

  team_params.oob.n_oob_eps = static_cast<uint32_t>(comm.GetWorldSize());
  team_params.oob.oob_ep = static_cast<uint32_t>(comm.GetRank());
  RETURN_CYLON_STATUS_IF_UCC_FAILED(ucc_team_create_post(&comm.uccContext, 1, &team_params,
                                                         &comm.uccTeam));

  while (UCC_INPROGRESS == (status = ucc_team_create_test(comm.uccTeam))) {
//    RETURN_CYLON_STATUS_IF_UCC_FAILED(ucc_context_progress(comm.uccContext));
  }

  RETURN_CYLON_STATUS_IF_UCC_FAILED(status);
  return Status::OK();
}

CommType UCXUCCCommunicator::GetCommType() const {
  return UCX;
}

std::unique_ptr<Channel> UCXUCCCommunicator::CreateChannel() const {
  return ucx_comm_->CreateChannel();
}

void UCXUCCCommunicator::Finalize() {
  if (!this->IsFinalized()) {
    ucc_status_t status;
    while (uccTeam && (UCC_INPROGRESS == (status = ucc_team_destroy(uccTeam)))) {
      if (UCC_OK != status) {
        LOG(ERROR) << "ucc_team_destroy failed";
        break;
      }
    }
    ucc_context_destroy(uccContext);
    ucx_comm_->Finalize();
    finalized = true;
  }
}

void UCXUCCCommunicator::Barrier() {
  return ucx_comm_->Barrier();
}

Status UCXUCCCommunicator::AllGather(const std::shared_ptr<Table> &table,
                                     std::vector<std::shared_ptr<Table>> *out) const {
  ucc::UccTableAllgatherImpl impl(uccTeam, uccContext, world_size);
  return impl.Execute(table, out);
}

Status UCXUCCCommunicator::Gather(const std::shared_ptr<Table> &table,
                                  int gather_root,
                                  bool gather_from_root,
                                  std::vector<std::shared_ptr<Table>> *out) const {
  ucc::UccTableGatherImpl impl(uccTeam, uccContext, rank, world_size);
  return impl.Execute(table, gather_root, gather_from_root, out);
}

Status UCXUCCCommunicator::Bcast(std::shared_ptr<Table> *table,
                                 int bcast_root,
                                 const std::shared_ptr<CylonContext> &ctx) const {
  ucc::UccTableBcastImpl impl(uccTeam, uccContext);
  // The ctx_ptr and the real context are not the same
  return impl.Execute(table, bcast_root, ctx);
}

Status UCXUCCCommunicator::AllReduce(const std::shared_ptr<Column> &column,
                                     net::ReduceOp reduce_op,
                                     std::shared_ptr<Column> *output) const {
  ucc::UccAllReduceImpl impl(uccTeam, uccContext);
  return impl.Execute(column, reduce_op, output);
}

Status UCXUCCCommunicator::AllReduce(const std::shared_ptr<Scalar> &values,
                                     net::ReduceOp reduce_op,
                                     std::shared_ptr<Scalar> *output) const {
  ucc::UccAllReduceImpl impl(uccTeam, uccContext);
  return impl.Execute(values, reduce_op, output);
}

Status UCXUCCCommunicator::Allgather(const std::shared_ptr<Column> &values,
                                     std::vector<std::shared_ptr<Column>> *output) const {
  ucc::UccAllGatherImpl impl(uccTeam, uccContext, world_size);
  return impl.Execute(values, world_size, output);
}

Status UCXUCCCommunicator::Allgather(const std::shared_ptr<Scalar> &value,
                                     std::shared_ptr<Column> *output) const {
  ucc::UccAllGatherImpl impl(uccTeam, uccContext, world_size);
  return impl.Execute(value, world_size, output);
}

#endif // BUILD_CYLON_UCC
}  // namespace net
}  // namespace cylon
