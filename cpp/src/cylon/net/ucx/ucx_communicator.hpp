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
#include <cylon/net/ucx/mpi_ucx_ucc_oob_context.hpp>


#include "cylon/util/macros.hpp"
#include "cylon/util/enum.hpp"

#ifdef BUILD_CYLON_UCC

#include <ucc/api/ucc.h>

#endif

namespace cylon {
    namespace net {


        BETTER_ENUM(UCXConfigMapV, char, PORT, ADDRESS)


        class UCXConfig : public CommConfig {


        public:
            CommType Type() override;

            explicit UCXConfig(std::shared_ptr<UCXOOBContext> oobContext);

            explicit UCXConfig(MPI_Comm comm = MPI_COMM_NULL);

            static std::shared_ptr<UCXConfig> Make(
                    std::shared_ptr<UCXOOBContext> oobContext);

            static std::shared_ptr<UCXConfig> Make(MPI_Comm comm = MPI_COMM_NULL);

            void setOOBContext(std::shared_ptr<UCXOOBContext> oobContext);

            std::shared_ptr<UCXOOBContext> getOOBContext();

            MPI_Comm GetMPIComm() const;

        private:
            std::shared_ptr<UCXOOBContext> oobContext = nullptr;
            MPI_Comm comm_;
        };

#ifdef BUILD_CYLON_UCC

        class UCCConfig : public CommConfig {


        public:
            CommType Type() override;

            explicit UCCConfig(std::shared_ptr<UCCOOBContext> oobContext);

            static std::shared_ptr<UCCConfig> Make(
                    std::shared_ptr<UCCOOBContext> &oobContext);

            void setOOBContext(std::shared_ptr<UCCOOBContext> oobContext);

            std::shared_ptr<UCCOOBContext> getOOBContext();

        private:
            std::shared_ptr<UCCOOBContext> oobContext;
        };

#endif

        class UCXCommunicator : public Communicator {

        private:
            enum UCX_ADDRESS_TYP {
                AUTO,
                OVERRIDE
            };

            UCX_ADDRESS_TYP addressTyp = UCX_ADDRESS_TYP::AUTO;
        public:
            explicit UCXCommunicator(MemoryPool *pool);

            UCXCommunicator(MemoryPool *pool, bool externally_init, MPI_Comm comm);

            UCXCommunicator(MemoryPool *pool, UCX_ADDRESS_TYP ucxAddressTyp);


            ~UCXCommunicator() override = default;

            std::unique_ptr<Channel> CreateChannel() const override;

            int GetRank() const override;

            int GetWorldSize() const override;

            void Finalize() override;

            void Barrier() override;

            CommType GetCommType() const override;

            UCX_ADDRESS_TYP getAddressTyp() const;





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

            static Status MakeOOB(const std::shared_ptr<CommConfig> &config,
                                  MemoryPool *pool, std::shared_ptr<Communicator> *out,
                                  const std::shared_ptr<CommConfig> &parent_config);


            // # UCX specific attributes - These need to be passed to the channels created
            // from the communicator The worker for receiving
            ucp_worker_h ucpRecvWorker{};
            // The worker for sending
            ucp_worker_h ucpSendWorker{};
            // Endpoint Map
            std::unordered_map<int, ucp_ep_h> endPointMap;
            // UCP Context - Holds a UCP communication instance's global information.
            ucp_context_h ucpContext{};

            std::shared_ptr<UCXOOBContext> oobContext = nullptr;

            bool externally_init = false;
            MPI_Comm mpi_comm;



        };

#ifdef BUILD_CYLON_UCC

        class UCXUCCCommunicator : public Communicator {
        public:
            explicit UCXUCCCommunicator(std::shared_ptr<Communicator> ucx_comm,
                                        std::shared_ptr<UCCOOBContext> &oobContext);

            explicit UCXUCCCommunicator(const std::shared_ptr<Communicator> &ucx_comm);


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
            std::shared_ptr<UCXCommunicator> ucx_comm_;
            std::shared_ptr<UCCOOBContext> oobContext;

        private:
            static Status MakeOOB(std::shared_ptr<UCCOOBContext> &ucc_oob_ctx,
                                  MemoryPool *pool, std::shared_ptr<Communicator> *out,
                                  const std::shared_ptr<CommConfig> &config);

        };

#endif
    }  // namespace net
}  // namespace cylon
#endif  // CYLON_SRC_CYLON_COMM_UCXCOMMUNICATOR_H_
