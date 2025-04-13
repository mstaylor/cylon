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

#ifndef CYLON_FMI_COMMUNICATOR_HPP
#define CYLON_FMI_COMMUNICATOR_HPP

#include <cylon/net/comm_config.hpp>
#include <cylon/net/communicator.hpp>
#include <cylon/thridparty/fmi/Communicator.hpp>


namespace cylon::net {

    class FMICommunicator;

    class FMIConfig : public CommConfig {
    public:
        explicit FMIConfig(int rank, int world_size, std::shared_ptr<FMI::Utils::Backends> &backend,
                           std::string &comm_name);

        FMIConfig(int rank, int world_size, std::string &host, int port, int maxtimeout,
                  bool resolveIp, std::string &comm_name);

        CommType Type() override;

        ~FMIConfig() override;

        static std::shared_ptr<FMIConfig> Make(int rank, int world_size,
                                               std::shared_ptr<FMI::Utils::Backends> &backend,
                                               std::string &comm_name);


        static std::shared_ptr<FMIConfig> Make(int rank, int world_size, std::string &host, int port, int maxtimeout,
                                               bool resolveIp, std::string &comm_name);

        int getRank() const;

        int getWorldSize() const;

        const std::string &getCommName() const;

        const std::shared_ptr<FMI::Utils::Backends> &getBackend() const;

    private:

        friend FMICommunicator;
        int rank_;
        int world_size_;
        std::string comm_name_;
        std::shared_ptr<FMI::Utils::Backends> backend_;
    };

    class FMICommunicator : public Communicator {
    public:
        FMICommunicator(MemoryPool *pool, int32_t rank, int32_t world_size,
                        const std::shared_ptr<FMI::Communicator>  &fmi_comm);
        ~FMICommunicator() override = default;
        std::unique_ptr<Channel> CreateChannel() const override;
        int GetRank() const override;
        int GetWorldSize() const override;
        void Finalize() override;
        void Barrier() override;
        void Barrier(FMI::Utils::Operation op);
        CommType GetCommType() const override;

        Status AllGather(const std::shared_ptr<Table> &table,
                         std::vector<std::shared_ptr<Table>> *out) const override;

        Status Gather(const std::shared_ptr<Table> &table, int gather_root,
                      bool gather_from_root, std::vector<std::shared_ptr<Table>> *out) const override;

        Status Bcast(std::shared_ptr<Table> *table,
                     int bcast_root,
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

        std::shared_ptr<FMI::Communicator> fmi_comm() const;

        static Status Make(const std::shared_ptr<CommConfig> &config,
                           MemoryPool *pool, std::shared_ptr<Communicator> *out);

    private:
        std::shared_ptr<FMI::Communicator> fmi_comm_  = nullptr;
        bool externally_init = false;
    };

}


#endif //CYLON_FMI_COMMUNICATOR_HPP
