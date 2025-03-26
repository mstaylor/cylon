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

#include "fmi_communicator.hpp"
#include "cylon/net/fmi/fmi_channel.hpp"
#include "cylon/net/fmi/fmi_operations.hpp"


namespace cylon::net {
    FMIConfig::FMIConfig(FMI::Communicator *comm) : comm_(comm) {}

    CommType FMIConfig::Type() {
        return FMI;
    }

    FMIConfig::~FMIConfig() = default;

    FMI::Communicator *FMIConfig::GetFMIComm() const {
        return comm_;
    }

    std::shared_ptr<FMIConfig> FMIConfig::Make(FMI::Communicator *comm) {
        return std::make_shared<FMIConfig>(comm);
    }


    FMICommunicator::FMICommunicator(MemoryPool *pool, int32_t rank, int32_t world_size,
                                     FMI::Communicator *fmi_comm) : Communicator(pool, rank, world_size),
                                     fmi_comm_(fmi_comm) {}

    std::unique_ptr<Channel> FMICommunicator::CreateChannel() const {
        return std::make_unique<fmi::FMIChannel>(fmi_comm_);
    }

    int FMICommunicator::GetRank() const {
        return this->rank;
    }

    int FMICommunicator::GetWorldSize() const {
        return this->world_size;
    }

    void FMICommunicator::Finalize() {}

    void FMICommunicator::Barrier() {
        fmi_comm_->barrier(FMI::Utils::DEFAULT);
    }

    void FMICommunicator::Barrier(FMI::Utils::Operation op) {
        fmi_comm_->barrier(op);
    }

    CommType FMICommunicator::GetCommType() const {
        return FMI;
    }

    Status
    FMICommunicator::AllGather(const std::shared_ptr<Table> &table, std::vector<std::shared_ptr<Table>> *out) const {
        fmi::FmiTableAllgatherImpl impl(fmi_comm_);
        return impl.Execute(table, out);
    }

    Status FMICommunicator::Gather(const std::shared_ptr<Table> &table, int gather_root, bool gather_from_root,
                                   std::vector<std::shared_ptr<Table>> *out) const {
        fmi::FmiTableGatherImpl impl(fmi_comm_);
        return impl.Execute(table, gather_root, gather_from_root, out);
    }

    Status FMICommunicator::Bcast(std::shared_ptr<Table> *table, int bcast_root,
                                  const std::shared_ptr<CylonContext> &ctx) const {
        fmi::FmiTableBcastImpl impl(fmi_comm_);
        return impl.Execute(table, bcast_root, ctx);
    }

    Status FMICommunicator::AllReduce(const std::shared_ptr<Column> &values, net::ReduceOp reduce_op,
                                      std::shared_ptr<Column> *output) const {
        fmi::FmiAllReduceImpl impl(fmi_comm_);
        return impl.Execute(values, reduce_op, output, pool);
    }

    Status FMICommunicator::AllReduce(const std::shared_ptr<Scalar> &value, net::ReduceOp reduce_op,
                                      std::shared_ptr<Scalar> *output) const {
        fmi::FmiAllReduceImpl impl(fmi_comm_);
        return impl.Execute(value, reduce_op, output, pool);
    }

    Status FMICommunicator::Allgather(const std::shared_ptr<Column> &values,
                                      std::vector<std::shared_ptr<Column>> *output) const {
        fmi::FmiAllgatherImpl impl(fmi_comm_);
        return impl.Execute(values, world_size, output, pool);
    }

    Status FMICommunicator::Allgather(const std::shared_ptr<Scalar> &value, std::shared_ptr<Column> *output) const {
        fmi::FmiAllgatherImpl impl(fmi_comm_);
        return impl.Execute(value, world_size, output, pool);
    }

    FMI::Communicator *FMICommunicator::fmi_comm() const {
        return fmi_comm_;
    }

    Status FMICommunicator::Make(const std::shared_ptr<CommConfig> &config, MemoryPool *pool,
                                 std::shared_ptr<Communicator> *out) {
        int ext_init, rank, world_size;
        // check if MPI is initialized

        auto fmi_comm = std::static_pointer_cast<FMIConfig>(config)->GetFMIComm();

        rank = fmi_comm->getPeerId();
        world_size = fmi_comm->getNumPeers();


        if (rank < 0 || world_size < 0 || rank >= world_size) {
            return {Code::ExecutionError, "Malformed rank :" + std::to_string(rank)
                                          + " or world size:" + std::to_string(world_size)};
        }

        *out = std::make_shared<FMICommunicator>(pool, rank, world_size, fmi_comm);
        return Status::OK();
    }




}
