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
#include "cylon/thridparty/fmi/utils/DirectBackend.hpp"
#include "cylon/thridparty/fmi/utils/RedisBackend.hpp"
#include "cylon/thridparty/fmi/utils/S3Backend.hpp"
#include <algorithm>
#include <cctype>


namespace cylon::net {
    FMIConfig::FMIConfig(int rank, int world_size,
                         std::shared_ptr<FMI::Utils::Backends> backend, std::string comm_name, bool nonblocking,
                         std::string redis_host, int redis_port, std::string redis_namespace) : rank_(rank),
                            world_size_(world_size), comm_name_(comm_name), backend_(backend),
                            nonblocking_(nonblocking), redis_host_(redis_host), redis_port_(redis_port),
                            redis_namespace_(redis_namespace){}

    FMIConfig::FMIConfig(int rank, int world_size, std::string host, int port,
                         int maxtimeout, bool resolveIp, std::string comm_name,
                         bool nonblocking, bool enablePing): rank_(rank), world_size_(world_size),
                                            comm_name_(comm_name),
                                            nonblocking_(nonblocking){
        auto backend = std::make_shared<FMI::Utils::DirectBackend>();
        backend->withHost(host.c_str());
        backend->withPort(port);
        backend->withMaxTimeout(maxtimeout);
        backend->setResolveBackendDNS(resolveIp);
        backend->setBlockingMode(nonblocking ? FMI::Utils::NONBLOCKING: FMI::Utils::BLOCKING);
        backend->setEnableHostPing(enablePing);
        backend_ = std::dynamic_pointer_cast<FMI::Utils::Backends>(backend);

    }


    FMIConfig::FMIConfig(int rank, int world_size, std::string host, int port,
                         int maxtimeout, bool resolveIp, std::string comm_name,
                         bool nonblocking) : FMIConfig(rank, world_size, host, port,
                                                               maxtimeout, resolveIp, comm_name,
                                                               nonblocking, false){}

    FMIConfig::FMIConfig(int rank, int world_size, std::string host, int port, int maxtimeout,
                         bool resolveIp, std::string comm_name, bool nonblocking,
                         bool enablePing, std::string redis_host,
                         int redis_port, std::string redis_namespace) : FMIConfig(rank, world_size, host, port,
                                                                                  maxtimeout, resolveIp,
                                                                                  comm_name, nonblocking,
                                                                                  enablePing) {
        this->redis_host_ = redis_host;
        this->redis_port_ = redis_port;
        this->redis_namespace_ = redis_namespace;

    }

    FMIConfig::FMIConfig(int rank, int world_size, std::string channel_type,
                         std::string host, int port, int maxtimeout,
                         std::string comm_name, bool nonblocking,
                         std::string redis_host, int redis_port, std::string redis_namespace,
                         std::string s3_bucket, std::string s3_region, int key_ttl,
                         int s3_retry_initial_ms, int s3_retry_max_ms) : rank_(rank), world_size_(world_size),
                                            comm_name_(comm_name), nonblocking_(nonblocking),
                                            redis_host_(redis_host), redis_port_(redis_port),
                                            redis_namespace_(redis_namespace), key_ttl_(key_ttl),
                                            s3_retry_initial_ms_(s3_retry_initial_ms),
                                            s3_retry_max_ms_(s3_retry_max_ms) {
        // Normalize channel_type to lowercase
        std::string type_lower = channel_type;
        std::transform(type_lower.begin(), type_lower.end(), type_lower.begin(),
                       [](unsigned char c){ return std::tolower(c); });
        this->channel_type_ = type_lower;

        // Default timeout for polling (configurable via maxtimeout for total wait)
        int timeout = maxtimeout > 0 ? std::min(100, maxtimeout) : 100;

        if (type_lower == "redis") {
            auto backend = std::make_shared<FMI::Utils::RedisBackend>();
            backend->withHost(redis_host.c_str());
            backend->withPort(redis_port);
            backend->withMaxTimeout(maxtimeout);
            backend->withTimeout(timeout);
            backend_ = std::dynamic_pointer_cast<FMI::Utils::Backends>(backend);
        } else if (type_lower == "s3") {
            auto backend = std::make_shared<FMI::Utils::S3Backend>();
            backend->withS3BacketName(const_cast<char*>(s3_bucket.c_str()));
            backend->withAWSRegion(const_cast<char*>(s3_region.c_str()));
            backend->withMaxTimeout(maxtimeout);
            backend->withTimeout(timeout);
            backend_ = std::dynamic_pointer_cast<FMI::Utils::Backends>(backend);
        } else {
            // Default to Direct backend
            auto backend = std::make_shared<FMI::Utils::DirectBackend>();
            backend->withHost(host.c_str());
            backend->withPort(port);
            backend->withMaxTimeout(maxtimeout);
            backend->setResolveBackendDNS(false);
            backend->setBlockingMode(nonblocking ? FMI::Utils::NONBLOCKING : FMI::Utils::BLOCKING);
            backend->setEnableHostPing(false);
            backend_ = std::dynamic_pointer_cast<FMI::Utils::Backends>(backend);
        }
    }





    FMIConfig::FMIConfig(int rank, int world_size, std::string host, int port,
                         int maxtimeout, bool resolveIp, std::string comm_name,
                         bool nonblocking): rank_(rank), world_size_(world_size),
                         comm_name_(comm_name),
                         nonblocking_(nonblocking){
        auto backend = std::make_shared<FMI::Utils::DirectBackend>();
        backend->withHost(host.c_str());
        backend->withPort(port);
        backend->withMaxTimeout(maxtimeout);
        backend->setResolveBackendDNS(resolveIp);
        backend->setBlockingMode(nonblocking ? FMI::Utils::NONBLOCKING: FMI::Utils::BLOCKING);
        backend->setEnableHostPing(enablePing);
        backend_ = std::dynamic_pointer_cast<FMI::Utils::Backends>(backend);

    }


    FMIConfig::FMIConfig(int rank, int world_size, std::string host, int port,
                         int maxtimeout, bool resolveIp, std::string comm_name,
                         bool nonblocking) : FMIConfig(rank, world_size, host, port,
                                                               maxtimeout, resolveIp, comm_name,
                                                               nonblocking, false){}

    FMIConfig::FMIConfig(int rank, int world_size, std::string host, int port, int maxtimeout, bool resolveIp,
                         std::string comm_name, bool nonblocking, std::string redis_host, int redis_port,
                         std::string redis_namespace) : FMIConfig(rank, world_size, host, port, maxtimeout,
                                                                  resolveIp, comm_name, nonblocking) {
        this->redis_host_ = redis_host;
        this->redis_port_ = redis_port;
        this->redis_namespace_ = redis_namespace;

    }





    CommType FMIConfig::Type() {
        return FMI;
    }

    FMIConfig::~FMIConfig() = default;


    std::shared_ptr<FMIConfig> FMIConfig::Make(int rank, int world_size,
                                               std::shared_ptr<FMI::Utils::Backends> backend,
                                               std::string comm_name,
                                               bool nonblocking,
                                               std::string redis_host,
                                               int redis_port,
                                               std::string redis_namespace) {
        return std::make_shared<FMIConfig>(rank, world_size, backend, comm_name, nonblocking,
                                           redis_host, redis_port, redis_namespace);
    }

    std::shared_ptr<FMIConfig>
    FMIConfig::Make(int rank, int world_size, std::string host, int port, int maxtimeout, bool resolveIp,
                    std::string comm_name, bool nonblocking) {
        return std::make_shared<FMIConfig>(rank, world_size, host, port, maxtimeout, resolveIp,
                                           comm_name, nonblocking);
    }

    std::shared_ptr<FMIConfig>
    FMIConfig::Make(int rank, int world_size, std::string host, int port, int maxtimeout, bool resolveIp,
                    std::string comm_name, bool nonblocking, bool enablePing) {
        return std::make_shared<FMIConfig>(rank, world_size, host, port, maxtimeout, resolveIp,
                                           comm_name, nonblocking, enablePing);
    }

    std::shared_ptr<FMIConfig>
    FMIConfig::Make(int rank, int world_size, std::string host, int port, int maxtimeout, bool resolveIp,
                    std::string comm_name, bool nonblocking, bool enableping,std::string redis_host, int redis_port,
                    std::string redis_namespace) {
        return std::make_shared<FMIConfig>(rank, world_size, host, port, maxtimeout, resolveIp,
                                          comm_name, nonblocking, enableping, redis_host,
                                          redis_port, redis_namespace);
    }

    std::shared_ptr<FMIConfig>
    FMIConfig::Make(int rank, int world_size, std::string channel_type,
                    std::string host, int port, int maxtimeout,
                    std::string comm_name, bool nonblocking,
                    std::string redis_host, int redis_port, std::string redis_namespace,
                    std::string s3_bucket, std::string s3_region, int key_ttl,
                    int s3_retry_initial_ms, int s3_retry_max_ms) {
        return std::make_shared<FMIConfig>(rank, world_size, channel_type, host, port, maxtimeout,
                                          comm_name, nonblocking, redis_host, redis_port,
                                          redis_namespace, s3_bucket, s3_region, key_ttl,
                                          s3_retry_initial_ms, s3_retry_max_ms);
    }



    int FMIConfig::getRank() const {
        return rank_;
    }

    int FMIConfig::getWorldSize() const {
        return world_size_;
    }


    const std::string &FMIConfig::getCommName() const {
        return comm_name_;
    }

    const std::shared_ptr<FMI::Utils::Backends> &FMIConfig::getBackend() const {
        return backend_;
    }

    bool FMIConfig::isNonblocking() const {
        return nonblocking_;
    }

    const std::string &FMIConfig::getRedisHost() const {
        return redis_host_;
    }

    int FMIConfig::getRedisPort() const {
        return redis_port_;
    }

    const std::string &FMIConfig::getRedisNamespace() const {
        return redis_namespace_;
    }

    const std::string &FMIConfig::getChannelType() const {
        return channel_type_;
    }

    int FMIConfig::getKeyTtl() const {
        return key_ttl_;
    }

    int FMIConfig::getS3RetryInitialMs() const {
        return s3_retry_initial_ms_;
    }

    int FMIConfig::getS3RetryMaxMs() const {
        return s3_retry_max_ms_;
    }

    FMICommunicator::FMICommunicator(MemoryPool *pool, int32_t rank, int32_t world_size,
                                     const std::shared_ptr<FMI::Communicator> &fmi_comm,
                                     bool nonblocking) : Communicator(pool, rank, world_size){}


    FMICommunicator::FMICommunicator(MemoryPool *pool, int32_t rank, int32_t world_size,
                                     const std::shared_ptr<FMI::Communicator> &fmi_comm,
                                     bool nonblocking, std::string redis_host, int redis_port,
                                     std::string redis_namespace) :
                                     Communicator(pool, rank, world_size),
                                     fmi_comm_(fmi_comm), nonblocking_(nonblocking),
                                     redis_host_(redis_host), redis_port_(redis_port),
                                     redis_namespace_(redis_namespace){}

    std::unique_ptr<Channel> FMICommunicator::CreateChannel() const {
        return std::make_unique<fmi::FMIChannel>(fmi_comm_, getBlockingMode(), redis_host_,
                                                 redis_port_, redis_namespace_);
    }

    int FMICommunicator::GetRank() const {
        return this->rank;
    }

    int FMICommunicator::GetWorldSize() const {
        return this->world_size;
    }

    void FMICommunicator::Finalize() {}

    void FMICommunicator::Barrier() {
        fmi_comm_->barrier();
    }

    CommType FMICommunicator::GetCommType() const {
        return FMI;
    }

    Status
    FMICommunicator::AllGather(const std::shared_ptr<Table> &table, std::vector<std::shared_ptr<Table>> *out) const {
        fmi::FmiTableAllgatherImpl impl(fmi_comm_, getBlockingMode());
        return impl.Execute(table, out);
    }

    Status FMICommunicator::Gather(const std::shared_ptr<Table> &table, int gather_root, bool gather_from_root,
                                   std::vector<std::shared_ptr<Table>> *out) const {
        fmi::FmiTableGatherImpl impl(fmi_comm_, getBlockingMode());
        return impl.Execute(table, gather_root, gather_from_root, out);
    }

    Status FMICommunicator::Bcast(std::shared_ptr<Table> *table, int bcast_root,
                                  const std::shared_ptr<CylonContext> &ctx) const {
        fmi::FmiTableBcastImpl impl(fmi_comm_, getBlockingMode());
        return impl.Execute(table, bcast_root, ctx);
    }

    Status FMICommunicator::AllReduce(const std::shared_ptr<Column> &values, net::ReduceOp reduce_op,
                                      std::shared_ptr<Column> *output) const {
        fmi::FmiAllReduceImpl impl(fmi_comm_, getBlockingMode( ));
        return impl.Execute(values, reduce_op, output, pool);
    }

    Status FMICommunicator::AllReduce(const std::shared_ptr<Scalar> &value, net::ReduceOp reduce_op,
                                      std::shared_ptr<Scalar> *output) const {
        fmi::FmiAllReduceImpl impl(fmi_comm_, getBlockingMode());
        return impl.Execute(value, reduce_op, output, pool);
    }

    Status FMICommunicator::Allgather(const std::shared_ptr<Column> &values,
                                      std::vector<std::shared_ptr<Column>> *output) const {
        fmi::FmiAllgatherImpl impl(fmi_comm_, getBlockingMode());
        return impl.Execute(values, world_size, output, pool);
    }

    Status FMICommunicator::Allgather(const std::shared_ptr<Scalar> &value, std::shared_ptr<Column> *output) const {
        fmi::FmiAllgatherImpl impl(fmi_comm_, getBlockingMode());
        return impl.Execute(value, world_size, output, pool);
    }

    std::shared_ptr<FMI::Communicator> FMICommunicator::fmi_comm() const {
        return fmi_comm_;
    }

    Status FMICommunicator::Make(const std::shared_ptr<CommConfig> &config,
                                 MemoryPool *pool,
                                 std::shared_ptr<Communicator> *out) {
        int rank, world_size;
        // check if MPI is initialized

        const auto &fmi_config = std::static_pointer_cast<FMIConfig>(config);



        auto fmi_comm = std::make_shared<FMI::Communicator>(fmi_config->getRank(),
                                                            fmi_config->getWorldSize(),
                                                            fmi_config->getBackend(),
                                                            fmi_config->getCommName(),
                                                            fmi_config->getRedisHost(),
                                                            fmi_config->getRedisPort(),
                                                            fmi_config->getRedisNamespace(),
                                                            fmi_config->getKeyTtl(),
                                                            fmi_config->getS3RetryInitialMs(),
                                                            fmi_config->getS3RetryMaxMs());

        rank = fmi_comm->getPeerId();
        world_size = fmi_comm->getNumPeers();



        if (rank < 0 || world_size < 0 || rank >= world_size) {
            return {Code::ExecutionError, "Malformed rank :" + std::to_string(rank)
                                          + " or world size:" + std::to_string(world_size)};
        }

        *out = std::make_shared<FMICommunicator>(pool, rank, world_size, fmi_comm,
                                                 fmi_config->isNonblocking(),
                                                 fmi_config->getRedisHost(),
                                                 fmi_config->getRedisPort(),
                                                 fmi_config->getRedisNamespace());

        return Status::OK();
    }

    FMI::Utils::Mode FMICommunicator::getBlockingMode() const{
        FMI::Utils::Mode mode;
        if (nonblocking_) {
            mode = FMI::Utils::NONBLOCKING;
        } else {
            mode = FMI::Utils::BLOCKING;
        }
        return mode;
    }



}
