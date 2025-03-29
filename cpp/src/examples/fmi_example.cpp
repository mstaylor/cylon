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

#include <glog/logging.h>

#include <cylon/net/fmi/fmi_communicator.hpp>
#include <cylon/ctx/cylon_context.hpp>
#include <cylon/table.hpp>
#include "thridparty/fmi/utils/DirectBackend.hpp"
#include "examples/example_utils.hpp"

#define CHECK_STATUS(status, msg) \
  if (!status.is_ok()) {          \
    LOG(ERROR) << msg << " " << status.get_msg(); \
    ctx->Finalize();              \
    return 1;                     \
  }

static constexpr int kCount = 10;
static constexpr double kDup = 0.9;

int main(int argc, char *argv[]) {

    if (argc < 2) {
        LOG(ERROR) << "There should be an argument for rank";
        return 1;
    }

    if (argc < 3) {
        LOG(ERROR) << "There should be an argument for worldsize";
        return 1;
    }


    if (argc < 4) {
        LOG(ERROR) << "There should be an argument for commname";
        return 1;
    }

    if (argc < 5) {
        LOG(ERROR) << "There should be an argument for host";
        return 1;
    }

    if (argc < 6) {
        LOG(ERROR) << "There should be an argument for port";
        return 1;
    }

    if (argc < 7) {
        LOG(ERROR) << "There should be an argument for maxTimeout";
        return 1;
    }



    auto rank = std::stoi(argv[1]);

    auto worldsize = std::stoi(argv[2]);

    auto com_name = std::string(argv[3]);

    auto host = std::string(argv[4]);

    auto port = std::stoi(argv[5]);

    auto maxTimout = std::stoi(argv[6]);

    auto backend = std::make_shared<FMI::Utils::DirectBackend>();

    backend->withHost(host.c_str());//rendezvous host
    backend->withPort(port);//rendezvous port
    backend->withMaxTimeout(maxTimout); //max timeout for direct connect
    backend->setResolveBackendDNS(true);//resolve rendezvous ip address

    std::shared_ptr<FMI::Utils::Backends> base_backend = std::dynamic_pointer_cast<FMI::Utils::Backends>(backend);

    auto config = std::make_shared<cylon::net::FMIConfig>(rank, worldsize,
                                                         base_backend, com_name);

    std::shared_ptr<cylon::CylonContext> ctx;

    if (!cylon::CylonContext::InitDistributed(config, &ctx).is_ok()) {
        return 1;
    }

    LOG(INFO) << "rank:" << ctx->GetRank() << " size:" << ctx->GetWorldSize();

    ctx->Barrier();

    std::shared_ptr<cylon::Table> first_table, second_table, out;

    cylon::examples::create_two_in_memory_tables(kCount, kDup, ctx, first_table, second_table);

    cylon::join::config::JoinConfig jc{cylon::join::config::JoinType::INNER, 0, 0,
                                       cylon::join::config::JoinAlgorithm::SORT, "l_", "r_"};

    auto status = cylon::DistributedJoin(first_table, second_table, jc, out);

    if (!status.is_ok()) {
        LOG(INFO) << "Table join failed ";
        return 1;
    }

    LOG(INFO) << "First table had : " << first_table->Rows() << " and Second table had : "
              << second_table->Rows() << ", Joined has : " << out->Rows();
    return 0;


}
