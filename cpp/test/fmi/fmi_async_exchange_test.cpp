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

/**
 * Publish-on-completion exchange over the FMI direct-redis channel.
 *
 * One ArrowAllToAll stays open for the whole run. Each round every rank works,
 * polling progress, then inserts a table for every peer tagged with the round.
 * One rank is slow. The test checks that the fast ranks receive each other's
 * tables while the slow rank is still working, and that every table arrives
 * exactly once after finish.
 *
 * Usage: fmi_async_exchange_test <rank> <world_size> <comm_name> <redis_host>
 *        <redis_port> <listen_port_base> <rounds> <fast_ms> <slow_ms> <slow_rank>
 * Prints one EVENT line per publish and receive, in ms since the post-setup
 * barrier, and returns 0 iff every check on this rank passed.
 */

#include <glog/logging.h>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <arrow/api.h>

#include <cylon/arrow/arrow_all_to_all.hpp>
#include <cylon/ctx/cylon_context.hpp>
#include <cylon/net/fmi/fmi_communicator.hpp>

namespace {

using Clock = std::chrono::steady_clock;

int g_failures = 0;

void Check(bool cond, int rank, const std::string &msg) {
  if (!cond) {
    LOG(ERROR) << "[rank " << rank << "] FAIL: " << msg;
    g_failures++;
  }
}

std::shared_ptr<arrow::Table> RoundTable(const std::shared_ptr<arrow::Schema> &schema,
                                         int rank, int round) {
  arrow::Int32Builder src, rnd;
  (void) src.Append(rank);
  (void) rnd.Append(round);
  std::shared_ptr<arrow::Array> a, b;
  (void) src.Finish(&a);
  (void) rnd.Finish(&b);
  return arrow::Table::Make(schema, {a, b});
}

double MsSince(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

}

int main(int argc, char *argv[]) {
  if (argc < 11) {
    LOG(ERROR) << "usage: " << argv[0] << " <rank> <world_size> <comm_name> <redis_host> "
               << "<redis_port> <listen_port_base> <rounds> <fast_ms> <slow_ms> <slow_rank>";
    return 2;
  }
  const int arg_rank = std::stoi(argv[1]);
  const int world_size = std::stoi(argv[2]);
  const std::string comm_name = argv[3];
  const std::string redis_host = argv[4];
  const int redis_port = std::stoi(argv[5]);
  const int listen_port_base = std::stoi(argv[6]);
  const int rounds = std::stoi(argv[7]);
  const int fast_ms = std::stoi(argv[8]);
  const int slow_ms = std::stoi(argv[9]);
  const int slow_rank = std::stoi(argv[10]);

  const auto setup_start = Clock::now();
  auto config = std::make_shared<cylon::net::FMIConfig>(
      arg_rank, world_size, "direct-redis", "", listen_port_base + arg_rank, 60000, comm_name,
      true, redis_host, redis_port, comm_name, "", "us-east-1", 3600, 100, 5000, "127.0.0.1");
  std::shared_ptr<cylon::CylonContext> ctx;
  if (!cylon::CylonContext::InitDistributed(config, &ctx).is_ok()) {
    LOG(ERROR) << "InitDistributed failed";
    return 2;
  }
  const int rank = ctx->GetRank();
  ctx->Barrier();
  const double setup_ms = MsSince(setup_start);
  std::cout << "SETUP rank=" << rank << " ms=" << setup_ms << std::endl;

  const auto t0 = Clock::now();
  const auto schema = arrow::schema({arrow::field("src", arrow::int32()),
                                     arrow::field("round", arrow::int32())});
  const std::vector<int> all_ranks = ctx->GetNeighbours(true);
  const std::vector<int> peers = ctx->GetNeighbours(false);

  std::map<std::pair<int, int>, int> received;
  std::map<std::pair<int, int>, double> arrival_ms;
  cylon::ArrowCallback on_table = [&](int source, const std::shared_ptr<arrow::Table> &table,
                                      int reference) {
    auto src = std::static_pointer_cast<arrow::Int32Array>(table->column(0)->chunk(0));
    auto rnd = std::static_pointer_cast<arrow::Int32Array>(table->column(1)->chunk(0));
    Check(table->num_rows() == 1 && src->Value(0) == source && rnd->Value(0) == reference, rank,
          "payload does not match its source and reference");
    const auto key = std::make_pair(source, reference);
    received[key]++;
    arrival_ms[key] = MsSince(t0);
    std::cout << "EVENT rank=" << rank << " recv src=" << source << " round=" << reference
              << " ms=" << arrival_ms[key] << std::endl;
    return true;
  };
  cylon::ArrowAllToAll exchange(ctx, all_ranks, all_ranks, ctx->GetNextSequence(), on_table,
                                schema);

  const int work_ms = rank == slow_rank ? slow_ms : fast_ms;
  for (int round = 0; round < rounds; round++) {
    const auto work_end = Clock::now() + std::chrono::milliseconds(work_ms);
    while (Clock::now() < work_end) {
      exchange.isComplete();
    }
    const auto table = RoundTable(schema, rank, round);
    for (int peer : peers) {
      exchange.insert(table, peer, round);
    }
    std::cout << "EVENT rank=" << rank << " publish round=" << round << " ms=" << MsSince(t0)
              << std::endl;
  }

  exchange.finish();
  while (!exchange.isComplete()) {
  }
  exchange.close();
  std::cout << "DONE rank=" << rank << " ms=" << MsSince(t0) << std::endl;

  for (int peer : peers) {
    for (int round = 0; round < rounds; round++) {
      const auto it = received.find(std::make_pair(peer, round));
      Check(it != received.end() && it->second == 1, rank,
            "table from rank " + std::to_string(peer) + " round " + std::to_string(round) +
                " not received exactly once");
    }
  }
  if (rank != slow_rank) {
    for (int peer : peers) {
      if (peer == slow_rank) {
        continue;
      }
      const auto it = arrival_ms.find(std::make_pair(peer, 0));
      Check(it != arrival_ms.end() && it->second < slow_ms, rank,
            "round 0 from fast rank " + std::to_string(peer) +
                " did not arrive while the slow rank was still working");
    }
  }

  ctx->Finalize();
  std::cout << "RESULT rank=" << rank << " failures=" << g_failures << std::endl;
  return g_failures == 0 ? 0 : 1;
}