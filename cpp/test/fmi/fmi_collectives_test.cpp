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
 * Multi-rank collective tests for the FMI communicator over the REDIS channel.
 *
 * This is an argv-driven standalone (one process per rank, coordinated through
 * Redis), modeled on cpp/src/examples/fmi_example.cpp — NOT a Catch2 single
 * process. Launch N copies with a shared, per-run-unique comm_name via
 * run_fmi_collectives.sh.
 *
 * Redis (ClientServer) overrides send/recv/bcast/reduce, so this file exercises:
 *   - Scatter  (Channel-base scatterv default: real send/recv)
 *   - Reduce   (ClientServer::reduce, delivered only at reduce_root)
 *   - Bcast    (round-trip; regression for the byte-count fix in FmiTableBcastImpl)
 * Table Gather/AllGather use the empty Channel-base gatherv stub on redis, so the
 * gather_root regression is covered by the Direct-channel test instead.
 *
 * Usage: fmi_collectives_test <rank> <world_size> <comm_name> <redis_host> <redis_port>
 * Returns 0 iff every assertion on this rank passed.
 */

#include <glog/logging.h>

#include <cstdlib>
#include <string>
#include <vector>

#include <arrow/api.h>

#include <cylon/net/fmi/fmi_communicator.hpp>
#include <cylon/ctx/cylon_context.hpp>
#include <cylon/table.hpp>
#include <cylon/column.hpp>

namespace {

int g_rank = -1;
int g_failures = 0;

#define TCHECK(cond, msg)                                                    \
  do {                                                                        \
    if (!(cond)) {                                                            \
      LOG(ERROR) << "[rank " << g_rank << "] FAIL: " << msg;                  \
      g_failures++;                                                           \
    }                                                                         \
  } while (0)

std::shared_ptr<cylon::Table> MakeInt64Table(
    const std::shared_ptr<cylon::CylonContext> &ctx,
    const std::vector<int64_t> &vals) {
  arrow::Int64Builder b;
  (void) b.AppendValues(vals);
  std::shared_ptr<arrow::Array> arr;
  (void) b.Finish(&arr);
  auto schema = arrow::schema({arrow::field("v", arrow::int64())});
  auto atable = arrow::Table::Make(schema, {arr});
  std::shared_ptr<cylon::Table> t;
  auto st = cylon::Table::FromArrowTable(ctx, atable, t);
  if (!st.is_ok()) {
    LOG(ERROR) << "MakeInt64Table failed: " << st.get_msg();
  }
  return t;
}

std::shared_ptr<cylon::Column> MakeInt64Column(const std::vector<int64_t> &vals) {
  arrow::Int64Builder b;
  (void) b.AppendValues(vals);
  std::shared_ptr<arrow::Array> arr;
  (void) b.Finish(&arr);
  return cylon::Column::Make(std::move(arr));
}

std::vector<int64_t> GetInt64Col(const std::shared_ptr<cylon::Table> &t) {
  std::vector<int64_t> out;
  if (!t || !t->get_table()) return out;
  const auto &chunked = t->get_table()->column(0);
  for (int c = 0; c < chunked->num_chunks(); c++) {
    auto arr = std::static_pointer_cast<arrow::Int64Array>(chunked->chunk(c));
    for (int64_t i = 0; i < arr->length(); i++) out.push_back(arr->Value(i));
  }
  return out;
}

std::vector<int64_t> GetInt64Column(const std::shared_ptr<cylon::Column> &c) {
  std::vector<int64_t> out;
  if (!c || !c->data()) return out;
  auto arr = std::static_pointer_cast<arrow::Int64Array>(c->data());
  for (int64_t i = 0; i < arr->length(); i++) out.push_back(arr->Value(i));
  return out;
}

}  // namespace

int main(int argc, char *argv[]) {
  if (argc < 6) {
    LOG(ERROR) << "usage: " << argv[0]
               << " <rank> <world_size> <comm_name> <redis_host> <redis_port>";
    return 2;
  }

  int arg_rank = std::stoi(argv[1]);
  int world_size = std::stoi(argv[2]);
  std::string comm_name = argv[3];
  std::string redis_host = argv[4];
  int redis_port = std::stoi(argv[5]);

  // Redis-channel config: host/port (rendezvous) are unused for the redis channel;
  // redis_host/redis_port carry coordination + data. comm_name doubles as the redis
  // namespace so concurrent runs never collide.
  auto config = cylon::net::FMIConfig::Make(
      arg_rank, world_size, /*channel_type=*/"redis", /*host=*/"", /*port=*/0,
      /*maxtimeout=*/5000, comm_name, /*nonblocking=*/false, redis_host,
      redis_port, /*redis_namespace=*/comm_name);

  std::shared_ptr<cylon::CylonContext> ctx;
  if (!cylon::CylonContext::InitDistributed(config, &ctx).is_ok()) {
    LOG(ERROR) << "InitDistributed failed";
    return 2;
  }

  const int rank = ctx->GetRank();
  const int ws = ctx->GetWorldSize();
  g_rank = rank;
  const auto &comm = ctx->GetCommunicator();

  ctx->Barrier();

  // --- Scatter A: root 0, equal 3-row shards. rank r receives [r*100 .. r*100+2].
  {
    std::vector<std::shared_ptr<cylon::Table>> tables;
    if (rank == 0) {
      for (int r = 0; r < ws; r++) {
        tables.push_back(MakeInt64Table(ctx, {r * 100, r * 100 + 1, r * 100 + 2}));
      }
    }
    std::shared_ptr<cylon::Table> out;
    auto st = comm->Scatter(tables, /*scatter_root=*/0, ctx, &out);
    TCHECK(st.is_ok(), "equal scatter status: " << st.get_msg());
    if (st.is_ok()) {
      std::vector<int64_t> exp = {rank * 100, rank * 100 + 1, rank * 100 + 2};
      TCHECK(GetInt64Col(out) == exp, "equal scatter values");
    }
  }
  ctx->Barrier();

  // --- Scatter B: non-zero root (ws-1), UNEVEN shards. rank r gets (r+1) rows
  // [r*100 .. r*100+r]. Exercises scatterv with variable counts AND a non-zero root.
  {
    int scatter_root = ws - 1;
    std::vector<std::shared_ptr<cylon::Table>> tables;
    if (rank == scatter_root) {
      for (int r = 0; r < ws; r++) {
        std::vector<int64_t> vals;
        for (int i = 0; i <= r; i++) vals.push_back(r * 100 + i);
        tables.push_back(MakeInt64Table(ctx, vals));
      }
    }
    std::shared_ptr<cylon::Table> out;
    auto st = comm->Scatter(tables, scatter_root, ctx, &out);
    TCHECK(st.is_ok(), "uneven scatter status: " << st.get_msg());
    if (st.is_ok()) {
      std::vector<int64_t> exp;
      for (int i = 0; i <= rank; i++) exp.push_back(rank * 100 + i);
      TCHECK(GetInt64Col(out) == exp, "uneven scatter values (n=" << exp.size() << ")");
    }
  }
  ctx->Barrier();

  // --- Reduce SUM/MAX/MIN at root 0. Each rank contributes [rank, rank+1, rank+2].
  {
    int reduce_root = 0;
    int64_t S = (int64_t) ws * (ws - 1) / 2;  // sum of ranks 0..ws-1

    // DIAGNOSTIC: single-element reduce first (count=1, the path the Frontiers paper
    // exercised via reduce_cost/reduce_metrics). If this passes but multi-element
    // fails, the issue is multi-element-specific, not the redis channel per se.
    {
      auto col1 = MakeInt64Column({(int64_t) rank});
      std::shared_ptr<cylon::Column> res1;
      auto st1 = comm->Reduce(col1, cylon::net::SUM, reduce_root, &res1);
      if (st1.is_ok() && rank == reduce_root) {
        auto got1 = GetInt64Column(res1);
        LOG(WARNING) << "[rank " << rank << "] DIAG reduce SUM count=1: got="
                     << (got1.empty() ? -999 : got1[0]) << " exp=" << S;
        TCHECK(got1.size() == 1 && got1[0] == S, "DIAG single-element reduce SUM");
      }
    }

    struct Case { cylon::net::ReduceOp op; std::vector<int64_t> exp; const char *name; };
    std::vector<Case> cases = {
        {cylon::net::SUM, {S, S + ws, S + 2 * ws}, "SUM"},
        {cylon::net::MAX, {ws - 1, ws, ws + 1}, "MAX"},
        {cylon::net::MIN, {0, 1, 2}, "MIN"},
    };
    for (const auto &tc : cases) {
      auto col = MakeInt64Column({rank, rank + 1, rank + 2});
      std::shared_ptr<cylon::Column> res;
      auto st = comm->Reduce(col, tc.op, reduce_root, &res);
      TCHECK(st.is_ok(), "reduce " << tc.name << " status: " << st.get_msg());
      if (st.is_ok() && rank == reduce_root) {
        auto got = GetInt64Column(res);
        std::string gs, es;
        for (auto v : got) gs += std::to_string(v) + " ";
        for (auto v : tc.exp) es += std::to_string(v) + " ";
        LOG(WARNING) << "[rank " << rank << "] DIAG reduce " << tc.name
                     << " got=[" << gs << "] exp=[" << es << "]";
        TCHECK(got == tc.exp, "reduce " << tc.name << " values at root");
      }
    }
  }
  ctx->Barrier();

  // --- Reduce of a non-numeric (string) column must fail cleanly, not crash.
  {
    arrow::StringBuilder sb;
    (void) sb.Append("a");
    std::shared_ptr<arrow::Array> sarr;
    (void) sb.Finish(&sarr);
    auto scol = cylon::Column::Make(std::move(sarr));
    std::shared_ptr<cylon::Column> res;
    auto st = comm->Reduce(scol, cylon::net::SUM, 0, &res);
    TCHECK(!st.is_ok(), "non-numeric reduce should fail (got ok)");
  }
  ctx->Barrier();

  // --- Bcast round-trip from root 0 (regression for the FmiTableBcastImpl byte-count
  // fix: a 4x size bug would corrupt or overflow the received buffer).
  {
    std::shared_ptr<cylon::Table> tbl;
    if (rank == 0) tbl = MakeInt64Table(ctx, {42, 43, 44, 45});
    auto st = comm->Bcast(&tbl, /*bcast_root=*/0, ctx);
    TCHECK(st.is_ok(), "bcast status: " << st.get_msg());
    if (st.is_ok()) {
      std::vector<int64_t> exp = {42, 43, 44, 45};
      TCHECK(GetInt64Col(tbl) == exp, "bcast values");
    }
  }
  ctx->Barrier();

  if (g_failures == 0) {
    LOG(INFO) << "[rank " << rank << "] ALL PASS (ws=" << ws << ")";
  } else {
    LOG(ERROR) << "[rank " << rank << "] " << g_failures << " FAILURES (ws=" << ws << ")";
  }

  ctx->Finalize();
  return g_failures == 0 ? 0 : 1;
}