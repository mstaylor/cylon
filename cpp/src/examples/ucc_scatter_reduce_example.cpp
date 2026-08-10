/*
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/**
 * Standalone validation of the new UCC Scatter / Reduce collective bindings.
 * Bootstraps via the redis OOB context (no mpirun) exactly like the Rivanna
 * experiments — launch with:
 *
 *   CYLON_SESSION_ID=uccsr \
 *   python python/pycylon/run_ucc_with_redis.py -n 4 -r 10.211.55.2:6379 \
 *       -e bin/ucc_scatter_reduce_example
 *
 * All data is built in-memory (no CSV dependency). Exits non-zero if any
 * assertion on this rank fails.
 */

#ifdef BUILD_CYLON_UCC
#include <iostream>
#include <string>
#include <vector>

#include <arrow/api.h>

#include <cylon/ctx/cylon_context.hpp>
#include <cylon/net/ucx/ucx_communicator.hpp>
#include <cylon/table.hpp>
#include <cylon/column.hpp>
#include "net/ucx/redis_ucx_ucc_oob_context.hpp"

namespace {

int g_rank = -1;
int g_failures = 0;

#define UCHECK(cond, msg)                                                     \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::cerr << "[rank " << g_rank << "] FAIL: " << msg << std::endl;       \
      g_failures++;                                                            \
    }                                                                          \
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
  if (!st.is_ok()) std::cerr << "MakeInt64Table failed: " << st.get_msg() << std::endl;
  return t;
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

std::shared_ptr<cylon::Column> MakeInt64Column(const std::vector<int64_t> &vals) {
  arrow::Int64Builder b;
  (void) b.AppendValues(vals);
  std::shared_ptr<arrow::Array> arr;
  (void) b.Finish(&arr);
  return cylon::Column::Make(std::move(arr));
}

// Reduce: root-only delivery; SUM/MAX/MIN over [rank, rank+1, rank+2].
void testColumnReduce(const std::shared_ptr<cylon::CylonContext> &ctx) {
  int rank = ctx->GetRank();
  int ws = ctx->GetWorldSize();
  int reduce_root = 0;
  int64_t S = (int64_t) ws * (ws - 1) / 2;
  struct Case { cylon::net::ReduceOp op; std::vector<int64_t> exp; const char *name; };
  std::vector<Case> cases = {
      {cylon::net::SUM, {S, S + ws, S + 2 * ws}, "SUM"},
      {cylon::net::MAX, {ws - 1, ws, ws + 1}, "MAX"},
      {cylon::net::MIN, {0, 1, 2}, "MIN"},
  };
  for (const auto &tc : cases) {
    auto col = MakeInt64Column({rank, rank + 1, rank + 2});
    std::shared_ptr<cylon::Column> res;
    auto st = ctx->GetCommunicator()->Reduce(col, tc.op, reduce_root, &res);
    UCHECK(st.is_ok(), "reduce " << tc.name << " status: " << st.get_msg());
    if (st.is_ok() && rank == reduce_root) {
      UCHECK(GetInt64Column(res) == tc.exp, "reduce " << tc.name << " values at root");
    }
  }
}

// Scatter (even): root 0, equal 3-row shards; rank r receives [r*100 .. r*100+2].
void testTableScatterEven(const std::shared_ptr<cylon::CylonContext> &ctx) {
  int rank = ctx->GetRank();
  int ws = ctx->GetWorldSize();
  std::vector<std::shared_ptr<cylon::Table>> tables;
  if (rank == 0) {
    for (int r = 0; r < ws; r++)
      tables.push_back(MakeInt64Table(ctx, {r * 100, r * 100 + 1, r * 100 + 2}));
  }
  std::shared_ptr<cylon::Table> out;
  auto st = ctx->GetCommunicator()->Scatter(tables, 0, ctx, &out);
  UCHECK(st.is_ok(), "even scatter status: " << st.get_msg());
  if (st.is_ok()) {
    std::vector<int64_t> exp = {rank * 100, rank * 100 + 1, rank * 100 + 2};
    UCHECK(GetInt64Col(out) == exp, "even scatter values");
  }
}

// Scatter (uneven): non-zero root (ws-1); rank r receives (r+1) rows.
void testTableScatterUneven(const std::shared_ptr<cylon::CylonContext> &ctx) {
  int rank = ctx->GetRank();
  int ws = ctx->GetWorldSize();
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
  auto st = ctx->GetCommunicator()->Scatter(tables, scatter_root, ctx, &out);
  UCHECK(st.is_ok(), "uneven scatter status: " << st.get_msg());
  if (st.is_ok()) {
    std::vector<int64_t> exp;
    for (int i = 0; i <= rank; i++) exp.push_back(rank * 100 + i);
    UCHECK(GetInt64Col(out) == exp, "uneven scatter values (n=" << exp.size() << ")");
  }
}

}  // namespace

int main() {
  // Match the proven pycylon setup (target/rivanna/.../ucc-ucx-redis/cylon_scaling.py):
  // the 2-arg UCCRedisOOBContext(world_size, "tcp://host:port"). Rank is assigned by
  // redis INCR on arrival. Read world_size + redis addr from env (set by the launcher).
  const char *ws_env = std::getenv("CYLON_UCX_OOB_WORLD_SIZE");
  const char *addr_env = std::getenv("CYLON_UCX_OOB_REDIS_ADDR");
  if (ws_env == nullptr || addr_env == nullptr) {
    std::cerr << "set CYLON_UCX_OOB_WORLD_SIZE and CYLON_UCX_OOB_REDIS_ADDR" << std::endl;
    return 2;
  }
  int world_size_env = std::atoi(ws_env);
  std::string redis_addr = "tcp://" + std::string(addr_env);
  std::shared_ptr<cylon::net::UCCOOBContext> oob_ctx =
      std::make_shared<cylon::net::UCCRedisOOBContext>(world_size_env, redis_addr);

  std::shared_ptr<cylon::CylonContext> ctx;
  auto ucc_config = std::make_shared<cylon::net::UCCConfig>(oob_ctx);
  if (!cylon::CylonContext::InitDistributed(ucc_config, &ctx).is_ok()) {
    std::cerr << "ctx init failed!" << std::endl;
    return 2;
  }

  g_rank = ctx->GetRank();
  int ws = ctx->GetWorldSize();

  testColumnReduce(ctx);
  ctx->Barrier();
  testTableScatterEven(ctx);
  ctx->Barrier();
  testTableScatterUneven(ctx);
  ctx->Barrier();

  if (g_failures == 0) {
    std::cout << "[rank " << g_rank << "] ALL PASS (ws=" << ws << ")" << std::endl;
  } else {
    std::cerr << "[rank " << g_rank << "] " << g_failures << " FAILURES (ws=" << ws << ")" << std::endl;
  }

  ctx->Finalize();
  return g_failures == 0 ? 0 : 1;
}
#else
int main() { return 0; }
#endif