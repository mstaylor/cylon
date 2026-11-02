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

#ifndef CYLON_ARROW_ASYNC_TABLE_EXCHANGE_HPP_
#define CYLON_ARROW_ASYNC_TABLE_EXCHANGE_HPP_

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <arrow/api.h>

#include "cylon/arrow/arrow_buffer.hpp"
#include "cylon/ctx/cylon_context.hpp"
#include "cylon/net/ops/all_to_all.hpp"
#include "cylon/net/serialize.hpp"
#include "cylon/status.hpp"

namespace cylon {

struct ReceivedTable {
  int source;
  int32_t reference;
  std::shared_ptr<arrow::Table> table;
};

/**
 * Publish-on-completion table exchange. One AllToAll stays open for the whole
 * run: Publish queues a table for every peer and returns, Drain progresses the
 * channel and hands back whatever peers' tables have arrived since the last
 * call, Idle says whether this rank's sends have all completed and no table is
 * half received, and FinishAndDrain progresses until every peer has finished
 * or the timeout passes, naming the peers it was still waiting for. Tables
 * travel in the CylonTableSerializer layout, the one AllGather uses, so nested
 * types such as fixed size lists survive the trip.
 *
 * Single-threaded like the rest of Cylon: every call must come from the thread
 * that owns the context. With no peers every call is a no-op.
 */
class AsyncTableExchange : public ReceiveCallback {
 public:
  AsyncTableExchange(const std::shared_ptr<CylonContext> &ctx,
                     std::shared_ptr<arrow::Schema> schema);
  ~AsyncTableExchange() override;

  Status Publish(const std::shared_ptr<arrow::Table> &table, int32_t reference);
  Status Drain(std::vector<ReceivedTable> *out);
  bool Idle() const;
  Status FinishAndDrain(double timeout_s, std::vector<ReceivedTable> *out);

  bool onReceive(int source, std::shared_ptr<Buffer> buffer, int length) override;
  bool onReceiveHeader(int source, int finished, int *buffer, int length) override;
  bool onSendComplete(int target, const void *buffer, int length) override;

 private:
  struct PendingReceive {
    int32_t reference = 0;
    int32_t num_rows = 0;
    int32_t slot = 0;
    int32_t messages_expected = 0;
    int32_t messages_received = 0;
    std::vector<std::shared_ptr<Buffer>> buffers;
    std::vector<int32_t> sizes;
  };

  Status TakeReceived(std::vector<ReceivedTable> *out);
  void Close();

  std::shared_ptr<CylonContext> ctx_;
  std::shared_ptr<arrow::Schema> schema_;
  std::vector<int> peers_;
  int32_t expected_buffers_;
  int64_t sends_inserted_ = 0;
  int64_t sends_completed_ = 0;
  std::unordered_set<int> finished_peers_;
  std::unique_ptr<ArrowAllocator> allocator_;
  std::shared_ptr<AllToAll> all_;
  std::vector<std::shared_ptr<TableSerializer>> in_flight_;
  std::unordered_map<int, PendingReceive> receives_;
  std::vector<ReceivedTable> received_;
  Status receive_status_;
  bool finished_ = false;
  bool closed_ = false;
};

}  // namespace cylon

#endif  // CYLON_ARROW_ASYNC_TABLE_EXCHANGE_HPP_