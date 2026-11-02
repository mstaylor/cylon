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

#include <cylon/arrow/async_table_exchange.hpp>

#include <chrono>
#include <string>
#include <utility>

#include <cylon/ctx/arrow_memory_pool_utils.hpp>
#include <cylon/serialize/table_serialize.hpp>
#include <cylon/table.hpp>
#include <cylon/util/macros.hpp>

namespace cylon {

namespace {
// The FMI channel sends 2 + headerLength ints but always reads 8, so headers must be 6 ints.
constexpr int kHeaderLength = 6;
constexpr int32_t kEmptyTableSlot = -1;
const int32_t kEmptyTableMarker = 0;
}

AsyncTableExchange::AsyncTableExchange(const std::shared_ptr<CylonContext> &ctx,
                                       std::shared_ptr<arrow::Schema> schema)
    : ctx_(ctx), schema_(std::move(schema)), peers_(ctx->GetNeighbours(false)),
      expected_buffers_(BufferSlots(schema_)) {
  if (peers_.empty()) {
    return;
  }
  allocator_ = std::make_unique<ArrowAllocator>(ToArrowPool(ctx_));
  const std::vector<int> all_ranks = ctx_->GetNeighbours(true);
  all_ = std::make_shared<AllToAll>(ctx_, all_ranks, all_ranks, ctx_->GetNextSequence(), this,
                                    allocator_.get());
  for (int peer : peers_) {
    receives_[peer] = PendingReceive{};
  }
}

AsyncTableExchange::~AsyncTableExchange() {
  Close();
}

Status AsyncTableExchange::Publish(const std::shared_ptr<arrow::Table> &table,
                                   int32_t reference) {
  if (!table->schema()->Equals(*schema_, false)) {
    return {Code::Invalid, "published table schema " + table->schema()->ToString() +
                               " does not match the exchange schema " + schema_->ToString()};
  }
  if (finished_) {
    return {Code::Invalid, "publish after finish on rank " + std::to_string(ctx_->GetRank())};
  }
  if (peers_.empty()) {
    return Status::OK();
  }

  std::shared_ptr<Table> cylon_table;
  RETURN_CYLON_STATUS_IF_FAILED(Table::FromArrowTable(ctx_, table, cylon_table));
  std::shared_ptr<TableSerializer> serializer;
  RETURN_CYLON_STATUS_IF_FAILED(CylonTableSerializer::Make(cylon_table, &serializer));

  const auto &sizes = serializer->getBufferSizes();
  const auto &buffers = serializer->getDataBuffers();
  const int32_t num_buffers = serializer->getNumberOfBuffers();
  const auto num_rows = static_cast<int32_t>(table->num_rows());

  std::vector<int32_t> slots;
  for (int32_t i = 0; i < num_buffers; i++) {
    if (sizes[i] > 0) {
      slots.push_back(i);
    }
  }
  const auto messages = static_cast<int32_t>(std::max<size_t>(slots.size(), 1));

  for (int peer : peers_) {
    if (slots.empty()) {
      int header[kHeaderLength] = {reference, kEmptyTableSlot, num_buffers, num_rows, messages};
      if (all_->insert(&kEmptyTableMarker, sizeof(kEmptyTableMarker), peer, header,
                       kHeaderLength) != 1) {
        return {Code::ExecutionError, "AllToAll refused an empty table for rank " +
                                          std::to_string(peer)};
      }
      sends_inserted_++;
      continue;
    }
    for (int32_t slot : slots) {
      int header[kHeaderLength] = {reference, slot, num_buffers, num_rows, messages};
      if (all_->insert(buffers[slot], sizes[slot], peer, header, kHeaderLength) != 1) {
        return {Code::ExecutionError, "AllToAll refused buffer " + std::to_string(slot) +
                                          " for rank " + std::to_string(peer)};
      }
      sends_inserted_++;
    }
  }
  in_flight_.push_back(std::move(serializer));
  return Status::OK();
}

Status AsyncTableExchange::Drain(std::vector<ReceivedTable> *out) {
  if (all_ && !closed_) {
    all_->isComplete();
  }
  return TakeReceived(out);
}

bool AsyncTableExchange::Idle() const {
  if (!all_ || closed_) {
    return true;
  }
  if (sends_completed_ != sends_inserted_) {
    return false;
  }
  for (const auto &entry : receives_) {
    if (entry.second.messages_expected != 0) {
      return false;
    }
  }
  return true;
}

Status AsyncTableExchange::FinishAndDrain(double timeout_s, std::vector<ReceivedTable> *out) {
  if (all_ && !closed_) {
    if (!finished_) {
      finished_ = true;
      all_->finish();
    }
    const auto deadline = std::chrono::steady_clock::now() +
        std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(timeout_s));
    while (!all_->isComplete()) {
      if (std::chrono::steady_clock::now() > deadline) {
        std::string waiting;
        for (int peer : peers_) {
          if (!finished_peers_.count(peer)) {
            waiting += (waiting.empty() ? "" : ", ") + std::to_string(peer);
          }
        }
        return {Code::ExecutionError, "finish timed out on rank " +
                                          std::to_string(ctx_->GetRank()) + " after " +
                                          std::to_string(timeout_s) + " s waiting for ranks [" +
                                          waiting + "]"};
      }
    }
    Close();
  }
  finished_ = true;
  return TakeReceived(out);
}

bool AsyncTableExchange::onReceiveHeader(int source, int finished, int *buffer, int length) {
  if (finished) {
    finished_peers_.insert(source);
    return true;
  }
  if (length < kHeaderLength) {
    receive_status_ = {Code::Invalid, "header of " + std::to_string(length) + " ints from rank " +
                                          std::to_string(source)};
    return false;
  }
  if (buffer[2] != expected_buffers_) {
    receive_status_ = {Code::Invalid, "rank " + std::to_string(source) + " sent a table of " +
                                          std::to_string(buffer[2]) + " buffers, the schema has " +
                                          std::to_string(expected_buffers_)};
    return false;
  }
  auto &pending = receives_[source];
  if (pending.messages_expected == 0) {
    pending.reference = buffer[0];
    pending.num_rows = buffer[3];
    pending.messages_expected = buffer[4];
    pending.buffers.assign(buffer[2], nullptr);
    pending.sizes.assign(buffer[2], 0);
  }
  pending.slot = buffer[1];
  return true;
}

bool AsyncTableExchange::onReceive(int source, std::shared_ptr<Buffer> buffer, int length) {
  auto &pending = receives_[source];
  if (pending.slot != kEmptyTableSlot) {
    pending.buffers[pending.slot] = std::move(buffer);
    pending.sizes[pending.slot] = length;
  }
  if (++pending.messages_received < pending.messages_expected) {
    return true;
  }

  Status status;
  for (auto &slot_buffer : pending.buffers) {
    if (!slot_buffer && status.is_ok()) {
      status = allocator_->Allocate(0, &slot_buffer);
    }
  }
  std::shared_ptr<Table> table;
  if (status.is_ok()) {
    status = DeserializeTable(ctx_, schema_, pending.buffers, pending.sizes, &table,
                              pending.num_rows);
  }
  if (status.is_ok()) {
    received_.push_back({source, pending.reference, table->get_table()});
  } else {
    receive_status_ = status;
  }
  pending = PendingReceive{};
  return true;
}

bool AsyncTableExchange::onSendComplete(int target, const void *buffer, int length) {
  sends_completed_++;
  CYLON_UNUSED(target);
  CYLON_UNUSED(buffer);
  CYLON_UNUSED(length);
  return true;
}

Status AsyncTableExchange::TakeReceived(std::vector<ReceivedTable> *out) {
  if (!receive_status_.is_ok()) {
    return receive_status_;
  }
  out->insert(out->end(), std::make_move_iterator(received_.begin()),
              std::make_move_iterator(received_.end()));
  received_.clear();
  return Status::OK();
}

void AsyncTableExchange::Close() {
  if (all_ && !closed_) {
    all_->close();
    in_flight_.clear();
  }
  closed_ = true;
}

}  // namespace cylon