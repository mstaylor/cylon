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

#include "checkpoint_coordinator.hpp"

#include <cylon/scalar.hpp>
#include <cylon/column.hpp>
#include <cylon/net/comm_operations.hpp>

#include <arrow/api.h>

namespace cylon {
namespace checkpoint {

// ─── LocalCoordinator ───

Status LocalCoordinator::BeginCheckpoint(uint64_t /*checkpoint_id*/,
                                         CheckpointDecision *decision) {
  *decision = CheckpointDecision::Proceed;
  return Status::OK();
}

Status LocalCoordinator::CommitCheckpoint(uint64_t /*checkpoint_id*/) {
  return Status::OK();
}

Status LocalCoordinator::AbortCheckpoint(uint64_t /*checkpoint_id*/) {
  return Status::OK();
}

Status LocalCoordinator::FindLatestCheckpoint(uint64_t *checkpoint_id) {
  *checkpoint_id = 0;
  return Status::OK();
}

// ─── DistributedCoordinator ───

DistributedCoordinator::DistributedCoordinator(
    std::shared_ptr<CylonContext> ctx)
    : ctx_(std::move(ctx)) {}

int DistributedCoordinator::GetRank() const {
  return ctx_->GetRank();
}

int DistributedCoordinator::GetWorldSize() const {
  return ctx_->GetWorldSize();
}

Status DistributedCoordinator::BeginCheckpoint(
    uint64_t /*checkpoint_id*/, CheckpointDecision *decision) {
  // All workers vote to proceed using AllReduce with LAND (logical AND).
  // Each sends 1 (yes), result is 1 only if all voted yes.
  auto vote = Scalar::Make(arrow::MakeScalar(static_cast<int32_t>(1)));

  std::shared_ptr<Scalar> result;
  RETURN_CYLON_STATUS_IF_FAILED(
      ctx_->GetCommunicator()->AllReduce(vote, net::LAND, &result));

  auto result_val = std::static_pointer_cast<arrow::Int32Scalar>(result->data());
  *decision = (result_val->value != 0)
                  ? CheckpointDecision::Proceed
                  : CheckpointDecision::Skip;
  return Status::OK();
}

Status DistributedCoordinator::CommitCheckpoint(uint64_t /*checkpoint_id*/) {
  // Barrier to ensure all workers have written their data
  ctx_->Barrier();
  return Status::OK();
}

Status DistributedCoordinator::AbortCheckpoint(uint64_t /*checkpoint_id*/) {
  // Barrier so all workers agree on abort
  ctx_->Barrier();
  return Status::OK();
}

Status DistributedCoordinator::FindLatestCheckpoint(
    uint64_t *checkpoint_id) {
  // AllReduce with MIN: each worker sends its latest checkpoint ID,
  // result is the minimum across all workers.
  auto my_id = Scalar::Make(
      arrow::MakeScalar(static_cast<int64_t>(*checkpoint_id)));

  std::shared_ptr<Scalar> result;
  RETURN_CYLON_STATUS_IF_FAILED(
      ctx_->GetCommunicator()->AllReduce(my_id, net::MIN, &result));

  auto result_val = std::static_pointer_cast<arrow::Int64Scalar>(result->data());
  *checkpoint_id = static_cast<uint64_t>(result_val->value);
  return Status::OK();
}

}  // namespace checkpoint
}  // namespace cylon