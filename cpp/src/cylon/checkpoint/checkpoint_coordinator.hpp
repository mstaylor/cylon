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

#ifndef CYLON_CHECKPOINT_COORDINATOR_HPP
#define CYLON_CHECKPOINT_COORDINATOR_HPP

#include "checkpoint_traits.hpp"

namespace cylon {
namespace checkpoint {

/// Local (single-process) coordinator — always agrees to checkpoint.
class LocalCoordinator : public CheckpointCoordinator {
 public:
  LocalCoordinator() = default;

  int GetRank() const override { return 0; }
  int GetWorldSize() const override { return 1; }

  Status BeginCheckpoint(uint64_t checkpoint_id,
                         CheckpointDecision *decision) override;
  Status CommitCheckpoint(uint64_t checkpoint_id) override;
  Status AbortCheckpoint(uint64_t checkpoint_id) override;
  Status FindLatestCheckpoint(uint64_t *checkpoint_id) override;
};

/// Distributed coordinator using the Cylon Communicator.
///
/// Uses allgather for voting and barrier for synchronization.
/// All operations are synchronous, matching Cylon's single-threaded model.
class DistributedCoordinator : public CheckpointCoordinator {
 public:
  explicit DistributedCoordinator(std::shared_ptr<CylonContext> ctx);

  int GetRank() const override;
  int GetWorldSize() const override;

  Status BeginCheckpoint(uint64_t checkpoint_id,
                         CheckpointDecision *decision) override;
  Status CommitCheckpoint(uint64_t checkpoint_id) override;
  Status AbortCheckpoint(uint64_t checkpoint_id) override;
  Status FindLatestCheckpoint(uint64_t *checkpoint_id) override;

 private:
  std::shared_ptr<CylonContext> ctx_;

  /// Vote among all workers: returns true if all voted yes
  Status AllVote(bool my_vote, bool *result);
};

}  // namespace checkpoint
}  // namespace cylon

#endif  // CYLON_CHECKPOINT_COORDINATOR_HPP