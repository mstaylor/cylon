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

#ifndef CYLON_CHECKPOINT_TYPES_HPP
#define CYLON_CHECKPOINT_TYPES_HPP

#include <cstdint>
#include <string>
#include <chrono>
#include <vector>

namespace cylon {
namespace checkpoint {

/// Operation types tracked by the trigger.
enum OperationType {
  Join,
  Filter,
  Sort,
  GroupBy,
  SetOp,
  Shuffle,
  Other
};

/// Checkpoint status.
enum CheckpointStatus {
  InProgress,
  Committed,
  Failed,
  Aborted
};

/// Urgency level for checkpoint triggers.
enum CheckpointUrgency {
  None,
  Low,
  Normal,
  High,
  Critical
};

/// Metadata stored alongside each checkpoint.
struct CheckpointMetadata {
  uint64_t checkpoint_id = 0;
  int world_size = 0;
  CheckpointStatus status = CheckpointStatus::InProgress;
  std::chrono::system_clock::time_point created_at;
  std::vector<std::string> table_names;
  std::string serializer_format;
};

/// Context provided to the trigger for decision making.
struct CheckpointContext {
  uint64_t operations_since_checkpoint = 0;
  uint64_t bytes_since_checkpoint = 0;
  std::chrono::steady_clock::time_point last_checkpoint_time;
  std::chrono::steady_clock::time_point start_time;
};

/// Decision from coordinator about whether to proceed with a checkpoint.
enum CheckpointDecision {
  Proceed,
  Skip,
  Abort
};

}  // namespace checkpoint
}  // namespace cylon

#endif  // CYLON_CHECKPOINT_TYPES_HPP