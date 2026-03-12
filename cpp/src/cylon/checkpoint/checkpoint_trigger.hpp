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

#ifndef CYLON_CHECKPOINT_TRIGGER_HPP
#define CYLON_CHECKPOINT_TRIGGER_HPP

#include "checkpoint_traits.hpp"
#include "checkpoint_config.hpp"

namespace cylon {
namespace checkpoint {

/// Trigger based on operation count and bytes processed.
class OperationCountTrigger : public CheckpointTrigger {
 public:
  explicit OperationCountTrigger(uint64_t op_threshold,
                                 uint64_t bytes_threshold = 0);

  void RecordOperation(OperationType op_type,
                       uint64_t bytes_processed) override;
  bool ShouldCheckpoint() const override;
  void ForceCheckpoint() override;
  void Reset() override;
  CheckpointUrgency Urgency() const override;
  CheckpointContext GetContext() const override;

 private:
  uint64_t op_threshold_;
  uint64_t bytes_threshold_;
  uint64_t op_count_ = 0;
  uint64_t bytes_count_ = 0;
  bool forced_ = false;
  std::chrono::steady_clock::time_point last_checkpoint_time_;
  std::chrono::steady_clock::time_point start_time_;
};

/// Trigger based on time interval.
class IntervalTrigger : public CheckpointTrigger {
 public:
  explicit IntervalTrigger(std::chrono::seconds interval);

  void RecordOperation(OperationType op_type,
                       uint64_t bytes_processed) override;
  bool ShouldCheckpoint() const override;
  void ForceCheckpoint() override;
  void Reset() override;
  CheckpointUrgency Urgency() const override;
  CheckpointContext GetContext() const override;

 private:
  std::chrono::seconds interval_;
  uint64_t op_count_ = 0;
  uint64_t bytes_count_ = 0;
  bool forced_ = false;
  std::chrono::steady_clock::time_point last_checkpoint_time_;
  std::chrono::steady_clock::time_point start_time_;
};

/// Composite trigger that combines multiple triggers (any-of semantics).
class CompositeTrigger : public CheckpointTrigger {
 public:
  CompositeTrigger() = default;

  void AddTrigger(std::unique_ptr<CheckpointTrigger> trigger);

  void RecordOperation(OperationType op_type,
                       uint64_t bytes_processed) override;
  bool ShouldCheckpoint() const override;
  void ForceCheckpoint() override;
  void Reset() override;
  CheckpointUrgency Urgency() const override;
  CheckpointContext GetContext() const override;

  /// Create a composite trigger from a TriggerConfig.
  static std::unique_ptr<CompositeTrigger> FromConfig(
      const TriggerConfig &config);

 private:
  std::vector<std::unique_ptr<CheckpointTrigger>> triggers_;
};

}  // namespace checkpoint
}  // namespace cylon

#endif  // CYLON_CHECKPOINT_TRIGGER_HPP