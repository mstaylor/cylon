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

#include "checkpoint_trigger.hpp"

namespace cylon {
namespace checkpoint {

// ─── OperationCountTrigger ───

OperationCountTrigger::OperationCountTrigger(uint64_t op_threshold,
                                             uint64_t bytes_threshold)
    : op_threshold_(op_threshold),
      bytes_threshold_(bytes_threshold),
      last_checkpoint_time_(std::chrono::steady_clock::now()),
      start_time_(std::chrono::steady_clock::now()) {}

void OperationCountTrigger::RecordOperation(OperationType /*op_type*/,
                                            uint64_t bytes_processed) {
  ++op_count_;
  bytes_count_ += bytes_processed;
}

bool OperationCountTrigger::ShouldCheckpoint() const {
  if (forced_) return true;
  if (op_threshold_ > 0 && op_count_ >= op_threshold_) return true;
  if (bytes_threshold_ > 0 && bytes_count_ >= bytes_threshold_) return true;
  return false;
}

void OperationCountTrigger::ForceCheckpoint() { forced_ = true; }

void OperationCountTrigger::Reset() {
  op_count_ = 0;
  bytes_count_ = 0;
  forced_ = false;
  last_checkpoint_time_ = std::chrono::steady_clock::now();
}

CheckpointUrgency OperationCountTrigger::Urgency() const {
  if (forced_) return CheckpointUrgency::Critical;
  if (op_threshold_ > 0 && op_count_ >= op_threshold_ * 2)
    return CheckpointUrgency::High;
  if (ShouldCheckpoint()) return CheckpointUrgency::Normal;
  return CheckpointUrgency::None;
}

CheckpointContext OperationCountTrigger::GetContext() const {
  return {op_count_, bytes_count_, last_checkpoint_time_, start_time_};
}

// ─── IntervalTrigger ───

IntervalTrigger::IntervalTrigger(std::chrono::seconds interval)
    : interval_(interval),
      last_checkpoint_time_(std::chrono::steady_clock::now()),
      start_time_(std::chrono::steady_clock::now()) {}

void IntervalTrigger::RecordOperation(OperationType /*op_type*/,
                                      uint64_t bytes_processed) {
  ++op_count_;
  bytes_count_ += bytes_processed;
}

bool IntervalTrigger::ShouldCheckpoint() const {
  if (forced_) return true;
  auto elapsed = std::chrono::steady_clock::now() - last_checkpoint_time_;
  return elapsed >= interval_;
}

void IntervalTrigger::ForceCheckpoint() { forced_ = true; }

void IntervalTrigger::Reset() {
  op_count_ = 0;
  bytes_count_ = 0;
  forced_ = false;
  last_checkpoint_time_ = std::chrono::steady_clock::now();
}

CheckpointUrgency IntervalTrigger::Urgency() const {
  if (forced_) return CheckpointUrgency::Critical;
  auto elapsed = std::chrono::steady_clock::now() - last_checkpoint_time_;
  if (elapsed >= interval_ * 3) return CheckpointUrgency::High;
  if (ShouldCheckpoint()) return CheckpointUrgency::Normal;
  return CheckpointUrgency::None;
}

CheckpointContext IntervalTrigger::GetContext() const {
  return {op_count_, bytes_count_, last_checkpoint_time_, start_time_};
}

// ─── CompositeTrigger ───

void CompositeTrigger::AddTrigger(
    std::unique_ptr<CheckpointTrigger> trigger) {
  triggers_.push_back(std::move(trigger));
}

void CompositeTrigger::RecordOperation(OperationType op_type,
                                       uint64_t bytes_processed) {
  for (auto &trigger : triggers_) {
    trigger->RecordOperation(op_type, bytes_processed);
  }
}

bool CompositeTrigger::ShouldCheckpoint() const {
  for (const auto &trigger : triggers_) {
    if (trigger->ShouldCheckpoint()) return true;
  }
  return false;
}

void CompositeTrigger::ForceCheckpoint() {
  for (auto &trigger : triggers_) {
    trigger->ForceCheckpoint();
  }
}

void CompositeTrigger::Reset() {
  for (auto &trigger : triggers_) {
    trigger->Reset();
  }
}

CheckpointUrgency CompositeTrigger::Urgency() const {
  auto max_urgency = CheckpointUrgency::None;
  for (const auto &trigger : triggers_) {
    auto u = trigger->Urgency();
    if (u > max_urgency) max_urgency = u;
  }
  return max_urgency;
}

CheckpointContext CompositeTrigger::GetContext() const {
  // Return context from the first trigger
  if (!triggers_.empty()) {
    return triggers_[0]->GetContext();
  }
  return {};
}

std::unique_ptr<CompositeTrigger> CompositeTrigger::FromConfig(
    const TriggerConfig &config) {
  auto composite = std::make_unique<CompositeTrigger>();

  if (config.operation_threshold > 0 || config.bytes_threshold > 0) {
    composite->AddTrigger(std::make_unique<OperationCountTrigger>(
        config.operation_threshold, config.bytes_threshold));
  }

  if (config.interval.count() > 0) {
    composite->AddTrigger(
        std::make_unique<IntervalTrigger>(config.interval));
  }

  return composite;
}

}  // namespace checkpoint
}  // namespace cylon