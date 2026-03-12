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

#ifndef CYLON_CHECKPOINT_CONFIG_HPP
#define CYLON_CHECKPOINT_CONFIG_HPP

#include <string>
#include <chrono>
#include <cstdint>

namespace cylon {
namespace checkpoint {

/// Trigger configuration for when to checkpoint.
struct TriggerConfig {
  /// Checkpoint after this many operations (0 = disabled)
  uint64_t operation_threshold = 100;
  /// Checkpoint after processing this many bytes (0 = disabled)
  uint64_t bytes_threshold = 100 * 1024 * 1024; // 100MB
  /// Checkpoint at this interval (0 = disabled)
  std::chrono::seconds interval{300}; // 5 minutes
  /// For serverless: checkpoint when remaining time drops below this
  std::chrono::seconds time_budget_threshold{0};
  /// For serverless: total time budget
  std::chrono::seconds total_time_budget{0};

  static TriggerConfig HPC(uint64_t ops, uint64_t bytes) {
    TriggerConfig c;
    c.operation_threshold = ops;
    c.bytes_threshold = bytes;
    c.interval = std::chrono::seconds{0};
    return c;
  }

  static TriggerConfig Serverless(std::chrono::seconds budget,
                                  std::chrono::seconds reserve) {
    TriggerConfig c;
    c.operation_threshold = 0;
    c.bytes_threshold = 0;
    c.interval = std::chrono::seconds{0};
    c.time_budget_threshold = reserve;
    c.total_time_budget = budget;
    return c;
  }
};

/// Retention policy for old checkpoints.
struct PrunePolicy {
  /// Maximum number of checkpoints to retain
  int max_checkpoints = 10;
  /// Maximum age in seconds (0 = no age limit)
  int max_age_seconds = 7 * 24 * 60 * 60; // 7 days
  /// Always keep at least this many recent checkpoints
  int min_retain = 3;
};

/// Main checkpoint configuration.
struct CheckpointConfig {
  /// Job identifier
  std::string job_id = "default";
  /// Base path for filesystem storage
  std::string storage_path = "/tmp/cylon_checkpoints";
  /// Trigger configuration
  TriggerConfig trigger;
  /// Retention policy
  PrunePolicy retention;

  CheckpointConfig() = default;

  explicit CheckpointConfig(const std::string &job_id)
      : job_id(job_id) {}

  CheckpointConfig &WithStoragePath(const std::string &path) {
    storage_path = path;
    return *this;
  }

  CheckpointConfig &WithTrigger(const TriggerConfig &t) {
    trigger = t;
    return *this;
  }

  CheckpointConfig &WithRetention(const PrunePolicy &p) {
    retention = p;
    return *this;
  }
};

}  // namespace checkpoint
}  // namespace cylon

#endif  // CYLON_CHECKPOINT_CONFIG_HPP