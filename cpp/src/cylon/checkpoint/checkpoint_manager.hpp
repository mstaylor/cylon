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

#ifndef CYLON_CHECKPOINT_MANAGER_HPP
#define CYLON_CHECKPOINT_MANAGER_HPP

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "checkpoint_config.hpp"
#include "checkpoint_traits.hpp"

namespace cylon {
namespace checkpoint {

/// Orchestrates the checkpoint lifecycle.
///
/// Manages registered tables, coordinates with other workers,
/// serializes data, and writes to storage. All operations are synchronous.
class CheckpointManager {
 public:
  CheckpointManager(std::shared_ptr<CylonContext> ctx,
                    std::unique_ptr<CheckpointCoordinator> coordinator,
                    std::unique_ptr<CheckpointStorage> storage,
                    std::unique_ptr<CheckpointSerializer> serializer,
                    std::unique_ptr<CheckpointTrigger> trigger,
                    CheckpointConfig config);

  /// Register a table for checkpointing.
  void RegisterTable(const std::string &name,
                     std::shared_ptr<Table> table);

  /// Update a previously registered table.
  void UpdateTable(const std::string &name,
                   std::shared_ptr<Table> table);

  /// Remove a table from checkpointing.
  void UnregisterTable(const std::string &name);

  /// Record an operation (forwarded to the trigger).
  void RecordOperation(OperationType op_type, uint64_t bytes_processed);

  /// Check if a checkpoint should be triggered.
  bool ShouldCheckpoint() const;

  /// Perform a checkpoint of all registered tables.
  /// Returns the checkpoint ID on success.
  Status Checkpoint(uint64_t *checkpoint_id);

  /// Restore from the latest available checkpoint.
  /// Populates the tables map with restored data.
  Status Restore(
      std::unordered_map<std::string, std::shared_ptr<Table>> *tables);

  /// Restore from a specific checkpoint.
  Status RestoreFrom(
      uint64_t checkpoint_id,
      std::unordered_map<std::string, std::shared_ptr<Table>> *tables);

  /// Prune old checkpoints according to retention policy.
  Status Prune();

  /// Get the next checkpoint ID.
  uint64_t NextCheckpointId() const { return next_checkpoint_id_; }

  /// Create a manager for local (single-process) use.
  static Status MakeLocal(const std::shared_ptr<CylonContext> &ctx,
                          const CheckpointConfig &config,
                          std::unique_ptr<CheckpointManager> *manager);

  /// Create a manager for distributed use.
  static Status MakeDistributed(const std::shared_ptr<CylonContext> &ctx,
                                const CheckpointConfig &config,
                                std::unique_ptr<CheckpointManager> *manager);

 private:
  std::shared_ptr<CylonContext> ctx_;
  std::unique_ptr<CheckpointCoordinator> coordinator_;
  std::unique_ptr<CheckpointStorage> storage_;
  std::unique_ptr<CheckpointSerializer> serializer_;
  std::unique_ptr<CheckpointTrigger> trigger_;
  CheckpointConfig config_;
  uint64_t next_checkpoint_id_ = 1;
  std::unordered_map<std::string, std::shared_ptr<Table>> tables_;
};

}  // namespace checkpoint
}  // namespace cylon

#endif  // CYLON_CHECKPOINT_MANAGER_HPP