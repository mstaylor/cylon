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

#ifndef CYLON_CHECKPOINT_TRAITS_HPP
#define CYLON_CHECKPOINT_TRAITS_HPP

#include <memory>
#include <string>
#include <vector>

#include <cylon/status.hpp>
#include <cylon/table.hpp>
#include <cylon/ctx/cylon_context.hpp>

#include "checkpoint_types.hpp"

namespace cylon {
namespace checkpoint {

/// Distributed coordination for checkpoints.
///
/// Handles when and how to synchronize checkpoints across workers.
/// All methods are synchronous.
class CheckpointCoordinator {
 public:
  virtual ~CheckpointCoordinator() = default;

  /// Get this worker's rank
  virtual int GetRank() const = 0;

  /// Total number of workers
  virtual int GetWorldSize() const = 0;

  /// Begin a checkpoint — coordinate with other workers (barrier + vote)
  virtual Status BeginCheckpoint(uint64_t checkpoint_id,
                                 CheckpointDecision *decision) = 0;

  /// Commit a checkpoint after all workers have written data
  virtual Status CommitCheckpoint(uint64_t checkpoint_id) = 0;

  /// Abort a checkpoint (rollback)
  virtual Status AbortCheckpoint(uint64_t checkpoint_id) = 0;

  /// Find the latest checkpoint that all workers agree on
  virtual Status FindLatestCheckpoint(uint64_t *checkpoint_id) = 0;

  /// Check if this worker is the leader (rank 0)
  virtual bool IsLeader() const { return GetRank() == 0; }
};

/// Handles reading and writing checkpoint data to storage.
///
/// All methods are synchronous filesystem or object store I/O.
class CheckpointStorage {
 public:
  virtual ~CheckpointStorage() = default;

  /// Write checkpoint data
  virtual Status Write(uint64_t checkpoint_id, int worker_id,
                       const std::string &key,
                       const uint8_t *data, size_t size) = 0;

  /// Read checkpoint data
  virtual Status Read(uint64_t checkpoint_id, int worker_id,
                      const std::string &key,
                      std::vector<uint8_t> *data) = 0;

  /// Check if checkpoint data exists
  virtual Status Exists(uint64_t checkpoint_id, int worker_id,
                        const std::string &key, bool *exists) = 0;

  /// List all keys for a checkpoint
  virtual Status ListKeys(uint64_t checkpoint_id, int worker_id,
                          std::vector<std::string> *keys) = 0;

  /// Delete a checkpoint
  virtual Status Delete(uint64_t checkpoint_id) = 0;

  /// List all available checkpoints (newest first)
  virtual Status ListCheckpoints(std::vector<uint64_t> *checkpoint_ids) = 0;

  /// Atomic move from staging to final location
  virtual Status CommitWrite(uint64_t checkpoint_id, int worker_id) = 0;

  /// Write checkpoint metadata
  virtual Status WriteMetadata(uint64_t checkpoint_id,
                               const CheckpointMetadata &metadata) = 0;

  /// Read checkpoint metadata
  virtual Status ReadMetadata(uint64_t checkpoint_id,
                              CheckpointMetadata *metadata) = 0;

  /// Get the base path/URI for this storage
  virtual const std::string &BasePath() const = 0;
};

/// Handles serialization of tables to/from bytes.
///
/// Uses Arrow IPC format for efficient zero-copy serialization.
class CheckpointSerializer {
 public:
  virtual ~CheckpointSerializer() = default;

  /// Serialize a table to bytes
  virtual Status SerializeTable(const std::shared_ptr<Table> &table,
                                std::vector<uint8_t> *data) = 0;

  /// Deserialize bytes to a table
  virtual Status DeserializeTable(const std::vector<uint8_t> &data,
                                  const std::shared_ptr<CylonContext> &ctx,
                                  std::shared_ptr<Table> *table) = 0;

  /// Get the format identifier
  virtual const std::string &FormatId() const = 0;
};

/// Determines when to checkpoint.
///
/// Tracks operations and decides based on configured thresholds.
class CheckpointTrigger {
 public:
  virtual ~CheckpointTrigger() = default;

  /// Update trigger state after an operation
  virtual void RecordOperation(OperationType op_type,
                               uint64_t bytes_processed) = 0;

  /// Check if we should checkpoint now
  virtual bool ShouldCheckpoint() const = 0;

  /// Force a checkpoint
  virtual void ForceCheckpoint() = 0;

  /// Reset trigger state after successful checkpoint
  virtual void Reset() = 0;

  /// Get urgency level
  virtual CheckpointUrgency Urgency() const = 0;

  /// Get current context for decision making
  virtual CheckpointContext GetContext() const = 0;
};

}  // namespace checkpoint
}  // namespace cylon

#endif  // CYLON_CHECKPOINT_TRAITS_HPP