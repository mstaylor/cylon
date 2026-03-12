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

#include "checkpoint_manager.hpp"

#include <algorithm>

#include <cylon/util/macros.hpp>

#include "checkpoint_coordinator.hpp"
#include "checkpoint_serializer.hpp"
#include "checkpoint_storage.hpp"
#include "checkpoint_trigger.hpp"

namespace cylon {
namespace checkpoint {

CheckpointManager::CheckpointManager(
    std::shared_ptr<CylonContext> ctx,
    std::unique_ptr<CheckpointCoordinator> coordinator,
    std::unique_ptr<CheckpointStorage> storage,
    std::unique_ptr<CheckpointSerializer> serializer,
    std::unique_ptr<CheckpointTrigger> trigger,
    CheckpointConfig config)
    : ctx_(std::move(ctx)),
      coordinator_(std::move(coordinator)),
      storage_(std::move(storage)),
      serializer_(std::move(serializer)),
      trigger_(std::move(trigger)),
      config_(std::move(config)) {}

void CheckpointManager::RegisterTable(const std::string &name,
                                      std::shared_ptr<Table> table) {
  tables_[name] = std::move(table);
}

void CheckpointManager::UpdateTable(const std::string &name,
                                    std::shared_ptr<Table> table) {
  tables_[name] = std::move(table);
}

void CheckpointManager::UnregisterTable(const std::string &name) {
  tables_.erase(name);
}

void CheckpointManager::RecordOperation(OperationType op_type,
                                        uint64_t bytes_processed) {
  trigger_->RecordOperation(op_type, bytes_processed);
}

bool CheckpointManager::ShouldCheckpoint() const {
  return trigger_->ShouldCheckpoint();
}

Status CheckpointManager::Checkpoint(uint64_t *checkpoint_id) {
  auto id = next_checkpoint_id_;

  // Phase 1: Coordinate — all workers agree to checkpoint
  CheckpointDecision decision;
  RETURN_CYLON_STATUS_IF_FAILED(
      coordinator_->BeginCheckpoint(id, &decision));
  if (decision != CheckpointDecision::Proceed) {
    *checkpoint_id = 0;
    return Status::OK();
  }

  // Phase 2: Serialize and write each table to staging
  auto worker_id = coordinator_->GetRank();

  if (!tables_.empty()) {
    for (const auto &[name, table] : tables_) {
      std::vector<uint8_t> data;
      RETURN_CYLON_STATUS_IF_FAILED(serializer_->SerializeTable(table, &data));
      RETURN_CYLON_STATUS_IF_FAILED(
          storage_->Write(id, worker_id, name, data.data(), data.size()));
    }

    // Phase 3: Commit staging to final location
    RETURN_CYLON_STATUS_IF_FAILED(storage_->CommitWrite(id, worker_id));
  }

  // Phase 4: Write metadata (leader only)
  if (coordinator_->IsLeader()) {
    CheckpointMetadata metadata;
    metadata.checkpoint_id = id;
    metadata.world_size = coordinator_->GetWorldSize();
    metadata.status = CheckpointStatus::Committed;
    metadata.created_at = std::chrono::system_clock::now();
    metadata.serializer_format = serializer_->FormatId();
    for (const auto &[name, table] : tables_) {
      metadata.table_names.push_back(name);
    }
    RETURN_CYLON_STATUS_IF_FAILED(storage_->WriteMetadata(id, metadata));
  }

  // Phase 5: Distributed commit — barrier to confirm all workers wrote
  RETURN_CYLON_STATUS_IF_FAILED(coordinator_->CommitCheckpoint(id));

  // Reset trigger and advance checkpoint ID
  trigger_->Reset();
  *checkpoint_id = id;
  ++next_checkpoint_id_;

  // Phase 6: Prune old checkpoints
  auto prune_status = Prune();
  // Don't fail the checkpoint if pruning fails
  (void)prune_status;

  return Status::OK();
}

Status CheckpointManager::Restore(
    std::unordered_map<std::string, std::shared_ptr<Table>> *tables) {
  // Find the latest checkpoint all workers agree on
  std::vector<uint64_t> checkpoints;
  RETURN_CYLON_STATUS_IF_FAILED(storage_->ListCheckpoints(&checkpoints));

  if (checkpoints.empty()) {
    return Status(Code::KeyError, "No checkpoints available for restore");
  }

  // Start with the newest checkpoint
  auto latest_id = checkpoints[0];

  // In distributed mode, agree on the checkpoint to restore from
  if (coordinator_->GetWorldSize() > 1) {
    RETURN_CYLON_STATUS_IF_FAILED(
        coordinator_->FindLatestCheckpoint(&latest_id));
    if (latest_id == 0) {
      return Status(Code::KeyError, "No valid checkpoint found");
    }
  }

  return RestoreFrom(latest_id, tables);
}

Status CheckpointManager::RestoreFrom(
    uint64_t checkpoint_id,
    std::unordered_map<std::string, std::shared_ptr<Table>> *tables) {
  auto worker_id = coordinator_->GetRank();

  // List all keys for this worker's checkpoint
  std::vector<std::string> keys;
  RETURN_CYLON_STATUS_IF_FAILED(
      storage_->ListKeys(checkpoint_id, worker_id, &keys));

  tables->clear();
  for (const auto &key : keys) {
    std::vector<uint8_t> data;
    RETURN_CYLON_STATUS_IF_FAILED(
        storage_->Read(checkpoint_id, worker_id, key, &data));

    std::shared_ptr<Table> table;
    RETURN_CYLON_STATUS_IF_FAILED(
        serializer_->DeserializeTable(data, ctx_, &table));

    (*tables)[key] = std::move(table);
  }

  // Update next checkpoint ID to continue from where we left off
  next_checkpoint_id_ = checkpoint_id + 1;

  return Status::OK();
}

Status CheckpointManager::Prune() {
  std::vector<uint64_t> checkpoints;
  RETURN_CYLON_STATUS_IF_FAILED(storage_->ListCheckpoints(&checkpoints));

  // Keep at least min_retain checkpoints
  auto retain = static_cast<size_t>(config_.retention.min_retain);
  if (checkpoints.size() <= retain) {
    return Status::OK();
  }

  // Delete checkpoints beyond max_checkpoints, preserving min_retain
  auto max_keep = static_cast<size_t>(config_.retention.max_checkpoints);
  size_t delete_from = std::max(max_keep, retain);

  for (size_t i = delete_from; i < checkpoints.size(); ++i) {
    auto status = storage_->Delete(checkpoints[i]);
    if (!status.is_ok()) {
      // Log but don't fail — pruning is best-effort
    }
  }

  return Status::OK();
}

Status CheckpointManager::MakeLocal(
    const std::shared_ptr<CylonContext> &ctx,
    const CheckpointConfig &config,
    std::unique_ptr<CheckpointManager> *manager) {
  auto coordinator = std::make_unique<LocalCoordinator>();
  auto storage = std::make_unique<FileSystemStorage>(
      config.storage_path, config.job_id);
  auto serializer = std::make_unique<ArrowIpcSerializer>();
  auto trigger = CompositeTrigger::FromConfig(config.trigger);

  *manager = std::make_unique<CheckpointManager>(
      ctx, std::move(coordinator), std::move(storage),
      std::move(serializer), std::move(trigger), config);
  return Status::OK();
}

Status CheckpointManager::MakeDistributed(
    const std::shared_ptr<CylonContext> &ctx,
    const CheckpointConfig &config,
    std::unique_ptr<CheckpointManager> *manager) {
  if (!ctx->IsDistributed()) {
    return Status(Code::Invalid,
                  "Cannot create distributed checkpoint manager "
                  "without distributed context");
  }

  auto coordinator = std::make_unique<DistributedCoordinator>(ctx);
  auto storage = std::make_unique<FileSystemStorage>(
      config.storage_path, config.job_id);
  auto serializer = std::make_unique<ArrowIpcSerializer>();
  auto trigger = CompositeTrigger::FromConfig(config.trigger);

  *manager = std::make_unique<CheckpointManager>(
      ctx, std::move(coordinator), std::move(storage),
      std::move(serializer), std::move(trigger), config);
  return Status::OK();
}

}  // namespace checkpoint
}  // namespace cylon