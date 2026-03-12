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

#ifndef CYLON_CHECKPOINT_STORAGE_HPP
#define CYLON_CHECKPOINT_STORAGE_HPP

#include "checkpoint_traits.hpp"

namespace cylon {
namespace checkpoint {

/// Filesystem-based checkpoint storage.
///
/// Directory layout:
///   {base_path}/{job_id}/checkpoint_{id}/worker_{rank}/{key}
///   {base_path}/{job_id}/checkpoint_{id}/metadata.json
///
/// Uses a staging directory that is renamed atomically on commit.
class FileSystemStorage : public CheckpointStorage {
 public:
  explicit FileSystemStorage(std::string base_path, std::string job_id);

  Status Write(uint64_t checkpoint_id, int worker_id,
               const std::string &key,
               const uint8_t *data, size_t size) override;

  Status Read(uint64_t checkpoint_id, int worker_id,
              const std::string &key,
              std::vector<uint8_t> *data) override;

  Status Exists(uint64_t checkpoint_id, int worker_id,
                const std::string &key, bool *exists) override;

  Status ListKeys(uint64_t checkpoint_id, int worker_id,
                  std::vector<std::string> *keys) override;

  Status Delete(uint64_t checkpoint_id) override;

  Status ListCheckpoints(std::vector<uint64_t> *checkpoint_ids) override;

  Status CommitWrite(uint64_t checkpoint_id, int worker_id) override;

  Status WriteMetadata(uint64_t checkpoint_id,
                       const CheckpointMetadata &metadata) override;

  Status ReadMetadata(uint64_t checkpoint_id,
                      CheckpointMetadata *metadata) override;

  const std::string &BasePath() const override { return base_path_; }

 private:
  std::string base_path_;
  std::string job_id_;

  std::string CheckpointDir(uint64_t checkpoint_id) const;
  std::string StagingDir(uint64_t checkpoint_id, int worker_id) const;
  std::string FinalDir(uint64_t checkpoint_id, int worker_id) const;
  std::string MetadataPath(uint64_t checkpoint_id) const;

  static Status CreateDirectories(const std::string &path);
  static Status RemoveRecursive(const std::string &path);
};

}  // namespace checkpoint
}  // namespace cylon

#endif  // CYLON_CHECKPOINT_STORAGE_HPP