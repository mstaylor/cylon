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

#include "checkpoint_storage.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>
#include <ftw.h>

namespace cylon {
namespace checkpoint {

FileSystemStorage::FileSystemStorage(std::string base_path, std::string job_id)
    : base_path_(std::move(base_path)), job_id_(std::move(job_id)) {}

std::string FileSystemStorage::CheckpointDir(uint64_t checkpoint_id) const {
  return base_path_ + "/" + job_id_ + "/checkpoint_" +
         std::to_string(checkpoint_id);
}

std::string FileSystemStorage::StagingDir(uint64_t checkpoint_id,
                                          int worker_id) const {
  return CheckpointDir(checkpoint_id) + "/staging_worker_" +
         std::to_string(worker_id);
}

std::string FileSystemStorage::FinalDir(uint64_t checkpoint_id,
                                        int worker_id) const {
  return CheckpointDir(checkpoint_id) + "/worker_" +
         std::to_string(worker_id);
}

std::string FileSystemStorage::MetadataPath(uint64_t checkpoint_id) const {
  return CheckpointDir(checkpoint_id) + "/metadata.json";
}

Status FileSystemStorage::CreateDirectories(const std::string &path) {
  std::string current;
  std::istringstream ss(path);
  std::string token;

  while (std::getline(ss, token, '/')) {
    if (token.empty() && current.empty()) {
      current = "/";
      continue;
    }
    if (!current.empty() && current.back() != '/') {
      current += "/";
    }
    current += token;
    if (::mkdir(current.c_str(), 0755) != 0 && errno != EEXIST) {
      return Status(Code::IOError,
                    "Failed to create directory: " + current +
                        " (" + std::strerror(errno) + ")");
    }
  }
  return Status::OK();
}

static int remove_callback(const char *fpath, const struct stat * /*sb*/,
                            int /*typeflag*/, struct FTW * /*ftwbuf*/) {
  return ::remove(fpath);
}

Status FileSystemStorage::RemoveRecursive(const std::string &path) {
  if (::nftw(path.c_str(), remove_callback, 64,
             FTW_DEPTH | FTW_PHYS) != 0) {
    if (errno != ENOENT) {
      return Status(Code::IOError,
                    "Failed to remove directory: " + path +
                        " (" + std::strerror(errno) + ")");
    }
  }
  return Status::OK();
}

Status FileSystemStorage::Write(uint64_t checkpoint_id, int worker_id,
                                const std::string &key,
                                const uint8_t *data, size_t size) {
  auto dir = StagingDir(checkpoint_id, worker_id);
  auto status = CreateDirectories(dir);
  if (!status.is_ok()) return status;

  auto file_path = dir + "/" + key;
  std::ofstream ofs(file_path, std::ios::binary | std::ios::trunc);
  if (!ofs) {
    return Status(Code::IOError, "Failed to open file for writing: " + file_path);
  }

  ofs.write(reinterpret_cast<const char *>(data), size);
  if (!ofs) {
    return Status(Code::IOError, "Failed to write data to: " + file_path);
  }
  ofs.close();

  return Status::OK();
}

Status FileSystemStorage::Read(uint64_t checkpoint_id, int worker_id,
                               const std::string &key,
                               std::vector<uint8_t> *data) {
  auto file_path = FinalDir(checkpoint_id, worker_id) + "/" + key;
  std::ifstream ifs(file_path, std::ios::binary | std::ios::ate);
  if (!ifs) {
    return Status(Code::IOError, "Failed to open file for reading: " + file_path);
  }

  auto file_size = ifs.tellg();
  ifs.seekg(0, std::ios::beg);

  data->resize(static_cast<size_t>(file_size));
  ifs.read(reinterpret_cast<char *>(data->data()), file_size);
  if (!ifs) {
    return Status(Code::IOError, "Failed to read data from: " + file_path);
  }

  return Status::OK();
}

Status FileSystemStorage::Exists(uint64_t checkpoint_id, int worker_id,
                                 const std::string &key, bool *exists) {
  auto file_path = FinalDir(checkpoint_id, worker_id) + "/" + key;
  struct stat st{};
  *exists = (::stat(file_path.c_str(), &st) == 0);
  return Status::OK();
}

Status FileSystemStorage::ListKeys(uint64_t checkpoint_id, int worker_id,
                                   std::vector<std::string> *keys) {
  auto dir = FinalDir(checkpoint_id, worker_id);
  DIR *dp = ::opendir(dir.c_str());
  if (!dp) {
    if (errno == ENOENT) {
      keys->clear();
      return Status::OK();
    }
    return Status(Code::IOError, "Failed to open directory: " + dir);
  }

  keys->clear();
  struct dirent *entry;
  while ((entry = ::readdir(dp)) != nullptr) {
    std::string name = entry->d_name;
    if (name != "." && name != "..") {
      keys->push_back(std::move(name));
    }
  }
  ::closedir(dp);
  return Status::OK();
}

Status FileSystemStorage::Delete(uint64_t checkpoint_id) {
  return RemoveRecursive(CheckpointDir(checkpoint_id));
}

Status FileSystemStorage::ListCheckpoints(
    std::vector<uint64_t> *checkpoint_ids) {
  auto job_dir = base_path_ + "/" + job_id_;
  DIR *dp = ::opendir(job_dir.c_str());
  if (!dp) {
    if (errno == ENOENT) {
      checkpoint_ids->clear();
      return Status::OK();
    }
    return Status(Code::IOError, "Failed to open directory: " + job_dir);
  }

  checkpoint_ids->clear();
  struct dirent *entry;
  const std::string prefix = "checkpoint_";
  while ((entry = ::readdir(dp)) != nullptr) {
    std::string name = entry->d_name;
    if (name.substr(0, prefix.size()) == prefix) {
      try {
        auto id = std::stoull(name.substr(prefix.size()));
        checkpoint_ids->push_back(id);
      } catch (...) {
        // skip malformed directory names
      }
    }
  }
  ::closedir(dp);

  // Sort newest first
  std::sort(checkpoint_ids->begin(), checkpoint_ids->end(),
            std::greater<uint64_t>());
  return Status::OK();
}

Status FileSystemStorage::CommitWrite(uint64_t checkpoint_id,
                                      int worker_id) {
  auto staging = StagingDir(checkpoint_id, worker_id);
  auto final_dir = FinalDir(checkpoint_id, worker_id);

  // Atomic rename from staging to final
  if (::rename(staging.c_str(), final_dir.c_str()) != 0) {
    return Status(Code::IOError,
                  "Failed to commit checkpoint: rename " + staging +
                      " -> " + final_dir +
                      " (" + std::strerror(errno) + ")");
  }
  return Status::OK();
}

Status FileSystemStorage::WriteMetadata(uint64_t checkpoint_id,
                                        const CheckpointMetadata &metadata) {
  auto dir = CheckpointDir(checkpoint_id);
  auto status = CreateDirectories(dir);
  if (!status.is_ok()) return status;

  auto path = MetadataPath(checkpoint_id);
  std::ofstream ofs(path, std::ios::trunc);
  if (!ofs) {
    return Status(Code::IOError, "Failed to open metadata file: " + path);
  }

  // Simple JSON serialization
  ofs << "{\n";
  ofs << "  \"checkpoint_id\": " << metadata.checkpoint_id << ",\n";
  ofs << "  \"world_size\": " << metadata.world_size << ",\n";
  ofs << "  \"status\": " << static_cast<int>(metadata.status) << ",\n";
  ofs << "  \"serializer_format\": \"" << metadata.serializer_format << "\",\n";
  ofs << "  \"table_names\": [";
  for (size_t i = 0; i < metadata.table_names.size(); ++i) {
    if (i > 0) ofs << ", ";
    ofs << "\"" << metadata.table_names[i] << "\"";
  }
  ofs << "]\n";
  ofs << "}\n";

  if (!ofs) {
    return Status(Code::IOError, "Failed to write metadata to: " + path);
  }
  return Status::OK();
}

Status FileSystemStorage::ReadMetadata(uint64_t checkpoint_id,
                                       CheckpointMetadata *metadata) {
  auto path = MetadataPath(checkpoint_id);
  std::ifstream ifs(path);
  if (!ifs) {
    return Status(Code::IOError, "Failed to open metadata file: " + path);
  }

  // Simple JSON parsing — look for key-value pairs
  std::string content((std::istreambuf_iterator<char>(ifs)),
                      std::istreambuf_iterator<char>());

  metadata->checkpoint_id = checkpoint_id;

  // Parse world_size
  auto pos = content.find("\"world_size\":");
  if (pos != std::string::npos) {
    metadata->world_size = std::stoi(content.substr(pos + 13));
  }

  // Parse status
  pos = content.find("\"status\":");
  if (pos != std::string::npos) {
    metadata->status =
        static_cast<CheckpointStatus>(std::stoi(content.substr(pos + 9)));
  }

  // Parse serializer_format
  pos = content.find("\"serializer_format\": \"");
  if (pos != std::string::npos) {
    auto start = pos + 22;
    auto end = content.find('"', start);
    if (end != std::string::npos) {
      metadata->serializer_format = content.substr(start, end - start);
    }
  }

  // Parse table_names
  metadata->table_names.clear();
  pos = content.find("\"table_names\":");
  if (pos != std::string::npos) {
    auto bracket_start = content.find('[', pos);
    auto bracket_end = content.find(']', bracket_start);
    if (bracket_start != std::string::npos &&
        bracket_end != std::string::npos) {
      auto names_str =
          content.substr(bracket_start + 1, bracket_end - bracket_start - 1);
      size_t search_pos = 0;
      while (true) {
        auto q1 = names_str.find('"', search_pos);
        if (q1 == std::string::npos) break;
        auto q2 = names_str.find('"', q1 + 1);
        if (q2 == std::string::npos) break;
        metadata->table_names.push_back(names_str.substr(q1 + 1, q2 - q1 - 1));
        search_pos = q2 + 1;
      }
    }
  }

  return Status::OK();
}

}  // namespace checkpoint
}  // namespace cylon