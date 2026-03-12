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

#ifndef CYLON_CHECKPOINT_SERIALIZER_HPP
#define CYLON_CHECKPOINT_SERIALIZER_HPP

#include "checkpoint_traits.hpp"

namespace cylon {
namespace checkpoint {

/// Arrow IPC serializer for checkpointing tables.
///
/// Serializes Arrow tables using the IPC stream format, which includes
/// the schema and all record batches in a self-describing binary format.
class ArrowIpcSerializer : public CheckpointSerializer {
 public:
  ArrowIpcSerializer() = default;

  Status SerializeTable(const std::shared_ptr<Table> &table,
                        std::vector<uint8_t> *data) override;

  Status DeserializeTable(const std::vector<uint8_t> &data,
                          const std::shared_ptr<CylonContext> &ctx,
                          std::shared_ptr<Table> *table) override;

  const std::string &FormatId() const override {
    static const std::string id = "arrow_ipc";
    return id;
  }
};

}  // namespace checkpoint
}  // namespace cylon

#endif  // CYLON_CHECKPOINT_SERIALIZER_HPP