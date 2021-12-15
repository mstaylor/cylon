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

#include <mpi.h>
#include <memory>

#include <cylon/net/communicator.hpp>
#include <cylon/net/mpi/mpi_communicator.hpp>
#include <cylon/net/mpi/mpi_channel.hpp>
#include <cylon/util/macros.hpp>

#include "cylon/arrow/arrow_buffer.hpp"
#include "cylon/util/arrow_utils.hpp"
#include "cylon/serialize/table_serialize.hpp"
#include "cylon/net/mpi/mpi_operations.hpp"

#include <arrow/ipc/api.h>
#include <arrow/io/api.h>

namespace cylon {
namespace net {
// configs
void MPIConfig::DummyConfig(int dummy) {
  this->AddConfig("Dummy", &dummy);
}
int MPIConfig::GetDummyConfig() {
  return *reinterpret_cast<int *>(this->GetConfig("Dummy"));
}

CommType MPIConfig::Type() {
  return CommType::MPI;
}
std::shared_ptr<MPIConfig> MPIConfig::Make() {
  return std::make_shared<MPIConfig>();
}

MPIConfig::~MPIConfig() = default;

std::unique_ptr<Channel> MPICommunicator::CreateChannel() const {
  return std::make_unique<MPIChannel>();
}

int MPICommunicator::GetRank() const {
  return this->rank;
}
int MPICommunicator::GetWorldSize() const {
  return this->world_size;
}
Status MPICommunicator::Init(const std::shared_ptr<CommConfig> &config) {
  CYLON_UNUSED(config);
  int initialized;
  MPI_Initialized(&initialized);
  if (!initialized) {
    MPI_Init(nullptr, nullptr);
  }

  MPI_Comm_rank(MPI_COMM_WORLD, &this->rank);
  MPI_Comm_size(MPI_COMM_WORLD, &this->world_size);

  return Status::OK();
}
void MPICommunicator::Finalize() {
  int finalized;
  MPI_Finalized(&finalized);
  if (!finalized) {
    MPI_Finalize();
  }
}
void MPICommunicator::Barrier() {
  MPI_Barrier(MPI_COMM_WORLD);
}

CommType MPICommunicator::GetCommType() const {
  return MPI;
}

/*
    |t_0, ..., t_m-1|...|t_0, ..., t_m-1|
     <--- buf_0 --->     <--- buf_n --->
                  to
    |b_0, ..., b_n-1|...|b_0, ..., b_n-1|
     <--- tbl_0 --->     <--- tbl_m --->
 */
std::vector<int32_t> ReshapeDispToPerTable(const std::vector<std::vector<int32_t>> &all_disps) {
  const size_t num_buf = all_disps.size();
  const size_t num_tables = all_disps[0].size(); // == world_size

  std::vector<int32_t> res(num_buf * num_tables, 0);
  for (size_t tid = 0; tid < num_tables; tid++) {
    for (size_t bid = 0; bid < num_buf; bid++) {
      res[tid * num_buf + bid] = all_disps[bid][tid];
    }
  }
  return res;
}

Status MPISyncCommunicator::AllGather(const std::shared_ptr<Table> &table,
                                      std::vector<std::shared_ptr<Table>> *out) const {
  std::shared_ptr<TableSerializer> serializer;
  RETURN_CYLON_STATUS_IF_FAILED(CylonTableSerializer::Make(table, &serializer));
  const auto &ctx = table->GetContext();
  auto *pool = ToArrowPool(ctx);

  const auto &allocator = std::make_shared<ArrowAllocator>(pool);
  std::vector<std::shared_ptr<Buffer>> receive_buffers;

  std::vector<int32_t> buffer_sizes_per_table;
  //  |b_0, ..., b_n-1|...|b_0, ..., b_n-1|
  //   <--- tbl_0 --->     <--- tbl_m --->

  std::vector<std::vector<int32_t>> all_disps;
  //  |t_0, ..., t_m-1|...|t_0, ..., t_m-1|
  //   <--- buf_0 --->     <--- buf_n --->

  RETURN_CYLON_STATUS_IF_FAILED(mpi::AllGather(serializer, allocator, buffer_sizes_per_table,
                                               receive_buffers, all_disps, ctx));

  // need to reshape all_disps for per-table basis
  auto buffer_offsets_per_table = ReshapeDispToPerTable(all_disps);

  const int num_tables = (int) all_disps[0].size();
  return DeserializeTables(ctx, table->get_table()->schema(), num_tables, receive_buffers,
                           buffer_sizes_per_table, buffer_offsets_per_table, out);
}

Status MPISyncCommunicator::Gather(const std::shared_ptr<Table> &table,
                                   int gather_root,
                                   bool gather_from_root,
                                   std::vector<std::shared_ptr<Table>> *out) const {
  std::shared_ptr<TableSerializer> serializer;
  RETURN_CYLON_STATUS_IF_FAILED(CylonTableSerializer::Make(table, &serializer));
  const auto &ctx = table->GetContext();
  auto *pool = ToArrowPool(ctx);

  const auto &allocator = std::make_shared<ArrowAllocator>(pool);
  std::vector<std::shared_ptr<Buffer>> receive_buffers;

  std::vector<int32_t> buffer_sizes_per_table;
  //  |b_0, ..., b_n-1|...|b_0, ..., b_n-1|
  //   <--- tbl_0 --->     <--- tbl_m --->

  std::vector<std::vector<int32_t>> all_disps;
  //  |t_0, ..., t_m-1|...|t_0, ..., t_m-1|
  //   <--- buf_0 --->     <--- buf_n --->

  RETURN_CYLON_STATUS_IF_FAILED(mpi::Gather(serializer, gather_root, gather_from_root,
                                            allocator, buffer_sizes_per_table, receive_buffers,
                                            all_disps, ctx));

  // need to reshape all_disps for per-table basis
  if (mpi::AmIRoot(gather_root, ctx)) {
    auto buffer_offsets_per_table = ReshapeDispToPerTable(all_disps);

    const int num_tables = (int) all_disps[0].size();
    return DeserializeTables(ctx, table->get_table()->schema(), num_tables, receive_buffers,
                             buffer_sizes_per_table, buffer_offsets_per_table, out);
  }
  return Status::OK();
}

Status BcastArrowSchema(const std::shared_ptr<CylonContext> &ctx,
                        std::shared_ptr<arrow::Schema> *schema,
                        int bcast_root) {
  std::shared_ptr<arrow::Buffer> buf;
  bool is_root = mpi::AmIRoot(bcast_root, ctx);

  if (is_root) {
    CYLON_ASSIGN_OR_RAISE(buf, arrow::ipc::SerializeSchema(**schema,
                                                           ToArrowPool(ctx)))
  }
  RETURN_CYLON_STATUS_IF_FAILED(mpi::BcastArrowBuffer(buf, bcast_root, ctx));

  if (!is_root) {
    assert(buf->data());
    arrow::io::BufferReader buf_reader(std::move(buf));
    CYLON_ASSIGN_OR_RAISE(*schema, arrow::ipc::ReadSchema(&buf_reader, nullptr))
  }

  return Status::OK();
}

Status MPISyncCommunicator::Bcast(const std::shared_ptr<CylonContext> &ctx,
                                  std::shared_ptr<Table> *table,
                                  int bcast_root) const {
  std::shared_ptr<arrow::Schema> schema;
  bool is_root = mpi::AmIRoot(bcast_root, ctx);
  auto *pool = ToArrowPool(ctx);

  if (is_root) {
    auto atable = (*table)->get_table();
    schema = atable->schema();
  }
  RETURN_CYLON_STATUS_IF_FAILED(BcastArrowSchema(ctx, &schema, bcast_root));

  std::shared_ptr<TableSerializer> serializer;
  if (is_root) {
    RETURN_CYLON_STATUS_IF_FAILED(CylonTableSerializer::Make(*table, &serializer));
  }
  const auto &allocator = std::make_shared<ArrowAllocator>(pool);
  std::vector<std::shared_ptr<Buffer>> receive_buffers;
  std::vector<int32_t> data_types;
  RETURN_CYLON_STATUS_IF_FAILED(
      mpi::Bcast(serializer, bcast_root, allocator, receive_buffers, data_types, ctx));

  if (!is_root) {
    if (receive_buffers.empty()) {
      std::shared_ptr<arrow::Table> atable;
      RETURN_CYLON_STATUS_IF_ARROW_FAILED(cylon::util::MakeEmptyArrowTable(schema, &atable, pool));
      return Table::FromArrowTable(ctx, std::move(atable), *table);
    } else {
      assert((int) receive_buffers.size() == 3 * schema->num_fields());
      RETURN_CYLON_STATUS_IF_FAILED(DeserializeTable(ctx, schema, receive_buffers, table));
    }
  }
  return Status::OK();
}
}  // namespace net
}  // namespace cylon
