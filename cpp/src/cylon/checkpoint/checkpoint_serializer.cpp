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

#include "checkpoint_serializer.hpp"

#include <cstring>

#include <arrow/api.h>
#include <arrow/io/memory.h>
#include <arrow/ipc/reader.h>
#include <arrow/ipc/writer.h>

#include <cylon/util/macros.hpp>

namespace cylon {
namespace checkpoint {

Status ArrowIpcSerializer::SerializeTable(const std::shared_ptr<Table> &table,
                                          std::vector<uint8_t> *data) {
  const auto &arrow_table = table->get_table();
  if (!arrow_table) {
    return {Code::Invalid, "Table has no underlying Arrow table"};
  }

  // Create an in-memory output stream
  CYLON_ASSIGN_OR_RAISE(auto stream,
                        arrow::io::BufferOutputStream::Create());

  // Create IPC stream writer
  CYLON_ASSIGN_OR_RAISE(auto writer,
                        arrow::ipc::MakeStreamWriter(stream,
                                                     arrow_table->schema()));

  // Write the table as record batches using TableBatchReader
  arrow::TableBatchReader batch_reader(*arrow_table);
  std::shared_ptr<arrow::RecordBatch> batch;
  while (true) {
    RETURN_CYLON_STATUS_IF_ARROW_FAILED(batch_reader.ReadNext(&batch));
    if (!batch) break;
    RETURN_CYLON_STATUS_IF_ARROW_FAILED(writer->WriteRecordBatch(*batch));
  }

  RETURN_CYLON_STATUS_IF_ARROW_FAILED(writer->Close());

  // Get the buffer
  CYLON_ASSIGN_OR_RAISE(auto buffer, stream->Finish());

  data->resize(buffer->size());
  std::memcpy(data->data(), buffer->data(), buffer->size());

  return Status::OK();
}

Status ArrowIpcSerializer::DeserializeTable(
    const std::vector<uint8_t> &data,
    const std::shared_ptr<CylonContext> &ctx,
    std::shared_ptr<Table> *table) {
  if (data.empty()) {
    return {Code::Invalid, "Empty data for deserialization"};
  }

  // Wrap data in a buffer reader
  auto buffer = arrow::Buffer::Wrap(data.data(), data.size());
  auto reader_input =
      std::make_shared<arrow::io::BufferReader>(std::move(buffer));

  // Open IPC stream reader
  CYLON_ASSIGN_OR_RAISE(auto reader,
                        arrow::ipc::RecordBatchStreamReader::Open(
                            std::move(reader_input)));

  // Read all record batches into a table
  CYLON_ASSIGN_OR_RAISE(auto arrow_table, reader->ToTable());

  return Table::FromArrowTable(ctx, std::move(arrow_table), *table);
}

}  // namespace checkpoint
}  // namespace cylon