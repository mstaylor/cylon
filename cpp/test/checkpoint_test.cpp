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

#include "common/test_header.hpp"
#include "test_utils.hpp"

#include <cylon/checkpoint/checkpoint_config.hpp>
#include <cylon/checkpoint/checkpoint_manager.hpp>
#include <cylon/checkpoint/checkpoint_serializer.hpp>
#include <cylon/checkpoint/checkpoint_storage.hpp>
#include <cylon/checkpoint/checkpoint_trigger.hpp>
#include <cylon/checkpoint/checkpoint_coordinator.hpp>

using namespace cylon;
using namespace cylon::checkpoint;

namespace {

// Helper to create a simple test table with int32 + double columns
Status CreateTestTable(const std::shared_ptr<CylonContext> &ctx,
                       int rows,
                       std::shared_ptr<Table> &output) {
  return cylon::test::CreateTable(ctx, rows, output);
}

// Helper to get a temp directory for test checkpoints
std::string TestCheckpointDir() {
  return "/tmp/cylon_checkpoint_test_" + std::to_string(getpid());
}

// Helper to clean up test checkpoint directory
void CleanupTestDir(const std::string &path) {
  std::string cmd = "rm -rf " + path;
  (void)system(cmd.c_str());
}

}  // namespace

TEST_CASE("Arrow IPC serializer round-trip", "[checkpoint]") {
  ArrowIpcSerializer serializer;

  SECTION("serialize and deserialize a table") {
    std::shared_ptr<Table> table;
    CHECK_CYLON_STATUS(CreateTestTable(ctx, 100, table));

    std::vector<uint8_t> data;
    CHECK_CYLON_STATUS(serializer.SerializeTable(table, &data));
    REQUIRE(!data.empty());

    std::shared_ptr<Table> restored;
    CHECK_CYLON_STATUS(serializer.DeserializeTable(data, ctx, &restored));

    REQUIRE(restored->Rows() == table->Rows());
    REQUIRE(restored->Columns() == table->Columns());

    // Verify data integrity
    auto orig_col = std::static_pointer_cast<arrow::Int32Array>(
        table->get_table()->column(0)->chunk(0));
    auto rest_col = std::static_pointer_cast<arrow::Int32Array>(
        restored->get_table()->column(0)->chunk(0));
    for (int64_t i = 0; i < orig_col->length(); ++i) {
      REQUIRE(orig_col->Value(i) == rest_col->Value(i));
    }
  }

  SECTION("format id") {
    REQUIRE(serializer.FormatId() == "arrow_ipc");
  }
}

TEST_CASE("FileSystem storage operations", "[checkpoint]") {
  auto base_path = TestCheckpointDir();
  CleanupTestDir(base_path);

  FileSystemStorage storage(base_path, "test_job");

  SECTION("write, commit, and read") {
    std::vector<uint8_t> write_data = {1, 2, 3, 4, 5, 6, 7, 8};
    CHECK_CYLON_STATUS(storage.Write(1, 0, "table_a",
                                     write_data.data(), write_data.size()));
    CHECK_CYLON_STATUS(storage.CommitWrite(1, 0));

    std::vector<uint8_t> read_data;
    CHECK_CYLON_STATUS(storage.Read(1, 0, "table_a", &read_data));
    REQUIRE(read_data == write_data);
  }

  SECTION("exists check") {
    std::vector<uint8_t> data = {10, 20};
    CHECK_CYLON_STATUS(storage.Write(2, 0, "exists_test",
                                     data.data(), data.size()));
    CHECK_CYLON_STATUS(storage.CommitWrite(2, 0));

    bool exists = false;
    CHECK_CYLON_STATUS(storage.Exists(2, 0, "exists_test", &exists));
    REQUIRE(exists);

    CHECK_CYLON_STATUS(storage.Exists(2, 0, "nonexistent", &exists));
    REQUIRE(!exists);
  }

  SECTION("list keys") {
    std::vector<uint8_t> data = {1};
    CHECK_CYLON_STATUS(storage.Write(3, 0, "key_a", data.data(), data.size()));
    CHECK_CYLON_STATUS(storage.Write(3, 0, "key_b", data.data(), data.size()));
    CHECK_CYLON_STATUS(storage.CommitWrite(3, 0));

    std::vector<std::string> keys;
    CHECK_CYLON_STATUS(storage.ListKeys(3, 0, &keys));
    REQUIRE(keys.size() == 2);
  }

  SECTION("list and delete checkpoints") {
    std::vector<uint8_t> data = {1};
    for (uint64_t id = 10; id <= 13; ++id) {
      CHECK_CYLON_STATUS(storage.Write(id, 0, "t", data.data(), data.size()));
      CHECK_CYLON_STATUS(storage.CommitWrite(id, 0));
    }

    std::vector<uint64_t> ids;
    CHECK_CYLON_STATUS(storage.ListCheckpoints(&ids));
    REQUIRE(ids.size() == 4);
    // Should be sorted newest first
    REQUIRE(ids[0] == 13);
    REQUIRE(ids[3] == 10);

    CHECK_CYLON_STATUS(storage.Delete(10));
    CHECK_CYLON_STATUS(storage.ListCheckpoints(&ids));
    REQUIRE(ids.size() == 3);
  }

  SECTION("metadata write and read") {
    CheckpointMetadata meta;
    meta.checkpoint_id = 42;
    meta.world_size = 4;
    meta.status = CheckpointStatus::Committed;
    meta.serializer_format = "arrow_ipc";
    meta.table_names = {"orders", "customers"};

    CHECK_CYLON_STATUS(storage.WriteMetadata(42, meta));

    CheckpointMetadata restored;
    CHECK_CYLON_STATUS(storage.ReadMetadata(42, &restored));
    REQUIRE(restored.checkpoint_id == 42);
    REQUIRE(restored.world_size == 4);
    REQUIRE(restored.status == CheckpointStatus::Committed);
    REQUIRE(restored.serializer_format == "arrow_ipc");
    REQUIRE(restored.table_names.size() == 2);
    REQUIRE(restored.table_names[0] == "orders");
    REQUIRE(restored.table_names[1] == "customers");
  }

  CleanupTestDir(base_path);
}

TEST_CASE("Operation count trigger", "[checkpoint]") {
  SECTION("triggers after threshold") {
    OperationCountTrigger trigger(5, 0);

    for (int i = 0; i < 4; ++i) {
      trigger.RecordOperation(OperationType::Join, 100);
      REQUIRE(!trigger.ShouldCheckpoint());
    }

    trigger.RecordOperation(OperationType::Join, 100);
    REQUIRE(trigger.ShouldCheckpoint());
  }

  SECTION("triggers on bytes threshold") {
    OperationCountTrigger trigger(0, 1000);

    trigger.RecordOperation(OperationType::Other, 500);
    REQUIRE(!trigger.ShouldCheckpoint());

    trigger.RecordOperation(OperationType::Other, 600);
    REQUIRE(trigger.ShouldCheckpoint());
  }

  SECTION("reset clears state") {
    OperationCountTrigger trigger(2, 0);
    trigger.RecordOperation(OperationType::Join, 0);
    trigger.RecordOperation(OperationType::Join, 0);
    REQUIRE(trigger.ShouldCheckpoint());

    trigger.Reset();
    REQUIRE(!trigger.ShouldCheckpoint());
  }

  SECTION("force checkpoint") {
    OperationCountTrigger trigger(1000, 0);
    REQUIRE(!trigger.ShouldCheckpoint());
    trigger.ForceCheckpoint();
    REQUIRE(trigger.ShouldCheckpoint());
    REQUIRE(trigger.Urgency() == CheckpointUrgency::Critical);
  }
}

TEST_CASE("Composite trigger from config", "[checkpoint]") {
  TriggerConfig config;
  config.operation_threshold = 10;
  config.bytes_threshold = 0;
  config.interval = std::chrono::seconds{0};

  auto trigger = CompositeTrigger::FromConfig(config);
  REQUIRE(!trigger->ShouldCheckpoint());

  for (int i = 0; i < 10; ++i) {
    trigger->RecordOperation(OperationType::Other, 0);
  }
  REQUIRE(trigger->ShouldCheckpoint());
}

TEST_CASE("Local coordinator", "[checkpoint]") {
  LocalCoordinator coord;

  REQUIRE(coord.GetRank() == 0);
  REQUIRE(coord.GetWorldSize() == 1);
  REQUIRE(coord.IsLeader());

  CheckpointDecision decision;
  CHECK_CYLON_STATUS(coord.BeginCheckpoint(1, &decision));
  REQUIRE(decision == CheckpointDecision::Proceed);

  CHECK_CYLON_STATUS(coord.CommitCheckpoint(1));
  CHECK_CYLON_STATUS(coord.AbortCheckpoint(1));
}

TEST_CASE("Checkpoint manager local end-to-end", "[checkpoint]") {
  auto base_path = TestCheckpointDir();
  CleanupTestDir(base_path);

  CheckpointConfig config("e2e_test");
  config.storage_path = base_path;
  config.trigger.operation_threshold = 2;
  config.trigger.bytes_threshold = 0;

  std::unique_ptr<CheckpointManager> manager;
  CHECK_CYLON_STATUS(CheckpointManager::MakeLocal(ctx, config, &manager));

  SECTION("checkpoint and restore") {
    // Create and register tables
    std::shared_ptr<Table> orders, customers;
    CHECK_CYLON_STATUS(CreateTestTable(ctx, 50, orders));
    CHECK_CYLON_STATUS(CreateTestTable(ctx, 30, customers));

    manager->RegisterTable("orders", orders);
    manager->RegisterTable("customers", customers);

    // Should not checkpoint yet (0 operations)
    REQUIRE(!manager->ShouldCheckpoint());

    // Record operations until trigger fires
    manager->RecordOperation(OperationType::Join, 0);
    manager->RecordOperation(OperationType::Filter, 0);
    REQUIRE(manager->ShouldCheckpoint());

    // Checkpoint
    uint64_t ckpt_id = 0;
    CHECK_CYLON_STATUS(manager->Checkpoint(&ckpt_id));
    REQUIRE(ckpt_id == 1);

    // Trigger should be reset
    REQUIRE(!manager->ShouldCheckpoint());

    // Restore
    std::unordered_map<std::string, std::shared_ptr<Table>> restored;
    CHECK_CYLON_STATUS(manager->RestoreFrom(ckpt_id, &restored));

    REQUIRE(restored.size() == 2);
    REQUIRE(restored.count("orders") == 1);
    REQUIRE(restored.count("customers") == 1);
    REQUIRE(restored["orders"]->Rows() == 50);
    REQUIRE(restored["customers"]->Rows() == 30);
  }

  SECTION("multiple checkpoints and restore latest") {
    std::shared_ptr<Table> table;
    CHECK_CYLON_STATUS(CreateTestTable(ctx, 10, table));
    manager->RegisterTable("data", table);

    // First checkpoint
    manager->RecordOperation(OperationType::Other, 0);
    manager->RecordOperation(OperationType::Other, 0);
    uint64_t id1 = 0;
    CHECK_CYLON_STATUS(manager->Checkpoint(&id1));
    REQUIRE(id1 == 1);

    // Update table and second checkpoint
    CHECK_CYLON_STATUS(CreateTestTable(ctx, 20, table));
    manager->UpdateTable("data", table);
    manager->RecordOperation(OperationType::Other, 0);
    manager->RecordOperation(OperationType::Other, 0);
    uint64_t id2 = 0;
    CHECK_CYLON_STATUS(manager->Checkpoint(&id2));
    REQUIRE(id2 == 2);

    // Restore latest
    std::unordered_map<std::string, std::shared_ptr<Table>> restored;
    CHECK_CYLON_STATUS(manager->Restore(&restored));
    REQUIRE(restored["data"]->Rows() == 20);
  }

  SECTION("unregister table") {
    std::shared_ptr<Table> table;
    CHECK_CYLON_STATUS(CreateTestTable(ctx, 10, table));
    manager->RegisterTable("temp", table);
    manager->UnregisterTable("temp");

    manager->RecordOperation(OperationType::Other, 0);
    manager->RecordOperation(OperationType::Other, 0);
    uint64_t ckpt_id = 0;
    CHECK_CYLON_STATUS(manager->Checkpoint(&ckpt_id));

    std::unordered_map<std::string, std::shared_ptr<Table>> restored;
    CHECK_CYLON_STATUS(manager->RestoreFrom(ckpt_id, &restored));
    REQUIRE(restored.empty());
  }

  CleanupTestDir(base_path);
}