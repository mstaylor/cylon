// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Tests for the checkpointing system.

use std::sync::Arc;
use std::time::Duration;
use tempfile::tempdir;
use tokio::sync::RwLock;

use arrow::array::{Int32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use cylon::checkpoint::{
    ArrowIpcSerializer, CheckpointConfig, CheckpointCoordinator, CheckpointDecision,
    CheckpointManagerBuilder, CheckpointSerializer, CheckpointStatus, CheckpointStorage,
    CheckpointTrigger, CompositeTrigger, FileSystemStorage, IntervalTrigger, LocalCoordinator,
    OperationCountTrigger, OperationType, StorageConfig, TimeBudgetTrigger, TriggerConfig,
    WorkerId,
};
use cylon::ctx::CylonContext;
use cylon::table::Table;

// ============================================================================
// WorkerId Tests
// ============================================================================

#[test]
fn test_worker_id_from_rank() {
    let worker = WorkerId::from_rank(5);
    match worker {
        WorkerId::Rank(r) => assert_eq!(r, 5),
        _ => panic!("Expected Rank variant"),
    }
}

#[test]
fn test_worker_id_serverless() {
    let worker = WorkerId::serverless("lambda-123");
    match worker {
        WorkerId::Serverless { worker_id } => assert_eq!(worker_id, "lambda-123"),
        _ => panic!("Expected Serverless variant"),
    }
}

#[test]
fn test_worker_id_to_path_string() {
    let rank_worker = WorkerId::from_rank(3);
    assert_eq!(rank_worker.to_path_string(), "rank_3");

    let serverless_worker = WorkerId::serverless("abc-123");
    assert_eq!(serverless_worker.to_path_string(), "worker_abc-123");
}

// ============================================================================
// Trigger Tests
// ============================================================================

#[test]
fn test_operation_count_trigger() {
    let trigger = OperationCountTrigger::new(10);

    // Initially should not trigger
    assert!(!trigger.should_checkpoint());
    assert_eq!(trigger.current_count(), 0);

    // Record 9 operations - should not trigger yet
    for _ in 0..9 {
        trigger.record_operation(OperationType::Join, 1000);
    }
    assert!(!trigger.should_checkpoint());
    assert_eq!(trigger.current_count(), 9);

    // 10th operation should trigger
    trigger.record_operation(OperationType::Sort, 1000);
    assert!(trigger.should_checkpoint());
    assert_eq!(trigger.current_count(), 10);

    // Reset should clear the counter
    trigger.reset();
    assert!(!trigger.should_checkpoint());
    assert_eq!(trigger.current_count(), 0);
}

#[test]
fn test_operation_count_trigger_force() {
    let trigger = OperationCountTrigger::new(100);

    assert!(!trigger.should_checkpoint());

    trigger.force_checkpoint();
    assert!(trigger.should_checkpoint());

    trigger.reset();
    assert!(!trigger.should_checkpoint());
}

#[test]
fn test_time_budget_trigger() {
    // Create a trigger with 10 second budget and 2 second reserve
    let trigger = TimeBudgetTrigger::new(Duration::from_secs(10), Duration::from_secs(2));

    // Initially should not trigger (remaining time > reserve)
    assert!(!trigger.should_checkpoint());
    assert!(!trigger.is_critical());

    // Remaining time should be approximately 10 seconds
    let remaining = trigger.remaining_time();
    assert!(remaining <= Duration::from_secs(10));
    assert!(remaining >= Duration::from_secs(9));
}

#[test]
fn test_time_budget_trigger_for_lambda() {
    let trigger = TimeBudgetTrigger::for_lambda(900, 60); // 15 min timeout, 1 min reserve

    assert!(!trigger.should_checkpoint());
    assert!(trigger.remaining_time() > Duration::from_secs(800));
}

#[test]
fn test_interval_trigger() {
    let trigger = IntervalTrigger::new(Duration::from_millis(100));

    // Initially should not trigger (just started)
    assert!(!trigger.should_checkpoint());

    // After waiting, should trigger
    std::thread::sleep(Duration::from_millis(150));
    assert!(trigger.should_checkpoint());

    // Reset and verify
    trigger.reset();
    assert!(!trigger.should_checkpoint());
}

#[test]
fn test_interval_trigger_minutes() {
    let trigger = IntervalTrigger::minutes(5);

    // Just verify it doesn't immediately trigger
    assert!(!trigger.should_checkpoint());
}

#[test]
fn test_composite_trigger() {
    let trigger = CompositeTrigger::new()
        .add(OperationCountTrigger::new(5))
        .add(IntervalTrigger::new(Duration::from_secs(3600))); // 1 hour

    // Neither should trigger initially
    assert!(!trigger.should_checkpoint());

    // Record operations - should trigger operation count
    for _ in 0..5 {
        trigger.record_operation(OperationType::Filter, 100);
    }

    // Now should trigger (operation count reached)
    assert!(trigger.should_checkpoint());
}

#[test]
fn test_composite_trigger_from_config() {
    let config = TriggerConfig::new()
        .with_operation_threshold(100)
        .with_interval(Duration::from_secs(300));

    let trigger = CompositeTrigger::from_config(&config);

    // Should not trigger initially
    assert!(!trigger.should_checkpoint());
}

// ============================================================================
// Serializer Tests
// ============================================================================

#[test]
fn test_arrow_ipc_serializer_state() {
    let serializer = ArrowIpcSerializer::new();

    // Test simple state serialization
    let state: i32 = 42;
    let data = serializer.serialize_state(&state).unwrap();
    let restored: i32 = serializer.deserialize_state(&data).unwrap();
    assert_eq!(state, restored);

    // Test complex state
    let complex_state = vec![1, 2, 3, 4, 5];
    let data = serializer.serialize_state(&complex_state).unwrap();
    let restored: Vec<i32> = serializer.deserialize_state(&data).unwrap();
    assert_eq!(complex_state, restored);
}

#[test]
fn test_arrow_ipc_serializer_format_id() {
    let serializer = ArrowIpcSerializer::new();
    assert_eq!(serializer.format_id(), "arrow_ipc");
}

// ============================================================================
// Storage Tests
// ============================================================================

#[tokio::test]
async fn test_filesystem_storage_write_read() {
    let temp_dir = tempdir().unwrap();
    let storage = FileSystemStorage::new(temp_dir.path(), "test-job");
    storage.initialize().await.unwrap();

    let worker_id = WorkerId::from_rank(0);
    let checkpoint_id = 1;
    let key = "test_data.bin";
    let data = b"Hello, checkpoint!";

    // Write data
    let path = storage
        .write(checkpoint_id, &worker_id, key, data)
        .await
        .unwrap();
    assert!(!path.is_empty());

    // Verify it exists
    assert!(storage.exists(checkpoint_id, &worker_id, key).await.unwrap());

    // Read it back
    let read_data = storage.read(checkpoint_id, &worker_id, key).await.unwrap();
    assert_eq!(read_data, data);
}

#[tokio::test]
async fn test_filesystem_storage_list_keys() {
    let temp_dir = tempdir().unwrap();
    let storage = FileSystemStorage::new(temp_dir.path(), "test-job");
    storage.initialize().await.unwrap();

    let worker_id = WorkerId::from_rank(0);
    let checkpoint_id = 1;

    // Write multiple files
    storage
        .write(checkpoint_id, &worker_id, "table1.arrow", b"data1")
        .await
        .unwrap();
    storage
        .write(checkpoint_id, &worker_id, "table2.arrow", b"data2")
        .await
        .unwrap();
    storage
        .write(checkpoint_id, &worker_id, "state.bin", b"state")
        .await
        .unwrap();

    // Commit the write
    storage.commit_write(checkpoint_id, &worker_id).await.unwrap();

    // List keys
    let keys = storage.list_keys(checkpoint_id, &worker_id).await.unwrap();
    assert_eq!(keys.len(), 3);
    assert!(keys.contains(&"table1.arrow".to_string()));
    assert!(keys.contains(&"table2.arrow".to_string()));
    assert!(keys.contains(&"state.bin".to_string()));
}

#[tokio::test]
async fn test_filesystem_storage_delete() {
    let temp_dir = tempdir().unwrap();
    let storage = FileSystemStorage::new(temp_dir.path(), "test-job");
    storage.initialize().await.unwrap();

    let worker_id = WorkerId::from_rank(0);
    let checkpoint_id = 1;

    // Write and commit
    storage
        .write(checkpoint_id, &worker_id, "data.bin", b"test")
        .await
        .unwrap();
    storage.commit_write(checkpoint_id, &worker_id).await.unwrap();

    // Verify exists
    assert!(storage.exists(checkpoint_id, &worker_id, "data.bin").await.unwrap());

    // Delete
    storage.delete(checkpoint_id).await.unwrap();

    // Verify gone
    assert!(!storage.exists(checkpoint_id, &worker_id, "data.bin").await.unwrap());
}

#[tokio::test]
async fn test_filesystem_storage_list_checkpoints() {
    let temp_dir = tempdir().unwrap();
    let storage = FileSystemStorage::new(temp_dir.path(), "test-job");
    storage.initialize().await.unwrap();

    let worker_id = WorkerId::from_rank(0);

    // Create multiple checkpoints
    for id in [1, 2, 3] {
        storage
            .write(id, &worker_id, "data.bin", b"test")
            .await
            .unwrap();
        storage.commit_write(id, &worker_id).await.unwrap();

        // Write metadata
        use cylon::checkpoint::CheckpointMetadata;
        let metadata = CheckpointMetadata::new(id, "test-job");
        storage.write_metadata(id, &metadata).await.unwrap();
    }

    // List checkpoints
    let checkpoints = storage.list_checkpoints().await.unwrap();
    assert_eq!(checkpoints.len(), 3);
    // Should be sorted newest first
    assert_eq!(checkpoints[0], 3);
    assert_eq!(checkpoints[1], 2);
    assert_eq!(checkpoints[2], 1);
}

// ============================================================================
// Coordinator Tests
// ============================================================================

#[tokio::test]
async fn test_local_coordinator() {
    let coordinator = LocalCoordinator::new();

    assert_eq!(coordinator.world_size(), 1);
    assert!(coordinator.is_leader());

    // Begin checkpoint should always proceed
    let decision = coordinator.begin_checkpoint(1).await.unwrap();
    assert!(matches!(decision, CheckpointDecision::Proceed(_)));

    // Commit should succeed
    coordinator.commit_checkpoint(1).await.unwrap();

    // Find latest should return the committed checkpoint
    let latest = coordinator.find_latest_checkpoint().await.unwrap();
    assert_eq!(latest, Some(1));
}

// ============================================================================
// Config Tests
// ============================================================================

#[test]
fn test_checkpoint_config_default() {
    let config = CheckpointConfig::default();
    assert_eq!(config.job_id, "default");
    assert!(config.async_io);
    assert!(!config.incremental);
}

#[test]
fn test_checkpoint_config_builder_pattern() {
    let config = CheckpointConfig::new("my-job")
        .with_async_io(false)
        .with_incremental(true);

    assert_eq!(config.job_id, "my-job");
    assert!(!config.async_io);
    assert!(config.incremental);
}

#[test]
fn test_trigger_config_serverless() {
    let config = TriggerConfig::serverless(Duration::from_secs(900), Duration::from_secs(60));

    assert!(config.time_budget_threshold.is_some());
    assert!(config.total_time_budget.is_some());
    assert!(config.operation_threshold.is_none());
}

#[test]
fn test_trigger_config_hpc() {
    let config = TriggerConfig::hpc(1000, 1024 * 1024 * 1024);

    assert_eq!(config.operation_threshold, Some(1000));
    assert_eq!(config.bytes_threshold, Some(1024 * 1024 * 1024));
    assert!(config.time_budget_threshold.is_none());
}

// ============================================================================
// Integration Tests (requires CylonContext)
// ============================================================================

fn create_test_table(ctx: Arc<CylonContext>) -> Table {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
    ]));

    let id_array = Int32Array::from(vec![1, 2, 3, 4, 5]);
    let name_array = StringArray::from(vec!["Alice", "Bob", "Charlie", "David", "Eve"]);

    let batch = RecordBatch::try_new(schema, vec![Arc::new(id_array), Arc::new(name_array)]).unwrap();

    Table::from_record_batches(ctx, vec![batch]).unwrap()
}

#[tokio::test]
async fn test_checkpoint_manager_local() {
    let temp_dir = tempdir().unwrap();

    let ctx = Arc::new(CylonContext::new(false));

    let config = CheckpointConfig::new("test-job")
        .with_storage(StorageConfig::filesystem(temp_dir.path()))
        .with_trigger(TriggerConfig::new().with_operation_threshold(10));

    let manager = CheckpointManagerBuilder::new()
        .with_config(config)
        .with_context(ctx.clone())
        .build_local()
        .await
        .unwrap();

    // Register a table
    let table = create_test_table(ctx.clone());
    manager
        .register_table("users", Arc::new(RwLock::new(table)))
        .await;

    // Create a checkpoint
    let checkpoint_id = manager.checkpoint().await.unwrap();
    assert_eq!(checkpoint_id, 1);

    // Verify checkpoint exists
    let checkpoints = manager.list_checkpoints().await.unwrap();
    assert!(checkpoints.contains(&1));

    // Get metadata
    let metadata = manager.get_metadata(checkpoint_id).await.unwrap();
    assert_eq!(metadata.checkpoint_id, 1);
    assert_eq!(metadata.status, CheckpointStatus::Committed);
    assert!(metadata.tables.contains(&"users".to_string()));
}

#[tokio::test]
async fn test_checkpoint_manager_restore() {
    let temp_dir = tempdir().unwrap();

    let ctx = Arc::new(CylonContext::new(false));

    let config = CheckpointConfig::new("test-job")
        .with_storage(StorageConfig::filesystem(temp_dir.path()));

    let manager = CheckpointManagerBuilder::new()
        .with_config(config)
        .with_context(ctx.clone())
        .build_local()
        .await
        .unwrap();

    // Create and register original table
    let original_table = create_test_table(ctx.clone());
    let table_ref = Arc::new(RwLock::new(original_table));
    manager.register_table("users", table_ref.clone()).await;

    // Checkpoint
    let checkpoint_id = manager.checkpoint().await.unwrap();

    // Modify the table (simulate processing)
    {
        let mut table = table_ref.write().await;
        *table = create_test_table(ctx.clone()); // Replace with new data
    }

    // Restore
    manager.restore_from(checkpoint_id).await.unwrap();

    // Verify table was restored
    let table = table_ref.read().await;
    assert_eq!(table.rows(), 5);
}

#[tokio::test]
async fn test_checkpoint_manager_multiple_checkpoints() {
    let temp_dir = tempdir().unwrap();

    let ctx = Arc::new(CylonContext::new(false));

    let config = CheckpointConfig::new("test-job")
        .with_storage(StorageConfig::filesystem(temp_dir.path()));

    let manager = CheckpointManagerBuilder::new()
        .with_config(config)
        .with_context(ctx.clone())
        .build_local()
        .await
        .unwrap();

    let table = create_test_table(ctx.clone());
    manager
        .register_table("users", Arc::new(RwLock::new(table)))
        .await;

    // Create multiple checkpoints
    let id1 = manager.checkpoint().await.unwrap();
    let id2 = manager.checkpoint().await.unwrap();
    let id3 = manager.checkpoint().await.unwrap();

    assert_eq!(id1, 1);
    assert_eq!(id2, 2);
    assert_eq!(id3, 3);

    // All should be listed
    let checkpoints = manager.list_checkpoints().await.unwrap();
    assert!(checkpoints.contains(&1));
    assert!(checkpoints.contains(&2));
    assert!(checkpoints.contains(&3));
}

#[tokio::test]
async fn test_checkpoint_manager_delete() {
    let temp_dir = tempdir().unwrap();

    let ctx = Arc::new(CylonContext::new(false));

    let config = CheckpointConfig::new("test-job")
        .with_storage(StorageConfig::filesystem(temp_dir.path()));

    let manager = CheckpointManagerBuilder::new()
        .with_config(config)
        .with_context(ctx.clone())
        .build_local()
        .await
        .unwrap();

    let table = create_test_table(ctx.clone());
    manager
        .register_table("users", Arc::new(RwLock::new(table)))
        .await;

    // Create checkpoint
    let checkpoint_id = manager.checkpoint().await.unwrap();

    // Delete it
    manager.delete_checkpoint(checkpoint_id).await.unwrap();

    // Should no longer be listed
    let checkpoints = manager.list_checkpoints().await.unwrap();
    assert!(!checkpoints.contains(&checkpoint_id));
}
