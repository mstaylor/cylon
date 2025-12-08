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

// ============================================================================
// Incremental Checkpoint Tests
// ============================================================================

use cylon::checkpoint::{
    ChangeTracker, DeltaTableInfo, DeltaType, IncrementalCheckpointInfo, IncrementalConfig,
    RowChangeType, RowRange,
};

#[test]
fn test_change_tracker_basic() {
    let tracker = ChangeTracker::new();

    // Initially no modifications
    assert!(!tracker.is_table_modified("users"));
    assert!(tracker.get_modified_tables().is_empty());

    // Mark table as modified
    tracker.mark_table_modified("users");
    assert!(tracker.is_table_modified("users"));
    assert!(!tracker.is_table_modified("orders"));

    // Check modified tables list
    let modified = tracker.get_modified_tables();
    assert_eq!(modified.len(), 1);
    assert!(modified.contains(&"users".to_string()));

    // Reset should clear
    tracker.reset();
    assert!(!tracker.is_table_modified("users"));
    assert!(tracker.get_modified_tables().is_empty());
}

#[test]
fn test_change_tracker_parent_checkpoint() {
    let tracker = ChangeTracker::new();

    // Initially no parent
    assert!(tracker.parent_checkpoint_id().is_none());

    // Set parent
    tracker.set_parent_checkpoint(42);
    assert_eq!(tracker.parent_checkpoint_id(), Some(42));

    // Create from checkpoint
    let tracker2 = ChangeTracker::from_checkpoint(100);
    assert_eq!(tracker2.parent_checkpoint_id(), Some(100));
}

#[test]
fn test_change_tracker_needs_checkpoint() {
    let tracker = ChangeTracker::new();

    // No parent - should always need checkpoint
    assert!(tracker.needs_checkpoint("users"));

    // With parent - only modified tables need checkpoint
    tracker.set_parent_checkpoint(1);
    assert!(!tracker.needs_checkpoint("users"));

    tracker.mark_table_modified("users");
    assert!(tracker.needs_checkpoint("users"));
    assert!(!tracker.needs_checkpoint("orders"));
}

#[test]
fn test_change_tracker_unchanged_tables() {
    let tracker = ChangeTracker::new();
    tracker.set_parent_checkpoint(1);

    tracker.mark_table_modified("users");
    tracker.mark_table_modified("orders");

    let all_tables = vec![
        "users".to_string(),
        "orders".to_string(),
        "products".to_string(),
        "inventory".to_string(),
    ];

    let unchanged = tracker.get_unchanged_tables(&all_tables);
    assert_eq!(unchanged.len(), 2);
    assert!(unchanged.contains(&"products".to_string()));
    assert!(unchanged.contains(&"inventory".to_string()));
}

#[test]
fn test_change_tracker_row_tracking() {
    let tracker = ChangeTracker::with_row_tracking();
    assert!(tracker.is_row_tracking_enabled());

    // Mark specific rows as modified
    tracker.mark_rows_modified(
        "users",
        RowRange::update(10, 20),
    );

    assert!(tracker.is_table_modified("users"));

    let ranges = tracker.get_modified_ranges("users");
    assert_eq!(ranges.len(), 1);
    assert_eq!(ranges[0].start, 10);
    assert_eq!(ranges[0].end, 20);
    assert_eq!(ranges[0].change_type, RowChangeType::Update);
}

#[test]
fn test_change_tracker_rows_appended() {
    let tracker = ChangeTracker::with_row_tracking();

    tracker.mark_rows_appended("users", 100, 50);

    assert!(tracker.is_table_modified("users"));

    let ranges = tracker.get_modified_ranges("users");
    assert_eq!(ranges.len(), 1);
    assert_eq!(ranges[0].start, 100);
    assert_eq!(ranges[0].end, 150);
    assert_eq!(ranges[0].change_type, RowChangeType::Append);
}

#[test]
fn test_change_tracker_delta_type() {
    let tracker = ChangeTracker::with_row_tracking();

    // Append only
    tracker.mark_rows_modified("append_table", RowRange::append(0, 10));
    assert_eq!(tracker.get_delta_type("append_table"), DeltaType::Append);

    // Update only
    tracker.mark_rows_modified("update_table", RowRange::update(5, 15));
    assert_eq!(tracker.get_delta_type("update_table"), DeltaType::Update);

    // Delete only
    tracker.mark_rows_modified("delete_table", RowRange::delete(0, 5));
    assert_eq!(tracker.get_delta_type("delete_table"), DeltaType::Delete);

    // Mixed operations
    tracker.mark_rows_modified("mixed_table", RowRange::append(100, 110));
    tracker.mark_rows_modified("mixed_table", RowRange::update(50, 60));
    assert_eq!(tracker.get_delta_type("mixed_table"), DeltaType::Mixed);
}

#[test]
fn test_change_tracker_without_row_tracking() {
    let tracker = ChangeTracker::new();
    assert!(!tracker.is_row_tracking_enabled());

    // Without row tracking, delta type is always Full
    tracker.mark_table_modified("users");
    assert_eq!(tracker.get_delta_type("users"), DeltaType::Full);
}

#[test]
fn test_row_range() {
    let range = RowRange::new(10, 20, RowChangeType::Update);
    assert_eq!(range.start, 10);
    assert_eq!(range.end, 20);
    assert_eq!(range.len(), 10);
    assert!(!range.is_empty());

    let empty_range = RowRange::new(10, 10, RowChangeType::Append);
    assert!(empty_range.is_empty());
    assert_eq!(empty_range.len(), 0);

    // Helper constructors
    let append = RowRange::append(0, 100);
    assert_eq!(append.change_type, RowChangeType::Append);

    let update = RowRange::update(50, 60);
    assert_eq!(update.change_type, RowChangeType::Update);

    let delete = RowRange::delete(0, 10);
    assert_eq!(delete.change_type, RowChangeType::Delete);
}

#[test]
fn test_delta_table_info() {
    // Append delta
    let append_info = DeltaTableInfo::append("users", 100, 500);
    assert_eq!(append_info.name, "users");
    assert_eq!(append_info.delta_type, DeltaType::Append);
    assert_eq!(append_info.affected_rows, 100);
    assert_eq!(append_info.append_start_row, Some(500));
    assert!(append_info.affected_indices.is_none());

    // Full table
    let full_info = DeltaTableInfo::full("products", 1000);
    assert_eq!(full_info.delta_type, DeltaType::Full);
    assert!(full_info.append_start_row.is_none());

    // Update delta
    let update_info = DeltaTableInfo::update("orders", vec![1, 5, 10, 20]);
    assert_eq!(update_info.delta_type, DeltaType::Update);
    assert_eq!(update_info.affected_rows, 4);
    assert_eq!(update_info.affected_indices, Some(vec![1, 5, 10, 20]));

    // Delete delta
    let delete_info = DeltaTableInfo::delete("inventory", vec![0, 1, 2]);
    assert_eq!(delete_info.delta_type, DeltaType::Delete);
    assert_eq!(delete_info.affected_rows, 3);
}

#[test]
fn test_incremental_checkpoint_info() {
    let mut info = IncrementalCheckpointInfo::new(42);
    assert_eq!(info.parent_checkpoint_id, 42);
    assert_eq!(info.chain_depth, 1);
    assert!(info.unchanged_tables.is_empty());
    assert!(info.delta_tables.is_empty());
    assert!(info.full_tables.is_empty());

    // Add tables
    info.add_unchanged("users");
    info.add_unchanged("products");
    info.add_full("orders");
    info.add_delta(DeltaTableInfo::append("logs", 100, 5000));

    assert_eq!(info.total_tables(), 4);
    assert_eq!(info.unchanged_tables.len(), 2);
    assert_eq!(info.full_tables.len(), 1);
    assert_eq!(info.delta_tables.len(), 1);

    // Not pure incremental (has full tables)
    assert!(!info.is_pure_incremental());

    // Savings ratio: 2 unchanged / 4 total = 0.5
    assert!((info.savings_ratio() - 0.5).abs() < 0.001);
}

#[test]
fn test_incremental_checkpoint_info_pure() {
    let mut info = IncrementalCheckpointInfo::new(1);
    info.add_unchanged("users");
    info.add_unchanged("products");
    info.add_delta(DeltaTableInfo::append("logs", 10, 100));

    // No full tables = pure incremental
    assert!(info.is_pure_incremental());
}

#[test]
fn test_incremental_config_default() {
    let config = IncrementalConfig::default();
    assert!(!config.enabled);
    assert!(!config.track_rows);
    assert_eq!(config.max_chain_depth, 10);
    assert!((config.min_savings_ratio - 0.2).abs() < 0.001);
}

#[test]
fn test_incremental_config_enabled() {
    let config = IncrementalConfig::enabled();
    assert!(config.enabled);
    assert!(!config.track_rows);
}

#[test]
fn test_incremental_config_builder() {
    let config = IncrementalConfig::enabled()
        .with_row_tracking()
        .with_max_chain_depth(5)
        .with_min_savings_ratio(0.3);

    assert!(config.enabled);
    assert!(config.track_rows);
    assert_eq!(config.max_chain_depth, 5);
    assert!((config.min_savings_ratio - 0.3).abs() < 0.001);
}

#[test]
fn test_incremental_config_savings_ratio_clamped() {
    let config = IncrementalConfig::enabled()
        .with_min_savings_ratio(1.5); // Over 1.0

    assert!((config.min_savings_ratio - 1.0).abs() < 0.001);

    let config2 = IncrementalConfig::enabled()
        .with_min_savings_ratio(-0.5); // Below 0.0

    assert!((config2.min_savings_ratio - 0.0).abs() < 0.001);
}

#[test]
fn test_checkpoint_config_with_incremental() {
    let config = CheckpointConfig::new("test-job")
        .with_incremental(true);

    assert!(config.incremental);
    assert!(config.incremental_config.enabled);
}

#[test]
fn test_checkpoint_config_with_incremental_config() {
    let inc_config = IncrementalConfig::enabled()
        .with_row_tracking()
        .with_max_chain_depth(3);

    let config = CheckpointConfig::new("test-job")
        .with_incremental_config(inc_config);

    assert!(config.incremental);
    assert!(config.incremental_config.enabled);
    assert!(config.incremental_config.track_rows);
    assert_eq!(config.incremental_config.max_chain_depth, 3);
}

#[tokio::test]
async fn test_checkpoint_manager_incremental_basic() {
    let temp_dir = tempdir().unwrap();

    let ctx = Arc::new(CylonContext::new(false));

    // Enable incremental checkpoints with min_savings_ratio of 0 to test with single table
    let inc_config = IncrementalConfig::enabled().with_min_savings_ratio(0.0);

    let config = CheckpointConfig::new("test-job")
        .with_storage(StorageConfig::filesystem(temp_dir.path()))
        .with_incremental_config(inc_config);

    let manager = CheckpointManagerBuilder::new()
        .with_config(config)
        .with_context(ctx.clone())
        .build_local()
        .await
        .unwrap();

    // Verify incremental is enabled
    assert!(manager.is_incremental_enabled());

    // Create a table
    let table = create_test_table(ctx.clone());
    manager
        .register_table("users", Arc::new(RwLock::new(table)))
        .await;

    // First checkpoint should be full (no parent)
    let id1 = manager.checkpoint().await.unwrap();
    assert_eq!(id1, 1);

    let metadata1 = manager.get_metadata(id1).await.unwrap();
    assert!(!metadata1.is_incremental());
    assert!(metadata1.parent_checkpoint_id.is_none());

    // Mark table as modified
    manager.mark_table_modified("users");

    // Second checkpoint should be incremental
    let id2 = manager.checkpoint().await.unwrap();
    assert_eq!(id2, 2);

    let metadata2 = manager.get_metadata(id2).await.unwrap();
    assert!(metadata2.is_incremental());
    assert_eq!(metadata2.parent_checkpoint_id, Some(1));
}

#[tokio::test]
async fn test_checkpoint_manager_incremental_unchanged_tables() {
    let temp_dir = tempdir().unwrap();

    let ctx = Arc::new(CylonContext::new(false));

    let config = CheckpointConfig::new("test-job")
        .with_storage(StorageConfig::filesystem(temp_dir.path()))
        .with_incremental(true);

    let manager = CheckpointManagerBuilder::new()
        .with_config(config)
        .with_context(ctx.clone())
        .build_local()
        .await
        .unwrap();

    // Create multiple tables
    let table1 = create_test_table(ctx.clone());
    let table2 = create_test_table(ctx.clone());

    manager
        .register_table("users", Arc::new(RwLock::new(table1)))
        .await;
    manager
        .register_table("products", Arc::new(RwLock::new(table2)))
        .await;

    // First checkpoint (full)
    let id1 = manager.checkpoint().await.unwrap();

    // Only modify one table
    manager.mark_table_modified("users");
    // "products" is NOT modified

    // Second checkpoint (incremental)
    let id2 = manager.checkpoint().await.unwrap();

    let metadata2 = manager.get_metadata(id2).await.unwrap();
    assert!(metadata2.is_incremental());

    // Check incremental info
    let inc_info = metadata2.incremental_info.as_ref().unwrap();
    assert!(inc_info.unchanged_tables.contains(&"products".to_string()));
}

#[tokio::test]
async fn test_checkpoint_manager_restore_incremental() {
    let temp_dir = tempdir().unwrap();

    let ctx = Arc::new(CylonContext::new(false));

    // Enable incremental with min_savings_ratio of 0 to test with single table
    let inc_config = IncrementalConfig::enabled().with_min_savings_ratio(0.0);

    let config = CheckpointConfig::new("test-job")
        .with_storage(StorageConfig::filesystem(temp_dir.path()))
        .with_incremental_config(inc_config);

    let manager = CheckpointManagerBuilder::new()
        .with_config(config)
        .with_context(ctx.clone())
        .build_local()
        .await
        .unwrap();

    // Create and register table
    let table = create_test_table(ctx.clone());
    let table_ref = Arc::new(RwLock::new(table));
    manager.register_table("users", table_ref.clone()).await;

    // First checkpoint (full)
    let _id1 = manager.checkpoint().await.unwrap();

    // Modify table
    manager.mark_table_modified("users");

    // Second checkpoint (incremental)
    let id2 = manager.checkpoint().await.unwrap();

    // Verify it's incremental
    let metadata = manager.get_metadata(id2).await.unwrap();
    assert!(metadata.is_incremental());

    // Restore from incremental checkpoint
    manager.restore_from(id2).await.unwrap();

    // Verify table was restored
    let table = table_ref.read().await;
    assert_eq!(table.rows(), 5);
}

#[tokio::test]
async fn test_checkpoint_manager_chain_depth_limit() {
    let temp_dir = tempdir().unwrap();

    let ctx = Arc::new(CylonContext::new(false));

    // Set max chain depth to 2 and min_savings_ratio to 0 to test with single table
    let inc_config = IncrementalConfig::enabled()
        .with_max_chain_depth(2)
        .with_min_savings_ratio(0.0);

    let config = CheckpointConfig::new("test-job")
        .with_storage(StorageConfig::filesystem(temp_dir.path()))
        .with_incremental_config(inc_config);

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

    // Checkpoint 1: Full
    let id1 = manager.checkpoint().await.unwrap();
    let meta1 = manager.get_metadata(id1).await.unwrap();
    assert!(!meta1.is_incremental());

    // Checkpoint 2: Incremental (chain depth 1)
    manager.mark_table_modified("users");
    let id2 = manager.checkpoint().await.unwrap();
    let meta2 = manager.get_metadata(id2).await.unwrap();
    assert!(meta2.is_incremental());
    assert_eq!(meta2.chain_depth(), 1);

    // Checkpoint 3: Incremental (chain depth 2)
    manager.mark_table_modified("users");
    let id3 = manager.checkpoint().await.unwrap();
    let meta3 = manager.get_metadata(id3).await.unwrap();
    assert!(meta3.is_incremental());
    assert_eq!(meta3.chain_depth(), 2);

    // Checkpoint 4: Should be full again (exceeded max chain depth)
    manager.mark_table_modified("users");
    let id4 = manager.checkpoint().await.unwrap();
    let meta4 = manager.get_metadata(id4).await.unwrap();
    assert!(!meta4.is_incremental());
}

#[tokio::test]
async fn test_checkpoint_manager_savings_ratio_threshold() {
    let temp_dir = tempdir().unwrap();

    let ctx = Arc::new(CylonContext::new(false));

    // Set min savings ratio to 0.5 (need at least 50% unchanged tables)
    let inc_config = IncrementalConfig::enabled().with_min_savings_ratio(0.5);

    let config = CheckpointConfig::new("test-job")
        .with_storage(StorageConfig::filesystem(temp_dir.path()))
        .with_incremental_config(inc_config);

    let manager = CheckpointManagerBuilder::new()
        .with_config(config)
        .with_context(ctx.clone())
        .build_local()
        .await
        .unwrap();

    // Create 2 tables
    let table1 = create_test_table(ctx.clone());
    let table2 = create_test_table(ctx.clone());

    manager
        .register_table("users", Arc::new(RwLock::new(table1)))
        .await;
    manager
        .register_table("products", Arc::new(RwLock::new(table2)))
        .await;

    // First checkpoint (full)
    let _id1 = manager.checkpoint().await.unwrap();

    // Modify BOTH tables - savings ratio would be 0%
    manager.mark_table_modified("users");
    manager.mark_table_modified("products");

    // Second checkpoint should be full (not incremental) because savings ratio is too low
    let id2 = manager.checkpoint().await.unwrap();
    let meta2 = manager.get_metadata(id2).await.unwrap();
    assert!(!meta2.is_incremental());
}

#[tokio::test]
async fn test_checkpoint_manager_change_tracker_access() {
    let temp_dir = tempdir().unwrap();

    let ctx = Arc::new(CylonContext::new(false));

    let config = CheckpointConfig::new("test-job")
        .with_storage(StorageConfig::filesystem(temp_dir.path()))
        .with_incremental(true);

    let manager = CheckpointManagerBuilder::new()
        .with_config(config)
        .with_context(ctx.clone())
        .build_local()
        .await
        .unwrap();

    // Access change tracker directly
    let tracker = manager.change_tracker();

    tracker.mark_table_modified("users");
    assert!(tracker.is_table_modified("users"));

    // Also test via manager method
    manager.mark_table_modified("orders");
    assert!(tracker.is_table_modified("orders"));
}

// ============================================================================
// Compression Tests
// ============================================================================

use cylon::checkpoint::{
    CompressionAlgorithm, CompressionConfig, Compressor, Lz4Compressor, NoCompressor,
    SnappyCompressor, ZstdCompressor,
};

#[test]
fn test_no_compression() {
    let compressor = NoCompressor;
    let test_data = b"Hello, this is test data for compression!";

    let compressed = compressor.compress(test_data).unwrap();
    assert_eq!(compressed, test_data);

    let decompressed = compressor.decompress(&compressed).unwrap();
    assert_eq!(decompressed, test_data);

    assert_eq!(compressor.algorithm(), CompressionAlgorithm::None);
    assert_eq!(compressor.extension(), "");
}

#[test]
fn test_lz4_compression() {
    let compressor = Lz4Compressor::new();
    let test_data = b"Hello, this is test data for compression! \
        It should compress reasonably well because it contains repeated patterns. \
        Repeated patterns. Repeated patterns. Repeated patterns.";

    let compressed = compressor.compress(test_data).unwrap();

    // LZ4 should compress this data
    assert!(compressed.len() < test_data.len());

    let decompressed = compressor.decompress(&compressed).unwrap();
    assert_eq!(decompressed, test_data);

    assert_eq!(compressor.algorithm(), CompressionAlgorithm::Lz4);
    assert_eq!(compressor.extension(), ".lz4");
}

#[test]
fn test_zstd_compression() {
    let compressor = ZstdCompressor::new();
    let test_data = b"Hello, this is test data for compression! \
        It should compress reasonably well because it contains repeated patterns. \
        Repeated patterns. Repeated patterns. Repeated patterns.";

    let compressed = compressor.compress(test_data).unwrap();

    // Zstd should compress this data well
    assert!(compressed.len() < test_data.len());

    let decompressed = compressor.decompress(&compressed).unwrap();
    assert_eq!(decompressed, test_data);

    assert_eq!(compressor.algorithm(), CompressionAlgorithm::Zstd);
    assert_eq!(compressor.extension(), ".zst");
}

#[test]
fn test_zstd_compression_levels() {
    let test_data = b"Hello, this is test data for compression! \
        It should compress reasonably well because it contains repeated patterns."
        .repeat(100);

    let low = ZstdCompressor::with_level(1);
    let high = ZstdCompressor::with_level(19);

    let compressed_low = low.compress(&test_data).unwrap();
    let compressed_high = high.compress(&test_data).unwrap();

    // Higher level should compress better (or at least as well)
    assert!(compressed_high.len() <= compressed_low.len());

    // Both should decompress correctly
    assert_eq!(low.decompress(&compressed_low).unwrap(), test_data);
    assert_eq!(high.decompress(&compressed_high).unwrap(), test_data);
}

#[test]
fn test_snappy_compression() {
    let compressor = SnappyCompressor::new();
    let test_data = b"Hello, this is test data for compression! \
        It should compress reasonably well because it contains repeated patterns. \
        Repeated patterns. Repeated patterns. Repeated patterns.";

    let compressed = compressor.compress(test_data).unwrap();

    // Snappy should compress this data
    assert!(compressed.len() < test_data.len());

    let decompressed = compressor.decompress(&compressed).unwrap();
    assert_eq!(decompressed, test_data);

    assert_eq!(compressor.algorithm(), CompressionAlgorithm::Snappy);
    assert_eq!(compressor.extension(), ".snappy");
}

#[test]
fn test_compression_empty_data() {
    let empty: &[u8] = &[];

    let lz4 = Lz4Compressor::new();
    let compressed = lz4.compress(empty).unwrap();
    let decompressed = lz4.decompress(&compressed).unwrap();
    assert_eq!(decompressed, empty);

    let zstd = ZstdCompressor::new();
    let compressed = zstd.compress(empty).unwrap();
    let decompressed = zstd.decompress(&compressed).unwrap();
    assert_eq!(decompressed, empty);

    let snappy = SnappyCompressor::new();
    let compressed = snappy.compress(empty).unwrap();
    let decompressed = snappy.decompress(&compressed).unwrap();
    assert_eq!(decompressed, empty);
}

#[test]
fn test_compression_config_builders() {
    let lz4 = CompressionConfig::lz4();
    assert_eq!(lz4.algorithm, CompressionAlgorithm::Lz4);
    assert!(lz4.level.is_none());

    let zstd = CompressionConfig::zstd();
    assert_eq!(zstd.algorithm, CompressionAlgorithm::Zstd);

    let zstd_level = CompressionConfig::zstd_with_level(10);
    assert_eq!(zstd_level.algorithm, CompressionAlgorithm::Zstd);
    assert_eq!(zstd_level.level, Some(10));

    let snappy = CompressionConfig::snappy();
    assert_eq!(snappy.algorithm, CompressionAlgorithm::Snappy);

    let none = CompressionConfig::none();
    assert_eq!(none.algorithm, CompressionAlgorithm::None);
}

#[test]
fn test_compression_config_with_level() {
    let config = CompressionConfig::zstd().with_level(15);
    assert_eq!(config.level, Some(15));
}

use cylon::checkpoint::{
    create_compressor, detect_algorithm_from_extension, strip_compression_extension,
};

#[test]
fn test_create_compressor() {
    let config = CompressionConfig::lz4();
    let compressor = create_compressor(&config);
    assert_eq!(compressor.algorithm(), CompressionAlgorithm::Lz4);

    let config = CompressionConfig::zstd_with_level(10);
    let compressor = create_compressor(&config);
    assert_eq!(compressor.algorithm(), CompressionAlgorithm::Zstd);

    let config = CompressionConfig::snappy();
    let compressor = create_compressor(&config);
    assert_eq!(compressor.algorithm(), CompressionAlgorithm::Snappy);

    let config = CompressionConfig::none();
    let compressor = create_compressor(&config);
    assert_eq!(compressor.algorithm(), CompressionAlgorithm::None);
}

#[test]
fn test_detect_algorithm_from_extension() {
    assert_eq!(
        detect_algorithm_from_extension("data.arrow.lz4"),
        CompressionAlgorithm::Lz4
    );
    assert_eq!(
        detect_algorithm_from_extension("data.arrow.zst"),
        CompressionAlgorithm::Zstd
    );
    assert_eq!(
        detect_algorithm_from_extension("data.arrow.zstd"),
        CompressionAlgorithm::Zstd
    );
    assert_eq!(
        detect_algorithm_from_extension("data.arrow.snappy"),
        CompressionAlgorithm::Snappy
    );
    assert_eq!(
        detect_algorithm_from_extension("data.arrow.snap"),
        CompressionAlgorithm::Snappy
    );
    assert_eq!(
        detect_algorithm_from_extension("data.arrow"),
        CompressionAlgorithm::None
    );
}

#[test]
fn test_strip_compression_extension() {
    assert_eq!(strip_compression_extension("data.arrow.lz4"), "data.arrow");
    assert_eq!(strip_compression_extension("data.arrow.zst"), "data.arrow");
    assert_eq!(
        strip_compression_extension("data.arrow.snappy"),
        "data.arrow"
    );
    assert_eq!(strip_compression_extension("data.arrow"), "data.arrow");
}

#[tokio::test]
async fn test_checkpoint_manager_with_lz4_compression() {
    let temp_dir = tempdir().unwrap();

    let ctx = Arc::new(CylonContext::new(false));

    let config = CheckpointConfig::new("test-job")
        .with_storage(StorageConfig::filesystem(temp_dir.path()))
        .with_compression(CompressionConfig::lz4());

    let manager = CheckpointManagerBuilder::new()
        .with_config(config)
        .with_context(ctx.clone())
        .build_local()
        .await
        .unwrap();

    // Verify compression is enabled
    assert!(manager.is_compression_enabled());
    assert_eq!(manager.compression_algorithm(), CompressionAlgorithm::Lz4);

    // Create and register a table
    let table = create_test_table(ctx.clone());
    let table_ref = Arc::new(RwLock::new(table));
    manager.register_table("users", table_ref.clone()).await;

    // Checkpoint
    let checkpoint_id = manager.checkpoint().await.unwrap();

    // Restore and verify
    manager.restore_from(checkpoint_id).await.unwrap();

    let restored_table = table_ref.read().await;
    assert_eq!(restored_table.rows(), 5);
}

#[tokio::test]
async fn test_checkpoint_manager_with_zstd_compression() {
    let temp_dir = tempdir().unwrap();

    let ctx = Arc::new(CylonContext::new(false));

    let config = CheckpointConfig::new("test-job")
        .with_storage(StorageConfig::filesystem(temp_dir.path()))
        .with_compression(CompressionConfig::zstd_with_level(5));

    let manager = CheckpointManagerBuilder::new()
        .with_config(config)
        .with_context(ctx.clone())
        .build_local()
        .await
        .unwrap();

    assert!(manager.is_compression_enabled());
    assert_eq!(manager.compression_algorithm(), CompressionAlgorithm::Zstd);

    let table = create_test_table(ctx.clone());
    let table_ref = Arc::new(RwLock::new(table));
    manager.register_table("users", table_ref.clone()).await;

    let checkpoint_id = manager.checkpoint().await.unwrap();
    manager.restore_from(checkpoint_id).await.unwrap();

    let restored_table = table_ref.read().await;
    assert_eq!(restored_table.rows(), 5);
}

#[tokio::test]
async fn test_checkpoint_manager_with_snappy_compression() {
    let temp_dir = tempdir().unwrap();

    let ctx = Arc::new(CylonContext::new(false));

    let config = CheckpointConfig::new("test-job")
        .with_storage(StorageConfig::filesystem(temp_dir.path()))
        .with_compression(CompressionConfig::snappy());

    let manager = CheckpointManagerBuilder::new()
        .with_config(config)
        .with_context(ctx.clone())
        .build_local()
        .await
        .unwrap();

    assert!(manager.is_compression_enabled());
    assert_eq!(
        manager.compression_algorithm(),
        CompressionAlgorithm::Snappy
    );

    let table = create_test_table(ctx.clone());
    let table_ref = Arc::new(RwLock::new(table));
    manager.register_table("users", table_ref.clone()).await;

    let checkpoint_id = manager.checkpoint().await.unwrap();
    manager.restore_from(checkpoint_id).await.unwrap();

    let restored_table = table_ref.read().await;
    assert_eq!(restored_table.rows(), 5);
}

#[tokio::test]
async fn test_checkpoint_compression_with_incremental() {
    let temp_dir = tempdir().unwrap();

    let ctx = Arc::new(CylonContext::new(false));

    // Enable both compression and incremental checkpoints
    let inc_config = IncrementalConfig::enabled().with_min_savings_ratio(0.0);

    let config = CheckpointConfig::new("test-job")
        .with_storage(StorageConfig::filesystem(temp_dir.path()))
        .with_compression(CompressionConfig::lz4())
        .with_incremental_config(inc_config);

    let manager = CheckpointManagerBuilder::new()
        .with_config(config)
        .with_context(ctx.clone())
        .build_local()
        .await
        .unwrap();

    assert!(manager.is_compression_enabled());
    assert!(manager.is_incremental_enabled());

    let table = create_test_table(ctx.clone());
    let table_ref = Arc::new(RwLock::new(table));
    manager.register_table("users", table_ref.clone()).await;

    // First checkpoint (full)
    let id1 = manager.checkpoint().await.unwrap();
    let meta1 = manager.get_metadata(id1).await.unwrap();
    assert!(!meta1.is_incremental());

    // Modify and create incremental checkpoint
    manager.mark_table_modified("users");
    let id2 = manager.checkpoint().await.unwrap();
    let meta2 = manager.get_metadata(id2).await.unwrap();
    assert!(meta2.is_incremental());

    // Restore from incremental checkpoint with compression
    manager.restore_from(id2).await.unwrap();

    let restored_table = table_ref.read().await;
    assert_eq!(restored_table.rows(), 5);
}

#[tokio::test]
async fn test_checkpoint_no_compression() {
    let temp_dir = tempdir().unwrap();

    let ctx = Arc::new(CylonContext::new(false));

    // Explicitly disable compression
    let config = CheckpointConfig::new("test-job")
        .with_storage(StorageConfig::filesystem(temp_dir.path()));
    // No compression config set

    let manager = CheckpointManagerBuilder::new()
        .with_config(config)
        .with_context(ctx.clone())
        .build_local()
        .await
        .unwrap();

    // Compression should not be enabled
    assert!(!manager.is_compression_enabled());
    assert_eq!(manager.compression_algorithm(), CompressionAlgorithm::None);

    let table = create_test_table(ctx.clone());
    let table_ref = Arc::new(RwLock::new(table));
    manager.register_table("users", table_ref.clone()).await;

    let checkpoint_id = manager.checkpoint().await.unwrap();
    manager.restore_from(checkpoint_id).await.unwrap();

    let restored_table = table_ref.read().await;
    assert_eq!(restored_table.rows(), 5);
}
