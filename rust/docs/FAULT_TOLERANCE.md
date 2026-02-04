# Fault Tolerance for Serverless Distributed Operations

This document describes the fault tolerance system implemented for Cylon's FMI (Function-as-a-service Message Interface) communication layer, designed specifically for serverless environments like AWS Lambda.

## Problem Statement

In serverless environments, workers can terminate unexpectedly due to:
- **Lambda 15-minute timeout**: Long-running jobs get terminated
- **Spot instance preemption**: Cost-optimized instances can be reclaimed
- **Network partitions**: NAT hole-punched connections can break
- **Resource limits**: Memory or CPU throttling

When a worker dies during a distributed operation (e.g., shuffle, join), other workers may hang indefinitely waiting for messages that will never arrive.

## Solution Architecture

The fault tolerance system provides three layers of protection:

```
┌─────────────────────────────────────────────────────────────────┐
│                    ResilientExecutor                            │
│     (Wraps operations with retry, abort, restore, rebalance)    │
└─────────────────────────────────────────────────────────────────┘
                               │
           ┌───────────────────┼───────────────────┐
           ▼                   ▼                   ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ HeartbeatWatcher│  │   WorkerPool    │  │ RecoveryHandler │
│ (Background     │  │ (Partition      │  │ (Checkpoint     │
│  thread + Redis)│  │  management)    │  │  restore)       │
└─────────────────┘  └─────────────────┘  └─────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FMICylonChannel                              │
│     (Progress loop checks atomic flag - instant, no I/O)        │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### 1. CheckpointRecoveryHandler (NEW)

Bridges the fault tolerance layer with the checkpointing system. This is the key integration point that connects `ResilientExecutor` with `CheckpointManager`.

```rust
use cylon::checkpoint::{
    CheckpointManager, CheckpointConfig, FileSystemStorage,
    ArrowIpcSerializer, CompositeTrigger, LocalCoordinator,
};
use cylon::net::fmi::{CheckpointRecoveryHandler, ResilientExecutor, FaultToleranceConfig};

// Create checkpoint manager
let manager = Arc::new(CheckpointManager::new(
    ctx.clone(),
    coordinator,
    storage,
    serializer,
    trigger,
    config,
));

// Create recovery handler wrapping the manager
let recovery_handler = Arc::new(CheckpointRecoveryHandler::new(manager));

// Register tables for checkpointing
recovery_handler.register_table("orders", orders_table).await;
recovery_handler.register_table("customers", customers_table).await;

// Create resilient executor with checkpoint support
let executor = ResilientExecutor::new(
    redis_coordinator,
    recovery_handler.clone(),
    "worker-0".to_string(),
    FaultToleranceConfig::for_serverless(),
);

// Operations will automatically checkpoint/restore on failure
executor.execute("shuffle", || shuffle_data(&channel)).await?;
```

### 2. HeartbeatWatcher

Monitors peer health via Redis heartbeats in a background thread.

**Key Design**: The background thread checks Redis and sets atomic flags. The progress loop only reads the atomic flag (instant, no blocking I/O).

```rust
use cylon::net::fmi::{HeartbeatWatcher, FaultToleranceConfig};

let config = FaultToleranceConfig::for_serverless();
let watcher = HeartbeatWatcher::new("worker-0".to_string(), config);

// Set peers to monitor (excludes self automatically)
watcher.set_expected_peers(vec![
    "worker-0".to_string(),
    "worker-1".to_string(),
    "worker-2".to_string(),
]);

// Start background monitoring
watcher.start(coordinator.clone());

// In progress loop - instant check, no I/O
if watcher.has_peer_failed() {
    let dead = watcher.get_dead_peers();
    // Handle failure...
}

// Clean up
watcher.stop();
```

### 2. WorkerPool

Manages dynamic worker membership and partition assignments with epoch-based tracking.

```rust
use cylon::net::fmi::WorkerPool;

let pool = WorkerPool::new("worker-0".to_string());

// Initialize with workers and partition count
pool.initialize(
    vec!["worker-0".to_string(), "worker-1".to_string(), "worker-2".to_string()],
    12, // 12 partitions
);

// Get my partitions (round-robin assignment)
let my_partitions = pool.my_partitions(); // [0, 3, 6, 9]

// After worker failure, rebalance
pool.rebalance_after_failure(&["worker-2".to_string()])?;

// Partitions redistributed, epoch incremented
assert_eq!(pool.epoch(), 2);
```

### 3. ResilientExecutor

Wraps distributed operations with automatic fault tolerance:
- Pre-flight checks (heartbeats, time budget)
- Retry with exponential backoff for transient failures
- Coordinated abort on peer failure
- Checkpoint restore via RecoveryHandler trait
- Partition rebalancing

```rust
use cylon::net::fmi::{ResilientExecutor, RecoveryHandler, FaultToleranceConfig};

// Implement custom recovery logic
struct MyRecoveryHandler { /* ... */ }

#[async_trait]
impl RecoveryHandler for MyRecoveryHandler {
    async fn restore_checkpoint(&self) -> CylonResult<Option<u64>> {
        // Restore from latest checkpoint
        Ok(Some(checkpoint_id))
    }

    async fn force_checkpoint(&self) -> CylonResult<u64> {
        // Create checkpoint before timeout
        Ok(new_checkpoint_id)
    }
}

// Create executor
let executor = ResilientExecutor::new(
    coordinator,
    Arc::new(MyRecoveryHandler::new()),
    "worker-0".to_string(),
    FaultToleranceConfig::for_serverless(),
);

executor.start_watcher();

// Execute with automatic fault tolerance
let result = executor.execute("shuffle", || {
    shuffle_data(&channel, &partitions)
}).await?;

executor.stop_watcher();
```

### 4. FMICylonChannel Integration

The channel's progress loop integrates with HeartbeatWatcher for instant failure detection:

```rust
// Set up channel with heartbeat watcher
let mut channel = FMICylonChannel::new(communicator, mode, redis_host, redis_port, namespace);
channel.set_heartbeat_watcher(watcher.clone());

// During progress_sends/progress_receives, the channel checks:
// - watcher.has_peer_failed() -> instant atomic read
// - Sets peer_failure_detected flag if true
// - Returns early from progress loop

// Check for detected failures
if channel.has_peer_failure() {
    // Handle failure, reset for retry
    channel.reset_peer_failure();
}
```

## Configuration

```rust
use std::time::Duration;
use cylon::net::fmi::FaultToleranceConfig;

let config = FaultToleranceConfig::default()
    // How often background thread checks Redis
    .with_heartbeat_check_interval(Duration::from_millis(200))
    // Timeout for individual operations
    .with_operation_timeout(Duration::from_secs(30))
    // Max retry attempts for transient failures
    .with_max_retries(3);

// Or use serverless-optimized defaults
let config = FaultToleranceConfig::for_serverless();
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `heartbeat_check_interval` | 200ms | How often to check peer heartbeats |
| `operation_timeout` | 30s | Timeout for individual operations |
| `max_retries` | 3 | Max retry attempts for transient failures |
| `initial_backoff` | 100ms | Initial retry backoff duration |
| `max_backoff` | 5s | Maximum retry backoff duration |
| `checkpoint_reserve` | 20s | Time reserved for checkpoint before timeout |
| `estimated_op_time` | 30s | Estimated time for a single operation |

## Recovery Flow

When a peer failure is detected:

```
1. HeartbeatWatcher detects dead peer via Redis
   └── Sets atomic flag (peer_failed = true)

2. Progress loop reads flag (instant, no I/O)
   └── Returns early, operation fails

3. ResilientExecutor catches failure
   ├── Signals abort to other workers via Redis
   ├── Calls RecoveryHandler.restore_checkpoint()
   ├── Calls WorkerPool.rebalance_after_failure()
   ├── Updates HeartbeatWatcher with new peer list
   └── Retries operation with new partition assignment
```

## Testing

### Unit Tests (no Redis required)

```bash
cargo test --features fmi,redis --test fault_tolerance_test
```

### Integration Tests (require Redis)

```bash
# Set Redis URL for your environment
REDIS_URL=redis://10.211.55.2:6379 cargo test --features fmi,redis --test fault_tolerance_test -- --ignored
```

### Redis Keys Created

The system creates keys with the pattern:
- `cylon:{job_id}:heartbeat:{worker_id}` - Worker heartbeats
- `cylon:{job_id}:workers` - Active worker set
- `cylon:{job_id}:checkpoint:{id}:*` - Checkpoint coordination

## Files

| File | Description |
|------|-------------|
| `src/net/fmi/fault_tolerance.rs` | Core fault tolerance components (HeartbeatWatcher, WorkerPool, ResilientExecutor, CheckpointRecoveryHandler) |
| `src/net/fmi/cylon_channel.rs` | Channel integration with heartbeat checking |
| `src/net/fmi/mod.rs` | Module exports |
| `src/checkpoint/coordinator.rs` | Redis coordinator with heartbeat support |
| `src/checkpoint/manager.rs` | CheckpointManager for actual checkpoint/restore |
| `tests/fault_tolerance_test.rs` | Unit and integration tests (18 tests total) |

## Dependencies

- `redis` feature flag required
- `fmi` feature flag required
- Redis server for coordination
- `tokio` runtime for async operations

## Limitations

1. **Redis dependency**: Requires external Redis server for coordination
2. **Detection latency**: ~200-500ms to detect dead peers (configurable)
3. **Recovery overhead**: Checkpoint restore adds latency to recovery
4. **Single point of failure**: Redis itself must be highly available

## Future Improvements

- [ ] Support for multiple Redis backends (cluster mode)
- [ ] Adaptive heartbeat intervals based on network conditions
- [ ] Partial operation recovery (resume from last successful partition)
- [ ] Integration with AWS Lambda remaining time API
