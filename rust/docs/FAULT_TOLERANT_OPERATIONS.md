# Fault-Tolerant Distributed Operations

This document describes the fault-tolerant architecture for distributed operations in Cylon's serverless environment, particularly for AWS Lambda with NAT hole-punching.

## Problem Statement

In serverless environments like AWS Lambda:

1. **Workers can die unexpectedly** - Lambda has a 15-minute maximum execution time
2. **NAT hole-punching connections are fragile** - Network state can be lost
3. **No persistent state** - Everything must be checkpointed externally
4. **Dynamic worker pools** - Workers may join or leave at any time

When a worker dies during a distributed operation (shuffle, join, reduce), the remaining workers need to:
- Detect the failure quickly
- Abort the current operation consistently
- Recover from the last checkpoint
- Continue with the surviving workers

## Execution Model

### Non-Blocking I/O with Progress Loop

Cylon uses **non-blocking socket I/O** with a **progress loop** pattern:

```rust
// Typical Cylon operation pattern
fn wait_all(&mut self) -> CylonResult<()> {
    while self.comm_ptr.communicator_event_progress(Operation::Default)
        == EventProcessStatus::Processing
    {
        // Progress loop - called repeatedly until operation completes
    }
    Ok(())
}
```

The progress loop:
1. Initiates non-blocking I/O operations
2. Polls for completion via `channel_event_progress()`
3. Loops until all operations complete

**Key Insight**: The progress loop is the ideal injection point for heartbeat checking.

### Async Checkpointing of Completed Data

While operations use non-blocking I/O with progress loops, checkpointing can be truly async:
- Only checkpoint **immutable/completed** data
- Never checkpoint data currently being modified
- Background thread handles checkpoint I/O

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Application Layer                             │
│                   (DataFrame operations, joins, etc.)                │
├─────────────────────────────────────────────────────────────────────┤
│                   ResilientOperationExecutor                         │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  • Pre-flight checks (heartbeats, time budget)              │    │
│  │  • Wraps operations with fault tolerance                    │    │
│  │  • Retry logic for transient failures                       │    │
│  │  • Coordinated abort on permanent failure                   │    │
│  │  • Async checkpoint of completed work                       │    │
│  └─────────────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────────────┤
│                   FaultTolerantChannel                               │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  • Wraps underlying channel (Direct)                        │    │
│  │  • Injects heartbeat check into progress loop               │    │
│  │  • Aborts operation early if peer heartbeat expires         │    │
│  │  • Configurable check interval                              │    │
│  └─────────────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────────────┤
│                         WorkerPool                                   │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  • Dynamic worker discovery via Redis heartbeats            │    │
│  │  • Partition assignment and tracking                        │    │
│  │  • Re-partitioning on worker failure                        │    │
│  │  • Epoch-based membership (consistent view)                 │    │
│  └─────────────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────────────┤
│                    Communication Layer                               │
│              (Direct channel with non-blocking I/O)                  │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  • Non-blocking socket I/O                                  │    │
│  │  • TCP keepalive for connection health                      │    │
│  │  • Progress-based completion model                          │    │
│  └─────────────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────────────┤
│                    Coordination Layer (Redis)                        │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  • Worker heartbeats (source of truth for liveness)         │    │
│  │  • Operation status tracking                                │    │
│  │  • Checkpoint coordination                                  │    │
│  │  • Distributed locking                                      │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. FaultTolerantChannel

Wraps the underlying channel to inject heartbeat checking into the progress loop:

```rust
pub struct FaultTolerantChannel<C: Channel> {
    /// Underlying channel
    inner: C,
    /// Redis coordinator for heartbeat checks
    coordinator: Arc<RedisCoordinator>,
    /// How often to check heartbeats during progress loop
    heartbeat_check_interval: Duration,
    /// Last heartbeat check time
    last_heartbeat_check: Instant,
    /// Expected peer workers
    expected_peers: Vec<String>,
    /// Abort flag (set when peer failure detected)
    abort_flag: AtomicBool,
    /// Abort reason
    abort_reason: Mutex<Option<String>>,
}

impl<C: Channel> FaultTolerantChannel<C> {
    /// Enhanced progress that checks heartbeats periodically
    pub fn channel_event_progress_with_heartbeat_check(
        &mut self,
        op: Operation,
    ) -> CylonResult<EventProcessStatus> {
        // Check if already aborted
        if self.abort_flag.load(Ordering::SeqCst) {
            let reason = self.abort_reason.lock().unwrap();
            return Err(CylonError::new(
                Code::ExecutionError,
                format!("Operation aborted: {}", reason.as_deref().unwrap_or("unknown")),
            ));
        }

        // Periodically check heartbeats
        if self.last_heartbeat_check.elapsed() >= self.heartbeat_check_interval {
            self.check_peer_heartbeats()?;
            self.last_heartbeat_check = Instant::now();
        }

        // Delegate to underlying channel
        Ok(self.inner.channel_event_progress(op))
    }

    fn check_peer_heartbeats(&self) -> CylonResult<()> {
        // Quick synchronous check of Redis heartbeats
        // Uses a cached Redis connection to minimize latency
        let dead_peers = self.coordinator.check_heartbeats_sync(&self.expected_peers)?;

        if !dead_peers.is_empty() {
            let reason = format!("Peers failed: {}", dead_peers.join(", "));
            *self.abort_reason.lock().unwrap() = Some(reason.clone());
            self.abort_flag.store(true, Ordering::SeqCst);

            // Signal abort to Redis so other workers know
            self.coordinator.signal_abort_sync(&reason)?;

            return Err(CylonError::new(
                Code::ExecutionError,
                format!("Operation aborted: {}", reason),
            ));
        }

        Ok(())
    }
}
```

### 2. Progress Loop Integration

The existing progress loop pattern:

```rust
// BEFORE: No fault tolerance
fn wait_all(&mut self) -> CylonResult<()> {
    while self.comm_ptr.communicator_event_progress(Operation::Default)
        == EventProcessStatus::Processing
    {}
    Ok(())
}
```

With fault tolerance:

```rust
// AFTER: With heartbeat checking in progress loop
fn wait_all(&mut self) -> CylonResult<()> {
    loop {
        // This now checks heartbeats periodically
        let status = self.fault_tolerant_channel
            .channel_event_progress_with_heartbeat_check(Operation::Default)?;

        if status != EventProcessStatus::Processing {
            break;
        }
    }
    Ok(())
}
```

### 3. ResilientOperationExecutor

Orchestrates fault-tolerant operation execution:

```rust
pub struct ResilientOperationExecutor {
    /// Fault-tolerant channel
    channel: FaultTolerantChannel<Direct>,
    /// Checkpoint manager
    checkpoint_manager: Arc<CheckpointManager>,
    /// Worker pool
    worker_pool: Arc<WorkerPool>,
    /// Configuration
    config: FaultToleranceConfig,
}

impl ResilientOperationExecutor {
    /// Execute operation with fault tolerance
    pub fn execute<F, T>(&mut self, op_name: &str, operation: F) -> Result<T, FaultError>
    where
        F: FnOnce(&mut FaultTolerantChannel<Direct>) -> CylonResult<T>,
    {
        let mut attempts = 0;
        let mut backoff = self.config.initial_backoff;

        loop {
            // Pre-flight check
            self.preflight_check()?;

            // Reset abort flag for new attempt
            self.channel.reset_abort();

            attempts += 1;

            match operation(&mut self.channel) {
                Ok(result) => {
                    // Mark for async checkpoint
                    self.mark_for_checkpoint(&result);
                    return Ok(result);
                }

                Err(error) => {
                    // Classify error
                    if self.channel.was_aborted() {
                        // Peer failure detected during operation
                        let dead_peers = self.channel.get_dead_peers();
                        return Err(FaultError::PeerFailure {
                            dead_peers,
                            error,
                        });
                    }

                    // Check if transient (all peers still alive)
                    let dead = self.check_peer_liveness()?;
                    if dead.is_empty() && attempts < self.config.max_retries {
                        // Transient error, retry
                        std::thread::sleep(backoff);
                        backoff = std::cmp::min(backoff * 2, self.config.max_backoff);
                        continue;
                    }

                    if !dead.is_empty() {
                        return Err(FaultError::PeerFailure {
                            dead_peers: dead,
                            error,
                        });
                    }

                    return Err(FaultError::RetriesExhausted { attempts, error });
                }
            }
        }
    }

    /// Pre-flight check before starting operation
    fn preflight_check(&self) -> Result<(), PreflightError> {
        // 1. Check all expected workers have valid heartbeats
        let dead_workers = self.worker_pool.check_heartbeats()?;
        if !dead_workers.is_empty() {
            return Err(PreflightError::WorkersDead(dead_workers));
        }

        // 2. Check remaining time budget (Lambda)
        if let Some(remaining) = self.get_remaining_time() {
            let required = self.config.estimated_op_time + self.config.checkpoint_reserve;
            if remaining < required {
                return Err(PreflightError::InsufficientTime { remaining, required });
            }
        }

        // 3. Check no abort signal pending
        if self.channel.coordinator.is_abort_signaled()? {
            return Err(PreflightError::AbortPending);
        }

        Ok(())
    }
}
```

### 4. WorkerPool

Manages dynamic worker membership:

```rust
pub struct WorkerPool {
    /// Current epoch (increments on membership change)
    epoch: AtomicU64,
    /// Redis coordinator
    coordinator: Arc<RedisCoordinator>,
    /// Current partition assignment
    partitions: RwLock<HashMap<String, Vec<u32>>>,
    /// This worker's ID
    worker_id: String,
}

impl WorkerPool {
    /// Join the worker pool
    pub fn join(&self) -> CylonResult<WorkerEpoch> {
        // Register with Redis
        self.coordinator.register_worker(&self.worker_id)?;

        // Start heartbeat thread
        self.start_heartbeat_thread();

        // Get current epoch
        self.sync_epoch()
    }

    /// Check heartbeats of expected workers
    pub fn check_heartbeats(&self) -> CylonResult<Vec<String>> {
        let partitions = self.partitions.read().unwrap();
        let expected: Vec<_> = partitions.keys().cloned().collect();
        self.coordinator.check_heartbeats_sync(&expected)
    }

    /// Rebalance partitions after worker failure
    pub fn rebalance(&self, active_workers: Vec<String>) -> CylonResult<WorkerEpoch> {
        let mut partitions = self.partitions.write().unwrap();

        // Collect orphaned partitions
        let mut orphaned: Vec<u32> = Vec::new();
        let dead: Vec<_> = partitions.keys()
            .filter(|w| !active_workers.contains(w))
            .cloned()
            .collect();

        for worker in &dead {
            if let Some(parts) = partitions.remove(worker) {
                orphaned.extend(parts);
            }
        }

        // Redistribute round-robin
        for (i, partition) in orphaned.into_iter().enumerate() {
            let worker = &active_workers[i % active_workers.len()];
            partitions.entry(worker.clone())
                .or_insert_with(Vec::new)
                .push(partition);
        }

        // Increment epoch
        let new_epoch = self.epoch.fetch_add(1, Ordering::SeqCst) + 1;

        Ok(WorkerEpoch {
            epoch: new_epoch,
            workers: active_workers,
            partitions: partitions.clone(),
        })
    }
}
```

## Failure Detection Timeline

With heartbeat checking in the progress loop:

```
T=0:    Workers A, B, C start shuffle operation
        Progress loop begins, checking heartbeats every 200ms

T=5:    Worker B hits Lambda timeout, dies
T=5:    B's TCP connections may start failing

T=5.2:  A's progress loop checks heartbeats
        B's heartbeat still valid (TTL not expired yet)

T=10:   B's heartbeat expires (TTL=10s, last sent at T=0)

T=10.2: A's progress loop checks heartbeats
        B's heartbeat MISSING - detected!
        A sets abort flag, signals abort in Redis
        A's operation returns Err(PeerFailure)

T=10.3: C's progress loop checks heartbeats
        Sees abort signal in Redis
        C's operation returns Err(PeerFailure)

T=11:   Both A and C begin coordinated recovery
```

**Detection time**: ~200ms after heartbeat expires (not minutes like TCP timeout)

## Heartbeat Check Performance

To avoid slowing down the progress loop:

1. **Batched checks**: Check all expected peers in one Redis call
2. **Cached connection**: Reuse Redis connection
3. **Configurable interval**: Default 200ms, tunable per environment
4. **Skip if recent**: Don't check if last check was < interval ago

```rust
// Efficient batch heartbeat check
pub fn check_heartbeats_sync(&self, workers: &[String]) -> CylonResult<Vec<String>> {
    let mut conn = self.get_cached_connection()?;
    let prefix = self.key_prefix();

    // Build pipeline for batch check
    let mut pipe = redis::pipe();
    for worker in workers {
        let key = format!("{}:worker:{}:heartbeat", prefix, worker);
        pipe.exists(&key);
    }

    // Execute batch
    let results: Vec<bool> = pipe.query(&mut conn)?;

    // Collect dead workers
    let dead: Vec<String> = workers.iter()
        .zip(results.iter())
        .filter(|(_, &exists)| !exists)
        .map(|(w, _)| w.clone())
        .collect();

    Ok(dead)
}
```

## Recovery Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Recovery Sequence                                │
│                                                                     │
│  ┌──────────────┐                                                   │
│  │ Peer Failure │                                                   │
│  │  Detected    │                                                   │
│  └──────┬───────┘                                                   │
│         │                                                           │
│         ▼                                                           │
│  ┌──────────────┐     Already set?    ┌──────────────┐             │
│  │ Signal Abort │────────────────────▶│    Skip      │             │
│  │  (Redis)     │      Yes            └──────────────┘             │
│  └──────┬───────┘                                                   │
│         │ No                                                        │
│         ▼                                                           │
│  ┌──────────────┐                                                   │
│  │Wait for Async│                                                   │
│  │ Checkpoint   │                                                   │
│  └──────┬───────┘                                                   │
│         │                                                           │
│         ▼                                                           │
│  ┌──────────────┐                                                   │
│  │  Restore     │                                                   │
│  │  Checkpoint  │                                                   │
│  └──────┬───────┘                                                   │
│         │                                                           │
│         ▼                                                           │
│  ┌──────────────┐                                                   │
│  │ Sync Epoch   │  (discover surviving workers)                     │
│  └──────┬───────┘                                                   │
│         │                                                           │
│         ▼                                                           │
│  ┌──────────────┐                                                   │
│  │ Rebalance    │  (redistribute dead worker's partitions)         │
│  │ Partitions   │                                                   │
│  └──────┬───────┘                                                   │
│         │                                                           │
│         ▼                                                           │
│  ┌──────────────┐                                                   │
│  │   Resume     │                                                   │
│  │  Operations  │                                                   │
│  └──────────────┘                                                   │
└─────────────────────────────────────────────────────────────────────┘
```

## Configuration

```rust
pub struct FaultToleranceConfig {
    // Heartbeat settings
    pub heartbeat_interval: Duration,           // 2s - how often to send heartbeat
    pub heartbeat_ttl: Duration,                // 10s - when heartbeat expires
    pub heartbeat_check_interval: Duration,     // 200ms - how often to check in progress loop

    // Retry settings
    pub max_retries: u32,                       // 3
    pub initial_backoff: Duration,              // 100ms
    pub max_backoff: Duration,                  // 5s

    // Time budget (Lambda)
    pub min_time_budget: Duration,              // 60s
    pub checkpoint_reserve: Duration,           // 20s
    pub estimated_op_time: Duration,            // 30s

    // Checkpoint settings
    pub checkpoint_trigger_threshold: usize,    // After N completed ops
    pub checkpoint_interval: Duration,          // Or every N seconds
}

impl FaultToleranceConfig {
    pub fn for_serverless() -> Self {
        Self {
            heartbeat_interval: Duration::from_secs(2),
            heartbeat_ttl: Duration::from_secs(10),
            heartbeat_check_interval: Duration::from_millis(200),
            max_retries: 3,
            initial_backoff: Duration::from_millis(100),
            max_backoff: Duration::from_secs(5),
            min_time_budget: Duration::from_secs(60),
            checkpoint_reserve: Duration::from_secs(20),
            estimated_op_time: Duration::from_secs(30),
            checkpoint_trigger_threshold: 5,
            checkpoint_interval: Duration::from_secs(60),
        }
    }
}
```

## Usage Example

```rust
use cylon::fault_tolerance::{
    FaultTolerantChannel, ResilientOperationExecutor,
    WorkerPool, FaultToleranceConfig,
};

// Setup
let config = FaultToleranceConfig::for_serverless();
let coordinator = RedisCoordinator::new(redis_config).await?;
let channel = Direct::new(&backend);
let ft_channel = FaultTolerantChannel::new(channel, coordinator.clone(), &config);

let executor = ResilientOperationExecutor::new(
    ft_channel,
    checkpoint_manager,
    WorkerPool::new(coordinator, worker_id),
    config,
);

// Join worker pool
let epoch = executor.worker_pool().join()?;
println!("Joined epoch {} with {} workers", epoch.epoch, epoch.workers.len());

// Execute operations with automatic fault tolerance
loop {
    match executor.execute("shuffle", |channel| {
        // Your distributed operation here
        // Heartbeats are checked automatically in progress loop
        shuffle_data(channel, &partitions)
    }) {
        Ok(result) => {
            // Success - result is marked for async checkpoint
            process_result(result);
        }
        Err(FaultError::PeerFailure { dead_peers, .. }) => {
            // Worker(s) died - recover and continue
            println!("Workers failed: {:?}, recovering...", dead_peers);
            executor.recover()?;
            continue;
        }
        Err(FaultError::InsufficientTime { remaining, .. }) => {
            // Running out of time - checkpoint and exit
            println!("Time budget low ({:?}), checkpointing...", remaining);
            executor.final_checkpoint()?;
            break;
        }
        Err(e) => {
            return Err(e.into());
        }
    }

    if no_more_work {
        break;
    }
}

// Final checkpoint before exit
executor.final_checkpoint()?;
```

## Summary

| Aspect | Implementation |
|--------|----------------|
| I/O Model | Non-blocking with progress loop |
| Failure Detection | Heartbeat check in progress loop (200ms interval) |
| Detection Latency | ~200ms after heartbeat expires |
| Retry Policy | Exponential backoff, max 3 attempts |
| Checkpointing | Async for completed/immutable data |
| Recovery | Coordinated abort → restore → rebalance |
| Time Management | Pre-flight budget check, checkpoint reserve |
