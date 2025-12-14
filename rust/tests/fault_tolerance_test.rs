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

//! Tests for fault tolerance components
//!
//! These tests verify the behavior of HeartbeatWatcher, WorkerPool,
//! and ResilientExecutor for serverless fault tolerance.

#[cfg(all(feature = "fmi", feature = "redis"))]
mod fault_tolerance_tests {
    use std::sync::Arc;
    use std::time::Duration;

    use cylon::net::fmi::{
        FaultToleranceConfig,
        HeartbeatWatcher,
        WorkerPool,
    };

    /// Get Redis URL from environment or use default
    /// Set REDIS_URL environment variable to override (e.g., "redis://host.docker.internal:6379")
    fn get_redis_url() -> String {
        std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://localhost:6379".to_string())
    }

    // ========================================================================
    // HeartbeatWatcher Tests
    // ========================================================================

    #[test]
    fn test_heartbeat_watcher_creation() {
        let config = FaultToleranceConfig::default();
        let watcher = HeartbeatWatcher::new("worker-0".to_string(), config);

        // Initially no failures
        assert!(!watcher.has_peer_failed());
        assert!(!watcher.is_abort_signaled());
        assert!(watcher.get_dead_peers().is_empty());
    }

    #[test]
    fn test_heartbeat_watcher_set_expected_peers() {
        let config = FaultToleranceConfig::default();
        let watcher = HeartbeatWatcher::new("worker-0".to_string(), config);

        // Set expected peers (should exclude self)
        watcher.set_expected_peers(vec![
            "worker-0".to_string(),
            "worker-1".to_string(),
            "worker-2".to_string(),
        ]);

        // Worker-0 should be filtered out since it's the local worker
    }

    #[test]
    fn test_heartbeat_watcher_signal_abort() {
        let config = FaultToleranceConfig::default();
        let watcher = HeartbeatWatcher::new("worker-0".to_string(), config);

        assert!(!watcher.is_abort_signaled());

        watcher.signal_abort("Test abort reason");

        assert!(watcher.is_abort_signaled());
        assert_eq!(
            watcher.get_abort_reason(),
            Some("Test abort reason".to_string())
        );
    }

    #[test]
    fn test_heartbeat_watcher_reset() {
        let config = FaultToleranceConfig::default();
        let watcher = HeartbeatWatcher::new("worker-0".to_string(), config);

        // Signal abort
        watcher.signal_abort("Test abort");
        assert!(watcher.is_abort_signaled());

        // Reset
        watcher.reset();

        assert!(!watcher.is_abort_signaled());
        assert!(!watcher.has_peer_failed());
        assert!(watcher.get_abort_reason().is_none());
        assert!(watcher.get_dead_peers().is_empty());
    }

    #[test]
    fn test_heartbeat_watcher_check_for_failure_no_failure() {
        let config = FaultToleranceConfig::default();
        let watcher = HeartbeatWatcher::new("worker-0".to_string(), config);

        // Should succeed when no failure
        assert!(watcher.check_for_failure().is_ok());
    }

    #[test]
    fn test_heartbeat_watcher_check_for_failure_with_abort() {
        let config = FaultToleranceConfig::default();
        let watcher = HeartbeatWatcher::new("worker-0".to_string(), config);

        watcher.signal_abort("Operation aborted");

        let result = watcher.check_for_failure();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("aborted"));
    }

    // ========================================================================
    // WorkerPool Tests
    // ========================================================================

    #[test]
    fn test_worker_pool_creation() {
        let pool = WorkerPool::new("worker-0".to_string());

        assert_eq!(pool.epoch(), 0);
        assert!(pool.workers().is_empty());
        assert!(pool.my_partitions().is_empty());
    }

    #[test]
    fn test_worker_pool_initialize() {
        let pool = WorkerPool::new("worker-0".to_string());

        pool.initialize(
            vec![
                "worker-0".to_string(),
                "worker-1".to_string(),
                "worker-2".to_string(),
            ],
            9,
        );

        assert_eq!(pool.epoch(), 1);
        assert_eq!(pool.workers().len(), 3);

        // Check partition distribution (round-robin)
        let my_partitions = pool.my_partitions();
        assert_eq!(my_partitions.len(), 3); // 9 partitions / 3 workers = 3 each
        assert!(my_partitions.contains(&0));
        assert!(my_partitions.contains(&3));
        assert!(my_partitions.contains(&6));
    }

    #[test]
    fn test_worker_pool_partition_distribution() {
        let pool = WorkerPool::new("worker-1".to_string());

        pool.initialize(
            vec![
                "worker-0".to_string(),
                "worker-1".to_string(),
                "worker-2".to_string(),
            ],
            9,
        );

        // worker-1 should get partitions 1, 4, 7
        let my_partitions = pool.my_partitions();
        assert_eq!(my_partitions.len(), 3);
        assert!(my_partitions.contains(&1));
        assert!(my_partitions.contains(&4));
        assert!(my_partitions.contains(&7));
    }

    #[test]
    fn test_worker_pool_rebalance_after_failure() {
        let pool = WorkerPool::new("worker-0".to_string());

        // Initial setup with 3 workers, 9 partitions
        pool.initialize(
            vec![
                "worker-0".to_string(),
                "worker-1".to_string(),
                "worker-2".to_string(),
            ],
            9,
        );

        let initial_epoch = pool.epoch();

        // Worker-2 fails
        let result = pool.rebalance_after_failure(&["worker-2".to_string()]);
        assert!(result.is_ok());

        // Epoch should increment
        assert_eq!(pool.epoch(), initial_epoch + 1);

        // Only 2 workers now
        assert_eq!(pool.workers().len(), 2);
        assert!(!pool.workers().contains(&"worker-2".to_string()));

        // All partitions should still be assigned
        let all = pool.all_partitions();
        let total: usize = all.values().map(|v| v.len()).sum();
        assert_eq!(total, 9);
    }

    #[test]
    fn test_worker_pool_multiple_failures() {
        let pool = WorkerPool::new("worker-0".to_string());

        pool.initialize(
            vec![
                "worker-0".to_string(),
                "worker-1".to_string(),
                "worker-2".to_string(),
                "worker-3".to_string(),
            ],
            12,
        );

        // Two workers fail simultaneously
        let result = pool.rebalance_after_failure(&[
            "worker-1".to_string(),
            "worker-3".to_string(),
        ]);
        assert!(result.is_ok());

        // Only 2 workers now
        assert_eq!(pool.workers().len(), 2);

        // All partitions should still be assigned
        let all = pool.all_partitions();
        let total: usize = all.values().map(|v| v.len()).sum();
        assert_eq!(total, 12);

        // Each remaining worker should have 6 partitions
        for (_, partitions) in all {
            assert_eq!(partitions.len(), 6);
        }
    }

    // ========================================================================
    // FaultToleranceConfig Tests
    // ========================================================================

    #[test]
    fn test_fault_tolerance_config_default() {
        let config = FaultToleranceConfig::default();

        assert_eq!(config.heartbeat_check_interval, Duration::from_millis(200));
        assert_eq!(config.operation_timeout, Duration::from_secs(30));
        assert_eq!(config.max_retries, 3);
    }

    #[test]
    fn test_fault_tolerance_config_builder() {
        let config = FaultToleranceConfig::for_serverless()
            .with_heartbeat_check_interval(Duration::from_millis(100))
            .with_operation_timeout(Duration::from_secs(60))
            .with_max_retries(5);

        assert_eq!(config.heartbeat_check_interval, Duration::from_millis(100));
        assert_eq!(config.operation_timeout, Duration::from_secs(60));
        assert_eq!(config.max_retries, 5);
    }

    // ========================================================================
    // Integration Tests (require Redis)
    // ========================================================================

    /// Integration test that requires a running Redis instance
    /// Run with: REDIS_URL=redis://host:port cargo test --features fmi,redis -- --ignored
    #[tokio::test]
    #[ignore]
    async fn test_heartbeat_watcher_with_redis() {
        use cylon::checkpoint::{RedisCoordinator, RedisCoordinatorConfig};

        let redis_url = get_redis_url();
        println!("Using Redis URL: {}", redis_url);

        let config = RedisCoordinatorConfig::new(
            redis_url,
            "fault-tolerance-test".to_string(),
            "worker-0".to_string(),
        )
        .with_heartbeat_interval(Duration::from_millis(100))
        .with_heartbeat_ttl(Duration::from_millis(500));

        let coordinator = match RedisCoordinator::new(config).await {
            Ok(c) => Arc::new(c),
            Err(e) => {
                println!("Skipping test: Redis not available: {}", e);
                return;
            }
        };

        let ft_config = FaultToleranceConfig::default()
            .with_heartbeat_check_interval(Duration::from_millis(100));

        let watcher = HeartbeatWatcher::new("worker-0".to_string(), ft_config);
        watcher.set_expected_peers(vec!["worker-1".to_string()]);

        // Start watcher
        watcher.start(coordinator.clone());

        // Give it time to detect the missing peer
        tokio::time::sleep(Duration::from_millis(700)).await;

        // worker-1 should be detected as dead (not sending heartbeats)
        assert!(watcher.has_peer_failed());
        let dead = watcher.get_dead_peers();
        assert!(dead.contains(&"worker-1".to_string()));

        // Clean up
        watcher.stop();
    }

    /// Integration test for ResilientExecutor
    /// Run with: REDIS_URL=redis://host:port cargo test --features fmi,redis -- --ignored
    #[tokio::test]
    #[ignore]
    async fn test_resilient_executor_with_redis() {
        use cylon::checkpoint::{RedisCoordinator, RedisCoordinatorConfig};
        use cylon::net::fmi::{ResilientExecutor, NoOpRecoveryHandler};

        let redis_url = get_redis_url();
        println!("Using Redis URL: {}", redis_url);

        let config = RedisCoordinatorConfig::new(
            redis_url,
            "resilient-executor-test".to_string(),
            "worker-0".to_string(),
        );

        let coordinator = match RedisCoordinator::new(config).await {
            Ok(c) => Arc::new(c),
            Err(e) => {
                println!("Skipping test: Redis not available: {}", e);
                return;
            }
        };

        let recovery = Arc::new(NoOpRecoveryHandler);
        let ft_config = FaultToleranceConfig::for_serverless();

        let executor = ResilientExecutor::new(
            coordinator,
            recovery,
            "worker-0".to_string(),
            ft_config,
        );

        // Simple operation should succeed
        let mut counter = 0;
        let result = executor
            .execute("test-op", || {
                counter += 1;
                Ok::<i32, cylon::error::CylonError>(counter)
            })
            .await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 1);
    }

    /// Test operation with transient failures and retries
    /// Run with: REDIS_URL=redis://host:port cargo test --features fmi,redis -- --ignored
    #[tokio::test]
    #[ignore]
    async fn test_resilient_executor_retry() {
        use cylon::checkpoint::{RedisCoordinator, RedisCoordinatorConfig};
        use cylon::net::fmi::{ResilientExecutor, NoOpRecoveryHandler};
        use cylon::error::{CylonError, Code};

        let redis_url = get_redis_url();
        println!("Using Redis URL: {}", redis_url);

        let config = RedisCoordinatorConfig::new(
            redis_url,
            "retry-test".to_string(),
            "worker-0".to_string(),
        );

        let coordinator = match RedisCoordinator::new(config).await {
            Ok(c) => Arc::new(c),
            Err(e) => {
                println!("Skipping test: Redis not available: {}", e);
                return;
            }
        };

        let recovery = Arc::new(NoOpRecoveryHandler);
        let ft_config = FaultToleranceConfig::for_serverless()
            .with_max_retries(3);

        let executor = ResilientExecutor::new(
            coordinator,
            recovery,
            "worker-0".to_string(),
            ft_config,
        );

        // Operation that fails twice then succeeds
        let mut attempts = 0;
        let result = executor
            .execute("retry-op", || {
                attempts += 1;
                if attempts < 3 {
                    Err(CylonError::new(Code::ExecutionError, "Transient failure"))
                } else {
                    Ok(attempts)
                }
            })
            .await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 3);
    }
}
