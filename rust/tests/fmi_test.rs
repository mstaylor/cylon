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

//! Tests for FMI (Function-as-a-service Message Interface) communicator
//!
//! ## Test Categories
//!
//! 1. **Unit Tests** - Test configuration, builder pattern, types (no network required)
//! 2. **Integration Tests** - Test communicator initialization (requires TCPunch server + Redis)
//! 3. **Multi-process Tests** - Test actual communication (requires multiple processes)
//!
//! ## Running Integration Tests
//!
//! Integration tests require:
//! - TCPunch server running (default: localhost:8080)
//! - Redis server running (default: localhost:6379)
//!
//! ```bash
//! # Start TCPunch server (in separate terminal)
//! # Start Redis server (in separate terminal)
//!
//! # Run tests
//! cargo test --features fmi fmi_test -- --ignored
//! ```
//!
//! ## Environment Variables
//!
//! - `CYLON_FMI_HOST` - TCPunch server host (default: localhost)
//! - `CYLON_FMI_PORT` - TCPunch server port (default: 8080)
//! - `CYLON_REDIS_HOST` - Redis host (default: localhost)
//! - `CYLON_REDIS_PORT` - Redis port (default: 6379)
//! - `CYLON_RANK` - Process rank for multi-process tests
//! - `CYLON_WORLD_SIZE` - Total number of processes

// ============================================================================
// Unit Tests - No network required
// ============================================================================

#[cfg(feature = "fmi")]
mod fmi_unit_tests {
    use cylon::net::fmi::{FMIConfig, FMIConfigBuilder};
    use cylon::net::{CommConfig, CommType};

    #[test]
    fn test_fmi_config_builder_defaults() {
        let config = FMIConfig::builder().build();

        assert_eq!(config.get_rank(), 0);
        assert_eq!(config.get_world_size(), 1);
        assert_eq!(config.get_comm_name(), "cylon");
        assert!(config.is_nonblocking());
        assert_eq!(config.get_redis_host(), "localhost");
        assert_eq!(config.get_redis_port(), 6379);
        assert_eq!(config.get_redis_namespace(), "cylon");

        println!("✓ FMIConfig builder defaults are correct");
    }

    #[test]
    fn test_fmi_config_builder_custom() {
        let config = FMIConfig::builder()
            .rank(2)
            .world_size(8)
            .host("tcpunch.example.com")
            .port(9000)
            .max_timeout(60000)
            .resolve_ip(true)
            .comm_name("my_comm")
            .nonblocking(false)
            .enable_ping(false)
            .redis_host("redis.example.com")
            .redis_port(6380)
            .redis_namespace("test_ns")
            .build();

        assert_eq!(config.get_rank(), 2);
        assert_eq!(config.get_world_size(), 8);
        assert_eq!(config.get_comm_name(), "my_comm");
        assert!(!config.is_nonblocking());
        assert_eq!(config.get_redis_host(), "redis.example.com");
        assert_eq!(config.get_redis_port(), 6380);
        assert_eq!(config.get_redis_namespace(), "test_ns");

        println!("✓ FMIConfig builder custom values work correctly");
    }

    #[test]
    fn test_fmi_config_comm_type() {
        let config = FMIConfig::builder().build();
        assert_eq!(config.get_type(), CommType::Fmi);

        println!("✓ FMIConfig returns correct CommType");
    }

    #[test]
    fn test_fmi_config_with_host() {
        let config = FMIConfig::with_host(
            1,           // rank
            4,           // world_size
            "localhost", // host
            8080,        // port
            30000,       // max_timeout
            false,       // resolve_ip
            "test",      // comm_name
            true,        // nonblocking
        );

        assert_eq!(config.get_rank(), 1);
        assert_eq!(config.get_world_size(), 4);
        assert_eq!(config.get_comm_name(), "test");
        assert!(config.is_nonblocking());

        println!("✓ FMIConfig::with_host works correctly");
    }

    #[test]
    fn test_fmi_config_with_host_and_ping() {
        let config = FMIConfig::with_host_and_ping(
            0,           // rank
            2,           // world_size
            "localhost", // host
            8080,        // port
            30000,       // max_timeout
            false,       // resolve_ip
            "test",      // comm_name
            true,        // nonblocking
            true,        // enable_ping
        );

        assert_eq!(config.get_rank(), 0);
        assert_eq!(config.get_world_size(), 2);

        println!("✓ FMIConfig::with_host_and_ping works correctly");
    }

    #[test]
    fn test_fmi_config_with_redis() {
        let config = FMIConfig::with_redis(
            0,           // rank
            4,           // world_size
            "localhost", // host
            8080,        // port
            30000,       // max_timeout
            false,       // resolve_ip
            "test",      // comm_name
            true,        // nonblocking
            true,        // enable_ping
            "redis.local", // redis_host
            6379,        // redis_port
            "cylon_test", // redis_namespace
        );

        assert_eq!(config.get_redis_host(), "redis.local");
        assert_eq!(config.get_redis_port(), 6379);
        assert_eq!(config.get_redis_namespace(), "cylon_test");

        println!("✓ FMIConfig::with_redis works correctly");
    }

    #[test]
    fn test_fmi_config_builder_is_default() {
        let builder1 = FMIConfigBuilder::new();
        let builder2 = FMIConfigBuilder::default();

        let config1 = builder1.build();
        let config2 = builder2.build();

        assert_eq!(config1.get_rank(), config2.get_rank());
        assert_eq!(config1.get_world_size(), config2.get_world_size());

        println!("✓ FMIConfigBuilder::new() == FMIConfigBuilder::default()");
    }

    #[test]
    fn test_fmi_config_clone() {
        let config1 = FMIConfig::builder()
            .rank(1)
            .world_size(4)
            .build();

        let config2 = config1.clone();

        assert_eq!(config1.get_rank(), config2.get_rank());
        assert_eq!(config1.get_world_size(), config2.get_world_size());

        println!("✓ FMIConfig clone works correctly");
    }

    #[test]
    fn test_fmi_comm_type_enum() {
        assert_eq!(CommType::Fmi, CommType::Fmi);
        assert_ne!(CommType::Fmi, CommType::Local);

        println!("✓ FMI CommType enum works correctly");
    }
}

// ============================================================================
// Common Types Unit Tests
// ============================================================================

#[cfg(feature = "fmi")]
mod fmi_common_types_tests {
    use cylon::net::fmi::{Mode, NbxStatus, Operation, Hint, BackendType, FmiContext};

    #[test]
    fn test_mode_enum() {
        assert_eq!(Mode::Blocking, Mode::Blocking);
        assert_eq!(Mode::NonBlocking, Mode::NonBlocking);
        assert_ne!(Mode::Blocking, Mode::NonBlocking);

        println!("✓ Mode enum works correctly");
    }

    #[test]
    fn test_nbx_status_enum() {
        assert_eq!(NbxStatus::Success, NbxStatus::Success);
        assert_eq!(NbxStatus::SendFailed, NbxStatus::SendFailed);
        assert_ne!(NbxStatus::Success, NbxStatus::SendFailed);

        println!("✓ NbxStatus enum works correctly");
    }

    #[test]
    fn test_operation_enum() {
        assert_eq!(Operation::Send, Operation::Send);
        assert_eq!(Operation::Bcast, Operation::Bcast);
        assert_eq!(Operation::Allgather, Operation::Allgather);
        assert_eq!(Operation::Barrier, Operation::Barrier);

        println!("✓ Operation enum works correctly");
    }

    #[test]
    fn test_hint_enum() {
        assert_eq!(Hint::Fast, Hint::Fast);
        assert_eq!(Hint::Cheap, Hint::Cheap);
        assert_ne!(Hint::Fast, Hint::Cheap);

        println!("✓ Hint enum works correctly");
    }

    #[test]
    fn test_backend_type_enum() {
        assert_eq!(BackendType::Direct, BackendType::Direct);
        assert_eq!(BackendType::Redis, BackendType::Redis);
        assert_eq!(BackendType::S3, BackendType::S3);

        println!("✓ BackendType enum works correctly");
    }

    #[test]
    fn test_fmi_context() {
        let mut ctx = FmiContext::new();
        assert_eq!(ctx.completed, 0);

        ctx.mark_completed();
        assert_eq!(ctx.completed, 1);
        assert!(ctx.is_completed());

        println!("✓ FmiContext works correctly");
    }

    #[test]
    fn test_fmi_context_default() {
        let ctx = FmiContext::default();
        assert_eq!(ctx.completed, 0);
        assert!(!ctx.is_completed());

        println!("✓ FmiContext default works correctly");
    }
}

// ============================================================================
// DirectBackend Unit Tests
// ============================================================================

#[cfg(feature = "fmi")]
mod fmi_backend_tests {
    use cylon::net::fmi::{DirectBackend, Mode, BackendType};

    #[test]
    fn test_direct_backend_builder() {
        let backend = DirectBackend::new()
            .with_host("tcpunch.example.com")
            .with_port(9000)
            .with_max_timeout(60000)
            .set_resolve_dns(true)
            .set_blocking_mode(Mode::NonBlocking)
            .set_enable_ping(true);

        assert_eq!(backend.get_host(), "tcpunch.example.com");
        assert_eq!(backend.get_port(), 9000);
        assert_eq!(backend.get_max_timeout(), 60000);
        assert!(backend.resolve_host_dns());
        assert_eq!(backend.get_blocking_mode(), Mode::NonBlocking);
        assert!(backend.enable_host_ping());
        assert_eq!(backend.get_backend_type(), BackendType::Direct);

        println!("✓ DirectBackend builder works correctly");
    }

    #[test]
    fn test_direct_backend_defaults() {
        let backend = DirectBackend::new();

        assert_eq!(backend.get_host(), "");
        assert_eq!(backend.get_port(), -1);
        assert_eq!(backend.get_max_timeout(), -1);
        assert!(!backend.resolve_host_dns());
        assert_eq!(backend.get_blocking_mode(), Mode::Blocking);
        assert!(!backend.enable_host_ping());

        println!("✓ DirectBackend defaults are correct");
    }

    #[test]
    fn test_direct_backend_clone() {
        let backend1 = DirectBackend::new()
            .with_host("localhost")
            .with_port(8080);

        let backend2 = backend1.clone();

        assert_eq!(backend1.get_host(), backend2.get_host());
        assert_eq!(backend1.get_port(), backend2.get_port());

        println!("✓ DirectBackend clone works correctly");
    }
}

// ============================================================================
// Integration Tests - Require TCPunch server and Redis
// ============================================================================

#[cfg(feature = "fmi")]
mod fmi_integration_tests {
    use cylon::net::fmi::{FMIConfig, FMICommunicator};
    use cylon::net::{Communicator, CommType};
    use cylon::ctx::CylonContext;

    fn get_test_config() -> (String, i32, String, i32) {
        let host = std::env::var("CYLON_FMI_HOST")
            .unwrap_or_else(|_| "localhost".to_string());
        let port = std::env::var("CYLON_FMI_PORT")
            .ok()
            .and_then(|p| p.parse().ok())
            .unwrap_or(8080);
        let redis_host = std::env::var("CYLON_REDIS_HOST")
            .unwrap_or_else(|_| "localhost".to_string());
        let redis_port = std::env::var("CYLON_REDIS_PORT")
            .ok()
            .and_then(|p| p.parse().ok())
            .unwrap_or(6379);

        (host, port, redis_host, redis_port)
    }

    #[test]
    #[ignore] // Requires TCPunch server and Redis
    fn test_fmi_communicator_init() {
        let (host, port, redis_host, redis_port) = get_test_config();
        let rank: i32 = std::env::var("CYLON_RANK")
            .ok()
            .and_then(|r| r.parse().ok())
            .unwrap_or(0);
        let world_size: i32 = std::env::var("CYLON_WORLD_SIZE")
            .ok()
            .and_then(|w| w.parse().ok())
            .unwrap_or(1);

        let config = FMIConfig::builder()
            .rank(rank)
            .world_size(world_size)
            .host(&host)
            .port(port)
            .redis_host(&redis_host)
            .redis_port(redis_port)
            .build();

        let comm_result = FMICommunicator::make(&config);

        match comm_result {
            Ok(comm) => {
                println!("✓ FMI communicator initialized");
                println!("  Rank: {}/{}", comm.get_rank(), comm.get_world_size());
                assert_eq!(comm.get_rank(), rank);
                assert_eq!(comm.get_world_size(), world_size);
                assert_eq!(comm.get_comm_type(), CommType::Fmi);
                assert!(!comm.is_finalized());
            }
            Err(e) => {
                eprintln!("Failed to initialize FMI communicator: {:?}", e);
                eprintln!("Make sure TCPunch server is running at: {}:{}", host, port);
                eprintln!("Make sure Redis is running at: {}:{}", redis_host, redis_port);
                panic!("FMI communicator initialization failed");
            }
        }
    }

    #[test]
    #[ignore] // Requires TCPunch server and Redis
    fn test_fmi_create_channel() {
        let (host, port, redis_host, redis_port) = get_test_config();

        let config = FMIConfig::builder()
            .rank(0)
            .world_size(1)
            .host(&host)
            .port(port)
            .redis_host(&redis_host)
            .redis_port(redis_port)
            .build();

        let comm = FMICommunicator::make(&config)
            .expect("Failed to create FMI communicator");

        let channel_result = comm.create_channel();
        assert!(channel_result.is_ok(), "Failed to create channel: {:?}", channel_result.err());

        println!("✓ FMI channel created successfully");
    }

    #[test]
    #[ignore] // Requires TCPunch server and Redis
    fn test_fmi_barrier() {
        let (host, port, redis_host, redis_port) = get_test_config();
        let rank: i32 = std::env::var("CYLON_RANK")
            .ok()
            .and_then(|r| r.parse().ok())
            .unwrap_or(0);
        let world_size: i32 = std::env::var("CYLON_WORLD_SIZE")
            .ok()
            .and_then(|w| w.parse().ok())
            .unwrap_or(1);

        let config = FMIConfig::builder()
            .rank(rank)
            .world_size(world_size)
            .host(&host)
            .port(port)
            .redis_host(&redis_host)
            .redis_port(redis_port)
            .build();

        let comm = FMICommunicator::make(&config)
            .expect("Failed to create FMI communicator");

        let result = comm.barrier();
        assert!(result.is_ok(), "Barrier failed: {:?}", result.err());

        println!("✓ Barrier completed on rank {}", rank);
    }

    #[test]
    #[ignore] // Requires TCPunch server and Redis
    fn test_fmi_context_init_distributed() {
        let (host, port, redis_host, redis_port) = get_test_config();
        let rank: i32 = std::env::var("CYLON_RANK")
            .ok()
            .and_then(|r| r.parse().ok())
            .unwrap_or(0);
        let world_size: i32 = std::env::var("CYLON_WORLD_SIZE")
            .ok()
            .and_then(|w| w.parse().ok())
            .unwrap_or(1);

        let config = FMIConfig::builder()
            .rank(rank)
            .world_size(world_size)
            .host(&host)
            .port(port)
            .redis_host(&redis_host)
            .redis_port(redis_port)
            .build();

        let ctx_result = CylonContext::init_distributed(&config);

        match ctx_result {
            Ok(ctx) => {
                println!("✓ CylonContext initialized with FMI");
                println!("  Rank: {}/{}", ctx.get_rank(), ctx.get_world_size());
                assert!(ctx.is_distributed());
                assert_eq!(ctx.get_rank(), rank);
                assert_eq!(ctx.get_world_size(), world_size);
                assert_eq!(ctx.get_comm_type(), CommType::Fmi);
            }
            Err(e) => {
                eprintln!("Failed to initialize CylonContext with FMI: {:?}", e);
                panic!("CylonContext initialization failed");
            }
        }
    }
}

// ============================================================================
// Multi-process Communication Tests
// ============================================================================

#[cfg(feature = "fmi")]
mod fmi_communication_tests {
    use cylon::net::fmi::{FMIConfig, FMICommunicator};
    use cylon::net::Communicator;

    fn get_test_config() -> (String, i32, String, i32, i32, i32) {
        let host = std::env::var("CYLON_FMI_HOST")
            .unwrap_or_else(|_| "localhost".to_string());
        let port = std::env::var("CYLON_FMI_PORT")
            .ok()
            .and_then(|p| p.parse().ok())
            .unwrap_or(8080);
        let redis_host = std::env::var("CYLON_REDIS_HOST")
            .unwrap_or_else(|_| "localhost".to_string());
        let redis_port = std::env::var("CYLON_REDIS_PORT")
            .ok()
            .and_then(|p| p.parse().ok())
            .unwrap_or(6379);
        let rank = std::env::var("CYLON_RANK")
            .ok()
            .and_then(|r| r.parse().ok())
            .unwrap_or(0);
        let world_size = std::env::var("CYLON_WORLD_SIZE")
            .ok()
            .and_then(|w| w.parse().ok())
            .unwrap_or(2);

        (host, port, redis_host, redis_port, rank, world_size)
    }

    #[test]
    #[ignore] // Requires multiple processes with TCPunch and Redis
    fn test_fmi_send_recv() {
        let (host, port, redis_host, redis_port, rank, world_size) = get_test_config();

        let config = FMIConfig::builder()
            .rank(rank)
            .world_size(world_size)
            .host(&host)
            .port(port)
            .redis_host(&redis_host)
            .redis_port(redis_port)
            .build();

        let comm = FMICommunicator::make(&config)
            .expect("Failed to create FMI communicator");

        if world_size < 2 {
            println!("Skipping send/recv test - need at least 2 processes");
            return;
        }

        if rank == 0 {
            // Send data to rank 1
            let data = vec![1u8, 2, 3, 4, 5];
            comm.send(&data, 1, 0).expect("Send failed");
            println!("✓ Rank 0 sent {} bytes to rank 1", data.len());
        } else if rank == 1 {
            // Receive data from rank 0
            let mut buffer = Vec::new();
            comm.recv(&mut buffer, 0, 0).expect("Recv failed");
            println!("✓ Rank 1 received {} bytes from rank 0", buffer.len());
            assert_eq!(buffer, vec![1u8, 2, 3, 4, 5]);
        }

        comm.barrier().expect("Barrier failed");
        println!("✓ Send/recv test completed on rank {}", rank);
    }

    #[test]
    #[ignore] // Requires multiple processes with TCPunch and Redis
    fn test_fmi_broadcast() {
        let (host, port, redis_host, redis_port, rank, world_size) = get_test_config();

        let config = FMIConfig::builder()
            .rank(rank)
            .world_size(world_size)
            .host(&host)
            .port(port)
            .redis_host(&redis_host)
            .redis_port(redis_port)
            .build();

        let comm = FMICommunicator::make(&config)
            .expect("Failed to create FMI communicator");

        let mut data = if rank == 0 {
            vec![42u8, 43, 44, 45]
        } else {
            vec![0u8; 4]
        };

        comm.broadcast(&mut data, 0).expect("Broadcast failed");

        assert_eq!(data, vec![42u8, 43, 44, 45]);
        println!("✓ Broadcast completed on rank {}, data: {:?}", rank, data);
    }

    #[test]
    #[ignore] // Requires multiple processes with TCPunch and Redis
    fn test_fmi_allgather() {
        let (host, port, redis_host, redis_port, rank, world_size) = get_test_config();

        let config = FMIConfig::builder()
            .rank(rank)
            .world_size(world_size)
            .host(&host)
            .port(port)
            .redis_host(&redis_host)
            .redis_port(redis_port)
            .build();

        let comm = FMICommunicator::make(&config)
            .expect("Failed to create FMI communicator");

        // Each rank sends its rank as data
        let send_data = vec![rank as u8];
        let result = comm.allgather(&send_data).expect("Allgather failed");

        // Should receive data from all ranks
        assert_eq!(result.len(), world_size as usize);
        for (i, data) in result.iter().enumerate() {
            assert_eq!(data[0], i as u8);
        }

        println!("✓ Allgather completed on rank {}, received from {} ranks", rank, result.len());
    }
}

// ============================================================================
// Feature disabled placeholder
// ============================================================================

#[cfg(not(feature = "fmi"))]
mod fmi_disabled {
    #[test]
    fn fmi_feature_not_enabled() {
        println!("FMI tests skipped - feature not enabled");
        println!("To run FMI tests, use: cargo test --features fmi");
    }
}
