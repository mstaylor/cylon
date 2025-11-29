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

//! Integration tests for CylonContext with various communicators
//!
//! Tests the CylonContext::init_distributed() method with MPI, UCX, and FMI backends.
//!
//! ## Running Tests
//!
//! ### MPI Tests
//! ```bash
//! # Single process
//! cargo test --features mpi --test cylon_context_test mpi -- --ignored
//!
//! # Multiple processes
//! mpirun -n 4 cargo test --features mpi --test cylon_context_test mpi -- --ignored
//! ```
//!
//! ### UCX Tests (requires Redis)
//! ```bash
//! export CYLON_REDIS_HOST=localhost
//! export CYLON_REDIS_PORT=6379
//! export CYLON_SESSION_ID=$(uuidgen)
//! cargo test --features ucx --test cylon_context_test ucx -- --ignored
//! ```
//!
//! ### FMI Tests (requires TCPunch server + Redis)
//! ```bash
//! export CYLON_FMI_HOST=localhost
//! export CYLON_FMI_PORT=8080
//! export CYLON_REDIS_HOST=localhost
//! export CYLON_REDIS_PORT=6379
//! cargo test --features fmi --test cylon_context_test fmi -- --ignored
//! ```

use cylon::ctx::CylonContext;
use cylon::net::CommType;

// ============================================================================
// Local Context Tests (no communicator required)
// ============================================================================

mod local_context_tests {
    use super::*;

    #[test]
    fn test_local_context_init() {
        let ctx = CylonContext::init();

        assert!(!ctx.is_distributed());
        assert_eq!(ctx.get_rank(), 0);
        assert_eq!(ctx.get_world_size(), 1);
        assert_eq!(ctx.get_comm_type(), CommType::Local);

        println!("✓ Local CylonContext initialized successfully");
    }

    #[test]
    fn test_local_context_config() {
        let ctx = CylonContext::init();

        ctx.add_config("test_key", "test_value");
        assert_eq!(ctx.get_config("test_key", "default"), "test_value");
        assert_eq!(ctx.get_config("missing_key", "default"), "default");

        println!("✓ Local CylonContext config works correctly");
    }

    #[test]
    fn test_local_context_neighbours() {
        let ctx = CylonContext::init();

        let neighbours_with_self = ctx.get_neighbours(true);
        let neighbours_without_self = ctx.get_neighbours(false);

        assert_eq!(neighbours_with_self, vec![0]);
        assert!(neighbours_without_self.is_empty());

        println!("✓ Local CylonContext neighbours work correctly");
    }

    #[test]
    fn test_local_context_sequence() {
        let ctx = CylonContext::init();

        let seq1 = ctx.get_next_sequence();
        let seq2 = ctx.get_next_sequence();
        let seq3 = ctx.get_next_sequence();

        assert_eq!(seq1, 1);
        assert_eq!(seq2, 2);
        assert_eq!(seq3, 3);

        println!("✓ Local CylonContext sequence numbers work correctly");
    }

    #[test]
    fn test_local_context_barrier() {
        let ctx = CylonContext::init();

        // Barrier should be a no-op for local context
        let result = ctx.barrier();
        assert!(result.is_ok());

        println!("✓ Local CylonContext barrier works correctly");
    }

    #[test]
    fn test_init_distributed_empty() {
        let ctx = CylonContext::init_distributed_empty();

        assert!(ctx.is_distributed());
        // No communicator set, so rank=0, world_size=1
        assert_eq!(ctx.get_rank(), 0);
        assert_eq!(ctx.get_world_size(), 1);

        println!("✓ CylonContext::init_distributed_empty works correctly");
    }
}

// ============================================================================
// MPI Context Tests
// ============================================================================

#[cfg(feature = "mpi")]
mod mpi_context_tests {
    use super::*;
    use cylon::net::mpi::MPICommunicator;
    #[allow(unused_imports)]
    use cylon::net::Communicator;

    #[test]
    #[ignore] // Requires MPI environment
    fn test_mpi_context_manual_setup() {
        // Create MPI communicator manually
        let comm = MPICommunicator::make()
            .expect("Failed to create MPI communicator");

        let rank = comm.get_rank();
        let world_size = comm.get_world_size();

        // Create context and set communicator
        let mut ctx = CylonContext::new(true);
        ctx.set_communicator(comm);

        assert!(ctx.is_distributed());
        assert_eq!(ctx.get_rank(), rank);
        assert_eq!(ctx.get_world_size(), world_size);
        assert_eq!(ctx.get_comm_type(), CommType::Mpi);

        println!("✓ MPI CylonContext (manual setup) - rank {}/{}", rank, world_size);
    }

    #[test]
    #[ignore] // Requires MPI environment
    fn test_mpi_context_barrier() {
        let comm = MPICommunicator::make()
            .expect("Failed to create MPI communicator");

        let rank = comm.get_rank();

        let mut ctx = CylonContext::new(true);
        ctx.set_communicator(comm);

        let result = ctx.barrier();
        assert!(result.is_ok());

        println!("✓ MPI barrier completed on rank {}", rank);
    }

    #[test]
    #[ignore] // Requires MPI environment
    fn test_mpi_context_neighbours() {
        let comm = MPICommunicator::make()
            .expect("Failed to create MPI communicator");

        let rank = comm.get_rank();
        let world_size = comm.get_world_size();

        let mut ctx = CylonContext::new(true);
        ctx.set_communicator(comm);

        let neighbours_with_self = ctx.get_neighbours(true);
        let neighbours_without_self = ctx.get_neighbours(false);

        assert_eq!(neighbours_with_self.len(), world_size as usize);
        assert_eq!(neighbours_without_self.len(), (world_size - 1) as usize);
        assert!(!neighbours_without_self.contains(&rank));

        println!("✓ MPI neighbours work correctly on rank {}", rank);
    }
}

// ============================================================================
// UCX Context Tests
// ============================================================================

#[cfg(feature = "ucx")]
mod ucx_context_tests {
    use super::*;
    use cylon::net::ucx::{UCXConfig, UCXCommunicator, RedisOOBContext};
    use cylon::net::Communicator;

    fn get_ucx_config() -> (String, u16, String, i32) {
        let redis_host = std::env::var("CYLON_REDIS_HOST")
            .unwrap_or_else(|_| "localhost".to_string());
        let redis_port: u16 = std::env::var("CYLON_REDIS_PORT")
            .ok()
            .and_then(|p| p.parse().ok())
            .unwrap_or(6379);
        let session_id = std::env::var("CYLON_SESSION_ID")
            .unwrap_or_else(|_| uuid::Uuid::new_v4().to_string());
        let world_size: i32 = std::env::var("CYLON_WORLD_SIZE")
            .ok()
            .and_then(|w| w.parse().ok())
            .unwrap_or(1);

        (redis_host, redis_port, session_id, world_size)
    }

    #[test]
    #[ignore] // Requires Redis server
    fn test_ucx_context_init_distributed() {
        let (redis_host, redis_port, session_id, world_size) = get_ucx_config();

        let config = UCXConfig::with_redis(
            &redis_host,
            redis_port,
            &session_id,
            world_size,
        );

        let ctx_result = CylonContext::init_distributed(&config);

        match ctx_result {
            Ok(ctx) => {
                assert!(ctx.is_distributed());
                assert_eq!(ctx.get_comm_type(), CommType::Ucx);
                assert!(ctx.get_rank() >= 0);
                assert!(ctx.get_rank() < ctx.get_world_size());

                println!("✓ UCX CylonContext initialized - rank {}/{}",
                    ctx.get_rank(), ctx.get_world_size());
            }
            Err(e) => {
                eprintln!("Failed to initialize UCX CylonContext: {:?}", e);
                eprintln!("Make sure Redis is running at {}:{}", redis_host, redis_port);
                panic!("UCX CylonContext initialization failed");
            }
        }
    }

    #[test]
    #[ignore] // Requires Redis server
    fn test_ucx_context_manual_setup() {
        let (redis_host, redis_port, session_id, world_size) = get_ucx_config();

        let oob = Box::new(RedisOOBContext::new(
            &redis_host,
            redis_port,
            &session_id,
            world_size,
        ));

        let comm = UCXCommunicator::make_oob(oob)
            .expect("Failed to create UCX communicator");

        let rank = comm.get_rank();

        let mut ctx = CylonContext::new(true);
        ctx.set_communicator(std::sync::Arc::new(comm));

        assert!(ctx.is_distributed());
        assert_eq!(ctx.get_comm_type(), CommType::Ucx);

        println!("✓ UCX CylonContext (manual setup) - rank {}", rank);
    }

    #[test]
    #[ignore] // Requires Redis server
    fn test_ucx_context_barrier() {
        let (redis_host, redis_port, session_id, world_size) = get_ucx_config();

        let config = UCXConfig::with_redis(
            &redis_host,
            redis_port,
            &session_id,
            world_size,
        );

        let ctx = CylonContext::init_distributed(&config)
            .expect("Failed to create UCX context");

        let result = ctx.barrier();
        assert!(result.is_ok());

        println!("✓ UCX barrier completed on rank {}", ctx.get_rank());
    }
}

// ============================================================================
// FMI Context Tests
// ============================================================================

#[cfg(feature = "fmi")]
mod fmi_context_tests {
    use super::*;
    use cylon::net::fmi::{FMIConfig, FMICommunicator};
    use cylon::net::Communicator;

    fn get_fmi_config() -> (String, i32, String, i32, i32, i32) {
        let host = std::env::var("CYLON_FMI_HOST")
            .unwrap_or_else(|_| "localhost".to_string());
        let port: i32 = std::env::var("CYLON_FMI_PORT")
            .ok()
            .and_then(|p| p.parse().ok())
            .unwrap_or(8080);
        let redis_host = std::env::var("CYLON_REDIS_HOST")
            .unwrap_or_else(|_| "localhost".to_string());
        let redis_port: i32 = std::env::var("CYLON_REDIS_PORT")
            .ok()
            .and_then(|p| p.parse().ok())
            .unwrap_or(6379);
        let rank: i32 = std::env::var("CYLON_RANK")
            .ok()
            .and_then(|r| r.parse().ok())
            .unwrap_or(0);
        let world_size: i32 = std::env::var("CYLON_WORLD_SIZE")
            .ok()
            .and_then(|w| w.parse().ok())
            .unwrap_or(1);

        (host, port, redis_host, redis_port, rank, world_size)
    }

    #[test]
    #[ignore] // Requires TCPunch server + Redis
    fn test_fmi_context_init_distributed() {
        let (host, port, redis_host, redis_port, rank, world_size) = get_fmi_config();

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
                assert!(ctx.is_distributed());
                assert_eq!(ctx.get_comm_type(), CommType::Fmi);
                assert_eq!(ctx.get_rank(), rank);
                assert_eq!(ctx.get_world_size(), world_size);

                println!("✓ FMI CylonContext initialized - rank {}/{}",
                    ctx.get_rank(), ctx.get_world_size());
            }
            Err(e) => {
                eprintln!("Failed to initialize FMI CylonContext: {:?}", e);
                eprintln!("Make sure TCPunch server is running at {}:{}", host, port);
                eprintln!("Make sure Redis is running at {}:{}", redis_host, redis_port);
                panic!("FMI CylonContext initialization failed");
            }
        }
    }

    #[test]
    #[ignore] // Requires TCPunch server + Redis
    fn test_fmi_context_manual_setup() {
        let (host, port, redis_host, redis_port, rank, world_size) = get_fmi_config();

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

        let mut ctx = CylonContext::new(true);
        ctx.set_communicator(comm as std::sync::Arc<dyn Communicator>);

        assert!(ctx.is_distributed());
        assert_eq!(ctx.get_comm_type(), CommType::Fmi);
        assert_eq!(ctx.get_rank(), rank);

        println!("✓ FMI CylonContext (manual setup) - rank {}", rank);
    }

    #[test]
    #[ignore] // Requires TCPunch server + Redis
    fn test_fmi_context_barrier() {
        let (host, port, redis_host, redis_port, rank, world_size) = get_fmi_config();

        let config = FMIConfig::builder()
            .rank(rank)
            .world_size(world_size)
            .host(&host)
            .port(port)
            .redis_host(&redis_host)
            .redis_port(redis_port)
            .build();

        let ctx = CylonContext::init_distributed(&config)
            .expect("Failed to create FMI context");

        let result = ctx.barrier();
        assert!(result.is_ok());

        println!("✓ FMI barrier completed on rank {}", rank);
    }

    #[test]
    #[ignore] // Requires TCPunch server + Redis
    fn test_fmi_context_neighbours() {
        let (host, port, redis_host, redis_port, rank, world_size) = get_fmi_config();

        let config = FMIConfig::builder()
            .rank(rank)
            .world_size(world_size)
            .host(&host)
            .port(port)
            .redis_host(&redis_host)
            .redis_port(redis_port)
            .build();

        let ctx = CylonContext::init_distributed(&config)
            .expect("Failed to create FMI context");

        let neighbours_with_self = ctx.get_neighbours(true);
        let neighbours_without_self = ctx.get_neighbours(false);

        assert_eq!(neighbours_with_self.len(), world_size as usize);
        assert_eq!(neighbours_without_self.len(), (world_size - 1) as usize);
        assert!(!neighbours_without_self.contains(&rank));

        println!("✓ FMI neighbours work correctly on rank {}", rank);
    }

    #[test]
    #[ignore] // Requires TCPunch server + Redis
    fn test_fmi_context_config() {
        let (host, port, redis_host, redis_port, rank, world_size) = get_fmi_config();

        let config = FMIConfig::builder()
            .rank(rank)
            .world_size(world_size)
            .host(&host)
            .port(port)
            .redis_host(&redis_host)
            .redis_port(redis_port)
            .build();

        let ctx = CylonContext::init_distributed(&config)
            .expect("Failed to create FMI context");

        ctx.add_config("distributed_key", "distributed_value");
        assert_eq!(ctx.get_config("distributed_key", "default"), "distributed_value");

        println!("✓ FMI context config works on rank {}", rank);
    }

    #[test]
    #[ignore] // Requires TCPunch server + Redis
    fn test_fmi_context_sequence() {
        let (host, port, redis_host, redis_port, rank, world_size) = get_fmi_config();

        let config = FMIConfig::builder()
            .rank(rank)
            .world_size(world_size)
            .host(&host)
            .port(port)
            .redis_host(&redis_host)
            .redis_port(redis_port)
            .build();

        let ctx = CylonContext::init_distributed(&config)
            .expect("Failed to create FMI context");

        let seq1 = ctx.get_next_sequence();
        let seq2 = ctx.get_next_sequence();

        assert!(seq2 > seq1);

        println!("✓ FMI context sequence works on rank {}", rank);
    }
}

// ============================================================================
// Cross-communicator comparison tests
// ============================================================================

mod context_api_tests {
    use super::*;

    #[test]
    fn test_context_api_consistency() {
        // Verify that the CylonContext API is consistent
        let ctx = CylonContext::init();

        // All these methods should exist and return sensible values
        let _ = ctx.is_distributed();
        let _ = ctx.get_rank();
        let _ = ctx.get_world_size();
        let _ = ctx.get_comm_type();
        let _ = ctx.get_neighbours(true);
        let _ = ctx.get_neighbours(false);
        let _ = ctx.get_next_sequence();
        let _ = ctx.get_communicator();
        let _ = ctx.get_memory_pool();
        let _ = ctx.barrier();

        ctx.add_config("key", "value");
        let _ = ctx.get_config("key", "default");

        println!("✓ CylonContext API is consistent");
    }
}
