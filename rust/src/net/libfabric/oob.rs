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

//! Out-of-band (OOB) communication for libfabric address exchange
//!
//! This module provides Redis-based OOB communication for exchanging
//! libfabric addresses between workers before they can communicate.
//!
//! ## Session Management
//!
//! **IMPORTANT:** Requires `CYLON_SESSION_ID` environment variable to prevent
//! conflicts with stale worker addresses in Redis. The launcher must generate
//! a unique session ID and pass it to all processes.
//!
//! Example:
//! ```bash
//! export CYLON_SESSION_ID=$(uuidgen)
//! ./my_libfabric_program
//! ```

use redis::{Commands, Client, Connection};

use crate::error::{CylonError, CylonResult, Code};
use super::LibfabricConfig;

/// Redis key prefix for libfabric OOB
const KEY_PREFIX: &str = "cylon:libfabric";

/// LibfabricRedisOOB provides out-of-band communication via Redis
///
/// Similar to UCXRedisOOBContext, uses Redis for out-of-band communication
/// to exchange libfabric endpoint addresses between workers.
pub struct LibfabricRedisOOB {
    /// Redis connection
    conn: Connection,
    /// Session ID for namespace isolation
    session_id: String,
    /// This worker's rank
    rank: i32,
    /// Total number of workers
    world_size: i32,
}

impl LibfabricRedisOOB {
    /// Create a new LibfabricRedisOOB instance
    ///
    /// # Arguments
    /// * `config` - Libfabric configuration containing Redis connection info
    ///
    /// # Errors
    /// Returns an error if:
    /// - Cannot connect to Redis server
    pub fn new(config: &LibfabricConfig) -> CylonResult<Self> {
        let url = format!("redis://{}:{}", config.redis_host, config.redis_port);
        let client = Client::open(url.as_str())
            .map_err(|e| CylonError::new(
                Code::IoError,
                format!("Failed to connect to Redis at {}: {}", url, e),
            ))?;

        let conn = client.get_connection()
            .map_err(|e| CylonError::new(
                Code::IoError,
                format!("Failed to get Redis connection: {}", e),
            ))?;

        Ok(Self {
            conn,
            session_id: config.session_id.clone(),
            rank: -1, // Will be assigned during get_world_size_and_rank
            world_size: config.world_size,
        })
    }

    /// Create from environment variables
    ///
    /// Reads configuration from:
    /// - `CYLON_SESSION_ID` - Required session ID
    /// - `CYLON_LIBFABRIC_REDIS_ADDR` - Redis address (default: 127.0.0.1:6379)
    /// - `CYLON_LIBFABRIC_WORLD_SIZE` - World size (required)
    pub fn from_env() -> CylonResult<Self> {
        let session_id = std::env::var("CYLON_SESSION_ID").map_err(|_| {
            CylonError::new(
                Code::Invalid,
                "CYLON_SESSION_ID environment variable not set. \
                 The launcher must set this to prevent conflicts with stale Redis data. \
                 Example: export CYLON_SESSION_ID=$(uuidgen)"
            )
        })?;

        let redis_addr = std::env::var("CYLON_LIBFABRIC_REDIS_ADDR")
            .unwrap_or_else(|_| "127.0.0.1:6379".to_string());

        let world_size: i32 = std::env::var("CYLON_LIBFABRIC_WORLD_SIZE")
            .map_err(|_| CylonError::new(Code::Invalid, "CYLON_LIBFABRIC_WORLD_SIZE not set"))?
            .parse()
            .map_err(|_| CylonError::new(Code::Invalid, "Invalid CYLON_LIBFABRIC_WORLD_SIZE"))?;

        let url = format!("redis://{}", redis_addr);
        let client = Client::open(url.as_str())
            .map_err(|e| CylonError::new(
                Code::IoError,
                format!("Failed to connect to Redis: {}", e),
            ))?;

        let conn = client.get_connection()
            .map_err(|e| CylonError::new(
                Code::IoError,
                format!("Failed to get Redis connection: {}", e),
            ))?;

        Ok(Self {
            conn,
            session_id,
            rank: -1,
            world_size,
        })
    }

    /// Generate a Redis key with the session prefix
    fn key(&self, suffix: &str) -> String {
        format!("{}:{}:{}", KEY_PREFIX, self.session_id, suffix)
    }

    /// Get world size and rank
    ///
    /// Atomically increments a counter to assign ranks.
    /// Returns (world_size, rank)
    pub fn get_world_size_and_rank(&mut self) -> CylonResult<(i32, i32)> {
        // Atomically increment to get rank assignment
        let key = self.key("num_cur_processes");
        let num_cur_processes: i32 = self.conn.incr(&key, 1)
            .map_err(|e| CylonError::new(Code::IoError, format!("Redis INCR failed: {}", e)))?;

        self.rank = num_cur_processes - 1;
        log::info!("Registered as rank {} in session {}", self.rank, self.session_id);

        Ok((self.world_size, self.rank))
    }

    /// Get this worker's rank
    pub fn rank(&self) -> i32 {
        self.rank
    }

    /// Get the world size
    pub fn world_size(&self) -> i32 {
        self.world_size
    }

    /// Perform an allgather operation for OOB address exchange
    ///
    /// Each worker publishes its address and waits for all other workers'
    /// addresses. Similar to UCXRedisOOBContext::OOBAllgather.
    ///
    /// # Arguments
    /// * `src` - Source buffer containing this process's address
    /// * `dst` - Destination buffer to receive all processes' addresses
    /// * `src_size` - Size of each address
    pub fn oob_allgather(
        &mut self,
        src: &[u8],
        dst: &mut [u8],
        src_size: usize,
    ) -> CylonResult<()> {
        let addr_map_key = self.key("fi_addr_mp");

        // Store this process's address in Redis hash
        let _: () = self.conn.hset(&addr_map_key, self.rank.to_string(), src)
            .map_err(|e| CylonError::new(Code::IoError, format!("Redis HSET failed: {}", e)))?;

        // Push signal to indicate we're ready
        let helper_key = self.key(&format!("fi_helper{}", self.rank));
        let zeros: Vec<i32> = vec![0; self.world_size as usize];
        for val in zeros {
            let _: () = self.conn.lpush(&helper_key, val)
                .map_err(|e| CylonError::new(Code::IoError, format!("Redis LPUSH failed: {}", e)))?;
        }

        // Gather addresses from all processes
        for i in 0..self.world_size {
            let offset = i as usize * src_size;

            if i == self.rank {
                // Copy own data
                dst[offset..offset + src_size].copy_from_slice(src);
                continue;
            }

            let i_str = i.to_string();
            let other_helper = self.key(&format!("fi_helper{}", i));

            // Wait for the other process to be ready and get their address
            loop {
                let val: Option<Vec<u8>> = self.conn.hget(&addr_map_key, &i_str)
                    .map_err(|e| CylonError::new(Code::IoError, format!("Redis HGET failed: {}", e)))?;

                if let Some(data) = val {
                    // Copy the address data
                    let copy_len = std::cmp::min(data.len(), src_size);
                    dst[offset..offset + copy_len].copy_from_slice(&data[..copy_len]);
                    break;
                }

                // Wait using blocking pop
                let _: Option<(String, i32)> = self.conn.blpop(&other_helper, 0.0)
                    .map_err(|e| CylonError::new(Code::IoError, format!("Redis BLPOP failed: {}", e)))?;
            }
        }

        log::debug!("OOB allgather complete: gathered {} addresses", self.world_size);

        Ok(())
    }

    /// Simple barrier using Redis
    ///
    /// All workers must call this before any can proceed.
    pub fn barrier(&mut self, barrier_id: &str) -> CylonResult<()> {
        let key = self.key(&format!("barrier:{}", barrier_id));

        // Increment barrier counter
        let count: i32 = self.conn.incr(&key, 1)
            .map_err(|e| CylonError::new(Code::IoError, format!("Redis INCR failed: {}", e)))?;

        log::debug!("Barrier {}: {}/{} workers arrived", barrier_id, count, self.world_size);

        // Wait for all workers to arrive using blpop on a notification key
        if count == self.world_size {
            // Last to arrive - notify all others
            let notify_key = self.key(&format!("barrier_notify:{}", barrier_id));
            for _ in 0..self.world_size - 1 {
                let _: () = self.conn.lpush(&notify_key, "done")
                    .map_err(|e| CylonError::new(Code::IoError, format!("Redis LPUSH failed: {}", e)))?;
            }
        } else {
            // Wait for notification
            let notify_key = self.key(&format!("barrier_notify:{}", barrier_id));
            let _: Option<(String, String)> = self.conn.blpop(&notify_key, 0.0)
                .map_err(|e| CylonError::new(Code::IoError, format!("Redis BLPOP failed: {}", e)))?;
        }

        Ok(())
    }

    /// Finalize the OOB context
    pub fn finalize(&mut self) -> CylonResult<()> {
        // Nothing to clean up - Redis keys will expire or be overwritten
        Ok(())
    }
}
