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

//! Redis-based Out-of-Band context for UCX/UCC
//!
//! Ported from cpp/src/cylon/net/ucx/redis_ucx_ucc_oob_context.hpp/cpp
//!
//! Uses the `redis` crate instead of hiredis C library

use redis::{Client, Connection, Commands};

use crate::error::{CylonError, CylonResult, Code};
use super::oob_context::UCXOOBContext;
use super::OOBType;

/// UCX Redis OOB Context
///
/// Corresponds to C++ UCXRedisOOBContext from redis_ucx_ucc_oob_context.hpp
///
/// Uses Redis for out-of-band communication to exchange UCX worker addresses.
pub struct UCXRedisOOBContext {
    /// Redis connection
    conn: Connection,
    /// World size (number of processes)
    world_size: i32,
    /// This process's rank
    rank: i32,
}

impl UCXRedisOOBContext {
    /// Create a new UCX Redis OOB context
    ///
    /// Corresponds to C++ UCXRedisOOBContext::UCXRedisOOBContext (redis_ucx_ucc_oob_context.cpp:7)
    ///
    /// # Arguments
    /// * `world_size` - Total number of processes
    /// * `redis_addr` - Redis server address (e.g., "redis://127.0.0.1:6379")
    pub fn new(world_size: i32, redis_addr: &str) -> CylonResult<Self> {
        let client = Client::open(redis_addr).map_err(|e| {
            CylonError::new(Code::IoError, format!("Failed to connect to Redis: {}", e))
        })?;

        let conn = client.get_connection().map_err(|e| {
            CylonError::new(Code::IoError, format!("Failed to get Redis connection: {}", e))
        })?;

        Ok(Self {
            conn,
            world_size,
            rank: -1,
        })
    }

    /// Create a new UCX Redis OOB context (factory method)
    ///
    /// Corresponds to C++ UCXRedisOOBContext::Make (redis_ucx_ucc_oob_context.cpp:50)
    pub fn make(world_size: i32, redis_addr: &str) -> CylonResult<Box<Self>> {
        Ok(Box::new(Self::new(world_size, redis_addr)?))
    }
}

impl UCXOOBContext for UCXRedisOOBContext {
    /// Initialize the OOB context
    ///
    /// Corresponds to C++ UCXRedisOOBContext::InitOOB (redis_ucx_ucc_oob_context.cpp:10)
    fn init_oob(&mut self) -> CylonResult<()> {
        Ok(())
    }

    /// Get world size and rank
    ///
    /// Corresponds to C++ UCXRedisOOBContext::getWorldSizeAndRank (redis_ucx_ucc_oob_context.cpp:12)
    ///
    /// Returns (world_size, rank)
    fn get_world_size_and_rank(&mut self) -> CylonResult<(i32, i32)> {
        // Atomically increment to get rank assignment
        // Corresponds to C++ line 14: int num_cur_processes = redis->incr("num_cur_processes");
        let num_cur_processes: i32 = self.conn.incr("num_cur_processes", 1).map_err(|e| {
            CylonError::new(Code::IoError, format!("Redis INCR failed: {}", e))
        })?;

        // Corresponds to C++ line 15: rank = this->rank = num_cur_processes - 1;
        self.rank = num_cur_processes - 1;
        Ok((self.world_size, self.rank))
    }

    /// Perform an allgather operation for OOB address exchange
    ///
    /// Corresponds to C++ UCXRedisOOBContext::OOBAllgather (redis_ucx_ucc_oob_context.cpp:20-44)
    ///
    /// # Arguments
    /// * `src` - Source buffer containing this process's data
    /// * `dst` - Destination buffer to receive all processes' data
    /// * `src_size` - Size of source data
    /// * `dst_size` - Total size of destination buffer
    fn oob_allgather(
        &mut self,
        src: &[u8],
        dst: &mut [u8],
        src_size: usize,
        _dst_size: usize,
    ) -> CylonResult<()> {
        // Corresponds to C++ line 23
        let ucp_worker_addr_mp_str = "ucp_worker_addr_mp";

        // Store this process's worker address in Redis hash
        // Corresponds to C++ lines 24-25:
        // redis->hset(ucc_worker_addr_mp_str, std::to_string(rank),
        //             std::string((char *)src, (char *)src + srcSize));
        let _: () = self.conn.hset(
            ucp_worker_addr_mp_str,
            self.rank.to_string(),
            src,
        ).map_err(|e| {
            CylonError::new(Code::IoError, format!("Redis HSET failed: {}", e))
        })?;

        // Push signal to indicate we're ready
        // Corresponds to C++ lines 26-27:
        // std::vector<int> v(world_size, 0);
        // redis->lpush("ucx_helper" + std::to_string(rank), v.begin(), v.end());
        let helper_key = format!("ucx_helper{}", self.rank);
        let zeros: Vec<i32> = vec![0; self.world_size as usize];
        for val in zeros {
            let _: () = self.conn.lpush(&helper_key, val).map_err(|e| {
                CylonError::new(Code::IoError, format!("Redis LPUSH failed: {}", e))
            })?;
        }

        // Gather addresses from all other processes
        // Corresponds to C++ lines 29-42
        for i in 0..self.world_size {
            if i == self.rank {
                // Copy own data
                // Corresponds to C++ line 30: continue;
                // (We copy instead of continue to match the logic)
                dst[i as usize * src_size..(i as usize + 1) * src_size]
                    .copy_from_slice(src);
                continue;
            }

            let i_str = i.to_string();
            let helper_name = format!("ucx_helper{}", i);

            // Wait for the other process to be ready and get their address
            // Corresponds to C++ lines 34-38
            loop {
                let val: Option<Vec<u8>> = self.conn.hget(ucp_worker_addr_mp_str, &i_str)
                    .map_err(|e| {
                        CylonError::new(Code::IoError, format!("Redis HGET failed: {}", e))
                    })?;

                if let Some(data) = val {
                    // Copy the address data
                    // Corresponds to C++ line 40: memcpy(dst + i * srcSize, val.value().data(), srcSize);
                    dst[i as usize * src_size..(i as usize + 1) * src_size]
                        .copy_from_slice(&data[..src_size]);
                    break;
                }

                // Wait using blocking pop
                // Corresponds to C++ line 36: redis->blpop(helperName);
                let _: Option<(String, i32)> = self.conn.blpop(&helper_name, 0.0)
                    .map_err(|e| {
                        CylonError::new(Code::IoError, format!("Redis BLPOP failed: {}", e))
                    })?;
            }
        }

        // Corresponds to C++ line 43
        Ok(())
    }

    /// Finalize the OOB context
    ///
    /// Corresponds to C++ UCXRedisOOBContext::Finalize (redis_ucx_ucc_oob_context.cpp:46)
    fn finalize(&mut self) -> CylonResult<()> {
        Ok(())
    }
}

/// UCC Redis OOB Context
///
/// Corresponds to C++ UCCRedisOOBContext from redis_ucx_ucc_oob_context.hpp
#[cfg(feature = "ucc")]
pub struct UCCRedisOOBContext {
    /// World size
    world_size: i32,
    /// Rank
    rank: i32,
    /// Redis client
    client: Client,
    /// Redis connection
    conn: Connection,
    /// Number of OOB allgather operations performed
    num_oob_allgather: i32,
    /// Redis address
    redis_addr: String,
}

#[cfg(feature = "ucc")]
impl UCCRedisOOBContext {
    /// Create a new UCC Redis OOB context
    ///
    /// Corresponds to C++ UCCRedisOOBContext::UCCRedisOOBContext (redis_ucx_ucc_oob_context.cpp:101)
    ///
    /// # Arguments
    /// * `world_size` - Total number of processes
    /// * `redis_addr` - Redis server address (e.g., "redis://127.0.0.1:6379")
    pub fn new(world_size: i32, redis_addr: &str) -> CylonResult<Self> {
        let client = Client::open(redis_addr).map_err(|e| {
            CylonError::new(Code::IoError, format!("Failed to connect to Redis: {}", e))
        })?;

        let conn = client.get_connection().map_err(|e| {
            CylonError::new(Code::IoError, format!("Failed to get Redis connection: {}", e))
        })?;

        Ok(Self {
            world_size,
            rank: -1,
            client,
            conn,
            num_oob_allgather: 0,
            redis_addr: redis_addr.to_string(),
        })
    }

    /// Create from environment variables
    ///
    /// Corresponds to C++ UCCRedisOOBContext::UCCRedisOOBContext() (redis_ucx_ucc_oob_context.cpp:105)
    ///
    /// Used with Python script `run_ucc_with_redis.py`
    /// Extracts CYLON_UCX_OOB_REDIS_ADDR and CYLON_UCX_OOB_WORLD_SIZE
    pub fn from_env() -> CylonResult<Self> {
        // Corresponds to C++ line 106: redis_addr = "tcp://" + std::string(getenv("CYLON_UCX_OOB_REDIS_ADDR"));
        let redis_addr = std::env::var("CYLON_UCX_OOB_REDIS_ADDR").map_err(|_| {
            CylonError::new(Code::Invalid, "CYLON_UCX_OOB_REDIS_ADDR not set")
        })?;
        let redis_addr = format!("tcp://{}", redis_addr);

        // Corresponds to C++ line 107: world_size = std::atoi(getenv("CYLON_UCX_OOB_WORLD_SIZE"));
        let world_size: i32 = std::env::var("CYLON_UCX_OOB_WORLD_SIZE")
            .map_err(|_| CylonError::new(Code::Invalid, "CYLON_UCX_OOB_WORLD_SIZE not set"))?
            .parse()
            .map_err(|_| CylonError::new(Code::Invalid, "Invalid CYLON_UCX_OOB_WORLD_SIZE"))?;

        Self::new(world_size, &redis_addr)
    }

    /// Create a new UCC Redis OOB context (factory method)
    ///
    /// Corresponds to C++ UCCRedisOOBContext::Make (redis_ucx_ucc_oob_context.cpp:133)
    pub fn make(world_size: i32, redis_addr: &str) -> CylonResult<Box<Self>> {
        Ok(Box::new(Self::new(world_size, redis_addr)?))
    }

    /// Get Redis connection
    ///
    /// Corresponds to C++ UCCRedisOOBContext::getRedis (redis_ucx_ucc_oob_context.cpp:123)
    pub fn get_redis(&mut self) -> &mut Connection {
        &mut self.conn
    }

    /// OOB allgather operation for UCC
    ///
    /// Corresponds to C++ UCCRedisOOBContext::oob_allgather (redis_ucx_ucc_oob_context.cpp:62)
    ///
    /// This is used as the callback for UCC team creation
    pub fn oob_allgather_impl(
        &mut self,
        sbuf: &[u8],
        rbuf: &mut [u8],
        msglen: usize,
    ) -> CylonResult<()> {
        // Corresponds to C++ lines 65-68
        let num_comm = self.num_oob_allgather;
        self.num_oob_allgather += 1;

        let map_key = format!("ucc_oob_mp{}", num_comm);

        // Store this process's data
        // Corresponds to C++ line 74: redis->hset("ucc_oob_mp" + std::to_string(num_comm), std::to_string(rank), s);
        let _: () = self.conn.hset(&map_key, self.rank.to_string(), sbuf)
            .map_err(|e| {
                CylonError::new(Code::IoError, format!("Redis HSET failed: {}", e))
            })?;

        // Signal readiness
        // Corresponds to C++ lines 75-77
        let helper_key = format!("ucc_helper{}:{}", num_comm, self.rank);
        let _: () = self.conn.lpush(&helper_key, "0").map_err(|e| {
            CylonError::new(Code::IoError, format!("Redis LPUSH failed: {}", e))
        })?;

        // Gather from all processes
        // Corresponds to C++ lines 79-96
        for i in 0..self.world_size {
            let offset = i as usize * msglen;

            if i == self.rank {
                // Copy own data
                // Corresponds to C++ line 81: memcpy((uint8_t*)rbuf + i * msglen, s.data(), msglen);
                rbuf[offset..offset + msglen].copy_from_slice(sbuf);
            } else {
                let other_helper = format!("ucc_helper{}:{}", num_comm, i);

                // Wait and get data from other process
                // Corresponds to C++ lines 83-95
                loop {
                    // Block until signaled
                    // Corresponds to C++ line 89: redis->brpoplpush(helperName, helperName, 0);
                    let _: Option<String> = self.conn.brpoplpush(&other_helper, &other_helper, 0.0)
                        .map_err(|e| {
                            CylonError::new(Code::IoError, format!("Redis BRPOPLPUSH failed: {}", e))
                        })?;

                    // Corresponds to C++ lines 90-91
                    let val: Option<Vec<u8>> = self.conn.hget(&map_key, i.to_string())
                        .map_err(|e| {
                            CylonError::new(Code::IoError, format!("Redis HGET failed: {}", e))
                        })?;

                    if let Some(data) = val {
                        // Corresponds to C++ line 94: memcpy((uint8_t*)rbuf + i * msglen, val.value().data(), msglen);
                        rbuf[offset..offset + msglen].copy_from_slice(&data[..msglen]);
                        break;
                    }
                }
            }
        }

        // Corresponds to C++ line 98: return UCC_OK;
        Ok(())
    }
}

#[cfg(feature = "ucc")]
impl super::oob_context::UCCOOBContext for UCCRedisOOBContext {
    /// Initialize OOB with the given rank
    ///
    /// Corresponds to C++ UCCRedisOOBContext::InitOOB (redis_ucx_ucc_oob_context.cpp:54)
    fn init_oob(&mut self, rank: i32) {
        self.rank = rank;
    }

    /// Create a UCX OOB context
    ///
    /// Corresponds to C++ UCCRedisOOBContext::makeUCXOOBContext (redis_ucx_ucc_oob_context.cpp:56)
    fn make_ucx_oob_context(&self) -> Box<dyn UCXOOBContext> {
        Box::new(
            UCXRedisOOBContext::new(self.world_size, &self.redis_addr)
                .expect("Failed to create UCX OOB context")
        )
    }

    /// Get the collective info pointer for UCC
    ///
    /// Corresponds to C++ UCCRedisOOBContext::getCollInfo (redis_ucx_ucc_oob_context.cpp:60)
    fn get_coll_info(&self) -> *mut std::ffi::c_void {
        self as *const _ as *mut std::ffi::c_void
    }

    /// Get the OOB type
    ///
    /// Corresponds to C++ UCCRedisOOBContext::Type (redis_ucx_ucc_oob_context.cpp:121)
    fn oob_type(&self) -> OOBType {
        OOBType::Redis
    }

    /// Get the world size
    ///
    /// Corresponds to C++ UCCRedisOOBContext::getWorldSize (redis_ucx_ucc_oob_context.cpp:127)
    fn get_world_size(&self) -> i32 {
        self.world_size
    }

    /// Get the rank
    ///
    /// Corresponds to C++ UCCRedisOOBContext::getRank (redis_ucx_ucc_oob_context.cpp:131)
    fn get_rank(&self) -> i32 {
        self.rank
    }

    /// Set the rank
    ///
    /// Corresponds to C++ UCCRedisOOBContext::setRank (redis_ucx_ucc_oob_context.cpp:129)
    fn set_rank(&mut self, rank: i32) {
        self.rank = rank;
    }

    /// Finalize the OOB context
    ///
    /// Corresponds to C++ UCCRedisOOBContext::Finalize (redis_ucx_ucc_oob_context.cpp:137)
    fn finalize(&mut self) -> CylonResult<()> {
        Ok(())
    }
}
