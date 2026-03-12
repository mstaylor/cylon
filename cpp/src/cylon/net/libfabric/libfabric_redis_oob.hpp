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

#ifndef CYLON_LIBFABRIC_REDIS_OOB_HPP
#define CYLON_LIBFABRIC_REDIS_OOB_HPP

#include <string>
#include <memory>
#include <vector>
#include <cylon/status.hpp>

#ifdef BUILD_CYLON_REDIS
#include <sw/redis++/redis++.h>
#endif

namespace cylon {
namespace net {
namespace libfabric {

#ifdef BUILD_CYLON_REDIS

/// Redis-based out-of-band context for libfabric address exchange.
///
/// Similar to UCXRedisOOBContext, uses Redis for:
/// - Rank assignment via atomic counter
/// - Address exchange via allgather
/// - Barrier synchronization
///
/// Requires CYLON_SESSION_ID environment variable for key isolation.
class LibfabricRedisOOB {
public:
    LibfabricRedisOOB(int world_size,
                      const std::string &redis_host,
                      int redis_port,
                      const std::string &session_id,
                      int ttl_seconds = 3600);

    /// Get world size and assign rank via Redis atomic increment
    Status getWorldSizeAndRank(int &world_size, int &rank);

    /// Allgather endpoint addresses via Redis
    /// Each rank publishes its address and polls for all peers.
    Status OOBAllgather(const uint8_t *src, uint8_t *dst,
                        size_t src_size, size_t dst_size);

    /// Barrier: all ranks must arrive before any can proceed
    Status Barrier(const std::string &barrier_id);

    /// Best-effort cleanup of session keys
    Status Finalize();

    static std::shared_ptr<LibfabricRedisOOB> Make(
        int world_size,
        const std::string &redis_host,
        int redis_port,
        const std::string &session_id,
        int ttl_seconds = 3600);

private:
    std::string key(const std::string &suffix) const;

    std::shared_ptr<sw::redis::Redis> redis_;
    int world_size_;
    int rank_ = -1;
    std::string session_id_;
    int ttl_seconds_;
};

#endif // BUILD_CYLON_REDIS

} // namespace libfabric
} // namespace net
} // namespace cylon

#endif // CYLON_LIBFABRIC_REDIS_OOB_HPP