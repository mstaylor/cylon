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

#include "libfabric_redis_oob.hpp"
#include <cstring>
#include <stdexcept>

namespace cylon {
namespace net {
namespace libfabric {

#ifdef BUILD_CYLON_REDIS

LibfabricRedisOOB::LibfabricRedisOOB(int world_size,
                                     const std::string &redis_host,
                                     int redis_port,
                                     const std::string &session_id,
                                     int ttl_seconds)
    : world_size_(world_size), session_id_(session_id), ttl_seconds_(ttl_seconds) {
    std::string redis_addr = "tcp://" + redis_host + ":" + std::to_string(redis_port);
    redis_ = std::make_shared<sw::redis::Redis>(redis_addr);
}

std::string LibfabricRedisOOB::key(const std::string &suffix) const {
    return "cylon:libfabric:" + session_id_ + ":" + suffix;
}

Status LibfabricRedisOOB::getWorldSizeAndRank(int &world_size, int &rank) {
    world_size = world_size_;
    auto k = key("num_cur_processes");
    int num_cur_processes = redis_->incr(k);
    redis_->expire(k, std::chrono::seconds(ttl_seconds_));
    rank = rank_ = num_cur_processes - 1;
    return Status::OK();
}

Status LibfabricRedisOOB::OOBAllgather(const uint8_t *src, uint8_t *dst,
                                       size_t src_size, size_t dst_size) {
    (void)dst_size;

    // Publish this rank's address
    auto addr_key = key("fi_addr_mp");
    redis_->hset(addr_key, std::to_string(rank_),
                 std::string(reinterpret_cast<const char *>(src),
                             reinterpret_cast<const char *>(src) + src_size));
    redis_->expire(addr_key, std::chrono::seconds(ttl_seconds_));

    // Push signal values so other ranks can wait on us
    auto helper_key = key("fi_helper" + std::to_string(rank_));
    std::vector<int> v(world_size_, 0);
    redis_->lpush(helper_key, v.begin(), v.end());
    redis_->expire(helper_key, std::chrono::seconds(ttl_seconds_));

    // Gather addresses from all ranks
    for (int i = 0; i < world_size_; i++) {
        if (i == rank_) {
            std::memcpy(dst + i * src_size, src, src_size);
            continue;
        }

        auto i_str = std::to_string(i);
        auto other_helper = key("fi_helper" + i_str);

        auto val = redis_->hget(addr_key, i_str);
        while (!val) {
            redis_->blpop(other_helper);
            val = redis_->hget(addr_key, i_str);
        }

        std::memcpy(dst + i * src_size, val.value().data(), src_size);
    }

    return Status::OK();
}

Status LibfabricRedisOOB::Barrier(const std::string &barrier_id) {
    auto k = key("barrier:" + barrier_id);

    int count = redis_->incr(k);
    redis_->expire(k, std::chrono::seconds(ttl_seconds_));

    if (count == world_size_) {
        // Last to arrive — notify all others
        auto notify_key = key("barrier_notify:" + barrier_id);
        for (int i = 0; i < world_size_ - 1; i++) {
            redis_->lpush(notify_key, "done");
        }
        redis_->expire(notify_key, std::chrono::seconds(ttl_seconds_));
    } else {
        // Wait for notification
        auto notify_key = key("barrier_notify:" + barrier_id);
        redis_->blpop(notify_key);
    }

    return Status::OK();
}

Status LibfabricRedisOOB::Finalize() {
    try {
        redis_->del(key("num_cur_processes"));
        redis_->del(key("fi_addr_mp"));
        for (int i = 0; i < world_size_; i++) {
            redis_->del(key("fi_helper" + std::to_string(i)));
        }
    } catch (...) {
        // Best-effort cleanup
    }
    return Status::OK();
}

std::shared_ptr<LibfabricRedisOOB> LibfabricRedisOOB::Make(
    int world_size,
    const std::string &redis_host,
    int redis_port,
    const std::string &session_id,
    int ttl_seconds) {
    return std::make_shared<LibfabricRedisOOB>(world_size, redis_host, redis_port,
                                               session_id, ttl_seconds);
}

#endif // BUILD_CYLON_REDIS

} // namespace libfabric
} // namespace net
} // namespace cylon