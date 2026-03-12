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

/**
 * Unit tests for LibfabricRedisOOB
 *
 * These tests require a running Redis server.
 * Set REDIS_TEST_HOST env var or defaults to 10.211.55.2 (Parallels host IP).
 *
 * All tests run single-process (world_size=1) so no MPI or multi-process
 * coordination is needed.
 *
 * Run with: ./libfabric_redis_oob_test
 */

#define CATCH_CONFIG_MAIN
#include <catch.hpp>
#include <glog/logging.h>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <string>

#if defined(BUILD_CYLON_LIBFABRIC) && defined(BUILD_CYLON_REDIS)

#include <cylon/net/libfabric/libfabric_redis_oob.hpp>

namespace cylon {
namespace test {

static std::string getRedisHost() {
    const char* host = std::getenv("REDIS_TEST_HOST");
    return host ? host : "10.211.55.2";
}

static const int REDIS_PORT = 6379;

// Use a unique session ID per test run to avoid collisions
static std::string uniqueSessionId(const std::string &base) {
    auto now = std::chrono::steady_clock::now().time_since_epoch();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
    return "test_lf_" + base + "_" + std::to_string(ms);
}

TEST_CASE("LibfabricRedisOOB Make", "[libfabric][redis-oob]") {
    auto session_id = uniqueSessionId("make");
    auto oob = net::libfabric::LibfabricRedisOOB::Make(
        1, getRedisHost(), REDIS_PORT, session_id, 60);

    REQUIRE(oob != nullptr);

    oob->Finalize();
}

TEST_CASE("LibfabricRedisOOB rank assignment single process",
          "[libfabric][redis-oob]") {
    auto session_id = uniqueSessionId("rank");
    auto oob = net::libfabric::LibfabricRedisOOB::Make(
        1, getRedisHost(), REDIS_PORT, session_id, 60);

    int world_size = -1;
    int rank = -1;
    auto st = oob->getWorldSizeAndRank(world_size, rank);

    REQUIRE(st.is_ok());
    REQUIRE(world_size == 1);
    REQUIRE(rank == 0);

    oob->Finalize();
}

TEST_CASE("LibfabricRedisOOB allgather single process",
          "[libfabric][redis-oob]") {
    auto session_id = uniqueSessionId("allgather");
    auto oob = net::libfabric::LibfabricRedisOOB::Make(
        1, getRedisHost(), REDIS_PORT, session_id, 60);

    // Must get rank before allgather (sets internal rank_)
    int world_size, rank;
    REQUIRE(oob->getWorldSizeAndRank(world_size, rank).is_ok());
    REQUIRE(rank == 0);

    // Simulate a 16-byte address
    uint8_t src_addr[16];
    for (int i = 0; i < 16; i++) src_addr[i] = static_cast<uint8_t>(0xA0 + i);

    uint8_t dst_addr[16];
    std::memset(dst_addr, 0, sizeof(dst_addr));

    auto st = oob->OOBAllgather(src_addr, dst_addr, 16, 16);

    REQUIRE(st.is_ok());
    // With world_size=1, dst should equal src
    REQUIRE(std::memcmp(src_addr, dst_addr, 16) == 0);

    oob->Finalize();
}

TEST_CASE("LibfabricRedisOOB barrier single process",
          "[libfabric][redis-oob]") {
    auto session_id = uniqueSessionId("barrier");
    auto oob = net::libfabric::LibfabricRedisOOB::Make(
        1, getRedisHost(), REDIS_PORT, session_id, 60);

    // Must get rank before barrier
    int world_size, rank;
    REQUIRE(oob->getWorldSizeAndRank(world_size, rank).is_ok());

    auto st = oob->Barrier("test_barrier_0");

    REQUIRE(st.is_ok());

    oob->Finalize();
}

TEST_CASE("LibfabricRedisOOB finalize is idempotent",
          "[libfabric][redis-oob]") {
    auto session_id = uniqueSessionId("finalize");
    auto oob = net::libfabric::LibfabricRedisOOB::Make(
        1, getRedisHost(), REDIS_PORT, session_id, 60);

    int world_size, rank;
    REQUIRE(oob->getWorldSizeAndRank(world_size, rank).is_ok());

    // Call finalize multiple times — should not throw
    REQUIRE(oob->Finalize().is_ok());
    REQUIRE(oob->Finalize().is_ok());
}

TEST_CASE("LibfabricRedisOOB session isolation",
          "[libfabric][redis-oob]") {
    // Two OOB instances with different session IDs should not interfere
    auto session_a = uniqueSessionId("iso_a");
    auto session_b = uniqueSessionId("iso_b");

    auto oob_a = net::libfabric::LibfabricRedisOOB::Make(
        1, getRedisHost(), REDIS_PORT, session_a, 60);
    auto oob_b = net::libfabric::LibfabricRedisOOB::Make(
        1, getRedisHost(), REDIS_PORT, session_b, 60);

    int ws_a, rank_a, ws_b, rank_b;
    REQUIRE(oob_a->getWorldSizeAndRank(ws_a, rank_a).is_ok());
    REQUIRE(oob_b->getWorldSizeAndRank(ws_b, rank_b).is_ok());

    // Both should get rank 0 since they're independent sessions
    REQUIRE(rank_a == 0);
    REQUIRE(rank_b == 0);

    oob_a->Finalize();
    oob_b->Finalize();
}

TEST_CASE("LibfabricRedisOOB allgather binary data",
          "[libfabric][redis-oob]") {
    auto session_id = uniqueSessionId("binary");
    auto oob = net::libfabric::LibfabricRedisOOB::Make(
        1, getRedisHost(), REDIS_PORT, session_id, 60);

    int world_size, rank;
    REQUIRE(oob->getWorldSizeAndRank(world_size, rank).is_ok());

    // Test with binary data that includes null bytes
    uint8_t src[] = {0x00, 0xFF, 0x00, 0xAB, 0xCD, 0x00, 0xEF, 0x01};
    uint8_t dst[8];
    std::memset(dst, 0x42, sizeof(dst));

    auto st = oob->OOBAllgather(src, dst, sizeof(src), sizeof(dst));

    REQUIRE(st.is_ok());
    REQUIRE(std::memcmp(src, dst, sizeof(src)) == 0);

    oob->Finalize();
}

} // namespace test
} // namespace cylon

#else // BUILD_CYLON_LIBFABRIC && BUILD_CYLON_REDIS

TEST_CASE("Libfabric Redis OOB tests skipped", "[libfabric]") {
#ifndef BUILD_CYLON_LIBFABRIC
    WARN("Skipped: BUILD_CYLON_LIBFABRIC not defined");
#endif
#ifndef BUILD_CYLON_REDIS
    WARN("Skipped: BUILD_CYLON_REDIS not defined");
#endif
}

#endif