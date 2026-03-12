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
 * Unit tests for LibfabricConfig
 *
 * These tests do NOT require libfabric, Redis, or MPI — they only test
 * the configuration object construction and accessors.
 *
 * Run with: ./libfabric_config_test
 */

#define CATCH_CONFIG_MAIN
#include <catch.hpp>

#ifdef BUILD_CYLON_LIBFABRIC

#include <cylon/net/libfabric/libfabric_communicator.hpp>
#include <cylon/net/comm_type.hpp>

namespace cylon {
namespace test {

TEST_CASE("LibfabricConfig construction", "[libfabric][config]") {
  SECTION("default provider and ttl") {
    auto config = cylon::net::LibfabricConfig::Make(
        4, "localhost", 6379, "test-session-001");

    REQUIRE(config != nullptr);
    REQUIRE(config->getWorldSize() == 4);
    REQUIRE(config->getRedisHost() == "localhost");
    REQUIRE(config->getRedisPort() == 6379);
    REQUIRE(config->getSessionId() == "test-session-001");
    REQUIRE(config->getKeyTtl() == 3600);
    REQUIRE(config->getProvider().empty());
  }

  SECTION("custom provider and ttl") {
    auto config = cylon::net::LibfabricConfig::Make(
        2, "10.0.0.1", 7000, "my-session", 600, "tcp");

    REQUIRE(config != nullptr);
    REQUIRE(config->getWorldSize() == 2);
    REQUIRE(config->getRedisHost() == "10.0.0.1");
    REQUIRE(config->getRedisPort() == 7000);
    REQUIRE(config->getSessionId() == "my-session");
    REQUIRE(config->getKeyTtl() == 600);
    REQUIRE(config->getProvider() == "tcp");
  }

  SECTION("efa provider") {
    auto config = cylon::net::LibfabricConfig::Make(
        8, "redis.example.com", 6380, "efa-session", 1800, "efa");

    REQUIRE(config->getProvider() == "efa");
    REQUIRE(config->getWorldSize() == 8);
  }

  SECTION("verbs provider") {
    auto config = cylon::net::LibfabricConfig::Make(
        16, "10.211.55.2", 6379, "verbs-session", 7200, "verbs");

    REQUIRE(config->getProvider() == "verbs");
  }
}

TEST_CASE("LibfabricConfig CommType", "[libfabric][config]") {
  auto config = cylon::net::LibfabricConfig::Make(
      2, "localhost", 6379, "type-test");

  REQUIRE(config->Type() == cylon::net::LIBFABRIC);
}

TEST_CASE("LibfabricConfig as CommConfig base", "[libfabric][config]") {
  std::shared_ptr<cylon::net::CommConfig> base_config =
      cylon::net::LibfabricConfig::Make(4, "localhost", 6379, "base-test");

  REQUIRE(base_config != nullptr);
  REQUIRE(base_config->Type() == cylon::net::LIBFABRIC);
}

TEST_CASE("LibfabricConfig single-node world", "[libfabric][config]") {
  auto config = cylon::net::LibfabricConfig::Make(
      1, "localhost", 6379, "single-node");

  REQUIRE(config->getWorldSize() == 1);
}

} // namespace test
} // namespace cylon

#else // BUILD_CYLON_LIBFABRIC

TEST_CASE("Libfabric config tests skipped - BUILD_CYLON_LIBFABRIC not defined",
          "[libfabric]") {
  WARN("Libfabric config tests are skipped because BUILD_CYLON_LIBFABRIC "
       "is not defined");
}

#endif // BUILD_CYLON_LIBFABRIC