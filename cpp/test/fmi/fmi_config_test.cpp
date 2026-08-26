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

#include <cylon/net/fmi/fmi_communicator.hpp>
#include <cylon/thridparty/fmi/utils/DirectBackend.hpp>
#include <gtest/gtest.h>
#include <string>

namespace {

std::shared_ptr<cylon::net::FMIConfig> MakeConfig(const std::string &channel_type) {
  return cylon::net::FMIConfig::Make(
      /*rank=*/0, /*world_size=*/2, channel_type,
      /*host=*/"127.0.0.1", /*port=*/18900, /*maxtimeout=*/1000,
      /*comm_name=*/"fmi_config_test", /*nonblocking=*/false,
      /*redis_host=*/"127.0.0.1", /*redis_port=*/6379, /*redis_namespace=*/"",
      /*s3_bucket=*/"", /*s3_region=*/"", /*key_ttl=*/0,
      /*s3_retry_initial_ms=*/0, /*s3_retry_max_ms=*/0);
}

}  // namespace

TEST(FMIConfigTest, DirectRedisMatches) {
  auto config = MakeConfig("direct-redis");
  auto *direct_backend =
      dynamic_cast<FMI::Utils::DirectBackend *>(config->getBackend().get());
  ASSERT_NE(direct_backend, nullptr);
  EXPECT_TRUE(direct_backend->useDirectRedis());
}

TEST(FMIConfigTest, DirectRedisMixedCaseMatches) {
  auto config = MakeConfig("Direct-Redis");
  auto *direct_backend =
      dynamic_cast<FMI::Utils::DirectBackend *>(config->getBackend().get());
  ASSERT_NE(direct_backend, nullptr);
  EXPECT_TRUE(direct_backend->useDirectRedis());
}

TEST(FMIConfigTest, DirectDefaultDoesNotEnableDirectRedis) {
  auto config = MakeConfig("direct");
  auto *direct_backend =
      dynamic_cast<FMI::Utils::DirectBackend *>(config->getBackend().get());
  ASSERT_NE(direct_backend, nullptr);
  EXPECT_FALSE(direct_backend->useDirectRedis());
}