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

#include "cylon/thridparty/fmi/comm/RedisDirectPair.hpp"
#include <gtest/gtest.h>
#include <cstring>
#include <string>
#include <thread>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/time.h>

TEST(RedisDirectPairTest, LowerRankListensHigherRankConnects) {
  std::string redis_host = std::getenv("CYLON_TEST_REDIS_HOST")
      ? std::getenv("CYLON_TEST_REDIS_HOST") : "127.0.0.1";
  int redis_port = std::getenv("CYLON_TEST_REDIS_PORT")
      ? std::atoi(std::getenv("CYLON_TEST_REDIS_PORT")) : 6379;
  std::string comm_name = "redis_direct_pair_test_" + std::to_string(getpid());

  FMI::Comm::RedisDirectEstablisher rank0, rank1;
  int rank0_fd = -1, rank1_fd = -1;

  std::thread t0([&] {
    rank0.Init(redis_host, redis_port, comm_name, /*self_rank=*/0, /*num_peers=*/2,
              /*listen_port=*/18881, "127.0.0.1");
    rank0_fd = rank0.Connect(0, 1, /*timeout_ms=*/5000, FMI::Utils::BLOCKING);
  });
  std::thread t1([&] {
    rank1.Init(redis_host, redis_port, comm_name, /*self_rank=*/1, /*num_peers=*/2,
              /*listen_port=*/18882, "127.0.0.1");
    rank1_fd = rank1.Connect(1, 0, /*timeout_ms=*/5000, FMI::Utils::BLOCKING);
  });
  t0.join();
  t1.join();

  ASSERT_GE(rank0_fd, 0);
  ASSERT_GE(rank1_fd, 0);

  const char *msg = "hello-from-rank-1";
  ASSERT_EQ(::send(rank1_fd, msg, strlen(msg), 0), (ssize_t)strlen(msg));
  char buf[64] = {0};
  ASSERT_EQ(::recv(rank0_fd, buf, sizeof(buf), 0), (ssize_t)strlen(msg));
  ASSERT_STREQ(buf, msg);

  rank0.Finalize();
  rank1.Finalize();
}

TEST(RedisDirectPairTest, BlockingAndNonBlockingPairsStayDistinct) {
  std::string redis_host = std::getenv("CYLON_TEST_REDIS_HOST")
      ? std::getenv("CYLON_TEST_REDIS_HOST") : "127.0.0.1";
  int redis_port = std::getenv("CYLON_TEST_REDIS_PORT")
      ? std::atoi(std::getenv("CYLON_TEST_REDIS_PORT")) : 6379;
  std::string comm_name = "redis_direct_pair_mode_test_" + std::to_string(getpid());

  FMI::Comm::RedisDirectEstablisher rank0, rank1;
  int rank0_blocking_fd = -1, rank0_nonblocking_fd = -1;
  int rank1_blocking_fd = -1, rank1_nonblocking_fd = -1;
  std::string rank0_error, rank1_error;

  std::thread t0([&] {
    try {
      rank0.Init(redis_host, redis_port, comm_name, /*self_rank=*/0, /*num_peers=*/2,
                /*listen_port=*/18883, "127.0.0.1");
      rank0_nonblocking_fd = rank0.Connect(0, 1, /*timeout_ms=*/5000, FMI::Utils::NONBLOCKING);
      rank0_blocking_fd = rank0.Connect(0, 1, /*timeout_ms=*/5000, FMI::Utils::BLOCKING);
    } catch (const std::exception &e) {
      rank0_error = e.what();
    }
  });
  std::thread t1([&] {
    try {
      rank1.Init(redis_host, redis_port, comm_name, /*self_rank=*/1, /*num_peers=*/2,
                /*listen_port=*/18884, "127.0.0.1");
      rank1_blocking_fd = rank1.Connect(1, 0, /*timeout_ms=*/5000, FMI::Utils::BLOCKING);
      rank1_nonblocking_fd = rank1.Connect(1, 0, /*timeout_ms=*/5000, FMI::Utils::NONBLOCKING);
    } catch (const std::exception &e) {
      rank1_error = e.what();
    }
  });
  t0.join();
  t1.join();

  ASSERT_TRUE(rank0_error.empty()) << "rank0 threw: " << rank0_error;
  ASSERT_TRUE(rank1_error.empty()) << "rank1 threw: " << rank1_error;
  ASSERT_GE(rank0_blocking_fd, 0);
  ASSERT_GE(rank0_nonblocking_fd, 0);
  ASSERT_GE(rank1_blocking_fd, 0);
  ASSERT_GE(rank1_nonblocking_fd, 0);
  ASSERT_NE(rank0_blocking_fd, rank0_nonblocking_fd);
  ASSERT_NE(rank1_blocking_fd, rank1_nonblocking_fd);

  struct timeval recv_timeout{5, 0};
  setsockopt(rank0_blocking_fd, SOL_SOCKET, SO_RCVTIMEO, &recv_timeout, sizeof(recv_timeout));
  setsockopt(rank0_nonblocking_fd, SOL_SOCKET, SO_RCVTIMEO, &recv_timeout, sizeof(recv_timeout));

  const char *blocking_msg = "payload-on-the-blocking-pair";
  const char *nonblocking_msg = "payload-on-the-nonblocking-pair";
  ASSERT_EQ(::send(rank1_blocking_fd, blocking_msg, strlen(blocking_msg), 0),
            (ssize_t) strlen(blocking_msg));
  ASSERT_EQ(::send(rank1_nonblocking_fd, nonblocking_msg, strlen(nonblocking_msg), 0),
            (ssize_t) strlen(nonblocking_msg));

  char blocking_buf[64] = {0};
  char nonblocking_buf[64] = {0};
  ASSERT_EQ(::recv(rank0_blocking_fd, blocking_buf, sizeof(blocking_buf), 0),
            (ssize_t) strlen(blocking_msg));
  ASSERT_EQ(::recv(rank0_nonblocking_fd, nonblocking_buf, sizeof(nonblocking_buf), 0),
            (ssize_t) strlen(nonblocking_msg));
  ASSERT_STREQ(blocking_buf, blocking_msg);
  ASSERT_STREQ(nonblocking_buf, nonblocking_msg);

  rank0.Finalize();
  rank1.Finalize();
}
