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

#include "cylon/thridparty/fmi/comm/Direct.hpp"
#include "cylon/thridparty/fmi/utils/DirectBackend.hpp"

#include <gtest/gtest.h>

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {

class TcpunchdFixture {
 public:
  explicit TcpunchdFixture(int port) {
    const char *server_path = std::getenv("CYLON_TEST_TCPUNCHD_PATH");
    std::string path = server_path ? server_path
        : "/home/parallels/TCPunch/server/rust/target/release/tcpunchd";

    pid_ = fork();
    if (pid_ == 0) {
      int devnull = open("/dev/null", O_RDWR);
      if (devnull >= 0) {
        dup2(devnull, STDIN_FILENO);
        dup2(devnull, STDOUT_FILENO);
        dup2(devnull, STDERR_FILENO);
        if (devnull > STDERR_FILENO) close(devnull);
      }
      std::string port_str = std::to_string(port);
      std::string health_port_str = std::to_string(port + 1000);
      execl(path.c_str(), path.c_str(), "-p", port_str.c_str(),
            "--health-port", health_port_str.c_str(), (char *) nullptr);
      _exit(127);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
  }

  ~TcpunchdFixture() {
    if (pid_ > 0) {
      kill(pid_, SIGTERM);
      int status = 0;
      waitpid(pid_, &status, 0);
    }
  }

  bool started() const { return pid_ > 0; }

 private:
  pid_t pid_ = -1;
};

std::string GetRedisHost() {
  const char *h = std::getenv("CYLON_TEST_REDIS_HOST");
  return h ? h : "127.0.0.1";
}

int GetRedisPort() {
  const char *p = std::getenv("CYLON_TEST_REDIS_PORT");
  return p ? std::atoi(p) : 6379;
}

int RunTcpunchRank(FMI::Utils::peer_num rank, int rendezvous_port,
                   const std::string &comm_name, const char *payload) {
  auto backend = std::make_shared<FMI::Utils::DirectBackend>();
  backend->withHost("127.0.0.1");
  backend->withPort(rendezvous_port);
  backend->withMaxTimeout(5000);
  backend->setResolveBackendDNS(false);
  backend->setBlockingMode(FMI::Utils::BLOCKING);

  FMI::Comm::Direct direct(backend);
  direct.set_peer_id(rank);
  direct.set_num_peers(2);
  direct.set_comm_name(comm_name);
  direct.init();

  size_t len = strlen(payload) + 1;
  try {
    if (rank == 0) {
      std::string received(len, '\0');
      auto buf = std::make_shared<channel_data>((void *) received.data(), len);
      direct.recv_object(buf, /*sender_id=*/1);
      std::memcpy(&received[0], buf->buf.get(), len);
      return std::memcmp(received.data(), payload, len) == 0 ? 0 : 1;
    } else {
      auto buf = std::make_shared<channel_data>((void *) payload, len);
      direct.send_object(buf, /*rcpt_id=*/0);
      return 0;
    }
  } catch (const std::exception &) {
    return 2;
  } catch (...) {
    return 3;
  }
}

int RunNonBlockingModeRank(FMI::Utils::peer_num rank, int listen_port,
                           const std::string &comm_name, const char *payload) {
  auto backend = std::make_shared<FMI::Utils::DirectBackend>();
  backend->setUseDirectRedis(true);
  backend->withHost("127.0.0.1");
  backend->withPort(listen_port);
  backend->withMaxTimeout(5000);
  backend->setResolveBackendDNS(false);
  backend->setBlockingMode(FMI::Utils::NONBLOCKING);

  FMI::Comm::Direct direct(backend);
  direct.set_redis_host(GetRedisHost());
  direct.set_redis_port(GetRedisPort());
  direct.set_comm_name(comm_name);
  direct.set_peer_id(rank);
  direct.set_num_peers(2);

  FMI::Utils::peer_num peer = rank == 0 ? 1 : 0;
  size_t len = strlen(payload) + 1;
  try {
    direct.init();
    if (!direct.checkSend(peer, FMI::Utils::NONBLOCKING)) return 1;

    if (rank == 0) {
      std::string received(len, '\0');
      auto buf = std::make_shared<channel_data>((void *) received.data(), len);
      direct.recv_object(buf, peer);
      std::memcpy(&received[0], buf->buf.get(), len);
      return std::memcmp(received.data(), payload, len) == 0 ? 0 : 4;
    }
    auto buf = std::make_shared<channel_data>((void *) payload, len);
    direct.send_object(buf, peer);
    return 0;
  } catch (const std::exception &) {
    return 2;
  } catch (...) {
    return 3;
  }
}

bool WaitForChildBounded(pid_t pid, int timeout_ms, int *status) {
  constexpr int kPollIntervalMs = 50;
  for (int waited = 0; waited < timeout_ms; waited += kPollIntervalMs) {
    if (waitpid(pid, status, WNOHANG) == pid) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(kPollIntervalMs));
  }
  return false;
}

}

TEST(DirectChannelRedisTest, TcpunchPathStillRoundTrips) {
  int rendezvous_port = 19000 + (getpid() % 200);
  TcpunchdFixture server(rendezvous_port);
  ASSERT_TRUE(server.started());

  std::string comm_name = "direct_tcpunch_regression_" + std::to_string(getpid());
  const char *payload = "hello-over-tcpunch";

  pid_t rank0_pid = fork();
  ASSERT_GE(rank0_pid, 0);
  if (rank0_pid == 0) {
    _exit(RunTcpunchRank(0, rendezvous_port, comm_name, payload));
  }

  pid_t rank1_pid = fork();
  ASSERT_GE(rank1_pid, 0);
  if (rank1_pid == 0) {
    _exit(RunTcpunchRank(1, rendezvous_port, comm_name, payload));
  }

  int status0 = 0, status1 = 0;
  ASSERT_EQ(waitpid(rank0_pid, &status0, 0), rank0_pid);
  ASSERT_EQ(waitpid(rank1_pid, &status1, 0), rank1_pid);

  ASSERT_TRUE(WIFEXITED(status0)) << "rank0 (receiver) process did not exit normally";
  ASSERT_TRUE(WIFEXITED(status1)) << "rank1 (sender) process did not exit normally";
  EXPECT_EQ(WEXITSTATUS(status0), 0) << "rank0 (receiver) failed (see codes in RunTcpunchRank)";
  EXPECT_EQ(WEXITSTATUS(status1), 0) << "rank1 (sender) failed (see codes in RunTcpunchRank)";
}

TEST(DirectChannelRedisTest, RedisPathRoundTripsBothDirections) {
  std::string redis_host = GetRedisHost();
  int redis_port = GetRedisPort();
  std::string comm_name = "direct_redis_test_" + std::to_string(getpid());

  int port0 = 19200 + (getpid() % 100) * 2;
  int port1 = port0 + 1;

  auto make_backend = [&](int listen_port) {
    auto backend = std::make_shared<FMI::Utils::DirectBackend>();
    backend->setUseDirectRedis(true);
    backend->withHost("127.0.0.1");
    backend->withPort(listen_port);
    backend->withMaxTimeout(5000);
    backend->setResolveBackendDNS(false);
    backend->setBlockingMode(FMI::Utils::BLOCKING);
    return backend;
  };

  auto backend0 = make_backend(port0);
  auto backend1 = make_backend(port1);

  FMI::Comm::Direct direct0(backend0);
  direct0.set_redis_host(redis_host);
  direct0.set_redis_port(redis_port);
  direct0.set_comm_name(comm_name);
  direct0.set_peer_id(0);
  direct0.set_num_peers(2);

  FMI::Comm::Direct direct1(backend1);
  direct1.set_redis_host(redis_host);
  direct1.set_redis_port(redis_port);
  direct1.set_comm_name(comm_name);
  direct1.set_peer_id(1);
  direct1.set_num_peers(2);

  direct0.init();
  direct1.init();

  const char *msg_to_0 = "hello-from-1-via-redis-direct";
  const char *msg_to_1 = "hello-from-0-via-redis-direct";
  size_t len0 = strlen(msg_to_0) + 1;
  size_t len1 = strlen(msg_to_1) + 1;

  std::string received_at_0(len0, '\0');
  std::string received_at_1(len1, '\0');
  bool rank0_ok = false, rank1_ok = false;
  std::string t0_error, t1_error;

  std::thread t0([&] {
    try {
      auto recv_buf = std::make_shared<channel_data>((void *) received_at_0.data(), len0);
      direct0.recv_object(recv_buf, /*sender_id=*/1);
      std::memcpy(&received_at_0[0], recv_buf->buf.get(), len0);

      auto send_buf = std::make_shared<channel_data>((void *) msg_to_1, len1);
      direct0.send_object(send_buf, /*rcpt_id=*/1);
      rank0_ok = true;
    } catch (const std::exception &e) {
      t0_error = e.what();
    } catch (...) {
      t0_error = "unknown exception";
    }
  });

  std::thread t1([&] {
    try {
      auto send_buf = std::make_shared<channel_data>((void *) msg_to_0, len0);
      direct1.send_object(send_buf, /*rcpt_id=*/0);

      auto recv_buf = std::make_shared<channel_data>((void *) received_at_1.data(), len1);
      direct1.recv_object(recv_buf, /*sender_id=*/0);
      std::memcpy(&received_at_1[0], recv_buf->buf.get(), len1);
      rank1_ok = true;
    } catch (const std::exception &e) {
      t1_error = e.what();
    } catch (...) {
      t1_error = "unknown exception";
    }
  });

  t0.join();
  t1.join();

  ASSERT_TRUE(t0_error.empty()) << "rank0 threw: " << t0_error;
  ASSERT_TRUE(t1_error.empty()) << "rank1 threw: " << t1_error;
  ASSERT_TRUE(rank0_ok);
  ASSERT_TRUE(rank1_ok);
  ASSERT_STREQ(received_at_0.c_str(), msg_to_0);
  ASSERT_STREQ(received_at_1.c_str(), msg_to_1);
}

TEST(DirectChannelRedisTest, RedisPathFourRankMeshRoundTrips) {
  constexpr int kNumRanks = 4;
  std::string redis_host = GetRedisHost();
  int redis_port = GetRedisPort();
  std::string comm_name = "direct_redis_mesh_test_" + std::to_string(getpid());

  int base_port = 19400 + (getpid() % 50) * 4;
  int ports[kNumRanks];
  for (int i = 0; i < kNumRanks; ++i) ports[i] = base_port + i;

  auto make_backend = [&](int listen_port) {
    auto backend = std::make_shared<FMI::Utils::DirectBackend>();
    backend->setUseDirectRedis(true);
    backend->withHost("127.0.0.1");
    backend->withPort(listen_port);
    backend->withMaxTimeout(5000);
    backend->setResolveBackendDNS(false);
    backend->setBlockingMode(FMI::Utils::BLOCKING);
    return backend;
  };

  std::vector<std::shared_ptr<FMI::Utils::DirectBackend>> backends;
  std::vector<std::unique_ptr<FMI::Comm::Direct>> directs;
  for (int i = 0; i < kNumRanks; ++i) {
    backends.push_back(make_backend(ports[i]));
    auto direct = std::make_unique<FMI::Comm::Direct>(backends[i]);
    direct->set_redis_host(redis_host);
    direct->set_redis_port(redis_port);
    direct->set_comm_name(comm_name);
    direct->set_peer_id(i);
    direct->set_num_peers(kNumRanks);
    directs.push_back(std::move(direct));
  }

  for (int i = 0; i < kNumRanks; ++i) {
    directs[i]->init();
  }

  auto make_tag = [](int from, int to) {
    return "from" + std::to_string(from) + "to" + std::to_string(to);
  };

  bool rank_ok[kNumRanks] = {false, false, false, false};
  std::string rank_error[kNumRanks];

  std::vector<std::thread> threads;
  for (int r = 0; r < kNumRanks; ++r) {
    threads.emplace_back([&, r] {
      try {
        FMI::Comm::Direct &direct = *directs[r];

        std::vector<std::string> outgoing(kNumRanks);
        for (int peer = 0; peer < kNumRanks; ++peer) {
          if (peer == r) continue;
          outgoing[peer] = make_tag(r, peer);
          auto send_buf = std::make_shared<channel_data>(
              (void *) outgoing[peer].c_str(), outgoing[peer].size() + 1);
          direct.send_object(send_buf, /*rcpt_id=*/peer);
        }

        for (int peer = 0; peer < kNumRanks; ++peer) {
          if (peer == r) continue;
          std::string expected = make_tag(peer, r);
          size_t len = expected.size() + 1;
          std::string received(len, '\0');
          auto recv_buf = std::make_shared<channel_data>((void *) received.data(), len);
          direct.recv_object(recv_buf, /*sender_id=*/peer);
          std::memcpy(&received[0], recv_buf->buf.get(), len);
          EXPECT_STREQ(received.c_str(), expected.c_str())
              << "rank " << r << " got wrong payload from rank " << peer;
        }

        rank_ok[r] = true;
      } catch (const std::exception &e) {
        rank_error[r] = e.what();
      } catch (...) {
        rank_error[r] = "unknown exception";
      }
    });
  }

  for (auto &t : threads) t.join();

  for (int r = 0; r < kNumRanks; ++r) {
    EXPECT_TRUE(rank_error[r].empty()) << "rank " << r << " threw: " << rank_error[r];
    EXPECT_TRUE(rank_ok[r]) << "rank " << r << " did not complete successfully";
  }
}

TEST(DirectChannelRedisTest, RedisNonBlockingModeEstablishesSockets) {
  constexpr int kRankTimeoutMs = 30000;
  std::string comm_name = "direct_redis_nbmode_test_" + std::to_string(getpid());
  const char *payload = "hello-from-1-in-nonblocking-mode";

  int port0 = 19600 + (getpid() % 100) * 2;
  int port1 = port0 + 1;

  pid_t rank0_pid = fork();
  ASSERT_GE(rank0_pid, 0);
  if (rank0_pid == 0) {
    _exit(RunNonBlockingModeRank(0, port0, comm_name, payload));
  }

  pid_t rank1_pid = fork();
  ASSERT_GE(rank1_pid, 0);
  if (rank1_pid == 0) {
    _exit(RunNonBlockingModeRank(1, port1, comm_name, payload));
  }

  int status0 = 0, status1 = 0;
  bool rank0_finished = WaitForChildBounded(rank0_pid, kRankTimeoutMs, &status0);
  bool rank1_finished = WaitForChildBounded(rank1_pid, kRankTimeoutMs, &status1);

  if (!rank0_finished) {
    kill(rank0_pid, SIGKILL);
    waitpid(rank0_pid, &status0, 0);
  }
  if (!rank1_finished) {
    kill(rank1_pid, SIGKILL);
    waitpid(rank1_pid, &status1, 0);
  }

  ASSERT_TRUE(rank0_finished) << "rank0 did not finish within " << kRankTimeoutMs << "ms";
  ASSERT_TRUE(rank1_finished) << "rank1 did not finish within " << kRankTimeoutMs << "ms";
  ASSERT_TRUE(WIFEXITED(status0)) << "rank0 process did not exit normally";
  ASSERT_TRUE(WIFEXITED(status1)) << "rank1 process did not exit normally";
  EXPECT_EQ(WEXITSTATUS(status0), 0)
      << "rank0 failed (1=no nonblocking socket, 2/3=threw, 4=payload mismatch)";
  EXPECT_EQ(WEXITSTATUS(status1), 0)
      << "rank1 failed (1=no nonblocking socket, 2/3=threw, 4=payload mismatch)";
}