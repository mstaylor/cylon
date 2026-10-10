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

#include "cylon/thridparty/TCPunch/client/tcpunch.hpp"

#include <gtest/gtest.h>

#include <fcntl.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int kPairTimeoutMs = 10000;

class TcpunchdProcess {
 public:
  explicit TcpunchdProcess(int port) {
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

  ~TcpunchdProcess() {
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

std::string PairName(const std::string &prefix, uint32_t spoke) {
  return prefix + "_hub_" + std::to_string(spoke);
}

int RunSpoke(const std::string &prefix, uint32_t spoke, int port) {
  signal(SIGPIPE, SIG_IGN);
  int socket_fd = -1;
  try {
    socket_fd = pair(PairName(prefix, spoke), "127.0.0.1", port, kPairTimeoutMs);
  } catch (...) {
    return 3;
  }
  if (socket_fd < 0) return 1;
  ssize_t sent = send(socket_fd, &spoke, sizeof(spoke), 0);
  char ack = 0;
  recv(socket_fd, &ack, 1, MSG_WAITALL);
  close(socket_fd);
  return sent == sizeof(spoke) ? 0 : 2;
}

uint32_t ReceiveSpokeId(int socket_fd) {
  struct timeval timeout{5, 0};
  setsockopt(socket_fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
  uint32_t spoke = UINT32_MAX;
  if (recv(socket_fd, &spoke, sizeof(spoke), MSG_WAITALL) != sizeof(spoke)) return UINT32_MAX;
  char ack = 1;
  send(socket_fd, &ack, 1, 0);
  return spoke;
}

}

TEST(TcpunchConcurrentPairTest, ConcurrentPairsEachReachTheirOwnPeer) {
  constexpr uint32_t kSpokes = 8;
  signal(SIGPIPE, SIG_IGN);
  int port = 21000 + (getpid() % 500);
  TcpunchdProcess server(port);
  ASSERT_TRUE(server.started());
  std::string prefix = "concurrent_pair_" + std::to_string(getpid());

  std::vector<pid_t> spokes;
  for (uint32_t spoke = 0; spoke < kSpokes; ++spoke) {
    pid_t pid = fork();
    ASSERT_GE(pid, 0);
    if (pid == 0) _exit(RunSpoke(prefix, spoke, port));
    spokes.push_back(pid);
  }

  std::vector<int> sockets(kSpokes, -1);
  std::vector<uint32_t> received(kSpokes, UINT32_MAX);
  std::vector<std::thread> hub;
  for (uint32_t spoke = 0; spoke < kSpokes; ++spoke) {
    hub.emplace_back([&, spoke]() {
      try {
        sockets[spoke] = pair(PairName(prefix, spoke), "127.0.0.1", port, kPairTimeoutMs);
      } catch (...) {
        sockets[spoke] = -1;
      }
      if (sockets[spoke] >= 0) received[spoke] = ReceiveSpokeId(sockets[spoke]);
    });
  }
  for (auto &t : hub) t.join();

  for (uint32_t spoke = 0; spoke < kSpokes; ++spoke) {
    EXPECT_GE(sockets[spoke], 0) << "hub failed to pair " << PairName(prefix, spoke);
    EXPECT_EQ(received[spoke], spoke) << "socket for " << PairName(prefix, spoke)
                                      << " reached a different peer";
    if (sockets[spoke] >= 0) close(sockets[spoke]);
  }
  for (pid_t pid : spokes) {
    int status = 0;
    waitpid(pid, &status, 0);
  }
}