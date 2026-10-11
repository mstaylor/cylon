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
#include "cylon/thridparty/TCPunch/client/tcpunch.hpp"

#include <gtest/gtest.h>

#include <fcntl.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>

namespace {

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

bool RegisterThenVanish(const std::string &pairing_name, int port) {
  int sock = socket(AF_INET, SOCK_STREAM, 0);
  if (sock < 0) return false;
  sockaddr_in server{};
  server.sin_family = AF_INET;
  server.sin_addr.s_addr = inet_addr("127.0.0.1");
  server.sin_port = htons(port);
  if (connect(sock, reinterpret_cast<sockaddr *>(&server), sizeof(server)) != 0) {
    close(sock);
    return false;
  }
  uint8_t req[CLIENT_REQUEST_SIZE];
  build_request(req, pairing_name);
  send(sock, req, CLIENT_REQUEST_SIZE, 0);

  uint8_t resp_buf[SERVER_RESPONSE_SIZE];
  ServerResponse resp{};
  do {
    if (recv(sock, resp_buf, SERVER_RESPONSE_SIZE, MSG_WAITALL) != (ssize_t) SERVER_RESPONSE_SIZE) {
      close(sock);
      return false;
    }
    parse_response(resp_buf, resp);
  } while (resp.status == PairingStatus::WAITING);
  close(sock);
  return resp.status == PairingStatus::PAIRED;
}

pid_t ForkVanishingPeer(const std::string &pairing_name, int port, int registrations) {
  pid_t pid = fork();
  if (pid == 0) {
    int paired = 0;
    for (int i = 0; i < registrations; ++i) {
      if (RegisterThenVanish(pairing_name, port)) ++paired;
    }
    _exit(paired == registrations ? 0 : 1);
  }
  return pid;
}

int WaitBounded(pid_t pid, int timeout_ms) {
  int status = 0;
  for (int waited = 0; waited < timeout_ms; waited += 50) {
    if (waitpid(pid, &status, WNOHANG) == pid) {
      return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  kill(pid, SIGKILL);
  waitpid(pid, &status, 0);
  return -2;
}

std::shared_ptr<FMI::Utils::DirectBackend> BlockingBackend(int port) {
  auto backend = std::make_shared<FMI::Utils::DirectBackend>();
  backend->withHost("127.0.0.1");
  backend->withPort(port);
  backend->withMaxTimeout(5000);
  backend->setResolveBackendDNS(false);
  backend->setBlockingMode(FMI::Utils::BLOCKING);
  return backend;
}

std::string PairingName(const std::string &comm_name) {
  return comm_name + "_fmi_pair0_1BLOCKING";
}

}

TEST(DirectPairFailureTest, BlockingSendThrowsWhenEveryHolePunchFails) {
  signal(SIGPIPE, SIG_IGN);
  setenv("FMI_HOLEPUNCH_CONNECT_TIMEOUT_MS", "1000", 1);
  setenv("FMI_TCPUNCH_MAX_TRIES", "2", 1);
  int port = 23000 + (getpid() % 500);
  TcpunchdProcess server(port);
  ASSERT_TRUE(server.started());

  std::string comm_name = "pair_failure_" + std::to_string(getpid());
  pid_t peer = ForkVanishingPeer(PairingName(comm_name), port, 2);
  ASSERT_GE(peer, 0);

  FMI::Comm::Direct direct(BlockingBackend(port));
  direct.set_peer_id(0);
  direct.set_num_peers(2);
  direct.set_comm_name(comm_name);
  direct.init();

  const char payload[] = "never-delivered";
  auto buf = std::make_shared<channel_data>((void *) payload, sizeof(payload));
  EXPECT_THROW(direct.send_object(buf, 1), FMI::Utils::Timeout);
  EXPECT_EQ(WaitBounded(peer, 5000), 0) << "vanishing peer did not pair twice";
}

TEST(DirectPairFailureTest, BlockingSendDeliversAfterAFailedHolePunch) {
  signal(SIGPIPE, SIG_IGN);
  setenv("FMI_HOLEPUNCH_CONNECT_TIMEOUT_MS", "2000", 1);
  setenv("FMI_TCPUNCH_MAX_TRIES", "4", 1);
  int port = 23600 + (getpid() % 500);
  TcpunchdProcess server(port);
  ASSERT_TRUE(server.started());

  std::string comm_name = "pair_recovery_" + std::to_string(getpid());
  const char payload[] = "delivered-on-retry";

  pid_t receiver = fork();
  ASSERT_GE(receiver, 0);
  if (receiver == 0) {
    if (!RegisterThenVanish(PairingName(comm_name), port)) _exit(4);
    try {
      FMI::Comm::Direct direct(BlockingBackend(port));
      direct.set_peer_id(1);
      direct.set_num_peers(2);
      direct.set_comm_name(comm_name);
      direct.init();
      char received[sizeof(payload)] = {0};
      auto buf = std::make_shared<channel_data>((void *) received, sizeof(received));
      direct.recv_object(buf, 0);
      _exit(std::memcmp(buf->buf.get(), payload, sizeof(payload)) == 0 ? 0 : 1);
    } catch (...) {
      _exit(2);
    }
  }

  FMI::Comm::Direct direct(BlockingBackend(port));
  direct.set_peer_id(0);
  direct.set_num_peers(2);
  direct.set_comm_name(comm_name);
  direct.init();

  auto buf = std::make_shared<channel_data>((void *) payload, sizeof(payload));
  EXPECT_NO_THROW(direct.send_object(buf, 1));
  EXPECT_EQ(WaitBounded(receiver, 20000), 0) << "receiver did not get the payload";
}