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

#include <arpa/inet.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <thread>

namespace {

constexpr int kPairTimeoutMs = 5000;

int ListeningSocket(uint16_t *port) {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = 0;
  if (bind(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0 || listen(fd, 4) != 0) {
    close(fd);
    return -1;
  }
  socklen_t len = sizeof(addr);
  getsockname(fd, reinterpret_cast<sockaddr *>(&addr), &len);
  *port = addr.sin_port;
  return fd;
}

void PairedResponse(uint8_t *buf, uint16_t reported_port, uint16_t peer_port) {
  memset(buf, 0, SERVER_RESPONSE_SIZE);
  uint32_t loopback = htonl(INADDR_LOOPBACK);
  buf[0] = static_cast<uint8_t>(PairingStatus::PAIRED);
  memcpy(buf + 1, &loopback, 4);
  memcpy(buf + 5, &reported_port, 2);
  memcpy(buf + 7, &loopback, 4);
  memcpy(buf + 11, &peer_port, 2);
}

class PortTranslatingRendezvous {
 public:
  explicit PortTranslatingRendezvous(uint16_t reported_port) : reported_port_(reported_port) {
    listen_fd_ = ListeningSocket(&port_);
    thread_ = std::thread([this]() { Serve(); });
  }

  ~PortTranslatingRendezvous() {
    if (thread_.joinable()) thread_.join();
    if (listen_fd_ >= 0) close(listen_fd_);
  }

  int port() const { return ntohs(port_); }

 private:
  void Serve() {
    int clients[2];
    uint16_t real_ports[2];
    for (int i = 0; i < 2; ++i) {
      sockaddr_in peer{};
      socklen_t len = sizeof(peer);
      clients[i] = accept(listen_fd_, reinterpret_cast<sockaddr *>(&peer), &len);
      real_ports[i] = peer.sin_port;
      uint8_t request[CLIENT_REQUEST_SIZE];
      recv(clients[i], request, CLIENT_REQUEST_SIZE, MSG_WAITALL);
    }
    for (int i = 0; i < 2; ++i) {
      uint8_t response[SERVER_RESPONSE_SIZE];
      PairedResponse(response, reported_port_, real_ports[1 - i]);
      send(clients[i], response, SERVER_RESPONSE_SIZE, 0);
    }
    std::this_thread::sleep_for(std::chrono::seconds(2));
    close(clients[0]);
    close(clients[1]);
  }

  uint16_t reported_port_;
  uint16_t port_ = 0;
  int listen_fd_ = -1;
  std::thread thread_;
};

}

TEST(TcpunchPortTranslationTest, PairsWhenTheServerReportsAPortOtherThanTheLocalOne) {
  signal(SIGPIPE, SIG_IGN);
  uint16_t occupied_port = 0;
  int occupied = ListeningSocket(&occupied_port);
  ASSERT_GE(occupied, 0);
  PortTranslatingRendezvous server(occupied_port);

  int sockets[2] = {-1, -1};
  std::thread peers[2];
  for (int i = 0; i < 2; ++i) {
    peers[i] = std::thread([&, i]() {
      try {
        sockets[i] = pair("translated_pair", "127.0.0.1", server.port(), kPairTimeoutMs);
      } catch (...) {
        sockets[i] = -1;
      }
    });
  }
  for (auto &t : peers) t.join();
  close(occupied);

  ASSERT_GE(sockets[0], 0) << "first peer failed to pair";
  ASSERT_GE(sockets[1], 0) << "second peer failed to pair";
  uint32_t sent = 0xC0FFEE, received = 0;
  ASSERT_EQ(send(sockets[0], &sent, sizeof(sent), 0), (ssize_t) sizeof(sent));
  ASSERT_EQ(recv(sockets[1], &received, sizeof(received), MSG_WAITALL), (ssize_t) sizeof(received));
  EXPECT_EQ(received, sent);
  close(sockets[0]);
  close(sockets[1]);
}