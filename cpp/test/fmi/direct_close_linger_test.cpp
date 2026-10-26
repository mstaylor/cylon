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

#include <gtest/gtest.h>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <vector>

namespace {

struct TcpPair {
  int sender = -1;
  int receiver = -1;
};

TcpPair ConnectLoopback(int receive_buffer_bytes, int send_buffer_bytes) {
  int listener = socket(AF_INET, SOCK_STREAM, 0);
  setsockopt(listener, SOL_SOCKET, SO_RCVBUF, &receive_buffer_bytes, sizeof(receive_buffer_bytes));
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  address.sin_port = 0;
  bind(listener, reinterpret_cast<sockaddr *>(&address), sizeof(address));
  socklen_t length = sizeof(address);
  getsockname(listener, reinterpret_cast<sockaddr *>(&address), &length);
  listen(listener, 1);

  TcpPair pair;
  pair.sender = socket(AF_INET, SOCK_STREAM, 0);
  setsockopt(pair.sender, SOL_SOCKET, SO_SNDBUF, &send_buffer_bytes, sizeof(send_buffer_bytes));
  connect(pair.sender, reinterpret_cast<sockaddr *>(&address), sizeof(address));
  pair.receiver = accept(listener, nullptr, nullptr);
  close(listener);
  return pair;
}

void FillUntilBlocked(int fd) {
  fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) | O_NONBLOCK);
  std::vector<char> chunk(64 * 1024, 'x');
  while (send(fd, chunk.data(), chunk.size(), 0) > 0) {
  }
}

double SecondsToClose(int fd, int linger_s) {
  auto start = std::chrono::steady_clock::now();
  FMI::Comm::close_after_acknowledged(fd, linger_s);
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

TEST(CloseAfterAcknowledgedTest, ReturnsAsSoonAsThePeerKernelHasTheBytesEvenIfItsApplicationNeverReads) {
  auto pair = ConnectLoopback(64 * 1024, 64 * 1024);
  char message[128] = {};
  ASSERT_EQ(send(pair.sender, message, sizeof(message), 0), static_cast<ssize_t>(sizeof(message)));

  EXPECT_LT(SecondsToClose(pair.sender, 5), 1.0);

  char received[sizeof(message)];
  EXPECT_EQ(recv(pair.receiver, received, sizeof(received), MSG_WAITALL), static_cast<ssize_t>(sizeof(received)));
  EXPECT_EQ(recv(pair.receiver, received, 1, 0), 0);
  close(pair.receiver);
}

TEST(CloseAfterAcknowledgedTest, WaitsUpToTheLingerWhileThePeerCannotAcknowledge) {
  auto pair = ConnectLoopback(4096, 4096);
  FillUntilBlocked(pair.sender);

  EXPECT_GE(SecondsToClose(pair.sender, 1), 0.9);
  close(pair.receiver);
}

TEST(CloseAfterAcknowledgedTest, ZeroLingerKeepsThePlainImmediateClose) {
  auto pair = ConnectLoopback(4096, 4096);
  FillUntilBlocked(pair.sender);

  EXPECT_LT(SecondsToClose(pair.sender, 0), 0.5);
  close(pair.receiver);
}

}