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

#include "RedisDirectPair.hpp"
#include "../../TCPunch/client/tcpunch.hpp"
#include <glog/logging.h>
#include <sw/redis++/redis++.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <unistd.h>
#include <poll.h>
#include <cstring>
#include <cerrno>
#include <chrono>
#include <thread>
#include <regex>

namespace {

std::string HttpGet(const std::string &host, int port, const std::string &path) {
  struct addrinfo hints{}, *res = nullptr;
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  if (getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &res) != 0) {
    return "";
  }
  int fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
  if (fd < 0) { freeaddrinfo(res); return ""; }
  struct timeval tv{2, 0};
  setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
  if (connect(fd, res->ai_addr, res->ai_addrlen) != 0) {
    close(fd); freeaddrinfo(res); return "";
  }
  freeaddrinfo(res);
  std::string req = "GET " + path + " HTTP/1.1\r\nHost: " + host +
                     "\r\nConnection: close\r\n\r\n";
  if (::send(fd, req.data(), req.size(), 0) < 0) { close(fd); return ""; }
  std::string resp;
  char buf[4096];
  ssize_t n;
  while ((n = ::recv(fd, buf, sizeof(buf), 0)) > 0) resp.append(buf, n);
  close(fd);
  auto body_start = resp.find("\r\n\r\n");
  return body_start == std::string::npos ? "" : resp.substr(body_start + 4);
}

constexpr int kAddrTtlSeconds = 3600;
constexpr int kPollIntervalMs = 200;
constexpr int kHandshakeTimeoutMs = 5000;

constexpr uint8_t kModeByteBlocking = 0;
constexpr uint8_t kModeByteNonBlocking = 1;

uint8_t EncodeModeByte(FMI::Utils::Mode mode) {
  return mode == FMI::Utils::NONBLOCKING ? kModeByteNonBlocking : kModeByteBlocking;
}

bool IsKnownModeByte(uint8_t mode_byte) {
  return mode_byte == kModeByteBlocking || mode_byte == kModeByteNonBlocking;
}

FMI::Utils::Mode DecodeModeByte(uint8_t mode_byte) {
  return mode_byte == kModeByteNonBlocking ? FMI::Utils::NONBLOCKING
                                           : FMI::Utils::BLOCKING;
}

const char *ModeName(FMI::Utils::Mode mode) {
  return mode == FMI::Utils::NONBLOCKING ? "NONBLOCKING" : "BLOCKING";
}

int PeerAndModeKey(int peer_id, FMI::Utils::Mode mode) {
  return peer_id * 2 + EncodeModeByte(mode);
}

bool RecvExactly(int fd, void *dst, size_t len) {
  size_t got = 0;
  while (got < len) {
    ssize_t n = ::recv(fd, static_cast<char *>(dst) + got, len - got, 0);
    if (n <= 0) return false;
    got += static_cast<size_t>(n);
  }
  return true;
}

}

std::string FMI::Comm::RedisDirectEstablisher::ResolveOwnAddress() const {
  if (!host_override_.empty()) {
    return host_override_ + ":" + std::to_string(listen_port_);
  }
  const char *metadata_uri = std::getenv("ECS_CONTAINER_METADATA_URI_V4");
  if (metadata_uri == nullptr) {
    LOG(ERROR) << "direct-redis: no host_override and "
               << "ECS_CONTAINER_METADATA_URI_V4 is not set — cannot resolve own address";
    throw ValidationFailure();
  }
  std::string url(metadata_uri);
  auto scheme_pos = url.find("://");
  if (scheme_pos == std::string::npos) {
    LOG(ERROR) << "direct-redis: ECS_CONTAINER_METADATA_URI_V4 is malformed (no scheme "
               << "separator): " << url;
    throw ValidationFailure();
  }
  auto scheme_end = scheme_pos + 3;
  auto path_start = url.find('/', scheme_end);
  if (path_start == std::string::npos) {
    LOG(ERROR) << "direct-redis: ECS_CONTAINER_METADATA_URI_V4 is malformed (no path "
               << "component): " << url;
    throw ValidationFailure();
  }
  std::string host = url.substr(scheme_end, path_start - scheme_end);
  std::string path = url.substr(path_start);
  std::string body = HttpGet(host, 80, path);
  std::smatch m;
  std::regex ip_re("\"IPv4Addresses\"\\s*:\\s*\\[\\s*\"([0-9.]+)\"");
  if (!std::regex_search(body, m, ip_re)) {
    LOG(ERROR) << "direct-redis: could not find IPv4Addresses in ECS metadata response";
    throw ValidationFailure();
  }
  return m[1].str() + ":" + std::to_string(listen_port_);
}

void FMI::Comm::RedisDirectEstablisher::PublishOwnAddress(const std::string &own_addr) const {
  auto opts = sw::redis::ConnectionOptions{};
  opts.host = redis_host_;
  opts.port = redis_port_;
  auto redis = std::make_shared<sw::redis::Redis>(opts);
  std::string key = comm_name_ + ":direct_redis_addrs";
  redis->hset(key, std::to_string(self_rank_), own_addr);
  redis->expire(key, std::chrono::seconds(kAddrTtlSeconds));
  LOG(INFO) << "direct-redis: published rank " << self_rank_ << " address " << own_addr;
}

std::string FMI::Comm::RedisDirectEstablisher::LookupPeerAddress(
    FMI::Utils::peer_num partner_id, int timeout_ms) const {
  auto opts = sw::redis::ConnectionOptions{};
  opts.host = redis_host_;
  opts.port = redis_port_;
  auto redis = std::make_shared<sw::redis::Redis>(opts);
  std::string key = comm_name_ + ":direct_redis_addrs";
  int waited_ms = 0;
  while (waited_ms < timeout_ms) {
    auto val = redis->hget(key, std::to_string(partner_id));
    if (val) return *val;
    std::this_thread::sleep_for(std::chrono::milliseconds(kPollIntervalMs));
    waited_ms += kPollIntervalMs;
  }
  LOG(WARNING) << "direct-redis: partner " << partner_id << " never published its "
               << "address within " << timeout_ms << "ms (key=" << key << ")";
  throw Timeout();
}

void FMI::Comm::RedisDirectEstablisher::Init(
    std::string redis_host, int redis_port, std::string comm_name,
    FMI::Utils::peer_num self_rank, FMI::Utils::peer_num num_peers,
    int listen_port, std::string host_override) {
  if (initialized_) {
    LOG(ERROR) << "direct-redis: Init() called more than once on rank " << self_rank_
               << " (comm_name=" << comm_name_ << ") — it must be called exactly once";
    throw ValidationFailure();
  }
  initialized_ = true;
  redis_host_ = std::move(redis_host);
  redis_port_ = redis_port;
  comm_name_ = std::move(comm_name);
  self_rank_ = self_rank;
  listen_port_ = listen_port;
  host_override_ = std::move(host_override);

  if (self_rank_ < num_peers - 1) {
    listen_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd_ < 0) {
      LOG(ERROR) << "direct-redis: socket() failed: " << strerror(errno);
      throw ValidationFailure();
    }
    int one = 1;
    setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(static_cast<uint16_t>(listen_port_));
    if (bind(listen_fd_, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0) {
      LOG(ERROR) << "direct-redis: bind() on port " << listen_port_
                 << " failed: " << strerror(errno);
      close(listen_fd_);
      listen_fd_ = -1;
      throw ValidationFailure();
    }
    if (listen(listen_fd_, /*backlog=*/num_peers) != 0) {
      LOG(ERROR) << "direct-redis: listen() failed: " << strerror(errno);
      close(listen_fd_);
      listen_fd_ = -1;
      throw ValidationFailure();
    }
    running_ = true;
    accept_thread_ = std::thread(&RedisDirectEstablisher::AcceptLoop, this);
  }

  PublishOwnAddress(ResolveOwnAddress());
}

void FMI::Comm::RedisDirectEstablisher::AcceptLoop() {
  while (running_) {
    struct pollfd pfd{};
    pfd.fd = listen_fd_;
    pfd.events = POLLIN;
    int rc = poll(&pfd, 1, kPollIntervalMs);
    if (rc == 0) {
      continue;
    }
    if (rc < 0) {
      if (errno == EINTR) continue;
      LOG(WARNING) << "direct-redis: poll() failed: " << strerror(errno)
                   << " — continuing to accept further peers";
      continue;
    }
    int fd = accept(listen_fd_, nullptr, nullptr);
    if (fd < 0) {
      if (!running_) return;
      LOG(WARNING) << "direct-redis: accept() failed: " << strerror(errno)
                   << " — continuing to accept further peers";
      continue;
    }
    if (!running_) {
      close(fd);
      return;
    }
    int32_t peer_id_net = 0;
    uint8_t mode_byte = 0;
    struct timeval handshake_tv{kHandshakeTimeoutMs / 1000,
                                (kHandshakeTimeoutMs % 1000) * 1000};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &handshake_tv, sizeof(handshake_tv));
    if (!RecvExactly(fd, &peer_id_net, sizeof(peer_id_net)) ||
        !RecvExactly(fd, &mode_byte, sizeof(mode_byte)) ||
        !IsKnownModeByte(mode_byte)) {
      LOG(WARNING) << "direct-redis: rank/mode handshake failed on an accepted "
                   << "connection, dropping it and continuing to accept further peers";
      close(fd);
      continue;
    }
    int from_peer = static_cast<int>(ntohl(peer_id_net));
    FMI::Utils::Mode from_mode = DecodeModeByte(mode_byte);
    {
      std::lock_guard<std::mutex> lock(mu_);
      const int key = PeerAndModeKey(from_peer, from_mode);
      auto existing = accepted_fd_by_peer_and_mode_.find(key);
      if (existing != accepted_fd_by_peer_and_mode_.end()) {
        LOG(WARNING) << "direct-redis: a second " << ModeName(from_mode)
                     << " connection from peer " << from_peer
                     << " arrived before the first was consumed — closing the superseded one";
        close(existing->second);
      }
      accepted_fd_by_peer_and_mode_[key] = fd;
    }
    LOG(INFO) << "direct-redis: accepted " << ModeName(from_mode)
              << " connection from peer " << from_peer;
    cv_.notify_all();
  }
}

int FMI::Comm::RedisDirectEstablisher::Connect(
    FMI::Utils::peer_num self_rank, FMI::Utils::peer_num partner_id, int timeout_ms,
    FMI::Utils::Mode mode) {
  if (partner_id < self_rank) {
    std::string addr = LookupPeerAddress(partner_id, timeout_ms);
    auto colon = addr.rfind(':');
    if (colon == std::string::npos) {
      LOG(ERROR) << "direct-redis: rank " << self_rank << " (comm_name=" << comm_name_
                 << ") read a malformed address for partner " << partner_id
                 << " — no host:port separator in \"" << addr << "\"";
      throw ValidationFailure();
    }
    std::string peer_host = addr.substr(0, colon);
    int peer_port = 0;
    try {
      peer_port = std::stoi(addr.substr(colon + 1));
    } catch (const std::exception &) {
      LOG(ERROR) << "direct-redis: rank " << self_rank << " (comm_name=" << comm_name_
                 << ") read a malformed address for partner " << partner_id
                 << " — unparseable port in \"" << addr << "\"";
      throw ValidationFailure();
    }

    struct addrinfo hints{}, *res = nullptr;
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(peer_host.c_str(), std::to_string(peer_port).c_str(), &hints,
                    &res) != 0) {
      throw ValidationFailure();
    }
    int fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (fd < 0 || connect(fd, res->ai_addr, res->ai_addrlen) != 0) {
      LOG(WARNING) << "direct-redis: connect() to peer " << partner_id << " at "
                   << addr << " failed: " << strerror(errno);
      if (fd >= 0) close(fd);
      freeaddrinfo(res);
      throw Timeout();
    }
    freeaddrinfo(res);
    int32_t self_rank_net = htonl(static_cast<uint32_t>(self_rank));
    if (::send(fd, &self_rank_net, sizeof(self_rank_net), 0) != sizeof(self_rank_net)) {
      close(fd);
      throw Timeout();
    }
    uint8_t mode_byte = EncodeModeByte(mode);
    if (::send(fd, &mode_byte, sizeof(mode_byte), 0) != sizeof(mode_byte)) {
      close(fd);
      throw Timeout();
    }
    return fd;
  }

  const int accept_key = PeerAndModeKey(partner_id, mode);
  std::unique_lock<std::mutex> lock(mu_);
  bool got = cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&] {
    return accepted_fd_by_peer_and_mode_.count(accept_key) > 0;
  });
  if (!got) throw Timeout();
  int fd = accepted_fd_by_peer_and_mode_[accept_key];
  accepted_fd_by_peer_and_mode_.erase(accept_key);
  return fd;
}

void FMI::Comm::RedisDirectEstablisher::Finalize() {
  running_ = false;
  if (accept_thread_.joinable()) accept_thread_.join();
  if (listen_fd_ >= 0) {
    close(listen_fd_);
    listen_fd_ = -1;
  }
  std::lock_guard<std::mutex> lock(mu_);
  for (const auto &entry : accepted_fd_by_peer_and_mode_) {
    close(entry.second);
  }
  accepted_fd_by_peer_and_mode_.clear();
}

FMI::Comm::RedisDirectEstablisher::~RedisDirectEstablisher() { Finalize(); }
