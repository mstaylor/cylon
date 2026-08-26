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

#ifndef CYLON_REDIS_DIRECT_PAIR_HPP
#define CYLON_REDIS_DIRECT_PAIR_HPP

#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <unordered_map>
#include "../utils/Common.hpp"

namespace FMI::Comm {

class RedisDirectEstablisher {
 public:
  RedisDirectEstablisher() = default;
  ~RedisDirectEstablisher();

  void Init(std::string redis_host, int redis_port, std::string comm_name,
            FMI::Utils::peer_num self_rank, FMI::Utils::peer_num num_peers,
            int listen_port, std::string host_override);

  int Connect(FMI::Utils::peer_num self_rank, FMI::Utils::peer_num partner_id,
              int timeout_ms, FMI::Utils::Mode mode);

  void Finalize();

 private:
  std::string redis_host_, comm_name_, host_override_;
  int redis_port_ = -1;
  int listen_port_ = -1;
  FMI::Utils::peer_num self_rank_ = -1;
  int listen_fd_ = -1;
  bool initialized_ = false;
  std::thread accept_thread_;
  std::atomic<bool> running_{false};
  std::mutex mu_;
  std::condition_variable cv_;
  std::unordered_map<int, int> accepted_fd_by_peer_and_mode_;

  std::string ResolveOwnAddress() const;
  std::string LookupPeerAddress(FMI::Utils::peer_num partner_id, int timeout_ms) const;
  void PublishOwnAddress(const std::string &own_addr) const;
  void AcceptLoop();
};

}

#endif