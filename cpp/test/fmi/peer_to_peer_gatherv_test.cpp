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

#include "cylon/thridparty/fmi/comm/PeerToPeer.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <set>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace {

using FMI::Comm::GatherTopology;
using FMI::Utils::peer_num;

class Mailboxes {
 public:
  void Put(peer_num src, peer_num dst, std::vector<char> message) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      queues_[{src, dst}].push_back(std::move(message));
    }
    ready_.notify_all();
  }

  std::vector<char> Take(peer_num src, peer_num dst) {
    std::unique_lock<std::mutex> lock(mutex_);
    auto &queue = queues_[{src, dst}];
    ready_.wait(lock, [&queue] { return !queue.empty(); });
    auto message = std::move(queue.front());
    queue.pop_front();
    return message;
  }

 private:
  std::mutex mutex_;
  std::condition_variable ready_;
  std::map<std::pair<peer_num, peer_num>, std::deque<std::vector<char>>> queues_;
};

class LoopbackPeer : public FMI::Comm::PeerToPeer {
 public:
  LoopbackPeer(std::shared_ptr<Mailboxes> mailboxes, peer_num rank, peer_num world_size)
      : mailboxes_(std::move(mailboxes)) {
    set_peer_id(rank);
    set_num_peers(world_size);
  }

  void send_object(const std::shared_ptr<channel_data> buf, peer_num dst) override {
    contacted.insert(dst);
    mailboxes_->Put(peer_id, dst, std::vector<char>(buf->buf.get(), buf->buf.get() + buf->len));
  }

  void recv_object(const std::shared_ptr<channel_data> buf, peer_num src) override {
    contacted.insert(src);
    auto message = mailboxes_->Take(src, peer_id);
    if (message.size() != buf->len) {
      throw std::runtime_error("receive length " + std::to_string(buf->len) + " does not match sent length " +
                               std::to_string(message.size()));
    }
    std::memcpy(buf->buf.get(), message.data(), message.size());
  }

  void send_object(std::shared_ptr<FMI::Comm::IOState>, peer_num, FMI::Utils::Mode) override {
    throw std::logic_error("nonblocking send is not used by the gatherv under test");
  }

  void recv_object(std::shared_ptr<FMI::Comm::IOState>, peer_num, FMI::Utils::Mode) override {
    throw std::logic_error("nonblocking receive is not used by the gatherv under test");
  }

  std::set<peer_num> contacted;

 private:
  std::shared_ptr<Mailboxes> mailboxes_;
};

std::vector<char> Contribution(peer_num rank) {
  std::size_t length = rank % 4 == 3 ? 0 : static_cast<std::size_t>(rank % 7 + 1) * 3;
  std::vector<char> bytes(length);
  for (std::size_t i = 0; i < length; ++i) {
    bytes[i] = static_cast<char>((rank * 31 + i) % 251);
  }
  return bytes;
}

std::set<peer_num> BinomialTreePeers(peer_num world_size, peer_num rank, peer_num root) {
  std::set<peer_num> peers;
  peer_num shifted = (rank - root + world_size) % world_size;
  int rounds = world_size > 1 ? static_cast<int>(std::ceil(std::log2(world_size))) : 0;
  for (int i = 0; i < rounds; ++i) {
    peer_num step = 1 << i;
    if (shifted % (2 * step) == 0 && shifted + step < world_size) {
      peers.insert((shifted + step + root) % world_size);
    } else if (shifted % step == 0 && shifted % (2 * step) != 0) {
      peers.insert((shifted - step + root) % world_size);
    }
  }
  return peers;
}

struct GatherOutcome {
  std::vector<char> gathered;
  std::vector<std::set<peer_num>> contacted;
  std::vector<int> callbacks;
};

GatherOutcome RunGatherv(peer_num world_size, peer_num root, GatherTopology topology) {
  auto mailboxes = std::make_shared<Mailboxes>();
  std::vector<int32_t> counts(world_size), displs(world_size);
  for (peer_num r = 0; r < world_size; ++r) {
    counts[r] = static_cast<int32_t>(Contribution(r).size());
  }
  std::exclusive_scan(counts.begin(), counts.end(), displs.begin(), 0);
  std::size_t total = static_cast<std::size_t>(std::accumulate(counts.begin(), counts.end(), 0));

  GatherOutcome outcome;
  outcome.gathered.assign(total, 0);
  outcome.contacted.resize(world_size);
  outcome.callbacks.assign(world_size, 0);
  std::vector<std::exception_ptr> failures(world_size);
  std::vector<std::thread> ranks;
  for (peer_num r = 0; r < world_size; ++r) {
    ranks.emplace_back([&, r] {
      try {
        LoopbackPeer peer(mailboxes, r, world_size);
        peer.set_gather_topology(topology);
        auto mine = Contribution(r);
        auto sendbuf = std::make_shared<channel_data>(mine.size());
        if (!mine.empty()) std::memcpy(sendbuf->buf.get(), mine.data(), mine.size());
        auto recvbuf = r == root ? std::make_shared<channel_data>(total) : std::make_shared<channel_data>();
        peer.gatherv(sendbuf, recvbuf, root, r == root ? counts : std::vector<int32_t>{},
                     r == root ? displs : std::vector<int32_t>{}, FMI::Utils::BLOCKING,
                     [&outcome, r](FMI::Utils::NbxStatus status, const std::string &, FMI::Utils::fmiContext *) {
                       if (status == FMI::Utils::SUCCESS) outcome.callbacks[r]++;
                     });
        if (r == root && total > 0) {
          std::memcpy(outcome.gathered.data(), recvbuf->buf.get(), total);
        }
        outcome.contacted[r] = peer.contacted;
      } catch (...) {
        failures[r] = std::current_exception();
      }
    });
  }
  for (auto &t : ranks) t.join();
  for (auto &failure : failures) {
    if (failure) std::rethrow_exception(failure);
  }
  return outcome;
}

std::vector<char> Expected(peer_num world_size) {
  std::vector<char> expected;
  for (peer_num r = 0; r < world_size; ++r) {
    auto bytes = Contribution(r);
    expected.insert(expected.end(), bytes.begin(), bytes.end());
  }
  return expected;
}

class BinomialGathervTest : public ::testing::TestWithParam<std::pair<peer_num, int>> {};

TEST_P(BinomialGathervTest, RootReceivesEveryRankInRankOrderOverTreeEdgesOnly) {
  peer_num world_size = GetParam().first;
  peer_num root = static_cast<peer_num>(GetParam().second % world_size);

  auto outcome = RunGatherv(world_size, root, GatherTopology::BINOMIAL);

  EXPECT_EQ(outcome.gathered, Expected(world_size));
  int rounds = world_size > 1 ? static_cast<int>(std::ceil(std::log2(world_size))) : 0;
  EXPECT_LE(static_cast<int>(outcome.contacted[root].size()), rounds);
  for (peer_num r = 0; r < world_size; ++r) {
    EXPECT_EQ(outcome.contacted[r], BinomialTreePeers(world_size, r, root)) << "rank " << r;
    EXPECT_EQ(outcome.callbacks[r], 1) << "rank " << r;
  }
}

INSTANTIATE_TEST_SUITE_P(
    WorldSizesAndRoots, BinomialGathervTest,
    ::testing::Values(std::make_pair(1, 0), std::make_pair(2, 0), std::make_pair(2, 1),
                      std::make_pair(3, 0), std::make_pair(3, 2), std::make_pair(5, 0),
                      std::make_pair(5, 3), std::make_pair(6, 5), std::make_pair(7, 4),
                      std::make_pair(8, 0), std::make_pair(13, 6), std::make_pair(517, 0),
                      std::make_pair(517, 300)));

TEST(LinearGathervTest, DefaultTopologyStillSendsEveryRankStraightToRoot) {
  peer_num world_size = 13;
  peer_num root = 4;

  auto outcome = RunGatherv(world_size, root, GatherTopology::LINEAR);

  EXPECT_EQ(outcome.gathered, Expected(world_size));
  EXPECT_EQ(outcome.contacted[root].size(), static_cast<std::size_t>(world_size - 1));
  for (peer_num r = 0; r < world_size; ++r) {
    if (r != root) EXPECT_EQ(outcome.contacted[r], std::set<peer_num>{root});
  }
}

TEST(GatherTopologyTest, ParsesTheConfiguredNameAndDefaultsToLinearWhenUnset) {
  EXPECT_EQ(FMI::Comm::parse_gather_topology(nullptr), GatherTopology::LINEAR);
  EXPECT_EQ(FMI::Comm::parse_gather_topology(""), GatherTopology::LINEAR);
  EXPECT_EQ(FMI::Comm::parse_gather_topology("linear"), GatherTopology::LINEAR);
  EXPECT_EQ(FMI::Comm::parse_gather_topology("binomial"), GatherTopology::BINOMIAL);
  EXPECT_THROW(FMI::Comm::parse_gather_topology("star"), std::invalid_argument);
}

}