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

#include <cmath>
#include <iostream>
#include <cstring>

#include "PeerToPeer.hpp"

void FMI::Comm::PeerToPeer::send(const std::shared_ptr<channel_data> buf, FMI::Utils::peer_num dest) {
    send_object(buf, dest);
}



void FMI::Comm::PeerToPeer::send(std::shared_ptr<channel_data> buf, FMI::Utils::peer_num dest,
                                 FMI::Utils::fmiContext *context,
                                 FMI::Utils::Mode mode,
                                 std::function<void(FMI::Utils::NbxStatus, const std::string &,
                                                    FMI::Utils::fmiContext *)> callback) {

    auto state = std::make_shared<IOState>();
    state->callbackResult = callback;
    state->context = context;
    state->setRequest(std::move(buf));
    state->processed = 0;
    state->operation = Utils::SEND;
    state->deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(getMaxTimeout());
    send_object(std::move(state), dest, mode);

}

void FMI::Comm::PeerToPeer::send(FMI::Utils::peer_num dest,
                                    Utils::Mode mode,
                                    std::shared_ptr<IOState> state) {
    send_object(state, dest, mode);
}

void FMI::Comm::PeerToPeer::recv(const std::shared_ptr<channel_data> buf, FMI::Utils::peer_num src) {
    recv_object(buf, src);
}

void FMI::Comm::PeerToPeer::recv(const std::shared_ptr<channel_data> buf, FMI::Utils::peer_num src,
                                 FMI::Utils::fmiContext * context,
                                 FMI::Utils::Mode mode,
                                 std::function<void(FMI::Utils::NbxStatus, const std::string&,
                                                    FMI::Utils::fmiContext *)> callback) {
    auto state = std::make_shared<IOState>();
    state->callbackResult = callback;
    state->context = context;
    state->setRequest(buf);
    state->processed = 0;
    state->operation = Utils::RECEIVE;
    state->deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(getMaxTimeout());
    recv_object(std::move(state), src, mode);
}

void FMI::Comm::PeerToPeer::recv(FMI::Utils::peer_num src,
                                     Utils::Mode mode,
                                     std::shared_ptr<IOState> state) {
    recv_object(state, src, mode);
}

void FMI::Comm::PeerToPeer::bcast(std::shared_ptr<channel_data> buf, FMI::Utils::peer_num root,
                                  FMI::Utils::Mode mode,
                                  std::function<void(FMI::Utils::NbxStatus, const std::string &,
                                                     FMI::Utils::fmiContext *)> callback) {
    int rounds = ceil(log2(num_peers));
    Utils::peer_num trans_peer_id = transform_peer_id(peer_id, root, true);
    for (int i = rounds - 1; i >= 0; i--) {
        Utils::peer_num rcpt = trans_peer_id + (Utils::peer_num) std::pow(2, i);
        if (trans_peer_id % (int) std::pow(2, i + 1) == 0 && rcpt < num_peers) {
            Utils::peer_num real_rcpt = transform_peer_id(rcpt, root, false);

            if (mode == Utils::BLOCKING) {
                send(buf, real_rcpt);
            } else {
                auto state = std::make_shared<IOState>();
                state->setRequest(buf);
                state->processed = 0;
                state->operation = Utils::SEND;
                state->callbackResult = callback;
                state->deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(getMaxTimeout());
                send(real_rcpt, mode, state);
            }


        } else if (trans_peer_id % (int) std::pow(2, i) == 0 && trans_peer_id % (int) std::pow(2, i + 1) != 0){
            Utils::peer_num real_src = transform_peer_id(trans_peer_id - (int) std::pow(2, i), root, false);
            if (mode == Utils::BLOCKING) {
                recv(buf, real_src);
            } else {
                auto state = std::make_shared<IOState>();
                state->setRequest(buf);
                state->processed = 0;
                state->operation = Utils::RECEIVE;
                state->callbackResult = callback;
                state->deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(getMaxTimeout());
                recv(real_src, mode, state);
            }

        }
    }
}

void FMI::Comm::PeerToPeer::barrier() {
    barrier(Utils::BLOCKING, nullptr);
}

void FMI::Comm::PeerToPeer::barrier(Utils::Mode mode,
                                    std::function<void(FMI::Utils::NbxStatus, const std::string&,
                                                       FMI::Utils::fmiContext *)> callback) {
    auto nop = [] (char* a, char* b) {};
    char send = 1;
    auto ctmp = std::make_shared<channel_data>(&send, sizeof(char));
    auto allReduceData = std::make_shared<channel_data>(&send,
                                                        sizeof(char));
    allreduce(allReduceData, ctmp, {nop, true, true}, mode, callback);
}

void FMI::Comm::PeerToPeer::reduce(const std::shared_ptr<channel_data> sendbuf,
                                   std::shared_ptr<channel_data> recvbuf,
                                   FMI::Utils::peer_num root, raw_function f) {
    bool left_to_right = !(f.commutative && f.associative);
    if (left_to_right) {
        reduce_ltr(sendbuf, recvbuf, root, f);
    } else {
        reduce_no_order(sendbuf, recvbuf, root, f);
    }
}

void FMI::Comm::PeerToPeer::reduce_ltr(const std::shared_ptr<channel_data> sendbuf,
                                       std::shared_ptr<channel_data> recvbuf,
                                       FMI::Utils::peer_num root, const raw_function& f) {
    if (peer_id == root) {
        std::size_t tmpbuf_len = sendbuf->len * num_peers;
        char* tmpbuf = new char[tmpbuf_len];
        auto tmpdata = std::make_shared<channel_data>(tmpbuf, tmpbuf_len);
        gather(sendbuf, tmpdata, root);
        std::memcpy(reinterpret_cast<void*>(recvbuf->buf.get()), tmpbuf, sendbuf->len);
        for (std::size_t i = sendbuf->len; i < tmpbuf_len; i += sendbuf->len) {

            f.f(recvbuf->buf.get(), tmpbuf + i);
        }
        delete[] tmpbuf;
    } else {
        auto tmpdata = std::make_shared<channel_data>();
        gather(sendbuf, tmpdata, root);
    }
}

void FMI::Comm::PeerToPeer::reduce_no_order(const std::shared_ptr<channel_data> sendbuf,
                                            const std::shared_ptr<channel_data> recvbuf,
                                            FMI::Utils::peer_num root, const raw_function& f) {
    int rounds = ceil(log2(num_peers));
    Utils::peer_num trans_peer_id = transform_peer_id(peer_id, root, true);
    std::shared_ptr<channel_data> recbufcpy = (peer_id == root)
            ? recvbuf
            : std::make_shared<channel_data>(sendbuf->len);
    for (int i = 0; i < rounds; i++) {
        Utils::peer_num src = trans_peer_id + (Utils::peer_num) std::pow(2, i);

        if (trans_peer_id % (int) std::pow(2, i + 1) == 0 && src < num_peers) {
            Utils::peer_num real_src = transform_peer_id(src, root, false);

            recv(recbufcpy, real_src);
            f.f(sendbuf->buf.get(), recbufcpy->buf.get());

        } else if (trans_peer_id % (int) std::pow(2, i) == 0 && trans_peer_id % (int) std::pow(2, i + 1) != 0){
            Utils::peer_num real_dst = transform_peer_id(trans_peer_id - (int) std::pow(2, i), root, false);
            send(sendbuf, real_dst);
        }
    }
    if (peer_id == root) {
        std::memcpy(recbufcpy->buf.get(), sendbuf->buf.get(), sendbuf->len);
    }
}

void FMI::Comm::PeerToPeer::allreduce(const std::shared_ptr<channel_data> sendbuf,
                                      std::shared_ptr<channel_data> recvbuf, raw_function f) {
    allreduce(sendbuf, recvbuf, f, Utils::BLOCKING, nullptr);
}

void FMI::Comm::PeerToPeer::allreduce(const std::shared_ptr<channel_data> sendbuf,
                                      std::shared_ptr<channel_data> recvbuf, raw_function f,
                                      Utils::Mode mode,
                                      std::function<void(FMI::Utils::NbxStatus, const std::string&,
                                                         FMI::Utils::fmiContext *)> callback) {
    bool left_to_right = !(f.commutative && f.associative);
    if (left_to_right) {
        reduce(sendbuf, recvbuf, 0, f);
        bcast(recvbuf, 0, mode, callback);
    } else {
        allreduce_no_order(sendbuf, recvbuf, f, mode, callback);
    }
}

void FMI::Comm::PeerToPeer::allreduce_no_order(const std::shared_ptr<channel_data> sendbuf,
                                               const std::shared_ptr<channel_data> recvbuf, const raw_function &f,
                                               Utils::Mode mode,
                                               std::function<void(FMI::Utils::NbxStatus, const std::string&,
                                                                  FMI::Utils::fmiContext *)> callback) {
    // Each round's reduction needs the peer's data before the next round can start, so — unlike
    // gatherv, which posts everything up front and drains once at the end — a nonblocking
    // exchange here posts this round's send+recv, then drains this channel's event loop before
    // returning, matching UCC's recursive-doubling allreduce (one step completes before the
    // next starts). Callers on the blocking path keep the original send-then-recv/recv-then-send
    // ordering (deadlock-free for two peers); nonblocking mode needs no such ordering since
    // neither call blocks the other.
    auto sendOne = [&](const std::shared_ptr<channel_data> &buf, Utils::peer_num peer) {
        if (mode == Utils::BLOCKING) {
            send(buf, peer);
        } else {
            send(buf, peer, nullptr, mode, callback);
            while (channel_event_progress(Utils::Operation::DEFAULT) == Utils::PROCESSING) {}
        }
    };
    auto recvOne = [&](const std::shared_ptr<channel_data> &buf, Utils::peer_num peer) {
        if (mode == Utils::BLOCKING) {
            recv(buf, peer);
        } else {
            recv(buf, peer, nullptr, mode, callback);
            while (channel_event_progress(Utils::Operation::DEFAULT) == Utils::PROCESSING) {}
        }
    };
    auto exchange = [&](Utils::peer_num peer) {
        if (mode == Utils::BLOCKING) {
            send(sendbuf, peer);
            recv(recvbuf, peer);
            return;
        }
        send(sendbuf, peer, nullptr, mode, callback);
        recv(recvbuf, peer, nullptr, mode, callback);
        while (channel_event_progress(Utils::Operation::DEFAULT) == Utils::PROCESSING) {}
    };

    // Non power of two N: First receive from processes with ID >= 2^ceil(log2(N)), send result after reduction
    int rounds = floor(log2(num_peers));
    int nearest_power_two = (int) std::pow(2, rounds);
    if (num_peers > nearest_power_two) {
        if (peer_id < nearest_power_two && peer_id + nearest_power_two < num_peers) {
            recvOne(recvbuf, peer_id + nearest_power_two);
            f.f(sendbuf->buf.get(), recvbuf->buf.get());
        } else if (peer_id >= nearest_power_two) {
            sendOne(sendbuf, peer_id - nearest_power_two);
        }
    }
    if (peer_id < nearest_power_two) {
        // Actual recursive doubling
        for (int i = 0; i < rounds; i++) {
            int peer = peer_id ^ (int) std::pow(2, i);
            exchange(peer);
            f.f(sendbuf->buf.get(), recvbuf->buf.get());
        }
    }
    if (num_peers > nearest_power_two) {
        if (peer_id < nearest_power_two && peer_id + nearest_power_two < num_peers) {
            sendOne(sendbuf, peer_id + nearest_power_two);
        } else if (peer_id >= nearest_power_two) {
            recvOne(sendbuf, peer_id - nearest_power_two);
        }
    }
    std::memcpy(recvbuf->buf.get(), sendbuf->buf.get(), sendbuf->len);
}

void FMI::Comm::PeerToPeer::scan(const std::shared_ptr<channel_data> sendbuf,
                                 std::shared_ptr<channel_data> recvbuf, raw_function f) {
    auto left_to_right = !(f.commutative && f.associative);
    if (left_to_right) {
        scan_ltr(sendbuf, recvbuf, f);
    } else {
        scan_no_order(sendbuf, recvbuf, f);
    }
}

void FMI::Comm::PeerToPeer::scan_ltr(const std::shared_ptr<channel_data> sendbuf,
                                     const std::shared_ptr<channel_data> recvbuf, const raw_function& f) {
    if (peer_id == 0) {
        send(sendbuf, 1);
        std::memcpy(recvbuf->buf.get(), sendbuf->buf.get(), sendbuf->len);
    } else {
        recv(recvbuf, peer_id - 1);
        f.f(recvbuf->buf.get(), sendbuf->buf.get());
        if (peer_id < num_peers - 1) {
            send(recvbuf, peer_id + 1);
        }
    }
}

void FMI::Comm::PeerToPeer::scan_no_order(const std::shared_ptr<channel_data> sendbuf,
                                          const std::shared_ptr<channel_data> recvbuf, const raw_function& f) {
    int rounds = floor(log2(num_peers));
    for (int i = 0; i < rounds; i ++) {
        if ((peer_id & ((int) std::pow(2, i + 1) - 1)) == (int) std::pow(2, i + 1) - 1) {
            Utils::peer_num src = peer_id - (int) std::pow(2, i);
            recv(recvbuf, src);
            f.f(sendbuf->buf.get(), recvbuf->buf.get());
        } else if ((peer_id & ((int) std::pow(2, i) - 1)) == (int) std::pow(2, i) - 1) {
            Utils::peer_num dst = peer_id + (int) std::pow(2, i);
            if (dst < num_peers) {
                send(sendbuf, dst);
                break;
            }
        }
    }
    for (int i = rounds; i > 0; i--) {
        if ((peer_id & ((int) std::pow(2, i) - 1)) == (int) std::pow(2, i) - 1) {
            Utils::peer_num dst = peer_id + (int) std::pow(2, i - 1);
            if (dst < num_peers) {
                send(sendbuf, dst);
            }
        } else if ((peer_id & ((int) std::pow(2, i - 1) - 1)) == (int) std::pow(2, i - 1) - 1) {
            int src = peer_id - (int) std::pow(2, i - 1);
            if (src > 0) {
                recv(recvbuf, src);
                f.f(sendbuf->buf.get(), recvbuf->buf.get());
            }
        }
    }
    std::memcpy(recvbuf->buf.get(), sendbuf->buf.get(), sendbuf->len);
}

void FMI::Comm::PeerToPeer::allgatherv(const std::shared_ptr<channel_data> sendbuf,
                                       std::shared_ptr<channel_data> recvbuf,
                                           FMI::Utils::peer_num root, const std::vector<int32_t> &recvcounts,
                                           const std::vector<int32_t> &displs,
                                       Utils::Mode mode,
                                       std::function<void(FMI::Utils::NbxStatus, const std::string&,
                                                          FMI::Utils::fmiContext *)> callback) {
    int rounds = ceil(log2(num_peers));
    Utils::peer_num trans_peer_id = transform_peer_id(peer_id, root, true);
    //channel_data recvbufcpy = {recvbuf.buf, recvbuf.len};

    // Calculate total size needed for the final buffer
    std::size_t total_buffer_size = 0;
    for (auto &count : recvcounts) total_buffer_size += count;

    // Allocate buffer for final gathered data
    //recvbuf.buf = new char[total_buffer_size];
    //recvbufcpy.buf = std::shared_ptr<char[]>(new char[total_buffer_size], std::default_delete<char[]>());
    //recvbufcpy.len = total_buffer_size;

    // Copy own data to correct position
    std::memcpy(recvbuf->buf.get() + displs[peer_id], sendbuf->buf.get(), sendbuf->len);

    // 🏗️ Gather Phase (Handle variable sizes using recvcounts and displs)
    for (int i = 0; i < rounds; i++) {
        Utils::peer_num src = trans_peer_id + (Utils::peer_num) std::pow(2, i);

        if (trans_peer_id % (int) std::pow(2, i + 1) == 0 && src < num_peers) {
            std::size_t offset = displs[src];
            std::size_t buf_len = 0;

            // 🔍 Calculate total length from responsible peers
            for (Utils::peer_num p = src; p < std::min(src + (Utils::peer_num) std::pow(2, i), num_peers); ++p) {
                buf_len += recvcounts[p];
            }

            Utils::peer_num real_src = transform_peer_id(src, root, false);
            auto request = channel_data::view(recvbuf, recvbuf->buf.get() + offset, buf_len);
            if (mode ==Utils::BLOCKING) {
                recv(request, real_src);
            } else {
                auto state = std::make_shared<IOState>();
                state->setRequest(request);
                state->processed = 0;
                state->operation = Utils::RECEIVE;
                state->callbackResult = callback;
                state->deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(getMaxTimeout());
                recv(real_src, mode, std::move(state));
            }
        } else if (trans_peer_id % (int) std::pow(2, i) == 0 && trans_peer_id % (int) std::pow(2, i + 1) != 0) {
            std::size_t offset = displs[trans_peer_id];
            std::size_t buf_len = 0;

            for (Utils::peer_num p = trans_peer_id; p < std::min(trans_peer_id + (Utils::peer_num) std::pow(2, i), num_peers); ++p) {
                buf_len += recvcounts[p];
            }

            Utils::peer_num real_dst = transform_peer_id(trans_peer_id - (int) std::pow(2, i),
                                                         root, false);
            auto request = channel_data::view(recvbuf, recvbuf->buf.get() + offset, buf_len);
            if (mode == Utils::BLOCKING) {
                send(request, real_dst);
            } else {
                auto state = std::make_shared<IOState>();
                state->setRequest(request);
                state->processed = 0;
                state->operation = Utils::SEND;
                state->callbackResult = callback;
                state->deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(getMaxTimeout());
                send(real_dst, mode, std::move(state));
            }

        }
    }

    // 🌐 Broadcast Phase (Share gathered variable-sized data with all)
    for (int i = 0; i < rounds; i++) {
        Utils::peer_num partner = trans_peer_id ^ (1 << i);
        if (partner < num_peers) {
            if ((trans_peer_id & (1 << i)) == 0) {
                auto request = channel_data::view(recvbuf, recvbuf->buf.get(), total_buffer_size);
                auto transformedPId = transform_peer_id(partner,
                                                        root, false);
                if (mode == Utils::BLOCKING) {
                    send(request, transformedPId);
                } else {
                    auto state = std::make_shared<IOState>();
                    state->request = request;
                    state->processed = 0;
                    state->operation = Utils::SEND;
                    state->callbackResult = callback;
                    state->deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(getMaxTimeout());
                    send(transformedPId, mode, std::move(state));
                }

            } else {
                auto request = channel_data::view(recvbuf, recvbuf->buf.get(), total_buffer_size);
                auto transformedPId = transform_peer_id(partner,
                                                        root, false);

                if (mode == Utils::BLOCKING) {
                    recv(request, transformedPId);
                } else {
                    auto state = std::make_shared<IOState>();
                    state->request = request;
                    state->processed = 0;
                    state->operation = Utils::RECEIVE;
                    state->callbackResult = callback;
                    state->deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(getMaxTimeout());
                    recv(transformedPId, mode, std::move(state));
                }

            }
        }
    }
}

void
FMI::Comm::PeerToPeer::allgather(const std::shared_ptr<channel_data> sendbuf,
                                 std::shared_ptr<channel_data> recvbuf, FMI::Utils::peer_num root,
                                 FMI::Utils::Mode mode,
                                 std::function<void(FMI::Utils::NbxStatus, const std::string &,
                                                    FMI::Utils::fmiContext *)> callback) {
    int rounds = ceil(log2(num_peers));
    Utils::peer_num trans_peer_id = transform_peer_id(peer_id, root, true);
    std::size_t single_buffer_size = sendbuf->len;
    std::size_t total_buffer_size = num_peers * single_buffer_size;

    //channel_data recvbufcpy = {recvbuf.buf, recvbuf.len};

    // Allocate buffer for the final gathered data
    /*if (peer_id == root) {
        //recvbuf.buf = new char[total_buffer_size];
        recvbufcpy.buf = std::shared_ptr<char[]>(new char[total_buffer_size], std::default_delete<char[]>());
        recvbufcpy.len = total_buffer_size;
    } else {
        //recvbuf.buf = new char[total_buffer_size];
        recvbufcpy.buf = std::shared_ptr<char[]>(new char[total_buffer_size], std::default_delete<char[]>());
        recvbufcpy.len = total_buffer_size;
    }*/
    std::memcpy(recvbuf->buf.get() + single_buffer_size * peer_id, sendbuf->buf.get(),
                single_buffer_size);

    for (int i = 0; i < rounds; i++) {
        Utils::peer_num src = trans_peer_id + (Utils::peer_num) std::pow(2, i);

        if (trans_peer_id % (int) std::pow(2, i + 1) == 0 && src < num_peers) {
            unsigned int responsible_peers = std::min((Utils::peer_num) std::pow(2, i), num_peers - src);
            std::size_t buf_len = responsible_peers * single_buffer_size;
            Utils::peer_num real_src = transform_peer_id(src, root, false);
            auto request = channel_data::view(recvbuf, recvbuf->buf.get() + real_src * single_buffer_size,
                                              buf_len);
            if (mode == Utils::BLOCKING) {
                recv(request, real_src);
            } else {
                auto state = std::make_shared<IOState>();
                state->setRequest(request);
                state->processed = 0;
                state->operation = Utils::RECEIVE;
                state->callbackResult = callback;
                state->deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(getMaxTimeout());
                recv(real_src, mode, std::move(state));
            }

        } else if (trans_peer_id % (int) std::pow(2, i) == 0 && trans_peer_id % (int) std::pow(2, i + 1) != 0) {
            unsigned int responsible_peers = std::min((Utils::peer_num) std::pow(2, i), num_peers - trans_peer_id);
            std::size_t buf_len = responsible_peers * single_buffer_size;
            Utils::peer_num real_dst = transform_peer_id(trans_peer_id - (int) std::pow(2, i), root, false);
            auto request = channel_data::view(recvbuf, recvbuf->buf.get() + trans_peer_id * single_buffer_size,
                                              buf_len);
            if (mode == Utils::BLOCKING) {
                send(request, real_dst);
            } else {
                auto state = std::make_shared<IOState>();
                state->setRequest(request);
                state->processed = 0;
                state->operation = Utils::SEND;
                state->callbackResult = callback;
                state->deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(getMaxTimeout());
                send(real_dst, mode, std::move(state));
            }

        }
    }

    // 🌐 Broadcast Phase (so all processes get the gathered data)
    for (int i = 0; i < rounds; i++) {
        Utils::peer_num partner = trans_peer_id ^ (1 << i);
        if (partner < num_peers) {
            if ((trans_peer_id & (1 << i)) == 0) {
                auto request = channel_data::view(recvbuf, recvbuf->buf.get(), total_buffer_size);
                auto transformedPId = transform_peer_id(partner,
                                                        root, false);
                if (mode == Utils::BLOCKING) {
                    send(request, transformedPId);
                } else {
                    auto state = std::make_shared<IOState>();
                    state->request = request;
                    state->processed = 0;
                    state->operation = Utils::SEND;
                    state->callbackResult = callback;
                    state->deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(getMaxTimeout());
                    send(transformedPId, mode, std::move(state));
                }

            } else {
                auto request = channel_data::view(recvbuf, recvbuf->buf.get(), total_buffer_size);
                auto transformedPId = transform_peer_id(partner,
                                                        root, false);

                if (mode == Utils::BLOCKING) {
                    recv(request, transformedPId);
                } else {
                    auto state = std::make_shared<IOState>();
                    state->request = request;
                    state->processed = 0;
                    state->operation = Utils::RECEIVE;
                    state->callbackResult = callback;
                    state->deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(getMaxTimeout());
                    recv(transformedPId, mode, std::move(state));
                }

            }
        }
    }
}

void FMI::Comm::PeerToPeer::gather(const std::shared_ptr<channel_data> sendbuf,
                                   std::shared_ptr<channel_data> recvbuf,
                                   FMI::Utils::peer_num root) {
    int rounds = ceil(log2(num_peers));
    Utils::peer_num trans_peer_id = transform_peer_id(peer_id, root, true);
    std::size_t single_buffer_size = sendbuf->len;
    // Find needed buffer size and allocate it
    if (peer_id != root) {
        unsigned int peers_in_buffer = 1;
        for (int i = rounds - 1; i >= 0; i--) {
            Utils::peer_num src = trans_peer_id + (Utils::peer_num) std::pow(2, i);
            if (trans_peer_id % (int) std::pow(2, i + 1) == 0 && src < num_peers) {
                peers_in_buffer += std::min((Utils::peer_num) std::pow(2, i), num_peers - src);
            }
        }
        // Non-root peers relay data for other peers in the binomial tree, so the
        // caller-supplied recvbuf (sized/valid for root only) is replaced here.
        recvbuf = std::make_shared<channel_data>(peers_in_buffer * single_buffer_size);
        std::memcpy(recvbuf->buf.get(), sendbuf->buf.get(), single_buffer_size);
    } else {
        std::memcpy(recvbuf->buf.get() + single_buffer_size * root, sendbuf->buf.get(),
                    single_buffer_size);
    }

    for (int i = 0; i < rounds; i++) {
        Utils::peer_num src = trans_peer_id + (Utils::peer_num) std::pow(2, i);

        if (trans_peer_id % (int) std::pow(2, i + 1) == 0 && src < num_peers) {
            unsigned int responsible_peers = std::min((Utils::peer_num) std::pow(2, i), num_peers - src);
            std::size_t buf_len = responsible_peers * single_buffer_size;
            Utils::peer_num real_src = transform_peer_id(src, root, false);

            if (peer_id == root) {
                if (real_src * single_buffer_size + buf_len > recvbuf->len) {
                    auto tmp = std::shared_ptr<char[]>(new char[buf_len], std::default_delete<char[]>());
                    auto ctmp = std::make_shared<channel_data>();
                    ctmp->buf = tmp;
                    ctmp->len = buf_len;
                    recv(ctmp, real_src);
                    unsigned int length_end = recvbuf->len - real_src * single_buffer_size;
                    std::memcpy(recvbuf->buf.get() + real_src * single_buffer_size, tmp.get(), length_end);
                    std::memcpy(recvbuf->buf.get(), tmp.get() + length_end, buf_len - length_end);
                } else {
                    auto ctmp = channel_data::view(recvbuf, recvbuf->buf.get() + real_src * single_buffer_size,
                                                   buf_len);
                    recv(ctmp, real_src);
                }
            } else {
                auto ctmp = channel_data::view(recvbuf, recvbuf->buf.get() + (src - trans_peer_id) * single_buffer_size,
                                               buf_len);
                recv(ctmp, real_src);
            }
        } else if (trans_peer_id % (int) std::pow(2, i) == 0 && trans_peer_id % (int) std::pow(2, i + 1) != 0){
            unsigned int responsible_peers = std::min((Utils::peer_num) std::pow(2, i), num_peers - trans_peer_id);
            std::size_t buf_len = responsible_peers * single_buffer_size;
            Utils::peer_num real_dst = transform_peer_id(trans_peer_id - (int) std::pow(2, i), root, false);
            auto ctmp = channel_data::view(recvbuf, recvbuf->buf.get(), buf_len);
            send(ctmp, real_dst);
        }
    }
    //if (peer_id != root) {
    //    delete[] recvbuf.buf;
    //}
}



void FMI::Comm::PeerToPeer::scatter(const std::shared_ptr<channel_data> sendbuf,
                                    std::shared_ptr<channel_data> recvbuf, FMI::Utils::peer_num root) {
    int rounds = ceil(log2(num_peers));
    Utils::peer_num trans_peer_id = transform_peer_id(peer_id, root, true);
    std::size_t single_buffer_size = recvbuf->len;
    auto sendbufcpy = std::make_shared<channel_data>(sendbuf->buf.get(), sendbuf->len);
    for (int i = rounds - 1; i >= 0; i--) {
        Utils::peer_num rcpt = trans_peer_id + (Utils::peer_num) std::pow(2, i);


        if (trans_peer_id % (int) std::pow(2, i + 1) == 0 && rcpt < num_peers) {
            unsigned int responsible_peers = std::min((Utils::peer_num) std::pow(2, i), num_peers - rcpt);
            std::size_t buf_len = responsible_peers * single_buffer_size;
            Utils::peer_num real_rcpt = transform_peer_id(rcpt, root, false);

            if (peer_id == root) {
                if (real_rcpt * single_buffer_size + buf_len > sendbuf->len) {
                    // Wrapping around, need to allocate a temporary buffer
                    char* tmp = new char[buf_len];
                    unsigned int length_end = sendbufcpy->len - real_rcpt * single_buffer_size; // How many bytes we need to send at end of buffer
                    std::memcpy(tmp, sendbufcpy->buf.get() + real_rcpt * single_buffer_size, length_end);
                    // Copy rest from beginning
                    std::memcpy(tmp + length_end, sendbuf->buf.get(), buf_len - length_end);
                    auto ctmp = std::make_shared<channel_data>(tmp, buf_len);
                    send(ctmp, real_rcpt);
                    delete[] tmp;
                } else {

                    auto ctmp = std::make_shared<channel_data>(sendbufcpy->buf.get() + real_rcpt * single_buffer_size,
                                         buf_len);
                    send(ctmp, real_rcpt);
                }
            } else {

                auto ctmp = std::make_shared<channel_data>(sendbufcpy->buf.get() + (rcpt - trans_peer_id) * single_buffer_size,
                    buf_len);
                send(ctmp, real_rcpt);
            }
        } else if (trans_peer_id % (int) std::pow(2, i) == 0 && trans_peer_id % (int) std::pow(2, i + 1) != 0){
            unsigned int responsible_peers = std::min((Utils::peer_num) std::pow(2, i), num_peers - trans_peer_id);
            std::size_t buf_len = responsible_peers * single_buffer_size;
            Utils::peer_num real_src = transform_peer_id(trans_peer_id - (int) std::pow(2, i), root, false);
            //sendbuf.buf = new char[buf_len];
            sendbufcpy->buf = std::shared_ptr<char[]>(new char[buf_len], std::default_delete<char[]>());
            sendbufcpy->len = buf_len;
            recv(sendbufcpy, real_src);
        }
    }
    if (peer_id == root) {
        std::memcpy(recvbuf->buf.get(), sendbufcpy->buf.get() + peer_id * single_buffer_size,
                    single_buffer_size);
    } else {
        std::memcpy(recvbuf->buf.get(), sendbufcpy->buf.get(), single_buffer_size);
        //delete[] sendbuf.buf;
    }
}

void FMI::Comm::PeerToPeer::scatterv(const std::shared_ptr<channel_data> sendbuf,
                                     std::shared_ptr<channel_data> recvbuf,
                                     FMI::Utils::peer_num root,
                                     const std::vector<int32_t> &sendcounts,
                                     const std::vector<int32_t> &displs,
                                     Utils::Mode mode,
                                     std::function<void(FMI::Utils::NbxStatus, const std::string&,
                                                        FMI::Utils::fmiContext *)> callback) {
    // Variable-length binomial scatter, mirroring the even PeerToPeer::scatter above but with
    // per-peer byte counts/displacements (byte-granular, like gatherv). Blocking send/recv; the
    // even scatter is blocking-only, so `mode` is accepted for interface parity and transfers
    // are synchronous. The root's transformed slices may wrap the real-id ordered sendbuf when
    // root != 0 — handled with a temporary buffer, exactly as PeerToPeer::gatherv does.
    (void) mode;
    int rounds = ceil(log2(num_peers));
    Utils::peer_num trans_peer_id = transform_peer_id(peer_id, root, true);

    // Transformed prefix sums, built once (O(P)): tpref[t] = sum of sendcounts over transformed
    // positions [0, t) — i.e. real ids (s + root) % num_peers. The byte length of any transformed
    // range [a, b) is then the O(1) difference tpref[b] - tpref[a], so no per-round rescans are
    // needed and the scatter stays O(log P) messages (the O(P) prefix pass is inherent to a
    // vector scatter — MPI_Scatterv / UCC SCATTERV do the same prefix bookkeeping internally).
    std::vector<std::size_t> tpref(num_peers + 1, 0);
    for (Utils::peer_num t = 0; t < num_peers; ++t) {
        tpref[t + 1] = tpref[t] + (std::size_t) sendcounts[transform_peer_id(t, root, false)];
    }

    std::size_t total = tpref[num_peers];
    std::size_t my_count = (std::size_t) sendcounts[peer_id];

    // sendbuf is only meaningful at root (matching MPI_Scatterv/UCC's contract —
    // TableScatterImpl::Execute passes nullptr for non-root's send_data), so only
    // root eagerly copies it here; every other peer's sendbufcpy is populated by
    // the round loop's recv() below before it's ever read.
    auto sendbufcpy = (peer_id == root)
                           ? std::make_shared<channel_data>(sendbuf->buf.get(), sendbuf->len)
                           : std::make_shared<channel_data>();
    for (int i = rounds - 1; i >= 0; i--) {
        Utils::peer_num rcpt = trans_peer_id + (Utils::peer_num) std::pow(2, i);

        if (trans_peer_id % (int) std::pow(2, i + 1) == 0 && rcpt < num_peers) {
            unsigned int responsible_peers = std::min((Utils::peer_num) std::pow(2, i),
                                                      num_peers - rcpt);
            std::size_t buf_len = tpref[rcpt + responsible_peers] - tpref[rcpt];
            Utils::peer_num real_rcpt = transform_peer_id(rcpt, root, false);

            if (peer_id == root) {
                if ((std::size_t) displs[real_rcpt] + buf_len > total) {
                    // Slice wraps the end of the real-id ordered buffer: assemble it contiguously
                    // in a temporary (kept alive until the blocking send returns).
                    auto tmp = std::shared_ptr<char[]>(new char[buf_len], std::default_delete<char[]>());
                    std::size_t length_end = total - (std::size_t) displs[real_rcpt];
                    std::memcpy(tmp.get(), sendbufcpy->buf.get() + displs[real_rcpt], length_end);
                    std::memcpy(tmp.get() + length_end, sendbuf->buf.get(), buf_len - length_end);
                    auto ctmp = std::make_shared<channel_data>();
                    ctmp->buf = tmp;
                    ctmp->len = buf_len;
                    send(ctmp, real_rcpt);
                } else {
                    auto ctmp = channel_data::view(sendbufcpy, sendbufcpy->buf.get() + displs[real_rcpt],
                                                   buf_len);
                    send(ctmp, real_rcpt);
                }
            } else {
                std::size_t local_off = tpref[rcpt] - tpref[trans_peer_id];
                auto ctmp = channel_data::view(sendbufcpy, sendbufcpy->buf.get() + local_off, buf_len);
                send(ctmp, real_rcpt);
            }
        } else if (trans_peer_id % (int) std::pow(2, i) == 0 &&
                   trans_peer_id % (int) std::pow(2, i + 1) != 0) {
            unsigned int responsible_peers = std::min((Utils::peer_num) std::pow(2, i),
                                                      num_peers - trans_peer_id);
            std::size_t buf_len = tpref[trans_peer_id + responsible_peers] - tpref[trans_peer_id];
            Utils::peer_num real_src = transform_peer_id(trans_peer_id - (int) std::pow(2, i),
                                                         root, false);
            sendbufcpy->buf = std::shared_ptr<char[]>(new char[buf_len], std::default_delete<char[]>());
            sendbufcpy->len = buf_len;
            recv(sendbufcpy, real_src);
        }
    }

    // Extract this rank's own shard: root reads it from its real-id offset in the original
    // buffer; every other peer's own block is first in its transformed-contiguous buffer.
    if (peer_id == root) {
        std::memcpy(recvbuf->buf.get(), sendbufcpy->buf.get() + displs[peer_id], my_count);
    } else {
        std::memcpy(recvbuf->buf.get(), sendbufcpy->buf.get(), my_count);
    }

    if (callback) {
        callback(FMI::Utils::SUCCESS, "", nullptr);
    }
}

FMI::Utils::peer_num FMI::Comm::PeerToPeer::transform_peer_id(FMI::Utils::peer_num id,
                                                              FMI::Utils::peer_num root,
                                                              bool forward) {
    if (forward) {
        return (id + num_peers - root) % num_peers; // Transform s.t. root has id 0
    } else {
        return (id + root) % num_peers;
    }
}

FMI::Utils::EventProcessStatus FMI::Comm::PeerToPeer::channel_event_progress(Utils::Operation op) {
    return Utils::NOOP;
}

void FMI::Comm::PeerToPeer::gatherv(const std::shared_ptr<channel_data> sendbuf,
                                    std::shared_ptr<channel_data> recvbuf, FMI::Utils::peer_num root,
                                    const std::vector<int32_t> &recvcounts,
                                    const std::vector<int32_t> &displs,
                                    Utils::Mode mode,
                                    std::function<void(FMI::Utils::NbxStatus, const std::string&,
                                                       FMI::Utils::fmiContext *)> callback) {
    // Linear (non-tree) algorithm, matching UCC's tl/ucp gatherv_linear: each
    // non-root peer sends its own contribution directly to root, so unlike
    // gather()/scatter() no peer other than root ever needs to know another
    // peer's size — recvcounts/displs (populated by the caller only at root,
    // same as MPI_Gatherv/UCC's contract) are read only in the root branch.
    if (peer_id == root) {
        std::memcpy(recvbuf->buf.get() + displs[peer_id], sendbuf->buf.get(), sendbuf->len);
        for (Utils::peer_num p = 0; p < num_peers; p++) {
            if (p == root) {
                continue;
            }
            auto request = channel_data::view(recvbuf, recvbuf->buf.get() + displs[p],
                                              (std::size_t) recvcounts[p]);
            if (mode == Utils::BLOCKING) {
                recv(request, p);
            } else {
                auto state = std::make_shared<IOState>();
                state->setRequest(request);
                state->processed = 0;
                state->operation = Utils::RECEIVE;
                state->callbackResult = callback;
                state->deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(getMaxTimeout());
                recv(p, mode, std::move(state));
            }
        }
    } else {
        if (mode == Utils::BLOCKING) {
            send(sendbuf, root);
        } else {
            send(sendbuf, root, nullptr, mode, callback);
        }
    }
}


