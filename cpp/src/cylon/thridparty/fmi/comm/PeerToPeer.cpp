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

void FMI::Comm::PeerToPeer::send(channel_data buf, FMI::Utils::peer_num dest) {
    send_object(buf, dest);
}

void FMI::Comm::PeerToPeer::send_nbx(channel_data buf, FMI::Utils::peer_num dest,
                                     std::function<void(FMI::Utils::NbxStatus, const std::string&)> callback) {
    send_object_nbx(buf, dest, callback);
}

void FMI::Comm::PeerToPeer::recv(channel_data buf, FMI::Utils::peer_num src) {
    recv_object(buf, src);
}

void FMI::Comm::PeerToPeer::recv_nbx(channel_data buf, FMI::Utils::peer_num src,
                                     std::function<void(FMI::Utils::NbxStatus, const std::string&)> callback) {
    recv_object_nbx(buf, src, callback);
}

void FMI::Comm::PeerToPeer::bcast(channel_data buf, FMI::Utils::peer_num root) {
    int rounds = ceil(log2(num_peers));
    Utils::peer_num trans_peer_id = transform_peer_id(peer_id, root, true);
    for (int i = rounds - 1; i >= 0; i--) {
        Utils::peer_num rcpt = trans_peer_id + (Utils::peer_num) std::pow(2, i);
        if (trans_peer_id % (int) std::pow(2, i + 1) == 0 && rcpt < num_peers) {
            Utils::peer_num real_rcpt = transform_peer_id(rcpt, root, false);
            send(buf, real_rcpt);
        } else if (trans_peer_id % (int) std::pow(2, i) == 0 && trans_peer_id % (int) std::pow(2, i + 1) != 0){
            Utils::peer_num real_src = transform_peer_id(trans_peer_id - (int) std::pow(2, i), root, false);
            recv(buf, real_src);
        }
    }
}

void FMI::Comm::PeerToPeer::barrier() {
    auto nop = [] (char* a, char* b) {};
    char send = 1;
    allreduce({&send, sizeof(char)}, {&send, sizeof(char)}, {nop, true, true});
}

void FMI::Comm::PeerToPeer::reduce(channel_data sendbuf, channel_data recvbuf, FMI::Utils::peer_num root, raw_function f) {
    bool left_to_right = !(f.commutative && f.associative);
    if (left_to_right) {
        reduce_ltr(sendbuf, recvbuf, root, f);
    } else {
        reduce_no_order(sendbuf, recvbuf, root, f);
    }
}

void FMI::Comm::PeerToPeer::reduce_ltr(channel_data sendbuf, channel_data recvbuf, FMI::Utils::peer_num root, const raw_function& f) {
    if (peer_id == root) {
        std::size_t tmpbuf_len = sendbuf.len * num_peers;
        char* tmpbuf = new char[tmpbuf_len];
        gather(sendbuf, {tmpbuf, tmpbuf_len}, root);
        std::memcpy(reinterpret_cast<void*>(recvbuf.buf), tmpbuf, sendbuf.len);
        for (std::size_t i = sendbuf.len; i < tmpbuf_len; i += sendbuf.len) {
            f.f(recvbuf.buf, tmpbuf + i);
        }
        delete[] tmpbuf;
    } else {
        gather(sendbuf, {}, root);
    }
}

void FMI::Comm::PeerToPeer::reduce_no_order(channel_data sendbuf, channel_data recvbuf, FMI::Utils::peer_num root, const raw_function& f) {
    int rounds = ceil(log2(num_peers));
    Utils::peer_num trans_peer_id = transform_peer_id(peer_id, root, true);
    if (peer_id != root) {
        recvbuf.buf = new char[sendbuf.len];
        recvbuf.len = sendbuf.len;
    }
    for (int i = 0; i < rounds; i++) {
        Utils::peer_num src = trans_peer_id + (Utils::peer_num) std::pow(2, i);

        if (trans_peer_id % (int) std::pow(2, i + 1) == 0 && src < num_peers) {
            Utils::peer_num real_src = transform_peer_id(src, root, false);
            recv({recvbuf.buf, recvbuf.len}, real_src);
            f.f(sendbuf.buf, recvbuf.buf);

        } else if (trans_peer_id % (int) std::pow(2, i) == 0 && trans_peer_id % (int) std::pow(2, i + 1) != 0){
            Utils::peer_num real_dst = transform_peer_id(trans_peer_id - (int) std::pow(2, i), root, false);
            send({sendbuf.buf, sendbuf.len}, real_dst);
        }
    }
    if (peer_id == root) {
        std::memcpy(recvbuf.buf, sendbuf.buf, sendbuf.len);
    } else {
        delete[] recvbuf.buf;
    }
}

void FMI::Comm::PeerToPeer::allreduce(channel_data sendbuf, channel_data recvbuf, raw_function f) {
    bool left_to_right = !(f.commutative && f.associative);
    if (left_to_right) {
        reduce(sendbuf, recvbuf, 0, f);
        bcast(recvbuf, 0);
    } else {
        allreduce_no_order(sendbuf, recvbuf, f);
    }
}

void FMI::Comm::PeerToPeer::allreduce_no_order(channel_data sendbuf, channel_data recvbuf, const raw_function &f) {
    // Non power of two N: First receive from processes with ID >= 2^ceil(log2(N)), send result after reduction
    int rounds = floor(log2(num_peers));
    int nearest_power_two = (int) std::pow(2, rounds);
    if (num_peers > nearest_power_two) {
        if (peer_id < nearest_power_two && peer_id + nearest_power_two < num_peers) {
            recv(recvbuf, peer_id + nearest_power_two);
            f.f(sendbuf.buf, recvbuf.buf);
        } else if (peer_id >= nearest_power_two) {
            send(sendbuf, peer_id - nearest_power_two);
        }
    }
    if (peer_id < nearest_power_two) {
        // Actual recursive doubling
        for (int i = 0; i < rounds; i++) {
            int peer = peer_id ^ (int) std::pow(2, i);
            if (peer < peer_id) {
                send(sendbuf, peer);
                recv(recvbuf, peer);
            } else {
                recv(recvbuf, peer);
                send(sendbuf, peer);
            }
            f.f(sendbuf.buf, recvbuf.buf);
        }
    }
    if (num_peers > nearest_power_two) {
        if (peer_id < nearest_power_two && peer_id + nearest_power_two < num_peers) {
            send(sendbuf, peer_id + nearest_power_two);
        } else if (peer_id >= nearest_power_two) {
            recv(sendbuf, peer_id - nearest_power_two);
        }
    }
    std::memcpy(recvbuf.buf, sendbuf.buf, sendbuf.len);
}

void FMI::Comm::PeerToPeer::scan(channel_data sendbuf, channel_data recvbuf, raw_function f) {
    bool left_to_right = !(f.commutative && f.associative);
    if (left_to_right) {
        scan_ltr(sendbuf, recvbuf, f);
    } else {
        scan_no_order(sendbuf, recvbuf, f);
    }
}

void FMI::Comm::PeerToPeer::scan_ltr(channel_data sendbuf, channel_data recvbuf, const raw_function& f) {
    if (peer_id == 0) {
        send(sendbuf, 1);
        std::memcpy(recvbuf.buf, sendbuf.buf, sendbuf.len);
    } else {
        recv(recvbuf, peer_id - 1);
        f.f(recvbuf.buf, sendbuf.buf);
        if (peer_id < num_peers - 1) {
            send(recvbuf, peer_id + 1);
        }
    }
}

void FMI::Comm::PeerToPeer::scan_no_order(channel_data sendbuf, channel_data recvbuf, const raw_function& f) {
    int rounds = floor(log2(num_peers));
    for (int i = 0; i < rounds; i ++) {
        if ((peer_id & ((int) std::pow(2, i + 1) - 1)) == (int) std::pow(2, i + 1) - 1) {
            Utils::peer_num src = peer_id - (int) std::pow(2, i);
            recv(recvbuf, src);
            f.f(sendbuf.buf, recvbuf.buf);
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
                f.f(sendbuf.buf, recvbuf.buf);
            }
        }
    }
    std::memcpy(recvbuf.buf, sendbuf.buf, sendbuf.len);
}

void FMI::Comm::PeerToPeer::allgatherv(channel_data sendbuf, channel_data &recvbuf, FMI::Utils::peer_num root,
                                       const std::vector<std::size_t> &recvcounts,
                                       const std::vector<std::size_t> &displs) {

    int rounds = ceil(log2(num_peers));
    Utils::peer_num trans_peer_id = transform_peer_id(peer_id, root, true);

    // Calculate total size needed for the final buffer
    std::size_t total_buffer_size = 0;
    for (auto &count : recvcounts) total_buffer_size += count;

    // Allocate buffer for final gathered data
    recvbuf.buf = new char[total_buffer_size];
    recvbuf.len = total_buffer_size;

    // Copy own data to correct position
    std::memcpy(recvbuf.buf + displs[peer_id], sendbuf.buf, sendbuf.len);

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
            recv({recvbuf.buf + offset, buf_len}, real_src);
        } else if (trans_peer_id % (int) std::pow(2, i) == 0 && trans_peer_id % (int) std::pow(2, i + 1) != 0) {
            std::size_t offset = displs[trans_peer_id];
            std::size_t buf_len = 0;

            for (Utils::peer_num p = trans_peer_id; p < std::min(trans_peer_id + (Utils::peer_num) std::pow(2, i), num_peers); ++p) {
                buf_len += recvcounts[p];
            }

            Utils::peer_num real_dst = transform_peer_id(trans_peer_id - (int) std::pow(2, i), root, false);
            send({recvbuf.buf + offset, buf_len}, real_dst);
        }
    }

    // 🌐 Broadcast Phase (Share gathered variable-sized data with all)
    for (int i = 0; i < rounds; i++) {
        Utils::peer_num partner = trans_peer_id ^ (1 << i);
        if (partner < num_peers) {
            if ((trans_peer_id & (1 << i)) == 0) {
                send({recvbuf.buf, total_buffer_size}, transform_peer_id(partner, root, false));
            } else {
                recv({recvbuf.buf, total_buffer_size}, transform_peer_id(partner, root, false));
            }
        }
    }

}

void FMI::Comm::PeerToPeer::allgather(channel_data sendbuf, channel_data recvbuf, FMI::Utils::peer_num root) {
    int rounds = ceil(log2(num_peers));
    Utils::peer_num trans_peer_id = transform_peer_id(peer_id, root, true);
    std::size_t single_buffer_size = sendbuf.len;
    std::size_t total_buffer_size = num_peers * single_buffer_size;

    // Allocate buffer for the final gathered data
    if (peer_id == root) {
        recvbuf.buf = new char[total_buffer_size];
        recvbuf.len = total_buffer_size;
    } else {
        recvbuf.buf = new char[total_buffer_size];
        recvbuf.len = total_buffer_size;
    }
    std::memcpy(recvbuf.buf + single_buffer_size * peer_id, sendbuf.buf, single_buffer_size);

    for (int i = 0; i < rounds; i++) {
        Utils::peer_num src = trans_peer_id + (Utils::peer_num) std::pow(2, i);

        if (trans_peer_id % (int) std::pow(2, i + 1) == 0 && src < num_peers) {
            unsigned int responsible_peers = std::min((Utils::peer_num) std::pow(2, i), num_peers - src);
            std::size_t buf_len = responsible_peers * single_buffer_size;
            Utils::peer_num real_src = transform_peer_id(src, root, false);

            recv({recvbuf.buf + real_src * single_buffer_size, buf_len}, real_src);
        } else if (trans_peer_id % (int) std::pow(2, i) == 0 && trans_peer_id % (int) std::pow(2, i + 1) != 0) {
            unsigned int responsible_peers = std::min((Utils::peer_num) std::pow(2, i), num_peers - trans_peer_id);
            std::size_t buf_len = responsible_peers * single_buffer_size;
            Utils::peer_num real_dst = transform_peer_id(trans_peer_id - (int) std::pow(2, i), root, false);
            send({recvbuf.buf + trans_peer_id * single_buffer_size, buf_len}, real_dst);
        }
    }

    // 🌐 Broadcast Phase (so all processes get the gathered data)
    for (int i = 0; i < rounds; i++) {
        Utils::peer_num partner = trans_peer_id ^ (1 << i);
        if (partner < num_peers) {
            if ((trans_peer_id & (1 << i)) == 0) {
                // Send the full gathered data to the partner
                send({recvbuf.buf, total_buffer_size}, transform_peer_id(partner, root, false));
            } else {
                // Receive the full gathered data from the partner
                recv({recvbuf.buf, total_buffer_size}, transform_peer_id(partner, root, false));
            }
        }
    }
}

void FMI::Comm::PeerToPeer::gather(channel_data sendbuf, channel_data recvbuf, FMI::Utils::peer_num root) {
    int rounds = ceil(log2(num_peers));
    Utils::peer_num trans_peer_id = transform_peer_id(peer_id, root, true);
    std::size_t single_buffer_size = sendbuf.len;
    // Find needed buffer size and allocate it
    if (peer_id != root) {
        unsigned int peers_in_buffer = 1;
        for (int i = rounds - 1; i >= 0; i--) {
            Utils::peer_num src = trans_peer_id + (Utils::peer_num) std::pow(2, i);
            if (trans_peer_id % (int) std::pow(2, i + 1) == 0 && src < num_peers) {
                peers_in_buffer += std::min((Utils::peer_num) std::pow(2, i), num_peers - src);
            }
        }
        recvbuf.buf = new char[peers_in_buffer * single_buffer_size];
        recvbuf.len = peers_in_buffer * single_buffer_size;
        std::memcpy(recvbuf.buf, sendbuf.buf, single_buffer_size);
    } else {
        std::memcpy(recvbuf.buf + single_buffer_size * root, sendbuf.buf, single_buffer_size);
    }

    for (int i = 0; i < rounds; i++) {
        Utils::peer_num src = trans_peer_id + (Utils::peer_num) std::pow(2, i);

        if (trans_peer_id % (int) std::pow(2, i + 1) == 0 && src < num_peers) {
            unsigned int responsible_peers = std::min((Utils::peer_num) std::pow(2, i), num_peers - src);
            std::size_t buf_len = responsible_peers * single_buffer_size;
            Utils::peer_num real_src = transform_peer_id(src, root, false);

            if (peer_id == root) {
                if (real_src * single_buffer_size + buf_len > recvbuf.len) {
                    // Need to wraparound with temporary buffer
                    char *tmp = new char[buf_len];
                    recv({tmp, buf_len}, real_src);
                    unsigned int length_end = recvbuf.len - real_src * single_buffer_size; // How many bytes to copy at end of buffer
                    std::memcpy(recvbuf.buf + real_src * single_buffer_size, tmp, length_end);
                    std::memcpy(recvbuf.buf, tmp + length_end, buf_len - length_end);
                    delete[] tmp;
                } else {
                    recv({recvbuf.buf + real_src * single_buffer_size, buf_len}, real_src);
                }
            } else {
                recv({recvbuf.buf + (src - trans_peer_id) * single_buffer_size, buf_len}, real_src);
            }
        } else if (trans_peer_id % (int) std::pow(2, i) == 0 && trans_peer_id % (int) std::pow(2, i + 1) != 0){
            unsigned int responsible_peers = std::min((Utils::peer_num) std::pow(2, i), num_peers - trans_peer_id);
            std::size_t buf_len = responsible_peers * single_buffer_size;
            Utils::peer_num real_dst = transform_peer_id(trans_peer_id - (int) std::pow(2, i), root, false);
            send({recvbuf.buf, buf_len}, real_dst);
        }
    }
    if (peer_id != root) {
        delete[] recvbuf.buf;
    }
}

void FMI::Comm::PeerToPeer::scatter(channel_data sendbuf, channel_data recvbuf, FMI::Utils::peer_num root) {
    int rounds = ceil(log2(num_peers));
    Utils::peer_num trans_peer_id = transform_peer_id(peer_id, root, true);
    std::size_t single_buffer_size = recvbuf.len;
    for (int i = rounds - 1; i >= 0; i--) {
        Utils::peer_num rcpt = trans_peer_id + (Utils::peer_num) std::pow(2, i);


        if (trans_peer_id % (int) std::pow(2, i + 1) == 0 && rcpt < num_peers) {
            unsigned int responsible_peers = std::min((Utils::peer_num) std::pow(2, i), num_peers - rcpt);
            std::size_t buf_len = responsible_peers * single_buffer_size;
            Utils::peer_num real_rcpt = transform_peer_id(rcpt, root, false);

            if (peer_id == root) {
                if (real_rcpt * single_buffer_size + buf_len > sendbuf.len) {
                    // Wrapping around, need to allocate a temporary buffer
                    char* tmp = new char[buf_len];
                    unsigned int length_end = sendbuf.len - real_rcpt * single_buffer_size; // How many bytes we need to send at end of buffer
                    std::memcpy(tmp, sendbuf.buf + real_rcpt * single_buffer_size, length_end);
                    // Copy rest from beginning
                    std::memcpy(tmp + length_end, sendbuf.buf, buf_len - length_end);
                    send({tmp, buf_len}, real_rcpt);
                    delete[] tmp;
                } else {
                    send({sendbuf.buf + real_rcpt * single_buffer_size, buf_len}, real_rcpt);
                }
            } else {
                send({sendbuf.buf + (rcpt - trans_peer_id) * single_buffer_size, buf_len}, real_rcpt);
            }
        } else if (trans_peer_id % (int) std::pow(2, i) == 0 && trans_peer_id % (int) std::pow(2, i + 1) != 0){
            unsigned int responsible_peers = std::min((Utils::peer_num) std::pow(2, i), num_peers - trans_peer_id);
            std::size_t buf_len = responsible_peers * single_buffer_size;
            Utils::peer_num real_src = transform_peer_id(trans_peer_id - (int) std::pow(2, i), root, false);
            sendbuf.buf = new char[buf_len];
            sendbuf.len = buf_len;
            recv(sendbuf, real_src);
        }
    }
    if (peer_id == root) {
        std::memcpy(recvbuf.buf, sendbuf.buf + peer_id * single_buffer_size, single_buffer_size);
    } else {
        std::memcpy(recvbuf.buf, sendbuf.buf, single_buffer_size);
        delete[] sendbuf.buf;
    }
}

FMI::Utils::peer_num FMI::Comm::PeerToPeer::transform_peer_id(FMI::Utils::peer_num id, FMI::Utils::peer_num root, bool forward) {
    if (forward) {
        return (id + num_peers - root) % num_peers; // Transform s.t. root has id 0
    } else {
        return (id + root) % num_peers;
    }
}

void FMI::Comm::PeerToPeer::communicator_event_progress() {

}








