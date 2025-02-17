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

#include <cstring>
#include "cylon/thridparty/fmi/comm/Channel.hpp"


std::shared_ptr<FMI::Comm::Channel>
FMI::Comm::Channel::get_channel(FMI::Utils::BackendType backendType, std::map<std::string, std::string> params,
                                std::map<std::string, std::string> model_params) {
    return nullptr;
}

void FMI::Comm::Channel::gather(channel_data sendbuf, channel_data recvbuf, FMI::Utils::peer_num root) {
    if (peer_id != root) {
        send(sendbuf, root);
    } else {
        auto buffer_length = sendbuf.len;
        for (int i = 0; i < num_peers; i++) {
            if (i == root) {
                std::memcpy(recvbuf.buf + root * buffer_length, sendbuf.buf, buffer_length);
            } else {
                channel_data peer_data {recvbuf.buf + i * buffer_length, buffer_length};
                recv(peer_data, i);
            }
        }
    }
}

void FMI::Comm::Channel::scatter(channel_data sendbuf, channel_data recvbuf, FMI::Utils::peer_num root) {
    if (peer_id == root) {
        auto buffer_length = recvbuf.len;
        for (int i = 0; i < num_peers; i++) {
            if (i == root) {
                std::memcpy(recvbuf.buf, sendbuf.buf + root * buffer_length, buffer_length);
            } else {
                channel_data peer_data {sendbuf.buf + i * buffer_length, buffer_length};
                send(peer_data, i);
            }
        }
    } else {
        recv(recvbuf, root);
    }
}

void FMI::Comm::Channel::allreduce(channel_data sendbuf, channel_data recvbuf, raw_function f) {
    reduce(sendbuf, recvbuf, 0, f);
    bcast(recvbuf, 0);
}

