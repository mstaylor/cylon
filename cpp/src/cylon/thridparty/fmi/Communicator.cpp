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

#include "Communicator.hpp"

FMI::Communicator::Communicator(FMI::Utils::peer_num peer_id, FMI::Utils::peer_num num_peers,
                                std::shared_ptr<FMI::Utils::Backends> &backend, std::string comm_name) {

    this->peer_id = peer_id;
    this->num_peers = num_peers;
    this->comm_name = comm_name;


    auto backend_name = backend->getName();
    register_channel(backend_name, Comm::Channel::get_channel(backend), Utils::DEFAULT);
    register_channel(backend_name, Comm::Channel::get_channel(backend), Utils::BCAST);
    register_channel(backend_name, Comm::Channel::get_channel(backend), Utils::GATHER);
    register_channel(backend_name, Comm::Channel::get_channel(backend), Utils::GATHERV);
    register_channel(backend_name, Comm::Channel::get_channel(backend), Utils::ALLGATHER);
    register_channel(backend_name, Comm::Channel::get_channel(backend), Utils::ALLGATHERV);
    register_channel(backend_name, Comm::Channel::get_channel(backend), Utils::RECEIVE);
    register_channel(backend_name, Comm::Channel::get_channel(backend), Utils::SEND);



}

void FMI::Communicator::register_channel(std::string name, std::shared_ptr<FMI::Comm::Channel> c,
                                         Utils::Operation op) {
    c->set_peer_id(peer_id);
    c->set_num_peers(num_peers);
    c->set_comm_name(comm_name);
    channel_map[op] = c;
    //channel = c;
}

FMI::Communicator::~Communicator() {
    channel_map[Utils::DEFAULT]->finalize();
    channel_map[Utils::BCAST]->finalize();
    channel_map[Utils::GATHER]->finalize();
    channel_map[Utils::GATHERV]->finalize();
    channel_map[Utils::ALLGATHER]->finalize();
    channel_map[Utils::ALLGATHERV]->finalize();
    channel_map[Utils::RECEIVE]->finalize();
    channel_map[Utils::SEND]->finalize();

    //channel->finalize();
}

FMI::Utils::peer_num FMI::Communicator::getNumPeers() const {
    return num_peers;
}

FMI::Utils::peer_num FMI::Communicator::getPeerId() const {
    return peer_id;
}
