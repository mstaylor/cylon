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

#ifndef CYLON_DIRECTNBX_HPP
#define CYLON_DIRECTNBX_HPP

#include "PeerToPeer.hpp"

namespace FMI::Comm {
    //! Channel that uses the TCPunch TCP NAT Hole Punching Library for connection establishment.

    enum DirectStatus {
        SUCCESS,
        CONNECTION_CLOSED_BY_PEER,
        SOCKET_CREATE_FAILED,
        TCP_NODELAY_FAILED
    };

    enum DirectOperation {
        SEND,
        RECEIVE
    };

    class DirectNbx : public PeerToPeer {
    public:
        explicit DirectNbx(std::shared_ptr<FMI::Utils::Backends> &backend);

        void send_object(channel_data buf, Utils::peer_num rcpt_id) override;

        void recv_object(channel_data buf, Utils::peer_num sender_id) override;


    private:
        //! Contains the socket file descriptor for the communication with the peers.
        std::vector<int> sockets;
        std::string hostname;
        int port;
        bool resolve_host_dns;
        unsigned int max_timeout;

        struct IOState {
            char* buffer;
            size_t length;
            size_t processed;
            DirectOperation operation;
            std::function<void(DirectStatus, const std::string&)> callback;
        };

        //! Checks if connection with a peer partner_id is already established, otherwise establishes it using TCPunch.
        void check_socket(Utils::peer_num partner_id, std::string pair_name);
    };
}

#endif //CYLON_DIRECTNBX_HPP
