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

#ifndef CYLON_DIRECT_HPP
#define CYLON_DIRECT_HPP

#include "PeerToPeer.hpp"
#include <cylon/thridparty/fmi/utils/Common.hpp>

namespace FMI::Comm {

        //! Channel that uses the TCPunch TCP NAT Hole Punching Library for connection establishment.
        class Direct : public PeerToPeer {
            public:
            explicit Direct(const std::shared_ptr<FMI::Utils::Backends> &backend);

            virtual ~Direct();

            void send_object(const channel_data &buf, Utils::peer_num rcpt_id) override;


            void send_object(const IOState &state, Utils::peer_num peer_id) override;

            void recv_object(const channel_data &buf, Utils::peer_num sender_id) override;

            void recv_object(const IOState &state, Utils::peer_num peer_id) override;

            Utils::EventProcessStatus channel_event_progress() override;

        private:
            //! Contains the socket file descriptor for the communication with the peers.
            std::vector<int> sockets;
            std::string hostname;
            int port;
            bool resolve_host_dns;
            unsigned int max_timeout;
            int epoll_fd;


            std::unordered_map<int, IOState> io_states;

            //! Checks if connection with a peer partner_id is already established, otherwise establishes it using TCPunch.
            void check_socket(Utils::peer_num partner_id, std::string pair_name);

            void check_socket_nbx(Utils::peer_num partner_id, std::string pair_name, const IOState& state);

            void add_epoll_event(int sockfd, Utils::Operation operation, const IOState& state) const;

            void handle_event(int sockfd);
        };
}

#endif //CYLON_DIRECT_HPP
