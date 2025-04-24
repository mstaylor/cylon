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
#include "Direct.hpp"

#include "../../TCPunch/client/tcpunch.hpp"
#include <sys/epoll.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <netdb.h>
#include <cstring>
#include <cerrno>
#include <iostream>
#include <vector>
#include <memory>
#include <functional>
#include <unordered_map>
#include "../utils/DirectBackend.hpp"
#include "cylon/net/channel.hpp"


#include <glog/logging.h>


FMI::Comm::Direct::Direct(const std::shared_ptr<FMI::Utils::Backends> &backend) {
    struct addrinfo hints, *res, *p;
    int status;
    char ipstr[INET6_ADDRSTRLEN];

    auto direct_backend = dynamic_cast<FMI::Utils::DirectBackend *>(backend.get());
    hostname = direct_backend->getHost();
    port = direct_backend->getPort();
    if (direct_backend->resolveHostDNS()) {

        memset(&hints, 0, sizeof hints);
        hints.ai_family = AF_INET;
        hints.ai_socktype = SOCK_STREAM;

        if ((status = getaddrinfo(hostname.c_str(), nullptr, &hints, &res)) != 0) {
            LOG(ERROR)  << "getaddrinfo error: " << gai_strerror(status) << std::endl;
        } else {
            // Iterate through the result list and convert each address to a string
            for(p = res; p != nullptr; p = p->ai_next) {
                void *addr;

                // Get the pointer to the address itself,
                struct sockaddr_in *ipv4 = (struct sockaddr_in *)p->ai_addr;
                addr = &(ipv4->sin_addr);

                // Convert the IP to a string and print it:
                inet_ntop(p->ai_family, addr, ipstr, sizeof ipstr);

                //TODO: remove
                std::cout << " resolved rendezvous DNS: " << ipstr << std::endl;
            }

            freeaddrinfo(res); // Free the linked list
            hostname = ipstr;

        }

    }

    max_timeout = direct_backend->getMaxTimeout();

    epoll_fd = epoll_create1(0);
    if (epoll_fd == -1) {
        LOG(ERROR) << "Failed to create epoll instance: " << strerror(errno) << std::endl;
    }

    sockets[Utils::NONBLOCKING] = {};
    sockets[Utils::BLOCKING] = {};



    io_states[Utils::Operation::SEND] = {};
    io_states[Utils::Operation::RECEIVE] = {};


}

inline const char* ModeToString(FMI::Utils::Mode mode) {
    switch (mode) {
        case FMI::Utils::Mode::BLOCKING: return "BLOCKING";
        case FMI::Utils::Mode::NONBLOCKING: return "NONBLOCKING";

        default: return "UNKNOWN_MODE";
    }
}


void FMI::Comm::Direct::init() {
    //iterator over world size and create all sockets for non-blocking based on multi-send/receives
    //create all the connections
    if (getNumPeers()> 0) {

        for (int i = 0; i < getNumPeers(); ++i) {

            if (i == peer_id) continue;


            std::string send_pairing_nb = get_pairing_name(peer_id, i, Utils::NONBLOCKING);

            check_socket_nbx(i, send_pairing_nb);

            /*std::string send_pairing_b = get_pairing_name(peer_id, i, Utils::BLOCKING);

            check_socket(i, send_pairing_b);*/

        }
    }
}

FMI::Comm::Direct::~Direct() {
    close(epoll_fd);
    for (auto sock : sockets[Utils::BLOCKING]) if (sock != -1) close(sock);
    for (auto sock : sockets[Utils::NONBLOCKING]) if (sock != -1) close(sock);

}

std::string FMI::Comm::Direct::get_pairing_name(FMI::Utils::peer_num a,
                                                FMI::Utils::peer_num b,
                                                FMI::Utils::Mode mode) {
    int min_id = std::min(a, b);
    int max_id = std::max(a, b);
    return "fmi_pair" + std::to_string(min_id) + "_" + std::to_string(max_id) + ModeToString(mode);
}

void FMI::Comm::Direct::send_object_blocking2(FMI::Comm::IOState &state, FMI::Utils::peer_num rcpt_id) {
    std::string pairing = get_pairing_name(peer_id, rcpt_id, Utils::BLOCKING);
    check_socket(rcpt_id, pairing);
    int socketfd = sockets[Utils::BLOCKING][rcpt_id];

    ssize_t sent_total = 0;

    // Zero-length message? Send dummy byte
    if (state.request.len == 0) {
        char dummy = 0;
        while (true) {
            ssize_t sent = ::send(socketfd, &dummy, 1, 0);
            if (sent == 1) {
                if (state.callback) state.callback();
                state.callbackResult(Utils::SUCCESS, "Zero-length message sent with dummy byte.", state.context);
                return;
            } else if (sent == -1) {
                if (errno == EINTR) continue;
                if (errno == EAGAIN || errno == EWOULDBLOCK) {
                    throw Utils::Timeout();
                }
                LOG(ERROR) << "Send error (dummy): " << strerror(errno);
                return;
            } else {
                state.callbackResult(Utils::DUMMY_SEND_FAILED, strerror(errno), state.context);
                return;
            }

        }

    }


    while (sent_total < state.request.len) {
        ssize_t sent = ::send(socketfd, state.request.buf.get() + sent_total,
                              state.request.len - sent_total, 0);

        if (sent == -1) {
            if (errno == EINTR) continue;
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                throw Utils::Timeout(); // or use callbackResult if needed
            }
            LOG(ERROR) << "Send error: " << strerror(errno);
            return;
        }

        sent_total += sent;
    }

    if (state.callback) state.callback();
    state.callbackResult(Utils::SUCCESS, "Blocking send complete", state.context);
}

void FMI::Comm::Direct::send_object(IOState &state, Utils::peer_num rcpt_id,
                                    Utils::Mode mode) {

    if (mode == Utils::NONBLOCKING) {
        std::string pairing = get_pairing_name(peer_id, rcpt_id, Utils::NONBLOCKING);

        // Use full-duplex socket for both send/recv
        check_socket_nbx(rcpt_id, pairing);
        int socketfd = sockets[Utils::NONBLOCKING][rcpt_id];


        // Zero-length message? Send dummy byte
        if (state.request.len == 0) {
            char dummy = 0;
            ssize_t sent = ::send(socketfd, &dummy, 1, 0);

            if (sent == 1) {
                if (state.callback) state.callback();
                state.callbackResult(Utils::SUCCESS, "Zero-length message sent with dummy byte.", state.context);
            } else if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) {
                // Need to retry via epoll
                io_states[Utils::Operation::SEND][socketfd] = state;
                add_epoll_event(socketfd, state);
            } else {
                state.callbackResult(Utils::DUMMY_SEND_FAILED, strerror(errno), state.context);
            }

            return;
        }

        // Normal message
        ssize_t processed = ::send(socketfd,
                                   state.request.buf.get() + state.processed,
                                   state.request.len - state.processed,
                                   0);

        if (processed > 0) {
            state.processed += processed;

            if (state.processed == state.request.len) {
                if (state.callback) state.callback();
                state.callbackResult(Utils::SUCCESS, "Send completed", state.context);
                return;
            }
        }

        // Still pending, register for epoll
        if (processed == -1 && (errno != EAGAIN && errno != EINTR)) {
            state.callbackResult(Utils::SEND_FAILED, strerror(errno), state.context);
            return;
        }

        // Save the state and try again via epoll
        io_states[Utils::Operation::SEND][socketfd] = state;
        add_epoll_event(socketfd, state);

    } else {
        send_object_blocking2(state, rcpt_id);
    }


}


void FMI::Comm::Direct::send_object(const channel_data &buf, FMI::Utils::peer_num rcpt_id) {
    std::string pairing = get_pairing_name(peer_id, rcpt_id, Utils::BLOCKING);
    check_socket(rcpt_id, /*comm_name + std::to_string(peer_id) + "_"
                    + std::to_string(rcpt_id)*/pairing);
    long sent = ::send(sockets[Utils::BLOCKING][rcpt_id], buf.buf.get(), buf.len, 0);
    if (sent == -1) {
        if (errno == EAGAIN) {
            throw Utils::Timeout();
        }
        LOG(ERROR) << peer_id << ": Error when sending: " << strerror(errno) ;
    }
}

void FMI::Comm::Direct::recv_object_blocking2(FMI::Comm::IOState &state, FMI::Utils::peer_num sender_id) {
    std::string pairing = get_pairing_name(peer_id, sender_id, Utils::BLOCKING);
    check_socket(sender_id, pairing);
    int sockfd = sockets[Utils::BLOCKING][sender_id];

    void *buffer = state.request.len == 0
                   ? static_cast<void *>(&state.dummy)
                   : state.request.buf.get() + state.processed;

    size_t remaining = state.request.len == 0 ? 1 : state.request.len - state.processed;

    while (remaining > 0) {
        ssize_t received = ::recv(sockfd, buffer, remaining, 0);

        if (received > 0) {
            state.processed += received;
            remaining -= received;
            buffer = static_cast<char *>(buffer) + received;

            LOG(INFO) << "processed receive bytes: " << state.processed << " of " << state.request.len;

            // 🔥 Check if full buffer received mid-loop
            if (state.processed == state.request.len) {
                // ✅ Protocol-level FIN check (YOUR LOGIC)
                if (state.request.len >= 8 * sizeof(int)) {
                    int *header = reinterpret_cast<int *>(state.request.buf.get());
                    if (header[0] == 0 && header[1] == CYLON_MSG_FIN) {
                        if (state.callback) state.callback();
                        state.callbackResult(Utils::SUCCESS, "Protocol FIN received", state.context);
                        return;
                    }
                }
                // ✅ Normal data completion
                if (state.callback) state.callback();
                state.callbackResult(Utils::SUCCESS, "Receive completed", state.context);
                return;
            }

        } else if (received == 0) {
            // Connection closed prematurely
            state.callbackResult(Utils::CONNECTION_CLOSED_BY_PEER,
                                 "Socket closed before full message or FIN was received", state.context);
            return;

        } else if (errno == EINTR) {
            continue; // Retry on interrupt

        } else {
            // Fatal recv error
            state.callbackResult(Utils::RECEIVE_FAILED, strerror(errno), state.context);
            return;
        }
    }

    // Handle zero-length message completion (if no loop body runs)
    if (state.request.len == 0) {
        if (state.callback) state.callback();
        state.callbackResult(Utils::SUCCESS, "Zero-length receive via dummy byte", state.context);
    }
}

void FMI::Comm::Direct::recv_object(IOState &state, Utils::peer_num sender_id,
                                    Utils::Mode mode) {

    if (mode == Utils::NONBLOCKING) {
        std::string pairing = get_pairing_name(peer_id, sender_id, Utils::NONBLOCKING);
        check_socket_nbx(sender_id, pairing);
        auto sender_socket = sockets[Utils::NONBLOCKING][sender_id];

        io_states[Utils::Operation::RECEIVE][sender_socket] = state;
        add_epoll_event(sender_socket, state);
    } else {
        recv_object_blocking2(state, sender_id);
    }
}

void FMI::Comm::Direct::recv_object(const channel_data &buf, FMI::Utils::peer_num sender_id) {
    std::string pairing = get_pairing_name(peer_id, sender_id, Utils::BLOCKING);
    check_socket(sender_id, /*comm_name + std::to_string(sender_id) + "_"
                                + std::to_string(peer_id)*/pairing);
    long received = ::recv(sockets[Utils::BLOCKING][sender_id], buf.buf.get(), buf.len, MSG_WAITALL);
    if (received == -1 || received < buf.len) {
        if (errno == EAGAIN) {
            throw Utils::Timeout();
        }
        LOG(ERROR) << peer_id << ": Error when receiving: " << strerror(errno);
    }
}

void FMI::Comm::Direct::check_socket(FMI::Utils::peer_num partner_id, std::string pair_name) {
    if (sockets[Utils::BLOCKING].empty()) {
        sockets[Utils::BLOCKING] = std::vector<int>(num_peers, -1);
    }
    if (sockets[Utils::BLOCKING][partner_id] == -1) {
        try {
            sockets[Utils::BLOCKING][partner_id] = pair(pair_name, hostname, port, max_timeout);
        } catch (Timeout) {
            throw Utils::Timeout();
        }

        struct timeval timeout;
        timeout.tv_sec = max_timeout / 1000;
        timeout.tv_usec = (max_timeout % 1000) * 1000;
        setsockopt(sockets[Utils::BLOCKING][partner_id], SOL_SOCKET, SO_RCVTIMEO, (const char*)&timeout, sizeof timeout);
        setsockopt(sockets[Utils::BLOCKING][partner_id], SOL_SOCKET, SO_SNDTIMEO, (const char*)&timeout, sizeof timeout);
        // Disable Nagle algorithm to avoid 40ms TCP ack delays
        int one = 1;
        // SOL_TCP not defined on macOS
#if !defined(SOL_TCP) && defined(IPPROTO_TCP)
#define SOL_TCP IPPROTO_TCP
#endif
        setsockopt(sockets[Utils::BLOCKING][partner_id], SOL_TCP, TCP_NODELAY, &one, sizeof(one));
    }
}

void FMI::Comm::Direct::add_epoll_event(int sockfd, const IOState &state) {
    uint32_t new_event = (state.operation == Utils::SEND) ? EPOLLOUT : EPOLLIN;

    auto it = socket_event_map.find(sockfd);

    if (it == socket_event_map.end()) {
        // First time adding this socket
        epoll_event ev{};
        ev.events = new_event;
        ev.data.fd = sockfd;

        if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, sockfd, &ev) == -1) {
            auto strError = "Failed to add socket to epoll: " + std::string(strerror(errno));
            state.callbackResult(Utils::ADD_EVENT_FAILED, strError, state.context);
            return;
        }

        socket_event_map[sockfd] = new_event;
    } else {
        // Socket already added — update interest if new event is needed
        uint32_t existing_events = it->second;

        if ((existing_events & new_event) == 0) {
            uint32_t combined_events = existing_events | new_event;

            epoll_event ev{};
            ev.events = combined_events;
            ev.data.fd = sockfd;

            if (epoll_ctl(epoll_fd, EPOLL_CTL_MOD, sockfd, &ev) == -1) {
                auto strError = "Failed to modify socket in epoll: " + std::string(strerror(errno));
                state.callbackResult(Utils::ADD_EVENT_FAILED, strError, state.context);
                return;
            }

            socket_event_map[sockfd] = combined_events;
        }
        // Else already tracking this event — nothing to do
    }
}

void FMI::Comm::Direct::check_timeouts(std::unordered_map<int, IOState> states) {
    auto now = std::chrono::steady_clock::now();
    for (auto it = states.begin(); it != states.end(); ) {
        if (now >= it->second.deadline) {
            it->second.callbackResult(Utils::NBX_TIMOUTOUT, "Operation timed out.", it->second.context);
            epoll_ctl(epoll_fd, EPOLL_CTL_DEL, it->first, nullptr);
            it = states.erase(it);
        } else {
            ++it;
        }
    }
}

FMI::Utils::EventProcessStatus
FMI::Comm::Direct::channel_event_progress(std::unordered_map<int, IOState> states, Utils::Operation op) {
    if (states.empty()) {
        return FMI::Utils::EMPTY;
    }

    constexpr int MAX_EVENTS = 10;
    epoll_event events[MAX_EVENTS];
    int n = epoll_wait(epoll_fd, events, MAX_EVENTS, 10);  // Use short timeout

    if (n == -1 && errno != EINTR) {
        // Report failure

        for (auto& [fd, state] : states) {
            state.callbackResult(Utils::EPOLL_WAIT_FAILED,
                                 "epoll_wait failed: " + std::string(strerror(errno)),
                                 state.context);
        }

        states.clear();
    }

    for (int i = 0; i < n; i++) {
        handle_event(events[i], states, op);
    }

    return FMI::Utils::PROCESSING;
}


FMI::Utils::EventProcessStatus FMI::Comm::Direct::channel_event_progress(Utils::Operation op) {

    if (op == Utils::DEFAULT) {
        FMI::Utils::EventProcessStatus status = FMI::Utils::EMPTY;
        for (auto& [operation, state] : io_states) {
            auto processStatus = channel_event_progress(state, op);
            if (processStatus != FMI::Utils::EMPTY) {
                status = processStatus;
            }
        }

        return status;
    } else {
        return channel_event_progress(io_states[op], op);
    }


}

void FMI::Comm::Direct::handle_event(epoll_event ev,
                                     std::unordered_map<int, IOState> &states,
                                     Utils::Operation op) const {
    int sockfd = ev.data.fd;

    // 🔹 Handle EPOLLOUT (send readiness)
    if (op == Utils::SEND && (ev.events & EPOLLOUT) && states.count(sockfd)) {
        IOState &state = states[sockfd];
        ssize_t processed = ::send(sockfd,
                                   state.request.buf.get() + state.processed,
                                   state.request.len - state.processed, 0);

        if (processed > 0) {
            state.processed += processed;
            if (state.processed == state.request.len) {
                if (state.callback) state.callback();
                state.callbackResult(Utils::SUCCESS, "Send completed", state.context);
                states.erase(sockfd);
                // Optional: remove EPOLLOUT if you're done sending
            }
        } else if (errno != EAGAIN && errno != EINTR) {
            state.callbackResult(Utils::SEND_FAILED, strerror(errno), state.context);
            states.erase(sockfd);
            epoll_ctl(epoll_fd, EPOLL_CTL_DEL, sockfd, nullptr);  // or MOD if still receiving
        }
    }

    // 🔹 Handle EPOLLIN (data available to read)
    if (op == Utils::RECEIVE && (ev.events & EPOLLIN) && states.count(sockfd)) {
        IOState &state = states[sockfd];

        void *buffer = state.request.len == 0
                       ? static_cast<void *>(&state.dummy)
                       : state.request.buf.get() + state.processed;

        size_t size = state.request.len == 0 ? 1 : state.request.len - state.processed;

        ssize_t received = ::recv(sockfd, buffer, size, 0);


        if (received > 0) {
            state.processed += received;

            LOG(INFO) << "processed receive bytes: " << state.processed << " of " << state.request.len;

            if (state.request.len == 0) {
                if (state.callback) state.callback();
                state.callbackResult(Utils::SUCCESS, "Zero-length receive via dummy byte", state.context);
                states.erase(sockfd);
                //epoll_ctl(epoll_fd, EPOLL_CTL_DEL, sockfd, nullptr);
            } else if (state.processed == state.request.len) {
                // Check for protocol-level FIN message
                if (state.request.len >= 8 * sizeof(int)) {
                    int *header = reinterpret_cast<int *>(state.request.buf.get());
                    if (header[0] == 0 && header[1] == CYLON_MSG_FIN) {
                        if (state.callback) state.callback();
                        state.callbackResult(Utils::SUCCESS, "Protocol FIN received", state.context);
                        // Clean up after FIN
                        states.erase(sockfd);
                        //epoll_ctl(epoll_fd, EPOLL_CTL_DEL, sockfd, nullptr);
                        return;
                    }
                }

                // Otherwise, normal data
                if (state.callback) state.callback();
                state.callbackResult(Utils::SUCCESS, "Receive completed", state.context);
                states.erase(sockfd);
                //epoll_ctl(epoll_fd, EPOLL_CTL_DEL, sockfd, nullptr);
            }

        } else if (received == 0) {
            // TCP-level connection close, but we didn't get a protocol FIN
            state.callbackResult(Utils::CONNECTION_CLOSED_BY_PEER,
                                 "Socket closed before full message or FIN was received", state.context);
            states.erase(sockfd);
            epoll_ctl(epoll_fd, EPOLL_CTL_DEL, sockfd, nullptr);

        } else if (errno != EAGAIN && errno != EINTR) {
            // Hard recv error
            state.callbackResult(Utils::RECEIVE_FAILED, strerror(errno), state.context);
            states.erase(sockfd);
            epoll_ctl(epoll_fd, EPOLL_CTL_DEL, sockfd, nullptr);
        }
    }
}


void FMI::Comm::Direct::check_socket_nbx(FMI::Utils::peer_num partner_id, std::string pair_name) {


    if (sockets[Utils::NONBLOCKING].empty()) {
        sockets[Utils::NONBLOCKING] = std::vector<int>(num_peers, -1);
    }
    if (sockets[Utils::NONBLOCKING][partner_id] == -1) {
        try {
            // 🔄 Use the original `pair()` function to establish the socket connection
            sockets[Utils::NONBLOCKING][partner_id] = pair(pair_name, hostname, port, max_timeout);
            LOG(INFO) << "Paired partnerId: " << partner_id << " to pair_name" << pair_name;
        } catch (const std::exception& e) {
            LOG(INFO) << "Socket pairing failed: " <<  std::string(e.what()) << " pairName: " << pair_name << "partnerId: " << partner_id;
            return;
        }

        // ✅ Set the socket to non-blocking mode
        int flags = fcntl(sockets[Utils::NONBLOCKING][partner_id], F_GETFL, 0);
        if (flags == -1 || fcntl(sockets[Utils::NONBLOCKING][partner_id], F_SETFL, flags | O_NONBLOCK) == -1) {
            LOG(INFO) << "Failed to set non-blocking mode: " << std::string(strerror(errno));
            return;
        }

        // ✅ Configure socket timeouts
        struct timeval timeout;
        timeout.tv_sec = max_timeout / 1000;
        timeout.tv_usec = (max_timeout % 1000) * 1000;
        if (setsockopt(sockets[Utils::NONBLOCKING][partner_id], SOL_SOCKET, SO_RCVTIMEO,
                       &timeout, sizeof(timeout)) == -1) {
            LOG(INFO) << "Failed to set SO_RCVTIMEO: " << std::string(strerror(errno));
        }
        if (setsockopt(sockets[Utils::NONBLOCKING][partner_id], SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) == -1) {
            LOG(INFO) << "Failed to set SO_SNDTIMEO: " + std::string(strerror(errno));
        }

        // ✅ Disable Nagle’s algorithm for low-latency communication
        int one = 1;
        if (setsockopt(sockets[Utils::NONBLOCKING][partner_id], IPPROTO_TCP,
                       TCP_NODELAY, &one, sizeof(one)) == -1) {
            LOG(INFO) << "Failed to set TCP_NODELAY: " << std::string(strerror(errno));
        }
    }


}

int FMI::Comm::Direct::getMaxTimeout() {
    return max_timeout;
}



















