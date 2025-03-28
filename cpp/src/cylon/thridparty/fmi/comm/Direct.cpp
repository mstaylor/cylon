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

}

FMI::Comm::Direct::~Direct() {
    close(epoll_fd);
    for (auto sock : sockets) if (sock != -1) close(sock);
}

void FMI::Comm::Direct::send_object(const IOState &state, Utils::peer_num peer_id) {
    check_socket_nbx(peer_id, comm_name + std::to_string(peer_id) + "_" + std::to_string(peer_id), state);
    io_states[sockets[peer_id]] = state;
    add_epoll_event(sockets[peer_id], Utils::SEND, state);
}


void FMI::Comm::Direct::send_object(const channel_data &buf, FMI::Utils::peer_num rcpt_id) {
    check_socket(rcpt_id, comm_name + std::to_string(peer_id) + "_" + std::to_string(rcpt_id));
    long sent = ::send(sockets[rcpt_id], buf.buf.get(), buf.len, 0);
    if (sent == -1) {
        if (errno == EAGAIN) {
            throw Utils::Timeout();
        }
        LOG(ERROR) << peer_id << ": Error when sending: " << strerror(errno) ;
    }
}

void FMI::Comm::Direct::recv_object(const IOState &state, Utils::peer_num peer_id) {
    check_socket_nbx(peer_id, comm_name + std::to_string(peer_id) + "_" + std::to_string(peer_id), state);
    io_states[sockets[peer_id]] = state;
    add_epoll_event(sockets[peer_id], Utils::RECEIVE, state);
}

void FMI::Comm::Direct::recv_object(const channel_data &buf, FMI::Utils::peer_num sender_id) {
    check_socket(sender_id, comm_name + std::to_string(sender_id) + "_" + std::to_string(peer_id));
    long received = ::recv(sockets[sender_id], buf.buf.get(), buf.len, MSG_WAITALL);
    if (received == -1 || received < buf.len) {
        if (errno == EAGAIN) {
            throw Utils::Timeout();
        }
        LOG(ERROR) << peer_id << ": Error when receiving: " << strerror(errno);
    }
}

void FMI::Comm::Direct::check_socket(FMI::Utils::peer_num partner_id, std::string pair_name) {
    if (sockets.empty()) {
        sockets = std::vector<int>(num_peers, -1);
    }
    if (sockets[partner_id] == -1) {
        try {
            sockets[partner_id] = pair(pair_name, hostname, port, max_timeout);
        } catch (Timeout) {
            throw Utils::Timeout();
        }

        struct timeval timeout;
        timeout.tv_sec = max_timeout / 1000;
        timeout.tv_usec = (max_timeout % 1000) * 1000;
        setsockopt(sockets[partner_id], SOL_SOCKET, SO_RCVTIMEO, (const char*)&timeout, sizeof timeout);
        setsockopt(sockets[partner_id], SOL_SOCKET, SO_SNDTIMEO, (const char*)&timeout, sizeof timeout);
        // Disable Nagle algorithm to avoid 40ms TCP ack delays
        int one = 1;
        // SOL_TCP not defined on macOS
#if !defined(SOL_TCP) && defined(IPPROTO_TCP)
#define SOL_TCP IPPROTO_TCP
#endif
        setsockopt(sockets[partner_id], SOL_TCP, TCP_NODELAY, &one, sizeof(one));
    }
}


void FMI::Comm::Direct::add_epoll_event(int sockfd, Utils::Operation operation, const IOState &state) const {
    epoll_event ev{};
    ev.events = Utils::SEND ? EPOLLOUT : EPOLLIN;
    ev.data.fd = sockfd;
    if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, sockfd, &ev) == -1) {
        state.callbackResult(Utils::ADD_EVENT_FAILED,
                             "Failed to add socket to epoll: " + std::string(strerror(errno)),
                             state.context);
    }
}

void FMI::Comm::Direct::check_timeouts() {
    auto now = std::chrono::steady_clock::now();
    for (auto it = io_states.begin(); it != io_states.end(); ) {
        if (now >= it->second.deadline) {
            it->second.callbackResult(Utils::NBX_TIMOUTOUT, "Operation timed out.", it->second.context);
            epoll_ctl(epoll_fd, EPOLL_CTL_DEL, it->first, nullptr);
            it = io_states.erase(it);
        } else {
            ++it;
        }
    }
}

FMI::Utils::EventProcessStatus FMI::Comm::Direct::channel_event_progress() {
    if (io_states.empty()) return FMI::Utils::EMPTY;

    constexpr int MAX_EVENTS = 10;
    epoll_event events[MAX_EVENTS];
    int n = epoll_wait(epoll_fd, events, MAX_EVENTS, 0);  // Immediate return
    if (n == -1 && errno != EINTR) {
        for (auto& [fd, state] : io_states) {
            state.callbackResult(Utils::EPOLL_WAIT_FAILED,
                                 "epoll_wait failed: " + std::string(strerror(errno)),state.context);
        }
        io_states.clear();
    }

    for (int i = 0; i < n; i++) {
        handle_event(events[i].data.fd);
    }

    // Check for timeouts
    check_timeouts();

    return FMI::Utils::PROCESSING;
}

void FMI::Comm::Direct::handle_event(int sockfd) {
    auto it = io_states.find(sockfd);
    if (it == io_states.end()) return;

    IOState& state = it->second;
    ssize_t processed = state.operation == Utils::SEND
                        ? ::send(sockfd, state.request.buf.get() + state.processed,
                                 state.request.len - state.processed, 0)
                        : ::recv(sockfd, state.request.buf.get() + state.processed,
                                 state.request.len - state.processed, 0);

    if (processed > 0) {
        state.processed += processed;
        if (state.processed == state.request.len) {
            if (state.callback) { //execute custom callback
                state.callback();
            }
            state.callbackResult(Utils::SUCCESS, "Operation completed successfully.", state.context);
            io_states.erase(it);
            epoll_ctl(epoll_fd, EPOLL_CTL_DEL, sockfd, nullptr);
        }
    } else if (errno == EAGAIN || errno == EWOULDBLOCK) {
        // No data ready now; just return and wait for epoll to trigger again
        return;
    } else if (errno == EINTR) {
        return; // or retry
    } else {
        state.callbackResult(Utils::CONNECTION_CLOSED_BY_PEER,
                             processed == 0 ? "Connection closed by peer." : strerror(errno),
                             state.context);
        io_states.erase(it);
        epoll_ctl(epoll_fd, EPOLL_CTL_DEL, sockfd, nullptr);
    }
}

void FMI::Comm::Direct::check_socket_nbx(FMI::Utils::peer_num partner_id, std::string pair_name,
                                         const IOState &state) {


    if (sockets.empty()) {
        sockets = std::vector<int>(num_peers, -1);
    }
    if (sockets[partner_id] == -1) {
        try {
            // 🔄 Use the original `pair()` function to establish the socket connection
            sockets[partner_id] = pair(pair_name, hostname, port, max_timeout);
        } catch (const std::exception& e) {
            state.callbackResult(Utils::SOCKET_PAIR_FAILED,
                                 "Socket pairing failed: " + std::string(e.what()), state.context);
            return;
        }

        // ✅ Set the socket to non-blocking mode
        int flags = fcntl(sockets[partner_id], F_GETFL, 0);
        if (flags == -1 || fcntl(sockets[partner_id], F_SETFL, flags | O_NONBLOCK) == -1) {
            state.callbackResult(Utils::SOCKET_SET_NONBLOCKING_FAILED,
                                 "Failed to set non-blocking mode: " + std::string(strerror(errno)),
                                 state.context);
            return;
        }

        // ✅ Configure socket timeouts
        struct timeval timeout;
        timeout.tv_sec = max_timeout / 1000;
        timeout.tv_usec = (max_timeout % 1000) * 1000;
        if (setsockopt(sockets[partner_id], SOL_SOCKET, SO_RCVTIMEO,
                       &timeout, sizeof(timeout)) == -1) {
            state.callbackResult(Utils::SOCKET_SET_SO_RCVTIMEO_FAILED,
                                 "Failed to set SO_RCVTIMEO: " + std::string(strerror(errno)),
                                 state.context);
        }
        if (setsockopt(sockets[partner_id], SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) == -1) {
            state.callbackResult(Utils::SOCKET_SET_SO_SNDTIMEO_FAILED,
                                 "Failed to set SO_SNDTIMEO: " + std::string(strerror(errno)),
                                 state.context);
        }

        // ✅ Disable Nagle’s algorithm for low-latency communication
        int one = 1;
        if (setsockopt(sockets[partner_id], IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one)) == -1) {
            state.callbackResult(Utils::TCP_NODELAY_FAILED,
                                 "Failed to set TCP_NODELAY: " + std::string(strerror(errno)),
                                 state.context);
        }
    }


}

int FMI::Comm::Direct::getMaxTimeout() {
    return max_timeout;
}








