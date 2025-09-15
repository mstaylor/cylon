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

#include "tcpunch.hpp"
#include <fcntl.h>
#include <csignal>
#include <cstring>
#include <cstdlib>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <cerrno>
#include <string>
#include <chrono>
#include <thread>
#include <stdexcept>
#include <iostream>
#include <atomic>
#include "../common/utils.hpp"


std::atomic<bool> connection_established(false);
std::atomic<int> accepting_socket(-1);

void* peer_listen(void* p) {
    auto* info = (PeerConnectionData*)p;

    // Create socket on the port that was previously used to contact the rendezvous server
    int listen_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_socket == -1) {
        error_exit_errno("Socket creation failed: ");
    }
    int enable_flag = 1;
    if (setsockopt(listen_socket, SOL_SOCKET, SO_REUSEADDR, &enable_flag, sizeof(int)) < 0 ||
        setsockopt(listen_socket, SOL_SOCKET, SO_REUSEPORT, &enable_flag, sizeof(int)) < 0) {
        error_exit_errno("Setting REUSE options failed: ");
    }

    // Set accept timeout for AWS Fargate environment (3 minutes)
    struct timeval accept_timeout;
    accept_timeout.tv_sec = 180;  // 3 minutes - enough for 120s connection + 15s validation + buffer
    accept_timeout.tv_usec = 0;
    if (setsockopt(listen_socket, SOL_SOCKET, SO_RCVTIMEO, &accept_timeout, sizeof(accept_timeout)) < 0) {
        error_exit_errno("Setting accept timeout failed: ");
    }

    struct sockaddr_in local_port_data{};
    local_port_data.sin_family = AF_INET;
    local_port_data.sin_addr.s_addr = INADDR_ANY;
    local_port_data.sin_port = info->port;

    if (bind(listen_socket, (const struct sockaddr *)&local_port_data, sizeof(local_port_data)) < 0) {
        error_exit_errno("Could not bind to local port: ");
    }

    if (listen(listen_socket, 1) == -1) {
        error_exit_errno("Listening on local port failed: ");
    }

    struct sockaddr_in peer_info{};
    unsigned int len = sizeof(peer_info);

    auto start_time = std::chrono::steady_clock::now();
    auto max_listen_time = std::chrono::seconds(180);
    
    while(true) {
        if (std::chrono::steady_clock::now() - start_time > max_listen_time) {
            LOG(INFO)  << "Peer listen timeout reached" << std::endl;
            break;
        }
        
        int peer = accept(listen_socket, (struct sockaddr*)&peer_info, &len);
        if (peer == -1) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                continue;
            }

            LOG(INFO) << "Error when connecting to peer: " << strerror(errno) << std::endl;

        } else {
            LOG(INFO) << "Succesfully connected to peer, accepting" << std::endl;

            accepting_socket = peer;
            connection_established = true;
            return 0;
        }
    }
    return 0;
}

void remove_pair(const std::string& pairing_name, const std::string& server_address, int port, int timeout_ms) {
    int socket_rendezvous;
    struct sockaddr_in server_data{};
    struct timeval timeout;
    timeout.tv_sec = timeout_ms / 1000;
    timeout.tv_usec = (timeout_ms % 1000) * 1000;


    socket_rendezvous = socket(AF_INET, SOCK_STREAM, 0);
    if (socket_rendezvous == -1) {
        error_exit_errno("Could not create socket for rendezvous server: ");
    }

    // Enable binding multiple sockets to the same local endpoint, see https://bford.info/pub/net/p2pnat/ for details
    int enable_flag = 1;
    if (setsockopt(socket_rendezvous, SOL_SOCKET, SO_REUSEADDR, &enable_flag, sizeof(int)) < 0 ||
        setsockopt(socket_rendezvous, SOL_SOCKET, SO_REUSEPORT, &enable_flag, sizeof(int)) < 0) {
        error_exit_errno("Setting REUSE options failed: ");
    }
    if (setsockopt(socket_rendezvous, SOL_SOCKET, SO_RCVTIMEO, (const char*)&timeout, sizeof timeout) < 0 ||
        setsockopt(socket_rendezvous, SOL_SOCKET, SO_REUSEPORT, &enable_flag, sizeof(int)) < 0) {
        error_exit_errno("Setting timeout failed: ");
    }

    server_data.sin_family = AF_INET;
    server_data.sin_addr.s_addr = inet_addr(server_address.c_str());
    server_data.sin_port = htons(port);

    if (connect(socket_rendezvous, (struct sockaddr *)&server_data, sizeof(server_data)) != 0) {
        error_exit_errno("Connection with the rendezvous server failed: ");
    }

    if(send(socket_rendezvous, pairing_name.c_str(), pairing_name.length(), MSG_DONTWAIT) == -1) {
        error_exit_errno("Failed to send data to rendezvous server: ");
    }
}


int pair(const std::string& pairing_name, const std::string& server_address, int port, int timeout_ms) {
    connection_established = false;
    accepting_socket = -1;
    struct timeval timeout;
    timeout.tv_sec = timeout_ms / 1000;
    timeout.tv_usec = (timeout_ms % 1000) * 1000;

    int socket_rendezvous;
    struct sockaddr_in server_data{};

    socket_rendezvous = socket(AF_INET, SOCK_STREAM, 0);
    if (socket_rendezvous == -1) {
        error_exit_errno("Could not create socket for rendezvous server: ");
    }

    // Enable binding multiple sockets to the same local endpoint, see https://bford.info/pub/net/p2pnat/ for details
    int enable_flag = 1;
    if (setsockopt(socket_rendezvous, SOL_SOCKET, SO_REUSEADDR, &enable_flag, sizeof(int)) < 0 ||
        setsockopt(socket_rendezvous, SOL_SOCKET, SO_REUSEPORT, &enable_flag, sizeof(int)) < 0) {
        error_exit_errno("Setting REUSE options failed: ");
    }
    if (setsockopt(socket_rendezvous, SOL_SOCKET, SO_RCVTIMEO, (const char*)&timeout, sizeof timeout) < 0 ||
        setsockopt(socket_rendezvous, SOL_SOCKET, SO_REUSEPORT, &enable_flag, sizeof(int)) < 0) {
        error_exit_errno("Setting timeout failed: ");
    }

    server_data.sin_family = AF_INET;
    server_data.sin_addr.s_addr = inet_addr(server_address.c_str());
    server_data.sin_port = htons(port);

    if (connect(socket_rendezvous, (struct sockaddr *)&server_data, sizeof(server_data)) != 0) {
        error_exit_errno("Connection with the rendezvous server failed: ");
    }

    if(send(socket_rendezvous, pairing_name.c_str(), pairing_name.length(), MSG_DONTWAIT) == -1) {
        error_exit_errno("Failed to send data to rendezvous server: ");
    }

    PeerConnectionData public_info;
    ssize_t bytes = recv(socket_rendezvous, &public_info, sizeof(public_info), MSG_WAITALL);
    if (bytes == -1) {
        error_exit_errno("Failed to get data from rendezvous server: ");
    } else if(bytes == 0) {
        error_exit("Server has disconnected");
    }

    pthread_t peer_listen_thread;
    int thread_return = pthread_create(&peer_listen_thread, nullptr, peer_listen, (void*) &public_info);
    if(thread_return) {
        error_exit_errno("Error when creating thread for listening: ");
    }

    PeerConnectionData peer_data;

    // Wait until rendezvous server sends info about peer
    ssize_t bytes_received = recv(socket_rendezvous, &peer_data, sizeof(peer_data), MSG_WAITALL);
    if(bytes_received == -1) {
        error_exit_errno("Failed to get peer data from rendezvous server: ");
    } else if(bytes_received == 0) {
        error_exit("Server has disconnected when waiting for peer data");
    }
#if DEBUG
    std::cout << "Peer: " << ip_to_string(&peer_data.ip.s_addr) << ":" << ntohs(peer_data.port) << std::endl;
#endif

    //We do NOT close the socket_rendezvous socket here, otherwise the next binds sometimes fail (although SO_REUSEADDR|SO_REUSEPORT is set)!

    int peer_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (setsockopt(peer_socket, SOL_SOCKET, SO_REUSEADDR, &enable_flag, sizeof(int)) < 0 ||
        setsockopt(peer_socket, SOL_SOCKET, SO_REUSEPORT, &enable_flag, sizeof(int)) < 0) {
        error_exit("Setting REUSE options failed");
    }

    //Set socket to non blocking for the following polling operations
    if(fcntl(peer_socket, F_SETFL, O_NONBLOCK) != 0) {
        error_exit_errno("Setting O_NONBLOCK failed: ");
    }

    struct sockaddr_in local_port_addr = {0};
    local_port_addr.sin_family = AF_INET;
    local_port_addr.sin_addr.s_addr = INADDR_ANY;
    local_port_addr.sin_port = public_info.port;

    if (bind(peer_socket, (const struct sockaddr *)&local_port_addr, sizeof(local_port_addr))) {
        error_exit_errno("Binding to same port failed");
    }

    struct sockaddr_in peer_addr = {0};
    peer_addr.sin_family = AF_INET;
    peer_addr.sin_addr.s_addr = peer_data.ip.s_addr;
    peer_addr.sin_port = peer_data.port;

    auto start_time = std::chrono::steady_clock::now();
    auto max_connection_time = std::chrono::milliseconds(timeout_ms > 0 ? timeout_ms : 30000);
    int attempt_count = 0;
    const int max_attempts = 100;

    while(!connection_established.load() && attempt_count < max_attempts) {
        if (std::chrono::steady_clock::now() - start_time > max_connection_time) {
            LOG(INFO) << "Max connection time exceeded....returning timeout";
            return -1; // Timeout
        }
        
        int peer_status = connect(peer_socket, (struct sockaddr *)&peer_addr, sizeof(struct sockaddr));
        if (peer_status != 0) {
            if (errno == EALREADY || errno == EAGAIN || errno == EINPROGRESS) {
                attempt_count++;
                continue;
            } else if(errno == EISCONN) {
                LOG(INFO) << "Succesfully connected to peer, EISCONN" << std::endl;
                break;
            } else {
                int base_delay = 100;
                int backoff_delay = base_delay * (1 + attempt_count / 10);
                std::this_thread::sleep_for(std::chrono::milliseconds(std::min(backoff_delay, 1000)));
                attempt_count++;
                continue;
            }
        } else {

            LOG(INFO) << "Succesfully connected to peer, peer_status" << std::endl;
            break;
        }
    }

    if (attempt_count >= max_attempts) {
        return -1; // Timeout after max attempts
    }

    if(connection_established.load()) {
        pthread_join(peer_listen_thread, nullptr);
        peer_socket = accepting_socket.load();
    }

    int flags = fcntl(peer_socket,  F_GETFL, 0);
    flags &= ~(O_NONBLOCK);
    fcntl(peer_socket, F_SETFL, flags);

    // Validation handshake to ensure both sides connected successfully
    ValidationMsg validation_msg;
    validation_msg.peer_id = 0; // Will be set by caller if needed
    validation_msg.timestamp = static_cast<uint32_t>(std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());

    // Set validation timeout (15 seconds for AWS Fargate environments)
    struct timeval validation_timeout;
    validation_timeout.tv_sec = 15;
    validation_timeout.tv_usec = 0;
    if (setsockopt(peer_socket, SOL_SOCKET, SO_RCVTIMEO, &validation_timeout, sizeof(validation_timeout)) < 0 ||
        setsockopt(peer_socket, SOL_SOCKET, SO_SNDTIMEO, &validation_timeout, sizeof(validation_timeout)) < 0) {
#if DEBUG
        std::cout << "Warning: Failed to set validation timeout" << std::endl;
#endif
    }

    // Send validation message
    ssize_t sent = send(peer_socket, &validation_msg, sizeof(validation_msg), 0);
    if (sent != sizeof(validation_msg)) {
        LOG(INFO) << "Validation handshake failed: could not send validation message for pair: " << pairing_name;
        close(peer_socket);
        return -2; // Validation failure
    }

    // Receive peer's validation message
    ValidationMsg peer_validation;
    ssize_t received = recv(peer_socket, &peer_validation, sizeof(peer_validation), 0);
    if (received != sizeof(peer_validation) || peer_validation.magic != 0xDEADBEEF) {
        LOG(INFO) << "Validation handshake failed: invalid or missing peer validation for pair: " << pairing_name;

        close(peer_socket);
        return -2; // Validation failure
    }


    LOG(INFO) << "Validation handshake completed successfully for pair: " << pairing_name;


    return peer_socket;
}
