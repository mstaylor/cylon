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
#include <unistd.h>
#include <pthread.h>
#include <cerrno>
#include <string>
#include <chrono>
#include <thread>
#include <stdexcept>
#include <iostream>
#include <atomic>
#include "../common/utils.hpp"

/// Maximum internal retries for protocol-level TIMEOUT/ERROR
static constexpr int MAX_PROTOCOL_RETRIES = 3;

// ============================================================================
// Protocol v2 Helpers
// ============================================================================

void build_request(uint8_t* buf, const std::string& pairing_name,
                   const std::string& token) {
    memset(buf, 0, CLIENT_REQUEST_SIZE);

    // Copy pairing name (max 99 chars + null terminator)
    size_t name_len = std::min(pairing_name.length(), MAX_PAIRING_NAME - 1);
    memcpy(buf, pairing_name.c_str(), name_len);

    // Copy reconnect token if present (offset 100, max 36 chars + null)
    if (!token.empty()) {
        size_t token_len = std::min(token.length(), TOKEN_LENGTH - 1);
        memcpy(buf + MAX_PAIRING_NAME, token.c_str(), token_len);
    }

    // Flags at offset 137 (4 bytes) — reserved, already zero
}

void parse_response(const uint8_t* buf, ServerResponse& resp) {
    // Status (1 byte at offset 0)
    resp.status = static_cast<PairingStatus>(buf[0]);

    // Your IP (4 bytes at offset 1, network byte order)
    memcpy(&resp.your_ip, buf + 1, 4);

    // Your port (2 bytes at offset 5, network byte order)
    memcpy(&resp.your_port, buf + 5, 2);

    // Peer IP (4 bytes at offset 7, network byte order)
    memcpy(&resp.peer_ip, buf + 7, 4);

    // Peer port (2 bytes at offset 11, network byte order)
    memcpy(&resp.peer_port, buf + 11, 2);

    // Token (37 bytes at offset 13, null-terminated)
    memcpy(resp.token, buf + 13, TOKEN_LENGTH);
    resp.token[TOKEN_LENGTH - 1] = '\0';
}

// ============================================================================
// Peer Listen Thread (unchanged — used for hole punching)
// ============================================================================

std::atomic<bool> connection_established(false);
std::atomic<int> accepting_socket(-1);

void* peer_listen(void* p) {
    auto* info = (PeerConnectionData*)p;

    // Create socket on the port that was previously used to contact the rendezvous server
    int listen_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_socket == -1) {
        LOG(ERROR) << "peer_listen: Socket creation failed: " << strerror(errno);
        return 0;
    }
    int enable_flag = 1;
    if (setsockopt(listen_socket, SOL_SOCKET, SO_REUSEADDR, &enable_flag, sizeof(int)) < 0 ||
        setsockopt(listen_socket, SOL_SOCKET, SO_REUSEPORT, &enable_flag, sizeof(int)) < 0) {
        LOG(ERROR) << "peer_listen: Setting REUSE options failed: " << strerror(errno);
        close(listen_socket);
        return 0;
    }

    // Short accept timeout so peer_listen checks connection_established frequently
    // and can exit promptly when the connect() path wins the race
    struct timeval accept_timeout;
    accept_timeout.tv_sec = 1;
    accept_timeout.tv_usec = 0;
    if (setsockopt(listen_socket, SOL_SOCKET, SO_RCVTIMEO, &accept_timeout, sizeof(accept_timeout)) < 0) {
        LOG(ERROR) << "peer_listen: Setting accept timeout failed: " << strerror(errno);
        close(listen_socket);
        return 0;
    }

    struct sockaddr_in local_port_data{};
    local_port_data.sin_family = AF_INET;
    local_port_data.sin_addr.s_addr = INADDR_ANY;
    local_port_data.sin_port = info->port;

    if (bind(listen_socket, (const struct sockaddr *)&local_port_data, sizeof(local_port_data)) < 0) {
        LOG(ERROR) << "peer_listen: Could not bind to local port: " << strerror(errno);
        close(listen_socket);
        return 0;
    }

    if (listen(listen_socket, 1) == -1) {
        LOG(ERROR) << "peer_listen: Listening on local port failed: " << strerror(errno);
        close(listen_socket);
        return 0;
    }

    struct sockaddr_in peer_info{};
    unsigned int len = sizeof(peer_info);
    int error_count = 0;

    while(true) {
        if (connection_established.load()) {
            break;
        }

        int peer = accept(listen_socket, (struct sockaddr*)&peer_info, &len);
        if (peer == -1) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                continue;
            }

            LOG(INFO) << "Error when connecting to peer: " << strerror(errno) << std::endl;

            // Add exponential backoff for persistent errors
            error_count++;
            if (error_count > 5) {
                int backoff_delay = std::min(100 * (1 << (error_count - 5)), 5000);
                std::this_thread::sleep_for(std::chrono::milliseconds(backoff_delay));
            }

        } else {
            LOG(INFO) << "Succesfully connected to peer, accepting" << std::endl;
            error_count = 0; // Reset error count on successful accept

            accepting_socket = peer;
            connection_established = true;
            close(listen_socket);
            return 0;
        }
    }
    close(listen_socket);
    return 0;
}

// ============================================================================
// Hole Punching (extracted from pair() — logic unchanged)
// ============================================================================

/// Perform hole punching given our public info and peer info.
/// Returns socket fd on success, -1 timeout, -2 validation failure, -3 bind failure.
static int do_hole_punch(const PeerConnectionData& public_info,
                         const PeerConnectionData& peer_data,
                         int socket_rendezvous,
                         const std::string& pairing_name,
                         int timeout_ms) {
    int enable_flag = 1;

    int peer_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (setsockopt(peer_socket, SOL_SOCKET, SO_REUSEADDR, &enable_flag, sizeof(int)) < 0 ||
        setsockopt(peer_socket, SOL_SOCKET, SO_REUSEPORT, &enable_flag, sizeof(int)) < 0) {
        error_exit("Setting REUSE options failed");
    }

    // Set socket to non blocking for the following polling operations
    if(fcntl(peer_socket, F_SETFL, O_NONBLOCK) != 0) {
        error_exit_errno("Setting O_NONBLOCK failed: ");
    }

    struct sockaddr_in local_port_addr = {0};
    local_port_addr.sin_family = AF_INET;
    local_port_addr.sin_addr.s_addr = INADDR_ANY;
    local_port_addr.sin_port = public_info.port;

    if (bind(peer_socket, (const struct sockaddr *)&local_port_addr, sizeof(local_port_addr))) {
        LOG(ERROR) << "pair: Binding to same port failed: " << strerror(errno);
        close(peer_socket);
        close(socket_rendezvous);
        connection_established = true;
        return -3;
    }

    struct sockaddr_in peer_addr = {0};
    peer_addr.sin_family = AF_INET;
    peer_addr.sin_addr.s_addr = peer_data.ip.s_addr;
    peer_addr.sin_port = peer_data.port;

    auto start_time = std::chrono::steady_clock::now();
    auto max_connection_time = std::chrono::milliseconds(timeout_ms > 0 ? timeout_ms : 30000);
    int attempt_count = 0;

    while(!connection_established.load()) {

        // Check overall timeout
        auto elapsed = std::chrono::steady_clock::now() - start_time;
        if (elapsed >= max_connection_time) {
            LOG(ERROR) << "pair: Connect loop timed out after "
                       << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << "ms";
            close(peer_socket);
            close(socket_rendezvous);
            connection_established = true;
            return -1;
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

    // Always signal peer_listen to exit and join the thread
    if (!connection_established.load()) {
        connection_established = true;
    }

    if (accepting_socket.load() >= 0) {
        // Connection was established via accept() — use that socket
        close(peer_socket);
        peer_socket = accepting_socket.load();
    }

    // Now safe to close socket_rendezvous — peer connection is established and holds the port
    close(socket_rendezvous);

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
        return -2;
    }

    // Receive peer's validation message
    ValidationMsg peer_validation;
    ssize_t received = recv(peer_socket, &peer_validation, sizeof(peer_validation), 0);
    if (received != sizeof(peer_validation) || peer_validation.magic != 0xDEADBEEF) {
        LOG(INFO) << "Validation handshake failed: invalid or missing peer validation for pair: " << pairing_name;
        close(peer_socket);
        return -2;
    }

    LOG(INFO) << "Validation handshake completed successfully for pair: " << pairing_name;

    return peer_socket;
}

// ============================================================================
// remove_pair — Protocol v2
// ============================================================================

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

    int enable_flag = 1;
    if (setsockopt(socket_rendezvous, SOL_SOCKET, SO_REUSEADDR, &enable_flag, sizeof(int)) < 0 ||
        setsockopt(socket_rendezvous, SOL_SOCKET, SO_REUSEPORT, &enable_flag, sizeof(int)) < 0) {
        error_exit_errno("Setting REUSE options failed: ");
    }
    if (setsockopt(socket_rendezvous, SOL_SOCKET, SO_RCVTIMEO, (const char*)&timeout, sizeof timeout) < 0) {
        error_exit_errno("Setting timeout failed: ");
    }

    server_data.sin_family = AF_INET;
    server_data.sin_addr.s_addr = inet_addr(server_address.c_str());
    server_data.sin_port = htons(port);

    if (connect(socket_rendezvous, (struct sockaddr *)&server_data, sizeof(server_data)) != 0) {
        error_exit_errno("Connection with the rendezvous server failed: ");
    }

    // Protocol v2: send fixed 141-byte request (no token, server will clean up after timeout)
    uint8_t req[CLIENT_REQUEST_SIZE];
    build_request(req, pairing_name);
    if (send(socket_rendezvous, req, CLIENT_REQUEST_SIZE, 0) == -1) {
        error_exit_errno("Failed to send data to rendezvous server: ");
    }

    close(socket_rendezvous);
}

// ============================================================================
// pair — Protocol v2
// ============================================================================

int pair(const std::string& pairing_name, const std::string& server_address, int port, int timeout_ms) {
    std::string reconnect_token;

    for (int attempt = 0; attempt < MAX_PROTOCOL_RETRIES; attempt++) {
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

        // Enable binding multiple sockets to the same local endpoint
        int enable_flag = 1;
        if (setsockopt(socket_rendezvous, SOL_SOCKET, SO_REUSEADDR, &enable_flag, sizeof(int)) < 0 ||
            setsockopt(socket_rendezvous, SOL_SOCKET, SO_REUSEPORT, &enable_flag, sizeof(int)) < 0) {
            error_exit_errno("Setting REUSE options failed: ");
        }
        if (setsockopt(socket_rendezvous, SOL_SOCKET, SO_RCVTIMEO, (const char*)&timeout, sizeof timeout) < 0) {
            error_exit_errno("Setting timeout failed: ");
        }

        server_data.sin_family = AF_INET;
        server_data.sin_addr.s_addr = inet_addr(server_address.c_str());
        server_data.sin_port = htons(port);

        if (connect(socket_rendezvous, (struct sockaddr *)&server_data, sizeof(server_data)) != 0) {
            error_exit_errno("Connection with the rendezvous server failed: ");
        }

        // Protocol v2: send fixed 141-byte request
        uint8_t req[CLIENT_REQUEST_SIZE];
        build_request(req, pairing_name, reconnect_token);
        if (send(socket_rendezvous, req, CLIENT_REQUEST_SIZE, 0) == -1) {
            error_exit_errno("Failed to send request to rendezvous server: ");
        }

        // Protocol v2: receive 51-byte response
        uint8_t resp_buf[SERVER_RESPONSE_SIZE];
        ssize_t bytes = recv(socket_rendezvous, resp_buf, SERVER_RESPONSE_SIZE, MSG_WAITALL);
        if (bytes == -1) {
            close(socket_rendezvous);
            error_exit_errno("Failed to get response from rendezvous server: ");
        } else if (bytes == 0) {
            close(socket_rendezvous);
            error_exit("Server has disconnected");
        } else if (bytes != SERVER_RESPONSE_SIZE) {
            close(socket_rendezvous);
            error_exit("Incomplete response from rendezvous server");
        }

        ServerResponse resp;
        parse_response(resp_buf, resp);

        // Save reconnect token for potential retry
        if (resp.token[0] != '\0') {
            reconnect_token = resp.token;
        }

        LOG(INFO) << "Rendezvous response: status=" << static_cast<int>(resp.status)
                  << ", token=" << resp.token;

        if (resp.status == PairingStatus::TIMEOUT) {
            LOG(INFO) << "Server timeout (attempt " << attempt + 1 << "), retrying with token";
            close(socket_rendezvous);
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            continue;
        }

        if (resp.status == PairingStatus::ERROR) {
            LOG(INFO) << "Server error (attempt " << attempt + 1 << "), clearing token and retrying";
            reconnect_token.clear();
            close(socket_rendezvous);
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            continue;
        }

        // Populate PeerConnectionData structs for the hole-punching code
        PeerConnectionData public_info;
        public_info.ip.s_addr = resp.your_ip;
        public_info.port = resp.your_port;

        PeerConnectionData peer_data;

        if (resp.status == PairingStatus::PAIRED) {
            // Got peer info immediately
            peer_data.ip.s_addr = resp.peer_ip;
            peer_data.port = resp.peer_port;

#if DEBUG
            std::cout << "Paired immediately. Peer: "
                      << ip_to_string(&peer_data.ip.s_addr) << ":" << ntohs(peer_data.port) << std::endl;
#endif
        } else {
            // status == WAITING — start listener thread, then wait for second response
            pthread_t peer_listen_thread;
            int thread_return = pthread_create(&peer_listen_thread, nullptr, peer_listen, (void*) &public_info);
            if (thread_return) {
                close(socket_rendezvous);
                error_exit_errno("Error when creating thread for listening: ");
            }

            // Wait for second 51-byte response with peer info
            bytes = recv(socket_rendezvous, resp_buf, SERVER_RESPONSE_SIZE, MSG_WAITALL);
            if (bytes == -1) {
                LOG(INFO) << "Timeout waiting for peer (attempt " << attempt + 1 << ")";
                close(socket_rendezvous);
                connection_established = true;
                pthread_join(peer_listen_thread, nullptr);
                continue;
            } else if (bytes == 0) {
                close(socket_rendezvous);
                connection_established = true;
                pthread_join(peer_listen_thread, nullptr);
                error_exit("Server has disconnected when waiting for peer data");
            }

            ServerResponse resp2;
            parse_response(resp_buf, resp2);

            if (resp2.status != PairingStatus::PAIRED) {
                LOG(INFO) << "Unexpected status after WAITING: " << static_cast<int>(resp2.status);
                close(socket_rendezvous);
                connection_established = true;
                pthread_join(peer_listen_thread, nullptr);
                continue;
            }

            peer_data.ip.s_addr = resp2.peer_ip;
            peer_data.port = resp2.peer_port;

#if DEBUG
            std::cout << "Peer: " << ip_to_string(&peer_data.ip.s_addr) << ":" << ntohs(peer_data.port) << std::endl;
#endif

            // Hole punch — do_hole_punch will join the listener thread via connection_established
            int result = do_hole_punch(public_info, peer_data, socket_rendezvous, pairing_name, timeout_ms);
            pthread_join(peer_listen_thread, nullptr);
            return result;
        }

        // PAIRED path — need to start listener thread for hole punching
        pthread_t peer_listen_thread;
        int thread_return = pthread_create(&peer_listen_thread, nullptr, peer_listen, (void*) &public_info);
        if (thread_return) {
            close(socket_rendezvous);
            error_exit_errno("Error when creating thread for listening: ");
        }

        int result = do_hole_punch(public_info, peer_data, socket_rendezvous, pairing_name, timeout_ms);
        pthread_join(peer_listen_thread, nullptr);
        return result;
    }

    // All retries exhausted
    throw Timeout();
}