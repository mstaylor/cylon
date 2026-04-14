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

#ifndef CYLON_TCPUNCH_HPP
#define CYLON_TCPUNCH_HPP

#include <iostream>
#include <string>
#include <cstdint>
#include <netinet/in.h>
#include <sys/socket.h>
#include <cstring>
#include <arpa/inet.h>
#include <exception>

#define DEBUG 1

struct Timeout : public std::exception {};
struct ValidationFailure : public std::exception {};

struct ValidationMsg {
    uint32_t magic = 0xDEADBEEF;
    uint32_t peer_id;
    uint32_t timestamp;
};

// ============================================================================
// Protocol v2 Constants and Types
// ============================================================================

/// Maximum length of pairing name field (including null terminator)
constexpr size_t MAX_PAIRING_NAME = 100;

/// Length of reconnection token field (UUID string + null)
constexpr size_t TOKEN_LENGTH = 37;

/// Client request size: name(100) + token(37) + flags(4) = 141 bytes
constexpr size_t CLIENT_REQUEST_SIZE = 141;

/// Server response size: status(1) + your_ip(4) + your_port(2) + peer_ip(4) + peer_port(2) + token(37) = 50 bytes
constexpr size_t SERVER_RESPONSE_SIZE = 50;

/// Pairing status returned by server
enum class PairingStatus : uint8_t {
    WAITING = 0,  // Registered, waiting for peer
    PAIRED  = 1,  // Peer found, proceed to hole punching
    TIMEOUT = 2,  // Server-side timeout, reconnect with token
    ERROR   = 3   // Invalid request/token, start fresh
};

/// Parsed server response (v2 protocol)
struct ServerResponse {
    PairingStatus status;
    uint32_t your_ip;     // network byte order
    uint16_t your_port;   // network byte order
    uint32_t peer_ip;     // network byte order
    uint16_t peer_port;   // network byte order
    char token[TOKEN_LENGTH];  // reconnection UUID (null-terminated)
};

/// Build a Protocol v2 client request (141 bytes)
void build_request(uint8_t* buf, const std::string& pairing_name,
                   const std::string& token = "");

/// Parse a Protocol v2 server response (51 bytes)
void parse_response(const uint8_t* buf, ServerResponse& resp);

// ============================================================================
// Public API (signatures unchanged for backward compatibility)
// ============================================================================

// Return values: socket fd on success, -1 timeout, -2 validation failure, -3 bind failure
int pair(const std::string& pairing_name, const std::string& server_address, int port = 10000, int timeout_ms = 0);

void remove_pair(const std::string& pairing_name, const std::string& server_address, int port = 10000, int timeout_ms = 0);
#endif //CYLON_TCPUNCH_HPP
