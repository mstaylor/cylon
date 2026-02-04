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

// Return values: socket fd on success, throws Timeout on timeout, throws ValidationFailure on validation failure
int pair(const std::string& pairing_name, const std::string& server_address, int port = 10000, int timeout_ms = 0);

void remove_pair(const std::string& pairing_name, const std::string& server_address, int port = 10000, int timeout_ms = 0);
#endif //CYLON_TCPUNCH_HPP
