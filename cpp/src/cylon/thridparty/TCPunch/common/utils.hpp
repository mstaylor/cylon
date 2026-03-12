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

#ifndef CYLON_UTILS_HPP
#define CYLON_UTILS_HPP

#include <arpa/inet.h>
#include <exception>
#include "../client/tcpunch.hpp"
#include <glog/logging.h>

typedef struct {
    struct in_addr ip;
    in_port_t      port;
} PeerConnectionData;

void error_exit(const std::string& error_string) {
    throw std::runtime_error{error_string};
}

void error_exit_errno(const std::string& error_string) {
    if (errno == EAGAIN) {
        LOG(INFO) << "error_exit_error: timeout - errorMsg: " << error_string;
        throw Timeout();
    } else {
        std::string err = error_string + strerror(errno);
        LOG(INFO) << "error_exit_error: " << err;
        throw std::runtime_error{err};
    }
}

std::string ip_to_string(in_addr_t *ip) {
    char str_buffer[20];
    inet_ntop(AF_INET, ip, str_buffer, sizeof(str_buffer));
    return {str_buffer};
}

#endif //CYLON_UTILS_HPP
