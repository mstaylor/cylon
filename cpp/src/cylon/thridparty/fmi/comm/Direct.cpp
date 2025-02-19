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

#include <tcpunch.h>
#include <sys/socket.h>
#include <thread>
#include <netinet/tcp.h>
#include <cmath>
#include <iostream>
#include <string>
#include <cstring>
#include <sys/types.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <memory>


FMI::Comm::Direct::Direct(std::shared_ptr<FMI::Utils::Backends> &backend) {
    struct addrinfo hints, *res, *p;
    int status;
    char ipstr[INET6_ADDRSTRLEN];

    hostname = params["host"];
    port = std::stoi(params["port"]);
    if (model_params["resolve_host_dns"] == "true") {

        memset(&hints, 0, sizeof hints);
        hints.ai_family = AF_INET;
        hints.ai_socktype = SOCK_STREAM;

        if ((status = getaddrinfo(hostname.c_str(), nullptr, &hints, &res)) != 0) {
            std::cerr << "getaddrinfo error: " << gai_strerror(status) << std::endl;
        } else {
            // Iterate through the result list and convert each address to a string
            for(p = res; p != nullptr; p = p->ai_next) {
                void *addr;

                // Get the pointer to the address itself,
                struct sockaddr_in *ipv4 = (struct sockaddr_in *)p->ai_addr;
                addr = &(ipv4->sin_addr);

                // Convert the IP to a string and print it:
                inet_ntop(p->ai_family, addr, ipstr, sizeof ipstr);
                std::cout << " resolved dns: " << ipstr << std::endl;
            }

            freeaddrinfo(res); // Free the linked list
            hostname = ipstr;

        }




        resolve_host_dns = true;
    } else {
        resolve_host_dns = false;
    }

    max_timeout = std::stoi(params["max_timeout"]);

    bandwidth = std::stod(model_params["bandwidth"]);




    if (model_params["include_infrastructure_costs"] == "true") {
        include_infrastructure_costs = true;
    } else {
        include_infrastructure_costs = false;
    }
}

