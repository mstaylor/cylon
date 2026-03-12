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

#ifndef CYLON_BACKENDS_HPP
#define CYLON_BACKENDS_HPP

#include <string>

namespace FMI::Utils {
    enum BackendType {
        S3, Redis, Direct
    };
    class Backends {
    private:
        bool enabled = false;
        std::string host = "";
        int port = -1;
        int max_timeout = -1;
        int timeout = -1;


    public:
        virtual ~Backends() = default;

        /**
            * Enabled the backend
            * @param enable
            */
        Backends * setEnabled(bool is_enabled);

        /**
        * Set the host
        * @param the host to set
        */
        Backends * withHost(const char * host);


        Backends * withPort(int port);

        /**
        * Set the max timeout
        * @param the max timeout to set
        */
        Backends * withMaxTimeout(int max_timeout);

        /**
        * Set the timeout
        * @param the timeout to set
        */
        Backends * withTimeout(int timeout);

        bool isEnabled();

        std::string getHost();

        int getTimeout();

        int getMaxTimeout();

        int getPort();

        virtual BackendType getBackendType();

        virtual std::string getName();


    };
}

#endif //CYLON_BACKENDS_HPP
