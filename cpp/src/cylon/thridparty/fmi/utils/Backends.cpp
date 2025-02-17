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

#include "cylon/thridparty/fmi/utils/Backends.hpp"

namespace FMI::Utils{

    Backends * Backends::setEnabled(bool is_enabled) {
        this->enabled = is_enabled;
        return this;
    }

    Backends * Backends::withHost(char *host) {
        this->host = host;
        return this;
    }

    Backends * Backends::withMaxTimeout(int max_timeout) {
        this->maxTimeout = max_timeout;
        return this;
    }

    Backends * Backends::withTimeout(int timeout) {
        this->timeout = timeout;
        return this;
    }

    bool Backends::isEnabled() {
        return this->enabled;
    }

    std::string Backends::getHost() {
        return this->host;
    }

    int Backends::getTimeout() {
        return this->timeout;
    }

    int Backends::getMaxTimeout() {
        return this->maxTimeout;
    }
}
