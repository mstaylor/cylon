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

#ifndef CYLON_DIRECTBACKEND_HPP
#define CYLON_DIRECTBACKEND_HPP

#include <string>
#include "Backends.hpp"

namespace FMI::Utils {
    class DirectBackend : public Backends {

    private:
        bool resolve_host_dns = false;

    public:
        DirectBackend() = default;

        std::string getName() override;

        BackendType getBackendType() override;

        /**
            * Enabled the resolve dns
            * @param enable
        */
        Backends * setResolveBackendDNS(bool do_resolve);


        bool resolveHostDNS() const;

    };

}

#endif //CYLON_DIRECTBACKEND_HPP
