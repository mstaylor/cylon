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
#include "Common.hpp"

namespace FMI::Utils {
    class DirectBackend : public Backends {

    private:
        bool resolve_host_dns = false;

        bool enable_host_ping = false;

        bool use_direct_redis = false;

        std::string advertise_host = "";

        Mode blockingMode = BLOCKING;

    public:
        DirectBackend() = default;

        std::string getName() override;

        BackendType getBackendType() override;

        Mode getBlockingMode();

        /**
            * Enabled the resolve dns
            * @param enable
        */
        Backends * setResolveBackendDNS(bool do_resolve);

        Backends * setEnableHostPing(bool do_enable);

        Backends * setUseDirectRedis(bool use_it);

        /**
            * Address to advertise to peers for the direct-redis channel, distinct
            * from the base Backends::host (which serves TCPunch's rendezvous-host
            * meaning). Left unset (empty) to let direct-redis fall through to ECS
            * metadata auto-discovery.
        */
        Backends * setAdvertiseHost(const char * host);

        Backends * setBlockingMode(Mode blockingMode);


        bool resolveHostDNS() const;

        bool enableHostPing() const;

        bool useDirectRedis() const;

        std::string getAdvertiseHost() const;

    };

}

#endif //CYLON_DIRECTBACKEND_HPP
