//
// Created by parallels on 7/2/24.
//

#include <cylon/net/fmi/fmi_config.hpp>

namespace cylon {
    namespace net {

        FMIOptions FMIOptions::withRendezvousHost(char * rendezvous_host) {
            this->rendezvous_host = rendezvous_host;
            return *this;
        }

        FMIOptions FMIOptions::withRendezvousPort(int rendezvouz_port) {
            this->port = rendezvouz_port;
            return *this;
        }

        FMIOptions FMIOptions::withMaxTimeout(int max_timeout) {
            this->maxTimeout = max_timeout;
            return *this;
        }

    }
}
