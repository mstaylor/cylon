//
// Created by parallels on 7/2/24.
//

#include <cylon/net/fmi/fmi_config.hpp>

namespace cylon {
    namespace net {

        FMIOptions FMIOptions::WithRendezvousHost(char * rendezvous_host) {
            this->rendezvous_host = rendezvous_host;
            return *this;
        }

    }
}
