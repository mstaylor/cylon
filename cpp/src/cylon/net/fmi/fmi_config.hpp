//
// Created by parallels on 7/2/24.
//

#ifndef CYLON_FMI_CONFIG_HPP
#define CYLON_FMI_CONFIG_HPP

#include <string>

namespace cylon {
    namespace net {
        class FMIOptions {
        private:
            std::string rendezvous_host = "cylon-rendezvous.aws-cylondata.com";
            int port = 10000;
            int maxTimeout = -1;

        public:
            FMIOptions();

            /**
            * Change the default host
            * @param delimiter character representing the delimiter
            */
            FMIOptions withRendezvousHost(char * rendezvous_host);

            FMIOptions withRendezvousPort(int rendezvouz_port);

            FMIOptions withMaxTimeout(int max_timeout);
        };

    }

}
#endif //CYLON_FMI_CONFIG_HPP
