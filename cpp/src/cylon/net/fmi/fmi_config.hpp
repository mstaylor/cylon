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

        public:
            FMIOptions();

            /**
            * Change the default delimiter(",")
            * @param delimiter character representing the delimiter
            */
            FMIOptions WithRendezvousHost(char delimiter);
        };

    }

}
#endif //CYLON_FMI_CONFIG_HPP
