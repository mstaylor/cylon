//
// Created by parallels on 7/2/24.
//

#ifndef CYLON_FMIOPTIONS_HPP
#define CYLON_FMIOPTIONS_HPP

#include <string>
#include <vector>
#include <memory>
#include <map>
#include "Backends.hpp"

namespace FMI::Utils {



        class FMIOptions {
        private:

            std::map<BackendType, std::shared_ptr<Backends>> backends;
        public:
            FMIOptions() = default;

            FMIOptions * withBackend(BackendType &backendType, std::shared_ptr<Backends> &backend);

            std::shared_ptr<Backends> getBackend(BackendType backendType);

        };

    }
#endif //CYLON_FMIOPTIONS_HPP
