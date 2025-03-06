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

#ifndef CYLON_REDIS_HPP
#define CYLON_REDIS_HPP

#include "ClientServer.hpp"
#include <map>
#include <string>
#ifdef BUILD_CYLON_REDIS
#include <hiredis/hiredis.h>
#endif

namespace FMI::Comm {
    //! Channel that uses Redis with the Hiredis client library as storage backend.
    #ifdef BUILD_CYLON_REDIS
    class Redis : public ClientServer {
    public:
        explicit Redis(std::shared_ptr<FMI::Utils::Backends> &backend);

        virtual ~Redis();

        void upload_object(const channel_data &buf, std::string name) override;

        bool download_object(const channel_data &buf, std::string name) override;

        void delete_object(std::string name) override;

        std::vector<std::string> get_object_names() override;


    private:
        redisContext* context;
        // Model params

    };
    #endif
}

#endif //CYLON_REDIS_HPP
