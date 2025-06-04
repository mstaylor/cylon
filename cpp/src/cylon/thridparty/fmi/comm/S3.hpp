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

#ifndef CYLON_S3_HPP
#define CYLON_S3_HPP

#include "ClientServer.hpp"
#include <map>
#include <string>
#include <aws/s3/S3Client.h>
#include <aws/core/Aws.h>
#include "../utils/Backends.hpp"

namespace FMI::Comm {
    //! Channel that uses AWS S3 as backend and uses the AWS SDK for C++ to access S3.
    class S3 : public ClientServer {
    public:
        explicit S3(const std::shared_ptr<FMI::Utils::Backends> &backend);

        virtual ~S3();

        void upload_object(const std::shared_ptr<channel_data> buf, std::string name) override;

        bool download_object(const std::shared_ptr<channel_data> buf, std::string name) override;

        void delete_object(std::string name) override;

        std::vector<std::string> get_object_names() override;


    private:
        std::string bucket_name;
        std::unique_ptr<Aws::S3::S3Client, Aws::Deleter<Aws::S3::S3Client>> client;
        Aws::SDKOptions options;
        //! Only one AWS SDK InitApi is allowed per application, we therefore track the number of instances (for multiple communicators) and call InitApi / ShutdownApi only on the first / last instance.
        inline static int instances = 0;

    };
}


#endif //CYLON_S3_HPP
