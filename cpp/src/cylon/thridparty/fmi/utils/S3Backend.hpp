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

#ifndef CYLON_S3BACKEND_HPP
#define CYLON_S3BACKEND_HPP

#include <string>
#include "Backends.hpp"

namespace FMI::Utils {
    class S3Backend : public Backends {
    private:
        std::string bucket_name;
        std::string region;
    public:
        S3Backend() = default;

        /**
        * Set the bucket name
        * @param the host to set
        */
        Backends * withS3BacketName(char * bucket);

        /**
        * Set the bucket name
        * @param the host to set
        */
        Backends * withAWSRegion(char * region);



    };

}

#endif //CYLON_S3BACKEND_HPP
