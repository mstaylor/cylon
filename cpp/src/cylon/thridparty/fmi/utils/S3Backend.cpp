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

#include "S3Backend.hpp"

FMI::Utils::Backends *FMI::Utils::S3Backend::withS3BacketName(char *bucket) {
    this->bucket_name = bucket;
    return this;
}

FMI::Utils::Backends *FMI::Utils::S3Backend::withAWSRegion(char *region) {
    this->region = region;
    return this;
}

std::string FMI::Utils::S3Backend::getBacketName() {
    return this->bucket_name;
}

std::string FMI::Utils::S3Backend::getAWSRegion() {
    return this->region;
}

std::string FMI::Utils::S3Backend::getName() {
    return "S3";
}

FMI::Utils::BackendType FMI::Utils::S3Backend::getBackendType() {
    return S3;
}
