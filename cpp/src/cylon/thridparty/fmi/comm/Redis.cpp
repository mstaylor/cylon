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
#include "Redis.hpp"

FMI::Comm::Redis::Redis(std::shared_ptr<FMI::Utils::Backends> &backend) : ClientServer(backend) {

}

FMI::Comm::Redis::~Redis() {

}

void FMI::Comm::Redis::upload_object(channel_data buf, std::string name) {

}

bool FMI::Comm::Redis::download_object(channel_data buf, std::string name) {
    return false;
}

void FMI::Comm::Redis::delete_object(std::string name) {

}

std::vector<std::string> FMI::Comm::Redis::get_object_names() {
    return std::vector<std::string>();
}

