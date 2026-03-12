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
#include "DirectBackend.hpp"
#include "Common.hpp"


std::string FMI::Utils::DirectBackend::getName() {
    return "Direct";
}

FMI::Utils::BackendType FMI::Utils::DirectBackend::getBackendType() {
    return Direct;
}

FMI::Utils::Backends *FMI::Utils::DirectBackend::setResolveBackendDNS(bool do_resolve) {
    this->resolve_host_dns = do_resolve;
    return this;
}

bool FMI::Utils::DirectBackend::resolveHostDNS() const {
    return this->resolve_host_dns;
}

FMI::Utils::Backends *FMI::Utils::DirectBackend::setBlockingMode(Mode blockingMode) {
    this->blockingMode = blockingMode;
    return this;
}

FMI::Utils::Mode FMI::Utils::DirectBackend::getBlockingMode() {
    return blockingMode;
}

FMI::Utils::Backends *FMI::Utils::DirectBackend::setEnableHostPing(bool do_enable) {
    this->enable_host_ping = do_enable;
    return this;
}

bool FMI::Utils::DirectBackend::enableHostPing() const {
    return enable_host_ping;
}
