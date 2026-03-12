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
#include <vector>
#include <memory>
#include <chrono>
#include <functional>
#include <unordered_map>

#ifdef BUILD_CYLON_REDIS
#include <hiredis/hiredis.h>
#endif

namespace FMI::Comm {
    //! Channel that uses Redis with the Hiredis client library as storage backend.
    //! Async operations use the blocking Redis client internally but expose the
    //! FMI async callback pattern. Downloads are polled on channel_event_progress.
    #ifdef BUILD_CYLON_REDIS

    //! Pending download operation — polled on channel_event_progress until the key appears
    struct RedisPendingDownload {
        std::shared_ptr<channel_data> buffer;
        std::string key_name;
        std::function<void(Utils::NbxStatus, const std::string&, Utils::fmiContext*)> callback;
        Utils::fmiContext* context = nullptr;
        std::chrono::steady_clock::time_point deadline;
    };

    class Redis : public ClientServer {
    public:
        explicit Redis(const std::shared_ptr<FMI::Utils::Backends> &backend);

        virtual ~Redis();

        void init() override;

        //! Blocking upload
        void upload_object(const std::shared_ptr<channel_data> buf, std::string name) override;

        //! Blocking download (single attempt)
        bool download_object(const std::shared_ptr<channel_data> buf, std::string name) override;

        void delete_object(std::string name) override;

        std::vector<std::string> get_object_names() override;

        //! Async upload: blocking SET + immediate callback (SET is fast)
        void upload_object_async(const std::shared_ptr<channel_data> buf,
                                 const std::string& name,
                                 Utils::fmiContext* context,
                                 std::function<void(Utils::NbxStatus, const std::string&, Utils::fmiContext*)> callback) override;

        //! Async download: registers pending download, polled on channel_event_progress
        void download_object_async(const std::shared_ptr<channel_data> buf,
                                   const std::string& name,
                                   Utils::fmiContext* context,
                                   std::function<void(Utils::NbxStatus, const std::string&, Utils::fmiContext*)> callback) override;

        //! Poll pending downloads and invoke callbacks for completed ones
        Utils::EventProcessStatus channel_event_progress(Utils::Operation op) override;

        bool has_pending_operations() const;

    private:
        redisContext* context;  // Blocking context

        // Hostname and port
        std::string redis_hostname;
        int redis_port;

        // Pending async downloads keyed by operation ID
        std::unordered_map<uint64_t, std::shared_ptr<RedisPendingDownload>> pending_downloads;
        uint64_t next_op_id = 0;
    };
    #endif
}

#endif //CYLON_REDIS_HPP
