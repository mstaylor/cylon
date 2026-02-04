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
#include <hiredis/async.h>
#endif

namespace FMI::Comm {
    //! Channel that uses Redis with the Hiredis client library as storage backend.
    #ifdef BUILD_CYLON_REDIS

    //! Async operation tracking for non-blocking Redis operations (similar to IOState in Direct.cpp)
    struct RedisAsyncOp {
        std::shared_ptr<channel_data> request;
        std::string object_name;
        Utils::Operation op_type;  // SEND (upload) or RECEIVE (download)
        std::function<void(Utils::NbxStatus, const std::string&, Utils::fmiContext*)> callbackResult;
        Utils::fmiContext* context = nullptr;
        std::chrono::steady_clock::time_point deadline;
        bool completed = false;
        bool success = false;
        std::string error_message;
    };

    class Redis : public ClientServer {
    public:
        explicit Redis(const std::shared_ptr<FMI::Utils::Backends> &backend);

        virtual ~Redis();

        //! Initialize async context for non-blocking operations
        void init() override;

        //! Blocking upload
        void upload_object(const std::shared_ptr<channel_data> buf, std::string name) override;

        //! Blocking download
        bool download_object(const std::shared_ptr<channel_data> buf, std::string name) override;

        void delete_object(std::string name) override;

        std::vector<std::string> get_object_names() override;

        //! Process pending async operations - polls Redis and completes ready operations
        Utils::EventProcessStatus channel_event_progress(Utils::Operation op) override;

        //! Start async upload operation
        void upload_object_async(const std::shared_ptr<channel_data> buf,
                                 const std::string& name,
                                 Utils::fmiContext* context,
                                 std::function<void(Utils::NbxStatus, const std::string&, Utils::fmiContext*)> callback);

        //! Start async download operation
        void download_object_async(const std::shared_ptr<channel_data> buf,
                                   const std::string& name,
                                   Utils::fmiContext* context,
                                   std::function<void(Utils::NbxStatus, const std::string&, Utils::fmiContext*)> callback);

        //! Check if there are pending async operations
        bool has_pending_operations() const;

    private:
        redisContext* context;  // Blocking context
        redisAsyncContext* async_context = nullptr;  // Async context

        // Async operation tracking - keyed by operation ID (similar to io_states in Direct.cpp)
        std::unordered_map<uint64_t, std::shared_ptr<RedisAsyncOp>> pending_ops;
        uint64_t next_op_id = 0;

        // Hostname and port for reconnection
        std::string redis_hostname;
        int redis_port;

        //! Initialize async context
        bool init_async_context();

        //! Process read/write events on async context
        void process_async_events();

        //! Handle completed operation
        void handle_completed_op(uint64_t op_id, bool success, const std::string& error_msg,
                                 const char* data = nullptr, size_t data_len = 0);

        //! Static callback for async SET commands
        static void set_callback(redisAsyncContext* c, void* reply, void* privdata);

        //! Static callback for async GET commands
        static void get_callback(redisAsyncContext* c, void* reply, void* privdata);
    };
    #endif
}

#endif //CYLON_REDIS_HPP
