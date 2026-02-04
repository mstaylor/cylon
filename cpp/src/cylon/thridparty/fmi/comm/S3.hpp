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
#include <vector>
#include <memory>
#include <chrono>
#include <functional>
#include <unordered_map>
#include <aws/s3/S3Client.h>
#include <aws/core/Aws.h>
#include "../utils/Backends.hpp"

namespace FMI::Comm {

    //! Async operation tracking for non-blocking S3 operations (similar to IOState in Direct.cpp)
    struct S3AsyncOp {
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

    //! Channel that uses AWS S3 as backend and uses the AWS SDK for C++ to access S3.
    class S3 : public ClientServer {
    public:
        explicit S3(const std::shared_ptr<FMI::Utils::Backends> &backend);

        virtual ~S3();

        //! Initialize for async operations
        void init() override;

        //! Blocking upload
        void upload_object(const std::shared_ptr<channel_data> buf, std::string name) override;

        //! Blocking download
        bool download_object(const std::shared_ptr<channel_data> buf, std::string name) override;

        void delete_object(std::string name) override;

        std::vector<std::string> get_object_names() override;

        //! Process pending async operations - checks completion of async S3 calls
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
        std::string bucket_name;
        std::unique_ptr<Aws::S3::S3Client, Aws::Deleter<Aws::S3::S3Client>> client;
        Aws::SDKOptions options;
        //! Only one AWS SDK InitApi is allowed per application, we therefore track the number of instances (for multiple communicators) and call InitApi / ShutdownApi only on the first / last instance.
        inline static int instances = 0;

        // Async operation tracking - keyed by operation ID (similar to io_states in Direct.cpp)
        std::unordered_map<uint64_t, std::shared_ptr<S3AsyncOp>> pending_ops;
        uint64_t next_op_id = 0;

        //! Handle completed operation
        void handle_completed_op(uint64_t op_id, bool success, const std::string& error_msg);
    };
}


#endif //CYLON_S3_HPP
