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

#include "S3.hpp"

#include <aws/core/auth/AWSCredentialsProvider.h>
#include <aws/s3/model/PutObjectRequest.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <aws/s3/model/DeleteObjectRequest.h>
#include <aws/s3/model/ListObjectsRequest.h>
#include <cstring>
#include <sstream>
#include <future>
#include "../utils/S3Backend.hpp"
#include <glog/logging.h>

char TAG[] = "S3Client";

// =============================================================================
// Extended S3AsyncOp to include futures for async operations
// =============================================================================

namespace FMI::Comm {
    // Extended struct to hold future for tracking async completion
    struct S3AsyncOpInternal : public S3AsyncOp {
        std::future<Aws::S3::Model::PutObjectOutcome> put_future;
        std::future<Aws::S3::Model::GetObjectOutcome> get_future;
        bool future_started = false;
        // Exponential backoff for GET retries on NoSuchKey
        int retry_count = 0;
        std::chrono::steady_clock::time_point next_retry_time;
        bool waiting_for_retry = false;
    };
}

// =============================================================================
// Constructor / Destructor
// =============================================================================

FMI::Comm::S3::S3(const std::shared_ptr<FMI::Utils::Backends> &backend) : ClientServer(backend) {
    auto s3backend = dynamic_cast<FMI::Utils::S3Backend *>(backend.get());

    if (instances == 0) {
        Aws::InitAPI(options);
    }
    instances++;
    bucket_name = s3backend->getBacketName();
    Aws::Client::ClientConfiguration config;
    config.region = s3backend->getAWSRegion();

    client = Aws::MakeUnique<Aws::S3::S3Client>(TAG, config);
}

FMI::Comm::S3::~S3() {
    pending_ops.clear();

    instances--;
    if (instances == 0) {
        Aws::ShutdownAPI(options);
    }
}

// =============================================================================
// Initialization
// =============================================================================

void FMI::Comm::S3::init() {
    // No special initialization needed - client is already configured
}

// =============================================================================
// Blocking Operations (existing)
// =============================================================================

bool FMI::Comm::S3::download_object(const std::shared_ptr<channel_data> buf, std::string name) {
    Aws::S3::Model::GetObjectRequest request;
    request.WithBucket(bucket_name).WithKey(name);
    auto outcome = client->GetObject(request);
    if (outcome.IsSuccess()) {
        auto& s = outcome.GetResult().GetBody();
        s.read(buf->buf.get(), buf->len);
        return true;
    } else {
        return false;
    }
}

void FMI::Comm::S3::upload_object(const std::shared_ptr<channel_data> buf, std::string name) {
    Aws::S3::Model::PutObjectRequest request;
    request.WithBucket(bucket_name).WithKey(name);

    auto data = std::make_shared<std::stringstream>(std::string(buf->buf.get(), buf->len));

    request.SetBody(data);
    auto outcome = client->PutObject(request);
    if (!outcome.IsSuccess()) {
        LOG(ERROR) << "Error when uploading to S3: " << outcome.GetError();
    }
}

void FMI::Comm::S3::delete_object(std::string name) {
    Aws::S3::Model::DeleteObjectRequest request;
    request.WithBucket(bucket_name).WithKey(name);
    auto outcome = client->DeleteObject(request);
    if (!outcome.IsSuccess()) {
        LOG(ERROR) << "Error when deleting from S3: " << outcome.GetError();
    }
}

std::vector<std::string> FMI::Comm::S3::get_object_names() {
    std::vector<std::string> object_names;
    Aws::S3::Model::ListObjectsRequest request;
    request.WithBucket(bucket_name);
    // Filter by comm_name prefix to avoid scanning entire bucket
    request.SetPrefix(comm_name);

    // Paginate through all results (ListObjects returns max 1000 per call)
    bool has_more = true;
    while (has_more) {
        auto outcome = client->ListObjects(request);
        if (outcome.IsSuccess()) {
            auto& result = outcome.GetResult();
            for (auto& object : result.GetContents()) {
                object_names.push_back(object.GetKey());
            }
            has_more = result.GetIsTruncated();
            if (has_more) {
                // Set marker to last key for next page
                request.SetMarker(object_names.back());
            }
        } else {
            LOG(ERROR) << "Error when listing objects from S3: " << outcome.GetError();
            has_more = false;
        }
    }
    return object_names;
}

// =============================================================================
// Async Operations
// =============================================================================

void FMI::Comm::S3::upload_object_async(
    const std::shared_ptr<channel_data> buf,
    const std::string& name,
    Utils::fmiContext* ctx,
    std::function<void(Utils::NbxStatus, const std::string&, Utils::fmiContext*)> callback) {

    auto op = std::make_shared<S3AsyncOpInternal>();
    op->request = buf;
    op->object_name = name;
    op->op_type = Utils::SEND;
    op->callbackResult = callback;
    op->context = ctx;
    op->deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(max_timeout);
    op->completed = false;
    op->success = false;

    uint64_t op_id = next_op_id++;

    // Create PUT request
    Aws::S3::Model::PutObjectRequest request;
    request.WithBucket(bucket_name).WithKey(name);

    auto data = std::make_shared<std::stringstream>(std::string(buf->buf.get(), buf->len));
    request.SetBody(data);

    // Launch async operation using Callable (returns future)
    op->put_future = client->PutObjectCallable(request);
    op->future_started = true;

    pending_ops[op_id] = op;
}

void FMI::Comm::S3::download_object_async(
    const std::shared_ptr<channel_data> buf,
    const std::string& name,
    Utils::fmiContext* ctx,
    std::function<void(Utils::NbxStatus, const std::string&, Utils::fmiContext*)> callback) {

    auto op = std::make_shared<S3AsyncOpInternal>();
    op->request = buf;
    op->object_name = name;
    op->op_type = Utils::RECEIVE;
    op->callbackResult = callback;
    op->context = ctx;
    op->deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(max_timeout);
    op->completed = false;
    op->success = false;

    uint64_t op_id = next_op_id++;

    // Create GET request
    Aws::S3::Model::GetObjectRequest request;
    request.WithBucket(bucket_name).WithKey(name);

    // Launch async operation using Callable (returns future)
    op->get_future = client->GetObjectCallable(request);
    op->future_started = true;

    pending_ops[op_id] = op;
}

// =============================================================================
// Event Processing
// =============================================================================

void FMI::Comm::S3::handle_completed_op(uint64_t op_id, bool success, const std::string& error_msg) {
    auto it = pending_ops.find(op_id);
    if (it == pending_ops.end()) {
        LOG(WARNING) << "S3 Operation " << op_id << " not found in pending operations";
        return;
    }

    auto& op = it->second;
    op->completed = true;
    op->success = success;
    op->error_message = error_msg;
}

FMI::Utils::EventProcessStatus FMI::Comm::S3::channel_event_progress(Utils::Operation op) {
    if (pending_ops.empty()) {
        return Utils::EMPTY;
    }

    auto now = std::chrono::steady_clock::now();
    std::vector<uint64_t> to_remove;

    for (auto& [op_id, pending_op] : pending_ops) {
        // Skip already completed
        if (pending_op->completed) {
            if (pending_op->callbackResult) {
                Utils::NbxStatus status = pending_op->success ? Utils::SUCCESS :
                    (pending_op->op_type == Utils::SEND ? Utils::SEND_FAILED : Utils::RECEIVE_FAILED);
                pending_op->callbackResult(status, pending_op->error_message, pending_op->context);
            }
            to_remove.push_back(op_id);
            continue;
        }

        // Check timeout
        if (now >= pending_op->deadline) {
            pending_op->completed = true;
            pending_op->success = false;
            pending_op->error_message = "Operation timed out";
            continue;
        }

        // Cast to internal type to access futures
        auto internal_op = std::static_pointer_cast<S3AsyncOpInternal>(pending_op);

        if (!internal_op->future_started) {
            continue;
        }

        // If waiting for backoff before retrying GET, check if it's time
        if (internal_op->waiting_for_retry) {
            if (now < internal_op->next_retry_time) {
                continue;  // Not yet time to retry
            }
            // Time to retry — re-launch GET
            Aws::S3::Model::GetObjectRequest request;
            request.WithBucket(bucket_name).WithKey(internal_op->object_name);
            internal_op->get_future = client->GetObjectCallable(request);
            internal_op->waiting_for_retry = false;
            continue;  // Will check the future on next progress call
        }

        // Check PUT future (non-blocking)
        if (internal_op->op_type == Utils::SEND && internal_op->put_future.valid()) {
            auto status = internal_op->put_future.wait_for(std::chrono::milliseconds(0));
            if (status == std::future_status::ready) {
                auto outcome = internal_op->put_future.get();
                internal_op->completed = true;
                internal_op->success = outcome.IsSuccess();
                if (!outcome.IsSuccess()) {
                    internal_op->error_message = outcome.GetError().GetMessage();
                }
            }
        }

        // Check GET future (non-blocking)
        if (internal_op->op_type == Utils::RECEIVE && internal_op->get_future.valid()) {
            auto status = internal_op->get_future.wait_for(std::chrono::milliseconds(0));
            if (status == std::future_status::ready) {
                auto outcome = internal_op->get_future.get();
                if (outcome.IsSuccess()) {
                    auto& body = outcome.GetResult().GetBody();
                    body.read(internal_op->request->buf.get(), internal_op->request->len);
                    internal_op->completed = true;
                    internal_op->success = true;
                } else {
                    // Key may not exist yet (sender hasn't uploaded) — retry with backoff
                    auto error_type = outcome.GetError().GetErrorType();
                    if (error_type == Aws::S3::S3Errors::NO_SUCH_KEY ||
                        error_type == Aws::S3::S3Errors::RESOURCE_NOT_FOUND) {
                        // Exponential backoff: initial_ms, 2x, 4x, ... capped at max_ms
                        int backoff_ms = std::min(s3_retry_initial_ms * (1 << internal_op->retry_count), s3_retry_max_ms);
                        internal_op->retry_count++;
                        internal_op->next_retry_time = now + std::chrono::milliseconds(backoff_ms);
                        internal_op->waiting_for_retry = true;
                    } else {
                        // Non-recoverable error
                        internal_op->completed = true;
                        internal_op->success = false;
                        internal_op->error_message = outcome.GetError().GetMessage();
                    }
                }
            }
        }
    }

    // Remove completed operations
    for (uint64_t id : to_remove) {
        pending_ops.erase(id);
    }

    return pending_ops.empty() ? Utils::EMPTY : Utils::PROCESSING;
}

bool FMI::Comm::S3::has_pending_operations() const {
    return !pending_ops.empty();
}