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
#include <glog/logging.h>
#include <cstring>
#include <algorithm>

#ifdef BUILD_CYLON_REDIS

// =============================================================================
// Constructor / Destructor
// =============================================================================

FMI::Comm::Redis::Redis(const std::shared_ptr<FMI::Utils::Backends> &backend) : ClientServer(backend) {
    auto redisBackend = backend.get();

    redis_hostname = redisBackend->getHost();
    redis_port = redisBackend->getPort();

    // Initialize blocking context
    context = redisConnect(redis_hostname.c_str(), redis_port);
    if (context == nullptr || context->err) {
        if (context) {
            LOG(ERROR) << "Error when connecting to Redis: " << context->errstr;
        } else {
            LOG(ERROR) << "Allocating Redis context not possible";
        }
    }
}

FMI::Comm::Redis::~Redis() {
    if (context != nullptr) {
        redisFree(context);
        context = nullptr;
    }
}

// =============================================================================
// Initialization
// =============================================================================

void FMI::Comm::Redis::init() {
    // Nothing to initialize — we use the blocking Redis client for both
    // sync and async FMI operations. Async downloads are polled via
    // channel_event_progress.
}

// =============================================================================
// Blocking Operations
// =============================================================================

void FMI::Comm::Redis::upload_object(const std::shared_ptr<channel_data> buf, std::string name) {
    std::string command;
    if (key_ttl_seconds > 0) {
        command = "SET " + name + " %b EX " + std::to_string(key_ttl_seconds);
    } else {
        command = "SET " + name + " %b";
    }
    auto* reply = (redisReply*) redisCommand(context, command.c_str(), buf->buf.get(), buf->len);
    if (reply == nullptr) {
        LOG(ERROR) << "Error when uploading to Redis: null reply";
        return;
    }
    if (reply->type == REDIS_REPLY_ERROR) {
        LOG(ERROR) << "Error when uploading to Redis: " << reply->str;
    }
    freeReplyObject(reply);
}

bool FMI::Comm::Redis::download_object(const std::shared_ptr<channel_data> buf, std::string name) {
    std::string command = "GET " + name;
    auto* reply = (redisReply*) redisCommand(context, command.c_str());
    if (reply == nullptr) {
        LOG(ERROR) << "Error when downloading from Redis: null reply";
        return false;
    }
    if (reply->type == REDIS_REPLY_NIL || reply->type == REDIS_REPLY_ERROR) {
        freeReplyObject(reply);
        return false;
    } else {
        std::memcpy(buf->buf.get(), reply->str, std::min(buf->len, reply->len));
        freeReplyObject(reply);
        return true;
    }
}

void FMI::Comm::Redis::delete_object(std::string name) {
    std::string command = "DEL " + name;
    auto* reply = (redisReply*) redisCommand(context, command.c_str());
    if (reply != nullptr) {
        freeReplyObject(reply);
    }
}

std::vector<std::string> FMI::Comm::Redis::get_object_names() {
    std::vector<std::string> keys;
    // Use comm_name prefix to avoid scanning the entire keyspace
    std::string pattern = comm_name + "*";
    std::string command = "KEYS " + pattern;
    auto* reply = (redisReply*) redisCommand(context, command.c_str());
    if (reply == nullptr) {
        return keys;
    }
    for (size_t i = 0; i < reply->elements; i++) {
        keys.emplace_back(reply->element[i]->str);
    }
    freeReplyObject(reply);
    return keys;
}

// =============================================================================
// Async Operations (using blocking Redis client with FMI callback pattern)
// =============================================================================

void FMI::Comm::Redis::upload_object_async(
    const std::shared_ptr<channel_data> buf,
    const std::string& name,
    Utils::fmiContext* ctx,
    std::function<void(Utils::NbxStatus, const std::string&, Utils::fmiContext*)> callback) {

    // Blocking SET completes in microseconds — do it immediately and callback
    upload_object(buf, name);

    if (ctx) {
        ctx->completed = 1;
    }
    if (callback) {
        callback(Utils::SUCCESS, "", ctx);
    }
}

void FMI::Comm::Redis::download_object_async(
    const std::shared_ptr<channel_data> buf,
    const std::string& name,
    Utils::fmiContext* ctx,
    std::function<void(Utils::NbxStatus, const std::string&, Utils::fmiContext*)> callback) {

    // Try immediate download — the key might already exist
    if (download_object(buf, name)) {
        if (ctx) {
            ctx->completed = 1;
        }
        if (callback) {
            callback(Utils::SUCCESS, "", ctx);
        }
        return;
    }

    // Key not available yet — register as pending, polled on channel_event_progress
    auto pending = std::make_shared<RedisPendingDownload>();
    pending->buffer = buf;
    pending->key_name = name;
    pending->callback = callback;
    pending->context = ctx;
    pending->deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(max_timeout);

    uint64_t op_id = next_op_id++;
    pending_downloads[op_id] = pending;
}

// =============================================================================
// Event Processing — poll pending downloads
// =============================================================================

FMI::Utils::EventProcessStatus FMI::Comm::Redis::channel_event_progress(Utils::Operation op) {
    if (pending_downloads.empty()) {
        return Utils::EMPTY;
    }

    auto now = std::chrono::steady_clock::now();
    std::vector<uint64_t> to_remove;

    for (auto& [op_id, pending] : pending_downloads) {
        // Try to download the key
        if (download_object(pending->buffer, pending->key_name)) {
            if (pending->context) {
                pending->context->completed = 1;
            }
            if (pending->callback) {
                pending->callback(Utils::SUCCESS, "", pending->context);
            }
            to_remove.push_back(op_id);
            continue;
        }

        // Check for timeout
        if (now >= pending->deadline) {
            LOG(ERROR) << "Redis async download timed out for key: " << pending->key_name;
            if (pending->context) {
                pending->context->completed = 1;
            }
            if (pending->callback) {
                pending->callback(Utils::RECEIVE_FAILED, "Download timed out", pending->context);
            }
            to_remove.push_back(op_id);
        }
    }

    for (uint64_t id : to_remove) {
        pending_downloads.erase(id);
    }

    return pending_downloads.empty() ? Utils::EMPTY : Utils::PROCESSING;
}

bool FMI::Comm::Redis::has_pending_operations() const {
    return !pending_downloads.empty();
}

#endif