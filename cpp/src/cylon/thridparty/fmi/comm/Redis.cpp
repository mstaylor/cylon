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
    // Free async context if initialized
    if (async_context != nullptr) {
        redisAsyncFree(async_context);
        async_context = nullptr;
    }

    // Free blocking context
    if (context != nullptr) {
        redisFree(context);
        context = nullptr;
    }
}

// =============================================================================
// Initialization
// =============================================================================

void FMI::Comm::Redis::init() {
    init_async_context();
}

bool FMI::Comm::Redis::init_async_context() {
    if (async_context != nullptr) {
        return true;  // Already initialized
    }

    async_context = redisAsyncConnect(redis_hostname.c_str(), redis_port);
    if (async_context == nullptr) {
        LOG(ERROR) << "Failed to allocate async Redis context";
        return false;
    }

    if (async_context->err) {
        LOG(ERROR) << "Async Redis connection error: " << async_context->errstr;
        redisAsyncFree(async_context);
        async_context = nullptr;
        return false;
    }

    // Store 'this' pointer for callbacks to access
    async_context->data = this;

    LOG(INFO) << "Redis async context initialized successfully";
    return true;
}

// =============================================================================
// Blocking Operations (existing)
// =============================================================================

void FMI::Comm::Redis::upload_object(const std::shared_ptr<channel_data> buf, std::string name) {
    std::string command = "SET " + name + " %b";
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
    std::string command = "KEYS *";
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
// Async Operations
// =============================================================================

void FMI::Comm::Redis::upload_object_async(
    const std::shared_ptr<channel_data> buf,
    const std::string& name,
    Utils::fmiContext* ctx,
    std::function<void(Utils::NbxStatus, const std::string&, Utils::fmiContext*)> callback) {

    // Ensure async context is initialized
    if (async_context == nullptr && !init_async_context()) {
        if (callback) {
            callback(Utils::SEND_FAILED, "Failed to initialize async context", ctx);
        }
        return;
    }

    // Create async operation tracking
    auto op = std::make_shared<RedisAsyncOp>();
    op->request = buf;
    op->object_name = name;
    op->op_type = Utils::SEND;
    op->callbackResult = callback;
    op->context = ctx;
    op->deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(max_timeout);
    op->completed = false;
    op->success = false;

    uint64_t op_id = next_op_id++;
    pending_ops[op_id] = op;

    // Issue async SET command - pass op_id as privdata
    uint64_t* op_id_ptr = new uint64_t(op_id);
    int status = redisAsyncCommand(async_context, set_callback, op_id_ptr,
                                   "SET %s %b", name.c_str(), buf->buf.get(), buf->len);

    if (status != REDIS_OK) {
        delete op_id_ptr;
        handle_completed_op(op_id, false, "Failed to issue async SET command");
        LOG(ERROR) << "Failed to issue async SET command for " << name;
    }
}

void FMI::Comm::Redis::download_object_async(
    const std::shared_ptr<channel_data> buf,
    const std::string& name,
    Utils::fmiContext* ctx,
    std::function<void(Utils::NbxStatus, const std::string&, Utils::fmiContext*)> callback) {

    // Ensure async context is initialized
    if (async_context == nullptr && !init_async_context()) {
        if (callback) {
            callback(Utils::RECEIVE_FAILED, "Failed to initialize async context", ctx);
        }
        return;
    }

    // Create async operation tracking
    auto op = std::make_shared<RedisAsyncOp>();
    op->request = buf;
    op->object_name = name;
    op->op_type = Utils::RECEIVE;
    op->callbackResult = callback;
    op->context = ctx;
    op->deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(max_timeout);
    op->completed = false;
    op->success = false;

    uint64_t op_id = next_op_id++;
    pending_ops[op_id] = op;

    // Issue async GET command
    uint64_t* op_id_ptr = new uint64_t(op_id);
    int status = redisAsyncCommand(async_context, get_callback, op_id_ptr,
                                   "GET %s", name.c_str());

    if (status != REDIS_OK) {
        delete op_id_ptr;
        handle_completed_op(op_id, false, "Failed to issue async GET command");
        LOG(ERROR) << "Failed to issue async GET command for " << name;
    }
}

// =============================================================================
// Async Callbacks (static)
// =============================================================================

void FMI::Comm::Redis::set_callback(redisAsyncContext* c, void* reply, void* privdata) {
    uint64_t* op_id_ptr = static_cast<uint64_t*>(privdata);
    if (op_id_ptr == nullptr) {
        return;
    }
    uint64_t op_id = *op_id_ptr;
    delete op_id_ptr;

    // Get the Redis instance from async context data
    Redis* self = static_cast<Redis*>(c->data);
    if (self == nullptr) {
        LOG(ERROR) << "Redis instance not found in async context";
        return;
    }

    redisReply* r = static_cast<redisReply*>(reply);
    if (r == nullptr) {
        self->handle_completed_op(op_id, false, "Null reply from Redis");
    } else if (r->type == REDIS_REPLY_ERROR) {
        self->handle_completed_op(op_id, false, r->str ? r->str : "Unknown error");
    } else {
        self->handle_completed_op(op_id, true, "");
    }
}

void FMI::Comm::Redis::get_callback(redisAsyncContext* c, void* reply, void* privdata) {
    uint64_t* op_id_ptr = static_cast<uint64_t*>(privdata);
    if (op_id_ptr == nullptr) {
        return;
    }
    uint64_t op_id = *op_id_ptr;
    delete op_id_ptr;

    // Get the Redis instance from async context data
    Redis* self = static_cast<Redis*>(c->data);
    if (self == nullptr) {
        LOG(ERROR) << "Redis instance not found in async context";
        return;
    }

    redisReply* r = static_cast<redisReply*>(reply);
    if (r == nullptr) {
        self->handle_completed_op(op_id, false, "Null reply from Redis");
    } else if (r->type == REDIS_REPLY_ERROR) {
        self->handle_completed_op(op_id, false, r->str ? r->str : "Unknown error");
    } else if (r->type == REDIS_REPLY_NIL) {
        self->handle_completed_op(op_id, false, "Key not found");
    } else if (r->type == REDIS_REPLY_STRING || r->type == REDIS_REPLY_STATUS) {
        // Copy data to buffer
        self->handle_completed_op(op_id, true, "", r->str, r->len);
    } else {
        self->handle_completed_op(op_id, false, "Unexpected reply type");
    }
}

// =============================================================================
// Event Processing
// =============================================================================

void FMI::Comm::Redis::handle_completed_op(uint64_t op_id, bool success,
                                            const std::string& error_msg,
                                            const char* data, size_t data_len) {
    auto it = pending_ops.find(op_id);
    if (it == pending_ops.end()) {
        LOG(WARNING) << "Operation " << op_id << " not found in pending operations";
        return;
    }

    auto& op = it->second;
    op->completed = true;
    op->success = success;
    op->error_message = error_msg;

    // For GET operations, copy data to buffer
    if (success && data != nullptr && data_len > 0 && op->op_type == Utils::RECEIVE) {
        size_t copy_len = std::min(op->request->len, data_len);
        std::memcpy(op->request->buf.get(), data, copy_len);
    }
}

FMI::Utils::EventProcessStatus FMI::Comm::Redis::channel_event_progress(Utils::Operation op) {
    if (pending_ops.empty()) {
        return Utils::EMPTY;
    }

    if (async_context == nullptr) {
        return Utils::NOOP;
    }

    // Process async events (calls our callbacks synchronously)
    process_async_events();

    // Check for timeouts and invoke callbacks for completed operations
    auto now = std::chrono::steady_clock::now();
    std::vector<uint64_t> to_remove;

    for (auto& [op_id, pending_op] : pending_ops) {
        // Check for timeout
        if (!pending_op->completed && now >= pending_op->deadline) {
            pending_op->completed = true;
            pending_op->success = false;
            pending_op->error_message = "Operation timed out";
        }

        // Invoke callback for completed operations
        if (pending_op->completed) {
            if (pending_op->callbackResult) {
                Utils::NbxStatus status = pending_op->success ? Utils::SUCCESS :
                    (pending_op->op_type == Utils::SEND ? Utils::SEND_FAILED : Utils::RECEIVE_FAILED);
                pending_op->callbackResult(status, pending_op->error_message, pending_op->context);
            }
            to_remove.push_back(op_id);
        }
    }

    // Remove completed operations
    for (uint64_t id : to_remove) {
        pending_ops.erase(id);
    }

    return pending_ops.empty() ? Utils::EMPTY : Utils::PROCESSING;
}

void FMI::Comm::Redis::process_async_events() {
    if (async_context == nullptr) {
        return;
    }

    // Get the file descriptor for the async connection
    int fd = async_context->c.fd;
    if (fd < 0) {
        return;
    }

    // Use select with zero timeout for non-blocking check
    fd_set read_fds, write_fds;
    FD_ZERO(&read_fds);
    FD_ZERO(&write_fds);

    // Always check for readable data
    FD_SET(fd, &read_fds);

    // Check if we have data to write
    if (async_context->c.obuf != nullptr && sdslen(async_context->c.obuf) > 0) {
        FD_SET(fd, &write_fds);
    }

    struct timeval tv = {0, 0};  // Non-blocking (zero timeout)
    int ret = select(fd + 1, &read_fds, &write_fds, nullptr, &tv);

    if (ret > 0) {
        if (FD_ISSET(fd, &write_fds)) {
            redisAsyncHandleWrite(async_context);
        }
        if (FD_ISSET(fd, &read_fds)) {
            redisAsyncHandleRead(async_context);
        }
    }
}

bool FMI::Comm::Redis::has_pending_operations() const {
    return !pending_ops.empty();
}

#endif