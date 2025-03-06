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
#include <cmath>

#ifdef BUILD_CYLON_REDIS

FMI::Comm::Redis::Redis(std::shared_ptr<FMI::Utils::Backends> &backend) : ClientServer(backend) {
    auto redisBackend = backend.get();


    std::string hostname = redisBackend->getHost();
    auto port = redisBackend->getPort();

    context = redisConnect(hostname.c_str(), port);
    if (context == nullptr || context->err) {
        if (context) {
            LOG(ERROR) << "Error when connecting to Redis: " << context->errstr;
        } else {
            LOG(ERROR) << "Allocating Redis context not possible";
        }
    }
}

FMI::Comm::Redis::~Redis() {

    redisFree(context);

}

void FMI::Comm::Redis::upload_object(const channel_data &buf, std::string name) {
    std::string command = "SET " + name + " %b";
    auto* reply = (redisReply*) redisCommand(context, command.c_str(), buf.buf.get(), buf.len);
    if (reply->type == REDIS_REPLY_ERROR) {
        LOG(ERROR) << "Error when uploading to Redis: " << reply->str;
    }
    freeReplyObject(reply);
}

bool FMI::Comm::Redis::download_object(const channel_data &buf, std::string name) {
    std::string command = "GET " + name;
    auto* reply = (redisReply*) redisCommand(context, command.c_str());
    if (reply->type == REDIS_REPLY_NIL || reply->type == REDIS_REPLY_ERROR) {
        freeReplyObject(reply);
        return false;
    } else {
        std::memcpy(buf.buf.get(), reply->str, std::min(buf.len, reply->len));
        freeReplyObject(reply);
        return true;
    }
}

void FMI::Comm::Redis::delete_object(std::string name) {
    std::string command = "DEL " + name;
    auto* reply = (redisReply*) redisCommand(context, command.c_str());
    freeReplyObject(reply);
}

std::vector<std::string> FMI::Comm::Redis::get_object_names() {
    std::vector<std::string> keys;
    std::string command = "KEYS *";
    auto* reply = (redisReply*) redisCommand(context, command.c_str());
    for (size_t i = 0; i < reply->elements; i++) {
        keys.emplace_back(reply->element[i]->str);
    }
    return keys;
}

#endif

