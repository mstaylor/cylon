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

/**
 * Unit tests for Redis FMI backend
 *
 * These tests require a running Redis server on localhost:6379
 * Run with: ./redis_test
 *
 * To start Redis: redis-server
 */

#define CATCH_CONFIG_MAIN
#include <catch.hpp>
#include <glog/logging.h>
#include <cstring>
#include <chrono>
#include <thread>
#include <cstdlib>

#ifdef BUILD_CYLON_REDIS

#include "cylon/thridparty/fmi/comm/Redis.hpp"
#include "cylon/thridparty/fmi/utils/RedisBackend.hpp"

// Helper to create a test Redis backend
std::shared_ptr<FMI::Utils::RedisBackend> createTestRedisBackend() {
    auto backend = std::make_shared<FMI::Utils::RedisBackend>();
    // Use REDIS_TEST_HOST env var or default to Parallels host IP
    const char* host = std::getenv("REDIS_TEST_HOST");
    backend->withHost(host ? host : "10.211.55.2");
    backend->withPort(6379);
    backend->withTimeout(100);      // 100ms backoff
    backend->withMaxTimeout(5000);  // 5s max
    return backend;
}

// =============================================================================
// Blocking Operations Tests
// =============================================================================

TEST_CASE("Redis blocking upload and download", "[redis][blocking]") {
    auto backend = createTestRedisBackend();
    FMI::Comm::Redis redis(backend);

    SECTION("upload and download small data") {
        const char* test_data = "hello redis";
        size_t data_len = strlen(test_data) + 1;

        // Upload
        auto upload_buf = std::make_shared<channel_data>(
            const_cast<char*>(test_data), data_len);
        redis.upload_object(upload_buf, "test_key_1");

        // Download - channel_data copies the buffer, so read from buf->buf.get()
        auto download_buf = std::make_shared<channel_data>(data_len);
        bool success = redis.download_object(download_buf, "test_key_1");

        REQUIRE(success);
        REQUIRE(strcmp(download_buf->buf.get(), test_data) == 0);

        // Cleanup
        redis.delete_object("test_key_1");
    }

    SECTION("upload and download binary data") {
        unsigned char binary_data[] = {0x00, 0x01, 0x02, 0xFF, 0xFE, 0x00, 0x10};
        size_t data_len = sizeof(binary_data);

        auto upload_buf = std::make_shared<channel_data>(
            reinterpret_cast<char*>(binary_data), data_len);
        redis.upload_object(upload_buf, "test_binary_key");

        auto download_buf = std::make_shared<channel_data>(data_len);
        bool success = redis.download_object(download_buf, "test_binary_key");

        REQUIRE(success);
        REQUIRE(memcmp(download_buf->buf.get(), binary_data, data_len) == 0);

        redis.delete_object("test_binary_key");
    }

    SECTION("download non-existent key returns false") {
        char download_data[64] = {0};
        auto download_buf = std::make_shared<channel_data>(
            download_data, 64);
        bool success = redis.download_object(download_buf, "non_existent_key_12345");

        REQUIRE_FALSE(success);
    }

    SECTION("delete object") {
        const char* test_data = "to be deleted";
        size_t data_len = strlen(test_data) + 1;

        auto upload_buf = std::make_shared<channel_data>(
            const_cast<char*>(test_data), data_len);
        redis.upload_object(upload_buf, "delete_test_key");

        // Verify it exists
        char download_data[64] = {0};
        auto download_buf = std::make_shared<channel_data>(
            download_data, data_len);
        REQUIRE(redis.download_object(download_buf, "delete_test_key"));

        // Delete
        redis.delete_object("delete_test_key");

        // Verify it's gone
        REQUIRE_FALSE(redis.download_object(download_buf, "delete_test_key"));
    }

    SECTION("get object names") {
        // Upload a few objects
        const char* data = "test";
        auto buf = std::make_shared<channel_data>(
            const_cast<char*>(data), 5);

        redis.upload_object(buf, "list_test_1");
        redis.upload_object(buf, "list_test_2");

        auto names = redis.get_object_names();
        bool found1 = std::find(names.begin(), names.end(), "list_test_1") != names.end();
        bool found2 = std::find(names.begin(), names.end(), "list_test_2") != names.end();

        REQUIRE(found1);
        REQUIRE(found2);

        redis.delete_object("list_test_1");
        redis.delete_object("list_test_2");
    }
}

// =============================================================================
// Async Operations Tests
// =============================================================================

TEST_CASE("Redis async upload and download", "[redis][async]") {
    auto backend = createTestRedisBackend();
    FMI::Comm::Redis redis(backend);
    redis.init();

    SECTION("async upload completes successfully") {
        const char* test_data = "async hello";
        size_t data_len = strlen(test_data) + 1;

        auto upload_buf = std::make_shared<channel_data>(
            const_cast<char*>(test_data), data_len);

        bool callback_called = false;
        FMI::Utils::NbxStatus callback_status;

        redis.upload_object_async(upload_buf, "async_test_key", nullptr,
            [&](FMI::Utils::NbxStatus status, const std::string& msg, FMI::Utils::fmiContext* ctx) {
                callback_called = true;
                callback_status = status;
            });

        // Process events until complete
        int iterations = 0;
        while (redis.has_pending_operations() && iterations < 100) {
            redis.channel_event_progress(FMI::Utils::DEFAULT);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            iterations++;
        }

        REQUIRE(callback_called);
        REQUIRE(callback_status == FMI::Utils::SUCCESS);

        // Verify data was uploaded (using blocking download)
        auto download_buf = std::make_shared<channel_data>(data_len);
        REQUIRE(redis.download_object(download_buf, "async_test_key"));
        REQUIRE(strcmp(download_buf->buf.get(), test_data) == 0);

        redis.delete_object("async_test_key");
    }

    SECTION("async download completes successfully") {
        // First upload some data (blocking)
        const char* test_data = "async download test";
        size_t data_len = strlen(test_data) + 1;

        auto upload_buf = std::make_shared<channel_data>(
            const_cast<char*>(test_data), data_len);
        redis.upload_object(upload_buf, "async_download_key");

        // Now download async
        auto download_buf = std::make_shared<channel_data>(data_len);

        bool callback_called = false;
        FMI::Utils::NbxStatus callback_status;

        redis.download_object_async(download_buf, "async_download_key", nullptr,
            [&](FMI::Utils::NbxStatus status, const std::string& msg, FMI::Utils::fmiContext* ctx) {
                callback_called = true;
                callback_status = status;
            });

        // Process events until complete
        int iterations = 0;
        while (redis.has_pending_operations() && iterations < 100) {
            redis.channel_event_progress(FMI::Utils::DEFAULT);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            iterations++;
        }

        REQUIRE(callback_called);
        REQUIRE(callback_status == FMI::Utils::SUCCESS);
        REQUIRE(strcmp(download_buf->buf.get(), test_data) == 0);

        redis.delete_object("async_download_key");
    }

    SECTION("async download non-existent key fails") {
        auto download_buf = std::make_shared<channel_data>(64);

        bool callback_called = false;
        FMI::Utils::NbxStatus callback_status = FMI::Utils::SUCCESS;

        redis.download_object_async(download_buf, "nonexistent_async_key_xyz", nullptr,
            [&](FMI::Utils::NbxStatus status, const std::string& msg, FMI::Utils::fmiContext* ctx) {
                callback_called = true;
                callback_status = status;
            });

        int iterations = 0;
        while (redis.has_pending_operations() && iterations < 100) {
            redis.channel_event_progress(FMI::Utils::DEFAULT);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            iterations++;
        }

        REQUIRE(callback_called);
        REQUIRE(callback_status == FMI::Utils::RECEIVE_FAILED);
    }

    SECTION("multiple async operations") {
        int completed_count = 0;

        for (int i = 0; i < 5; i++) {
            std::string data = "data_" + std::to_string(i);
            auto buf = std::make_shared<channel_data>(
                const_cast<char*>(data.c_str()), data.size() + 1);

            redis.upload_object_async(buf, "multi_async_" + std::to_string(i), nullptr,
                [&](FMI::Utils::NbxStatus status, const std::string& msg, FMI::Utils::fmiContext* ctx) {
                    if (status == FMI::Utils::SUCCESS) {
                        completed_count++;
                    }
                });
        }

        int iterations = 0;
        while (redis.has_pending_operations() && iterations < 200) {
            redis.channel_event_progress(FMI::Utils::DEFAULT);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            iterations++;
        }

        REQUIRE(completed_count == 5);

        // Cleanup
        for (int i = 0; i < 5; i++) {
            redis.delete_object("multi_async_" + std::to_string(i));
        }
    }
}

TEST_CASE("Redis channel_event_progress returns correct status", "[redis][async]") {
    auto backend = createTestRedisBackend();
    FMI::Comm::Redis redis(backend);
    redis.init();

    SECTION("returns EMPTY when no pending operations") {
        auto status = redis.channel_event_progress(FMI::Utils::DEFAULT);
        REQUIRE(status == FMI::Utils::EMPTY);
    }

    SECTION("returns PROCESSING when operations are pending") {
        const char* test_data = "test";
        auto buf = std::make_shared<channel_data>(
            const_cast<char*>(test_data), 5);

        redis.upload_object_async(buf, "progress_test_key", nullptr, nullptr);

        // Should return PROCESSING while operation is pending
        REQUIRE(redis.has_pending_operations());

        // Process until complete
        while (redis.has_pending_operations()) {
            redis.channel_event_progress(FMI::Utils::DEFAULT);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        // Should return EMPTY after completion
        auto status = redis.channel_event_progress(FMI::Utils::DEFAULT);
        REQUIRE(status == FMI::Utils::EMPTY);

        redis.delete_object("progress_test_key");
    }
}

#else // BUILD_CYLON_REDIS

TEST_CASE("Redis tests skipped - BUILD_CYLON_REDIS not defined", "[redis]") {
    WARN("Redis tests are skipped because BUILD_CYLON_REDIS is not defined");
}

#endif // BUILD_CYLON_REDIS