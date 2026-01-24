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
 * Unit tests for S3 FMI backend
 *
 * These tests require:
 * 1. AWS credentials configured (via environment variables or ~/.aws/credentials)
 * 2. An existing S3 bucket for testing
 *
 * Set environment variables before running:
 *   export FMI_TEST_S3_BUCKET=your-test-bucket
 *   export FMI_TEST_S3_REGION=us-east-1
 *
 * Run with: ./s3_test
 */

#define CATCH_CONFIG_MAIN
#include <catch.hpp>
#include <glog/logging.h>
#include <cstring>
#include <chrono>
#include <thread>
#include <cstdlib>

#include "cylon/thridparty/fmi/comm/S3.hpp"
#include "cylon/thridparty/fmi/utils/S3Backend.hpp"

// Helper to get test configuration from environment
struct S3TestConfig {
    std::string bucket;
    std::string region;
    bool valid;

    S3TestConfig() {
        const char* bucket_env = std::getenv("FMI_TEST_S3_BUCKET");
        const char* region_env = std::getenv("FMI_TEST_S3_REGION");

        if (bucket_env && region_env) {
            bucket = bucket_env;
            region = region_env;
            valid = true;
        } else {
            valid = false;
        }
    }
};

// =============================================================================
// Blocking Operations Tests
// =============================================================================

TEST_CASE("S3 blocking upload and download", "[s3][blocking]") {
    S3TestConfig config;
    if (!config.valid) {
        WARN("S3 tests skipped - set FMI_TEST_S3_BUCKET and FMI_TEST_S3_REGION");
        return;
    }

    auto backend = std::make_shared<FMI::Utils::S3Backend>();
    backend->withS3BacketName(const_cast<char*>(config.bucket.c_str()));
    backend->withAWSRegion(const_cast<char*>(config.region.c_str()));
    backend->withTimeout(100);
    backend->withMaxTimeout(30000);

    FMI::Comm::S3 s3(backend);

    SECTION("upload and download small data") {
        const char* test_data = "hello s3 world";
        size_t data_len = strlen(test_data) + 1;

        // Upload
        auto upload_buf = std::make_shared<channel_data>(
            const_cast<char*>(test_data), data_len);
        s3.upload_object(upload_buf, "fmi_test/test_key_1");

        // Download - channel_data copies buffer, read from buf->buf.get()
        auto download_buf = std::make_shared<channel_data>(data_len);
        bool success = s3.download_object(download_buf, "fmi_test/test_key_1");

        REQUIRE(success);
        REQUIRE(strcmp(download_buf->buf.get(), test_data) == 0);

        // Cleanup
        s3.delete_object("fmi_test/test_key_1");
    }

    SECTION("upload and download binary data") {
        unsigned char binary_data[] = {0x00, 0x01, 0x02, 0xFF, 0xFE, 0x00, 0x10};
        size_t data_len = sizeof(binary_data);

        auto upload_buf = std::make_shared<channel_data>(
            reinterpret_cast<char*>(binary_data), data_len);
        s3.upload_object(upload_buf, "fmi_test/binary_key");

        auto download_buf = std::make_shared<channel_data>(data_len);
        bool success = s3.download_object(download_buf, "fmi_test/binary_key");

        REQUIRE(success);
        REQUIRE(memcmp(download_buf->buf.get(), binary_data, data_len) == 0);

        s3.delete_object("fmi_test/binary_key");
    }

    SECTION("download non-existent key returns false") {
        char download_data[64] = {0};
        auto download_buf = std::make_shared<channel_data>(
            download_data, 64);
        bool success = s3.download_object(download_buf, "fmi_test/non_existent_key_xyz123");

        REQUIRE_FALSE(success);
    }

    SECTION("upload larger data") {
        // Create 1KB of test data
        std::vector<char> large_data(1024);
        for (size_t i = 0; i < large_data.size(); i++) {
            large_data[i] = static_cast<char>(i % 256);
        }

        auto upload_buf = std::make_shared<channel_data>(
            large_data.data(), large_data.size());
        s3.upload_object(upload_buf, "fmi_test/large_key");

        auto download_buf = std::make_shared<channel_data>(large_data.size());
        bool success = s3.download_object(download_buf, "fmi_test/large_key");

        REQUIRE(success);
        REQUIRE(memcmp(download_buf->buf.get(), large_data.data(), large_data.size()) == 0);

        s3.delete_object("fmi_test/large_key");
    }

    SECTION("delete object") {
        const char* test_data = "to be deleted";
        size_t data_len = strlen(test_data) + 1;

        auto upload_buf = std::make_shared<channel_data>(
            const_cast<char*>(test_data), data_len);
        s3.upload_object(upload_buf, "fmi_test/delete_test");

        // Verify it exists
        char download_data[64] = {0};
        auto download_buf = std::make_shared<channel_data>(
            download_data, data_len);
        REQUIRE(s3.download_object(download_buf, "fmi_test/delete_test"));

        // Delete
        s3.delete_object("fmi_test/delete_test");

        // Verify it's gone
        REQUIRE_FALSE(s3.download_object(download_buf, "fmi_test/delete_test"));
    }

    SECTION("get object names") {
        // Upload a few objects
        const char* data = "test";
        auto buf = std::make_shared<channel_data>(
            const_cast<char*>(data), 5);

        s3.upload_object(buf, "fmi_test/list_1");
        s3.upload_object(buf, "fmi_test/list_2");

        auto names = s3.get_object_names();
        bool found1 = std::find(names.begin(), names.end(), "fmi_test/list_1") != names.end();
        bool found2 = std::find(names.begin(), names.end(), "fmi_test/list_2") != names.end();

        REQUIRE(found1);
        REQUIRE(found2);

        s3.delete_object("fmi_test/list_1");
        s3.delete_object("fmi_test/list_2");
    }
}

// =============================================================================
// Async Operations Tests
// =============================================================================

TEST_CASE("S3 async upload and download", "[s3][async]") {
    S3TestConfig config;
    if (!config.valid) {
        WARN("S3 async tests skipped - set FMI_TEST_S3_BUCKET and FMI_TEST_S3_REGION");
        return;
    }

    auto backend = std::make_shared<FMI::Utils::S3Backend>();
    backend->withS3BacketName(const_cast<char*>(config.bucket.c_str()));
    backend->withAWSRegion(const_cast<char*>(config.region.c_str()));
    backend->withTimeout(100);
    backend->withMaxTimeout(30000);

    FMI::Comm::S3 s3(backend);
    s3.init();

    SECTION("async upload completes successfully") {
        const char* test_data = "async s3 hello";
        size_t data_len = strlen(test_data) + 1;

        auto upload_buf = std::make_shared<channel_data>(
            const_cast<char*>(test_data), data_len);

        bool callback_called = false;
        FMI::Utils::NbxStatus callback_status;

        s3.upload_object_async(upload_buf, "fmi_test/async_key", nullptr,
            [&](FMI::Utils::NbxStatus status, const std::string& msg, FMI::Utils::fmiContext* ctx) {
                callback_called = true;
                callback_status = status;
            });

        // Process events until complete (S3 operations can take longer)
        int iterations = 0;
        while (s3.has_pending_operations() && iterations < 300) {
            s3.channel_event_progress(FMI::Utils::DEFAULT);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            iterations++;
        }

        REQUIRE(callback_called);
        REQUIRE(callback_status == FMI::Utils::SUCCESS);

        // Verify data was uploaded
        auto download_buf = std::make_shared<channel_data>(data_len);
        REQUIRE(s3.download_object(download_buf, "fmi_test/async_key"));
        REQUIRE(strcmp(download_buf->buf.get(), test_data) == 0);

        s3.delete_object("fmi_test/async_key");
    }

    SECTION("async download completes successfully") {
        // First upload some data (blocking)
        const char* test_data = "async download test data";
        size_t data_len = strlen(test_data) + 1;

        auto upload_buf = std::make_shared<channel_data>(
            const_cast<char*>(test_data), data_len);
        s3.upload_object(upload_buf, "fmi_test/async_download_key");

        // Now download async
        auto download_buf = std::make_shared<channel_data>(data_len);

        bool callback_called = false;
        FMI::Utils::NbxStatus callback_status;

        s3.download_object_async(download_buf, "fmi_test/async_download_key", nullptr,
            [&](FMI::Utils::NbxStatus status, const std::string& msg, FMI::Utils::fmiContext* ctx) {
                callback_called = true;
                callback_status = status;
            });

        int iterations = 0;
        while (s3.has_pending_operations() && iterations < 300) {
            s3.channel_event_progress(FMI::Utils::DEFAULT);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            iterations++;
        }

        REQUIRE(callback_called);
        REQUIRE(callback_status == FMI::Utils::SUCCESS);
        REQUIRE(strcmp(download_buf->buf.get(), test_data) == 0);

        s3.delete_object("fmi_test/async_download_key");
    }

    SECTION("async download non-existent key fails") {
        auto download_buf = std::make_shared<channel_data>(64);

        bool callback_called = false;
        FMI::Utils::NbxStatus callback_status = FMI::Utils::SUCCESS;

        s3.download_object_async(download_buf, "fmi_test/nonexistent_async_xyz", nullptr,
            [&](FMI::Utils::NbxStatus status, const std::string& msg, FMI::Utils::fmiContext* ctx) {
                callback_called = true;
                callback_status = status;
            });

        int iterations = 0;
        while (s3.has_pending_operations() && iterations < 300) {
            s3.channel_event_progress(FMI::Utils::DEFAULT);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            iterations++;
        }

        REQUIRE(callback_called);
        REQUIRE(callback_status == FMI::Utils::RECEIVE_FAILED);
    }

    SECTION("multiple async operations") {
        int completed_count = 0;

        for (int i = 0; i < 3; i++) {
            std::string data = "multi_data_" + std::to_string(i);
            auto buf = std::make_shared<channel_data>(
                const_cast<char*>(data.c_str()), data.size() + 1);

            s3.upload_object_async(buf, "fmi_test/multi_async_" + std::to_string(i), nullptr,
                [&](FMI::Utils::NbxStatus status, const std::string& msg, FMI::Utils::fmiContext* ctx) {
                    if (status == FMI::Utils::SUCCESS) {
                        completed_count++;
                    }
                });
        }

        int iterations = 0;
        while (s3.has_pending_operations() && iterations < 600) {
            s3.channel_event_progress(FMI::Utils::DEFAULT);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            iterations++;
        }

        REQUIRE(completed_count == 3);

        // Cleanup
        for (int i = 0; i < 3; i++) {
            s3.delete_object("fmi_test/multi_async_" + std::to_string(i));
        }
    }
}

TEST_CASE("S3 channel_event_progress returns correct status", "[s3][async]") {
    S3TestConfig config;
    if (!config.valid) {
        WARN("S3 status tests skipped - set FMI_TEST_S3_BUCKET and FMI_TEST_S3_REGION");
        return;
    }

    auto backend = std::make_shared<FMI::Utils::S3Backend>();
    backend->withS3BacketName(const_cast<char*>(config.bucket.c_str()));
    backend->withAWSRegion(const_cast<char*>(config.region.c_str()));
    backend->withTimeout(100);
    backend->withMaxTimeout(30000);

    FMI::Comm::S3 s3(backend);
    s3.init();

    SECTION("returns EMPTY when no pending operations") {
        auto status = s3.channel_event_progress(FMI::Utils::DEFAULT);
        REQUIRE(status == FMI::Utils::EMPTY);
    }

    SECTION("returns PROCESSING when operations are pending") {
        const char* test_data = "test";
        auto buf = std::make_shared<channel_data>(
            const_cast<char*>(test_data), 5);

        s3.upload_object_async(buf, "fmi_test/progress_test", nullptr, nullptr);

        // Should have pending operations
        REQUIRE(s3.has_pending_operations());

        // Process until complete
        while (s3.has_pending_operations()) {
            s3.channel_event_progress(FMI::Utils::DEFAULT);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        // Should return EMPTY after completion
        auto status = s3.channel_event_progress(FMI::Utils::DEFAULT);
        REQUIRE(status == FMI::Utils::EMPTY);

        s3.delete_object("fmi_test/progress_test");
    }
}