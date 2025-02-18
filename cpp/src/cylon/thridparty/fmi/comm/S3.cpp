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
#include <cmath>
#include "../utils/S3Backend.hpp"
#include <glog/logging.h>

char TAG[] = "S3Client";

FMI::Comm::S3::S3(std::shared_ptr<FMI::Utils::Backends> &backend) : ClientServer(backend) {
    auto s3backend = dynamic_cast<FMI::Utils::S3Backend *>(backend.get());

    if (instances == 0) {
        // Only one call allowed (https://github.com/aws/aws-sdk-cpp/issues/456), give possible multiple clients control over initialization
        Aws::InitAPI(options);
    }
    instances++;
    bucket_name = s3backend->getBacketName();
    Aws::Client::ClientConfiguration config;
    config.region = s3backend->getAWSRegion();


    //auto credentialsProvider = Aws::MakeShared<Aws::Auth::EnvironmentAWSCredentialsProvider>(TAG);
    //client = Aws::MakeUnique<Aws::S3::S3Client>(TAG, credentialsProvider, config);
    client = Aws::MakeUnique<Aws::S3::S3Client>(TAG, config);

}

FMI::Comm::S3::~S3() {
    instances--;
    if (instances == 0) {
        Aws::ShutdownAPI(options);
    }
}

bool FMI::Comm::S3::download_object(channel_data buf, std::string name) {
    Aws::S3::Model::GetObjectRequest request;
    request.WithBucket(bucket_name).WithKey(name);
    auto outcome = client->GetObject(request);
    if (outcome.IsSuccess()) {
        auto& s = outcome.GetResult().GetBody();
        s.read(buf.buf, buf.len);
        return true;
    } else {
        return false;
    }
}

void FMI::Comm::S3::upload_object(channel_data buf, std::string name) {
    Aws::S3::Model::PutObjectRequest request;
    request.WithBucket(bucket_name).WithKey(name);

    //const std::shared_ptr<Aws::IOStream> data = Aws::MakeShared<boost::interprocess::bufferstream>(TAG, buf.buf, buf.len);

    auto data = std::make_shared<std::stringstream>(std::string(buf.buf, buf.len));


    request.SetBody(data);
    auto outcome = client->PutObject(request);
    if (!outcome.IsSuccess()) {
        LOG(ERROR)  << "Error when uploading to S3: " << outcome.GetError();
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
    auto outcome = client->ListObjects(request);
    if (outcome.IsSuccess()) {
        auto objects = outcome.GetResult().GetContents();
        for (auto& object : objects) {
            object_names.push_back(object.GetKey());
        }
    } else {
        LOG(ERROR) << "Error when listing objects from S3: " << outcome.GetError();
    }
    return object_names;
}
