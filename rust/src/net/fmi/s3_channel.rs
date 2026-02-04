// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! S3 channel implementation for FMI
//!
//! This module corresponds to cpp/src/cylon/thridparty/fmi/comm/S3.hpp/cpp
//!
//! Uses AWS S3 as an object store for message passing between peers.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use aws_sdk_s3::Client as S3Client;
use aws_sdk_s3::primitives::ByteStream;
use aws_sdk_s3::config::{Builder as S3ConfigBuilder, Region, Credentials};
use aws_config::BehaviorVersion;

use crate::error::{CylonError, CylonResult, Code};
use super::client_server::{ClientServer, StorageBackend};
use super::common::*;

/// Async operation tracking (matches S3AsyncOp in C++)
struct S3AsyncOp {
    request: Arc<ChannelData>,
    object_name: String,
    op_type: Operation,  // Send (upload) or Receive (download)
    callback: Option<NbxCallback>,
    deadline: Instant,
    completed: bool,
    success: bool,
    error_message: String,
}

/// S3 storage backend implementation
pub struct S3Storage {
    // AWS S3 client
    client: Option<S3Client>,
    runtime: tokio::runtime::Runtime,

    // Configuration
    bucket_name: String,
    region: String,
    endpoint: Option<String>,
    timeout: i32,
    max_timeout: i32,

    // Async operation tracking
    pending_ops: HashMap<u64, S3AsyncOp>,
    next_op_id: u64,
}

impl S3Storage {
    /// Create a new S3 storage backend from configuration
    pub fn new(backend: &S3Backend) -> CylonResult<Self> {
        // Create tokio runtime for blocking on async operations
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| CylonError::new(Code::IoError, format!("Failed to create tokio runtime: {}", e)))?;

        Ok(Self {
            client: None,
            runtime,
            bucket_name: backend.get_bucket_name().to_string(),
            region: backend.get_region().to_string(),
            endpoint: backend.get_endpoint().map(|s| s.to_string()),
            timeout: backend.get_timeout(),
            max_timeout: backend.get_max_timeout(),
            pending_ops: HashMap::new(),
            next_op_id: 0,
        })
    }

    /// Initialize the S3 client (lazy initialization)
    fn ensure_client(&mut self) -> CylonResult<&S3Client> {
        if self.client.is_none() {
            let client = self.runtime.block_on(async {
                let region = Region::new(self.region.clone());

                let mut config_builder = S3ConfigBuilder::new()
                    .behavior_version(BehaviorVersion::latest())
                    .region(region);

                // If endpoint is specified (e.g., for MinIO or LocalStack), configure it
                if let Some(ref endpoint) = self.endpoint {
                    config_builder = config_builder
                        .endpoint_url(endpoint)
                        .force_path_style(true);
                }

                // Load credentials from environment
                let sdk_config = aws_config::load_defaults(BehaviorVersion::latest()).await;
                if let Some(creds_provider) = sdk_config.credentials_provider() {
                    config_builder = config_builder.credentials_provider(creds_provider.clone());
                }

                let config = config_builder.build();
                S3Client::from_conf(config)
            });

            self.client = Some(client);
        }

        Ok(self.client.as_ref().unwrap())
    }

    /// Process a single pending async operation
    fn process_async_op(&mut self, op_id: u64) -> bool {
        let op = match self.pending_ops.get_mut(&op_id) {
            Some(op) => op,
            None => return false,
        };

        if op.completed {
            return true;
        }

        // Check deadline
        if Instant::now() > op.deadline {
            op.completed = true;
            op.success = false;
            op.error_message = "Timeout".to_string();

            if let Some(ref callback) = op.callback {
                let mut ctx = FmiContext::new();
                callback(NbxStatus::NbxTimeout, &op.error_message, &mut ctx);
            }
            return true;
        }

        // Try to complete the operation
        match op.op_type {
            Operation::Send => {
                // For send (upload), try to execute the PUT operation
                let bucket = self.bucket_name.clone();
                let key = op.object_name.clone();
                let data = op.request.as_slice().to_vec();

                let result = self.runtime.block_on(async {
                    let client = match &self.client {
                        Some(c) => c,
                        None => return Err("Client not initialized".to_string()),
                    };

                    client.put_object()
                        .bucket(&bucket)
                        .key(&key)
                        .body(ByteStream::from(data))
                        .send()
                        .await
                        .map_err(|e| e.to_string())
                });

                match result {
                    Ok(_) => {
                        op.completed = true;
                        op.success = true;
                        if let Some(ref callback) = op.callback {
                            let mut ctx = FmiContext::new();
                            callback(NbxStatus::Success, "", &mut ctx);
                        }
                    }
                    Err(e) => {
                        op.completed = true;
                        op.success = false;
                        op.error_message = e;
                        if let Some(ref callback) = op.callback {
                            let mut ctx = FmiContext::new();
                            callback(NbxStatus::SendFailed, &op.error_message, &mut ctx);
                        }
                    }
                }
                true
            }
            Operation::Receive => {
                // For receive (download), try to execute the GET operation
                let bucket = self.bucket_name.clone();
                let key = op.object_name.clone();

                let result = self.runtime.block_on(async {
                    let client = match &self.client {
                        Some(c) => c,
                        None => return Err("Client not initialized".to_string()),
                    };

                    match client.get_object()
                        .bucket(&bucket)
                        .key(&key)
                        .send()
                        .await
                    {
                        Ok(output) => {
                            match output.body.collect().await {
                                Ok(data) => Ok(Some(data.into_bytes().to_vec())),
                                Err(e) => Err(e.to_string()),
                            }
                        }
                        Err(e) => {
                            // Check if it's a "not found" error
                            let err_str = e.to_string();
                            if err_str.contains("NoSuchKey") || err_str.contains("not found") {
                                Ok(None)
                            } else {
                                Err(err_str)
                            }
                        }
                    }
                });

                match result {
                    Ok(Some(data)) => {
                        // Copy data to buffer
                        let mut buf = op.request.as_mut_slice();
                        let copy_len = std::cmp::min(data.len(), buf.len());
                        buf[..copy_len].copy_from_slice(&data[..copy_len]);

                        op.completed = true;
                        op.success = true;
                        if let Some(ref callback) = op.callback {
                            let mut ctx = FmiContext::new();
                            callback(NbxStatus::Success, "", &mut ctx);
                        }
                        true
                    }
                    Ok(None) => {
                        // Key doesn't exist yet, keep trying
                        false
                    }
                    Err(e) => {
                        op.completed = true;
                        op.success = false;
                        op.error_message = e;
                        if let Some(ref callback) = op.callback {
                            let mut ctx = FmiContext::new();
                            callback(NbxStatus::ReceiveFailed, &op.error_message, &mut ctx);
                        }
                        true
                    }
                }
            }
            _ => {
                op.completed = true;
                op.success = false;
                op.error_message = "Unsupported operation type".to_string();
                true
            }
        }
    }
}

impl StorageBackend for S3Storage {
    fn upload_object(&self, data: &[u8], name: &str) -> CylonResult<()> {
        // Create a temporary client for this operation (since we need &self)
        let bucket = self.bucket_name.clone();
        let key = name.to_string();
        let data_vec = data.to_vec();
        let region = self.region.clone();
        let endpoint = self.endpoint.clone();

        // Use a new runtime for the blocking operation
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| CylonError::new(Code::IoError, format!("Runtime error: {}", e)))?;

        rt.block_on(async {
            let region = Region::new(region);
            let mut config_builder = S3ConfigBuilder::new()
                .behavior_version(BehaviorVersion::latest())
                .region(region);

            if let Some(ref ep) = endpoint {
                config_builder = config_builder
                    .endpoint_url(ep)
                    .force_path_style(true);
            }

            let sdk_config = aws_config::load_defaults(BehaviorVersion::latest()).await;
            if let Some(creds_provider) = sdk_config.credentials_provider() {
                config_builder = config_builder.credentials_provider(creds_provider.clone());
            }

            let config = config_builder.build();
            let client = S3Client::from_conf(config);

            client.put_object()
                .bucket(&bucket)
                .key(&key)
                .body(ByteStream::from(data_vec))
                .send()
                .await
                .map_err(|e| CylonError::new(Code::IoError, format!("S3 PUT failed: {}", e)))?;

            Ok(())
        })
    }

    fn download_object(&self, buf: &mut [u8], name: &str) -> CylonResult<bool> {
        let bucket = self.bucket_name.clone();
        let key = name.to_string();
        let region = self.region.clone();
        let endpoint = self.endpoint.clone();

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| CylonError::new(Code::IoError, format!("Runtime error: {}", e)))?;

        rt.block_on(async {
            let region = Region::new(region);
            let mut config_builder = S3ConfigBuilder::new()
                .behavior_version(BehaviorVersion::latest())
                .region(region);

            if let Some(ref ep) = endpoint {
                config_builder = config_builder
                    .endpoint_url(ep)
                    .force_path_style(true);
            }

            let sdk_config = aws_config::load_defaults(BehaviorVersion::latest()).await;
            if let Some(creds_provider) = sdk_config.credentials_provider() {
                config_builder = config_builder.credentials_provider(creds_provider.clone());
            }

            let config = config_builder.build();
            let client = S3Client::from_conf(config);

            match client.get_object()
                .bucket(&bucket)
                .key(&key)
                .send()
                .await
            {
                Ok(output) => {
                    let data = output.body.collect().await
                        .map_err(|e| CylonError::new(Code::IoError, format!("S3 GET body read failed: {}", e)))?;
                    let bytes = data.into_bytes();
                    let copy_len = std::cmp::min(bytes.len(), buf.len());
                    buf[..copy_len].copy_from_slice(&bytes[..copy_len]);
                    Ok(true)
                }
                Err(e) => {
                    let err_str = e.to_string();
                    if err_str.contains("NoSuchKey") || err_str.contains("not found") {
                        Ok(false)
                    } else {
                        Err(CylonError::new(Code::IoError, format!("S3 GET failed: {}", err_str)))
                    }
                }
            }
        })
    }

    fn delete_object(&self, name: &str) -> CylonResult<()> {
        let bucket = self.bucket_name.clone();
        let key = name.to_string();
        let region = self.region.clone();
        let endpoint = self.endpoint.clone();

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| CylonError::new(Code::IoError, format!("Runtime error: {}", e)))?;

        rt.block_on(async {
            let region = Region::new(region);
            let mut config_builder = S3ConfigBuilder::new()
                .behavior_version(BehaviorVersion::latest())
                .region(region);

            if let Some(ref ep) = endpoint {
                config_builder = config_builder
                    .endpoint_url(ep)
                    .force_path_style(true);
            }

            let sdk_config = aws_config::load_defaults(BehaviorVersion::latest()).await;
            if let Some(creds_provider) = sdk_config.credentials_provider() {
                config_builder = config_builder.credentials_provider(creds_provider.clone());
            }

            let config = config_builder.build();
            let client = S3Client::from_conf(config);

            client.delete_object()
                .bucket(&bucket)
                .key(&key)
                .send()
                .await
                .map_err(|e| CylonError::new(Code::IoError, format!("S3 DELETE failed: {}", e)))?;

            Ok(())
        })
    }

    fn get_object_names(&self) -> CylonResult<Vec<String>> {
        let bucket = self.bucket_name.clone();
        let region = self.region.clone();
        let endpoint = self.endpoint.clone();

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| CylonError::new(Code::IoError, format!("Runtime error: {}", e)))?;

        rt.block_on(async {
            let region = Region::new(region);
            let mut config_builder = S3ConfigBuilder::new()
                .behavior_version(BehaviorVersion::latest())
                .region(region);

            if let Some(ref ep) = endpoint {
                config_builder = config_builder
                    .endpoint_url(ep)
                    .force_path_style(true);
            }

            let sdk_config = aws_config::load_defaults(BehaviorVersion::latest()).await;
            if let Some(creds_provider) = sdk_config.credentials_provider() {
                config_builder = config_builder.credentials_provider(creds_provider.clone());
            }

            let config = config_builder.build();
            let client = S3Client::from_conf(config);

            let mut names = Vec::new();
            let mut continuation_token: Option<String> = None;

            loop {
                let mut request = client.list_objects_v2()
                    .bucket(&bucket);

                if let Some(token) = continuation_token {
                    request = request.continuation_token(token);
                }

                let output = request.send().await
                    .map_err(|e| CylonError::new(Code::IoError, format!("S3 LIST failed: {}", e)))?;

                if let Some(contents) = output.contents {
                    for obj in contents {
                        if let Some(key) = obj.key {
                            names.push(key);
                        }
                    }
                }

                if output.is_truncated.unwrap_or(false) {
                    continuation_token = output.next_continuation_token;
                } else {
                    break;
                }
            }

            Ok(names)
        })
    }

    fn upload_object_async(
        &self,
        data: Arc<ChannelData>,
        name: String,
        callback: Option<NbxCallback>,
    ) -> CylonResult<()> {
        // Note: This requires mutable access. In practice, use S3StorageMut
        // For now, we execute synchronously and call the callback
        let data_slice = data.as_slice();
        self.upload_object(&data_slice, &name)?;

        if let Some(cb) = callback {
            let mut ctx = FmiContext::new();
            cb(NbxStatus::Success, "", &mut ctx);
        }

        Ok(())
    }

    fn download_object_async(
        &self,
        buf: Arc<ChannelData>,
        name: String,
        callback: Option<NbxCallback>,
    ) -> CylonResult<()> {
        // Note: This requires mutable access. In practice, use S3StorageMut
        // For now, we execute synchronously and call the callback
        let mut data = buf.as_mut_slice();
        let found = self.download_object(&mut data, &name)?;

        if let Some(cb) = callback {
            let mut ctx = FmiContext::new();
            if found {
                cb(NbxStatus::Success, "", &mut ctx);
            } else {
                cb(NbxStatus::ReceiveFailed, "Key not found", &mut ctx);
            }
        }

        Ok(())
    }

    fn process_pending_operations(&self) -> EventProcessStatus {
        // Immutable version - no pending ops can be processed
        EventProcessStatus::Empty
    }

    fn has_pending_operations(&self) -> bool {
        false
    }

    fn get_timeout(&self) -> i32 {
        self.timeout
    }

    fn get_max_timeout(&self) -> i32 {
        self.max_timeout
    }
}

/// Mutable S3 storage for async operations
pub struct S3StorageMut {
    inner: S3Storage,
}

impl S3StorageMut {
    pub fn new(backend: &S3Backend) -> CylonResult<Self> {
        Ok(Self {
            inner: S3Storage::new(backend)?,
        })
    }

    /// Initialize the S3 client
    pub fn init(&mut self) -> CylonResult<()> {
        self.inner.ensure_client()?;
        Ok(())
    }

    /// Start async upload
    pub fn upload_object_async_mut(
        &mut self,
        data: Arc<ChannelData>,
        name: String,
        callback: Option<NbxCallback>,
    ) -> CylonResult<()> {
        let op_id = self.inner.next_op_id;
        self.inner.next_op_id += 1;

        let deadline = Instant::now() + Duration::from_millis(self.inner.max_timeout as u64);

        let op = S3AsyncOp {
            request: data,
            object_name: name,
            op_type: Operation::Send,
            callback,
            deadline,
            completed: false,
            success: false,
            error_message: String::new(),
        };

        self.inner.pending_ops.insert(op_id, op);
        Ok(())
    }

    /// Start async download
    pub fn download_object_async_mut(
        &mut self,
        buf: Arc<ChannelData>,
        name: String,
        callback: Option<NbxCallback>,
    ) -> CylonResult<()> {
        let op_id = self.inner.next_op_id;
        self.inner.next_op_id += 1;

        let deadline = Instant::now() + Duration::from_millis(self.inner.max_timeout as u64);

        let op = S3AsyncOp {
            request: buf,
            object_name: name,
            op_type: Operation::Receive,
            callback,
            deadline,
            completed: false,
            success: false,
            error_message: String::new(),
        };

        self.inner.pending_ops.insert(op_id, op);
        Ok(())
    }

    /// Process pending operations (non-blocking poll)
    pub fn process_pending_operations_mut(&mut self) -> EventProcessStatus {
        if self.inner.pending_ops.is_empty() {
            return EventProcessStatus::Empty;
        }

        let op_ids: Vec<u64> = self.inner.pending_ops.keys().copied().collect();
        let mut any_processing = false;

        for op_id in op_ids {
            let completed = self.inner.process_async_op(op_id);
            if !completed {
                any_processing = true;
            }
        }

        // Remove completed operations
        self.inner.pending_ops.retain(|_, op| !op.completed);

        if any_processing || !self.inner.pending_ops.is_empty() {
            EventProcessStatus::Processing
        } else {
            EventProcessStatus::Empty
        }
    }

    pub fn has_pending_operations(&self) -> bool {
        !self.inner.pending_ops.is_empty()
    }
}

/// Type alias for S3-backed ClientServer channel
pub type S3Channel = ClientServer<S3Storage>;

/// Create a new S3 channel from backend configuration
pub fn new_s3_channel(backend: &S3Backend) -> CylonResult<S3Channel> {
    let storage = S3Storage::new(backend)?;
    Ok(ClientServer::new(storage))
}