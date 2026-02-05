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

//! Redis channel implementation for FMI
//!
//! This module corresponds to cpp/src/cylon/thridparty/fmi/comm/Redis.hpp/cpp
//!
//! Uses Redis as a key-value store for message passing between peers.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use redis::{Client, Commands, Connection};

use crate::error::{CylonError, CylonResult, Code};
use super::client_server::{ClientServer, StorageBackend};
use super::common::*;

/// Async operation tracking (matches RedisAsyncOp in C++)
struct RedisAsyncOp {
    request: Arc<ChannelData>,
    object_name: String,
    op_type: Operation,  // Send (upload) or Receive (download)
    context: Option<Arc<FmiContext>>,
    callback: Option<NbxCallback>,
    deadline: Instant,
    completed: bool,
    success: bool,
    error_message: String,
}

/// Redis storage backend implementation
pub struct RedisStorage {
    // Connection
    client: Client,
    connection: Option<Connection>,

    // Configuration
    host: String,
    port: i32,
    timeout: i32,
    max_timeout: i32,

    // Async operation tracking
    pending_ops: HashMap<u64, RedisAsyncOp>,
    next_op_id: u64,
}

impl RedisStorage {
    /// Create a new Redis storage backend from configuration
    pub fn new(backend: &RedisBackend) -> CylonResult<Self> {
        let host = backend.get_host();
        let port = backend.get_port();
        let url = format!("redis://{}:{}", host, port);

        let client = Client::open(url.as_str())
            .map_err(|e| CylonError::new(Code::IoError, format!("Redis connection failed: {}", e)))?;

        Ok(Self {
            client,
            connection: None,
            host: host.to_string(),
            port,
            timeout: backend.get_timeout(),
            max_timeout: backend.get_max_timeout(),
            pending_ops: HashMap::new(),
            next_op_id: 0,
        })
    }

    /// Ensure connection is established
    fn ensure_connection(&mut self) -> CylonResult<&mut Connection> {
        if self.connection.is_none() {
            let conn = self.client.get_connection()
                .map_err(|e| CylonError::new(Code::IoError, format!("Redis connection failed: {}", e)))?;
            self.connection = Some(conn);
        }
        Ok(self.connection.as_mut().unwrap())
    }

    /// Process a single pending async operation
    fn process_async_op(&mut self, op_id: u64) -> bool {
        // First check if operation exists and is not completed
        let (op_type, deadline, completed, object_name, request) = {
            let op = match self.pending_ops.get(&op_id) {
                Some(op) => op,
                None => return false,
            };
            if op.completed {
                return true;
            }
            (op.op_type, op.deadline, op.completed, op.object_name.clone(), op.request.clone())
        };

        // Check deadline
        if Instant::now() > deadline {
            if let Some(op) = self.pending_ops.get_mut(&op_id) {
                op.completed = true;
                op.success = false;
                op.error_message = "Timeout".to_string();
                // Always mark context completed, regardless of callback
                if let Some(ref ctx) = op.context {
                    ctx.mark_completed();
                    if let Some(ref callback) = op.callback {
                        callback(NbxStatus::NbxTimeout, &op.error_message, ctx);
                    }
                }
            }
            return true;
        }

        // Ensure connection is established (separate borrow scope)
        if let Err(e) = self.ensure_connection() {
            if let Some(op) = self.pending_ops.get_mut(&op_id) {
                op.completed = true;
                op.success = false;
                op.error_message = e.to_string();
            }
            return true;
        }

        // Try to complete the operation
        match op_type {
            Operation::Send => {
                // For send (upload), execute the SET command
                let data = request.as_slice();
                let result: redis::RedisResult<()> = self.connection.as_mut().unwrap()
                    .set(&object_name, data.as_slice());

                if let Some(op) = self.pending_ops.get_mut(&op_id) {
                    match result {
                        Ok(()) => {
                            op.completed = true;
                            op.success = true;
                            // Always mark context completed, regardless of callback
                            if let Some(ref ctx) = op.context {
                                ctx.mark_completed();
                                if let Some(ref callback) = op.callback {
                                    callback(NbxStatus::Success, "", ctx);
                                }
                            }
                        }
                        Err(e) => {
                            op.completed = true;
                            op.success = false;
                            op.error_message = e.to_string();
                            // Always mark context completed, regardless of callback
                            if let Some(ref ctx) = op.context {
                                ctx.mark_completed();
                                if let Some(ref callback) = op.callback {
                                    callback(NbxStatus::SendFailed, &op.error_message, ctx);
                                }
                            }
                        }
                    }
                }
                true
            }
            Operation::Receive => {
                // For receive (download), execute the GET command
                let result: redis::RedisResult<Option<Vec<u8>>> = self.connection.as_mut().unwrap()
                    .get(&object_name);

                match result {
                    Ok(Some(data)) => {
                        // Copy data to buffer
                        let mut buf = request.as_mut_slice();
                        let copy_len = std::cmp::min(data.len(), buf.len());
                        buf[..copy_len].copy_from_slice(&data[..copy_len]);

                        if let Some(op) = self.pending_ops.get_mut(&op_id) {
                            op.completed = true;
                            op.success = true;
                            // Always mark context completed, regardless of callback
                            if let Some(ref ctx) = op.context {
                                ctx.mark_completed();
                                if let Some(ref callback) = op.callback {
                                    callback(NbxStatus::Success, "", ctx);
                                }
                            }
                        }
                        true
                    }
                    Ok(None) => {
                        // Key doesn't exist yet, keep trying
                        false
                    }
                    Err(e) => {
                        if let Some(op) = self.pending_ops.get_mut(&op_id) {
                            op.completed = true;
                            op.success = false;
                            op.error_message = e.to_string();
                            // Always mark context completed, regardless of callback
                            if let Some(ref ctx) = op.context {
                                ctx.mark_completed();
                                if let Some(ref callback) = op.callback {
                                    callback(NbxStatus::ReceiveFailed, &op.error_message, ctx);
                                }
                            }
                        }
                        true
                    }
                }
            }
            _ => {
                if let Some(op) = self.pending_ops.get_mut(&op_id) {
                    op.completed = true;
                    op.success = false;
                    op.error_message = "Unsupported operation type".to_string();
                }
                true
            }
        }
    }
}

impl StorageBackend for RedisStorage {
    fn upload_object(&self, data: &[u8], name: &str) -> CylonResult<()> {
        // Need mutable access - this is a design limitation
        // In practice, we use the mutable version through ClientServer
        let url = format!("redis://{}:{}", self.host, self.port);
        let client = Client::open(url.as_str())
            .map_err(|e| CylonError::new(Code::IoError, format!("Redis connection failed: {}", e)))?;
        let mut conn = client.get_connection()
            .map_err(|e| CylonError::new(Code::IoError, format!("Redis connection failed: {}", e)))?;

        conn.set::<_, _, ()>(name, data)
            .map_err(|e| CylonError::new(Code::IoError, format!("Redis SET failed: {}", e)))?;

        Ok(())
    }

    fn download_object(&self, buf: &mut [u8], name: &str) -> CylonResult<bool> {
        let url = format!("redis://{}:{}", self.host, self.port);
        let client = Client::open(url.as_str())
            .map_err(|e| CylonError::new(Code::IoError, format!("Redis connection failed: {}", e)))?;
        let mut conn = client.get_connection()
            .map_err(|e| CylonError::new(Code::IoError, format!("Redis connection failed: {}", e)))?;

        let result: redis::RedisResult<Option<Vec<u8>>> = conn.get(name);

        match result {
            Ok(Some(data)) => {
                let copy_len = std::cmp::min(data.len(), buf.len());
                buf[..copy_len].copy_from_slice(&data[..copy_len]);
                Ok(true)
            }
            Ok(None) => Ok(false),
            Err(e) => Err(CylonError::new(Code::IoError, format!("Redis GET failed: {}", e))),
        }
    }

    fn delete_object(&self, name: &str) -> CylonResult<()> {
        let url = format!("redis://{}:{}", self.host, self.port);
        let client = Client::open(url.as_str())
            .map_err(|e| CylonError::new(Code::IoError, format!("Redis connection failed: {}", e)))?;
        let mut conn = client.get_connection()
            .map_err(|e| CylonError::new(Code::IoError, format!("Redis connection failed: {}", e)))?;

        conn.del::<_, ()>(name)
            .map_err(|e| CylonError::new(Code::IoError, format!("Redis DEL failed: {}", e)))?;

        Ok(())
    }

    fn get_object_names(&self) -> CylonResult<Vec<String>> {
        let url = format!("redis://{}:{}", self.host, self.port);
        let client = Client::open(url.as_str())
            .map_err(|e| CylonError::new(Code::IoError, format!("Redis connection failed: {}", e)))?;
        let mut conn = client.get_connection()
            .map_err(|e| CylonError::new(Code::IoError, format!("Redis connection failed: {}", e)))?;

        let keys: Vec<String> = conn.keys("*")
            .map_err(|e| CylonError::new(Code::IoError, format!("Redis KEYS failed: {}", e)))?;

        Ok(keys)
    }

    fn upload_object_async(
        &self,
        data: Arc<ChannelData>,
        name: String,
        context: Option<Arc<FmiContext>>,
        callback: Option<NbxCallback>,
    ) -> CylonResult<()> {
        // Note: This requires mutable access. In practice, use RedisStorageMut
        // For now, we execute synchronously and call the callback
        let data_slice = data.as_slice();
        self.upload_object(&data_slice, &name)?;

        // Always mark context completed, regardless of callback
        if let Some(ref ctx) = context {
            ctx.mark_completed();
            if let Some(cb) = callback {
                cb(NbxStatus::Success, "", ctx);
            }
        }

        Ok(())
    }

    fn download_object_async(
        &self,
        buf: Arc<ChannelData>,
        name: String,
        context: Option<Arc<FmiContext>>,
        callback: Option<NbxCallback>,
    ) -> CylonResult<()> {
        // Note: This requires mutable access. In practice, use RedisStorageMut
        // For now, we execute synchronously and call the callback
        let mut data = buf.as_mut_slice();
        let found = self.download_object(&mut data, &name)?;

        // Always mark context completed, regardless of callback
        if let Some(ref ctx) = context {
            ctx.mark_completed();
            if let Some(cb) = callback {
                if found {
                    cb(NbxStatus::Success, "", ctx);
                } else {
                    cb(NbxStatus::ReceiveFailed, "Key not found", ctx);
                }
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

/// Mutable Redis storage for async operations
pub struct RedisStorageMut {
    inner: RedisStorage,
}

impl RedisStorageMut {
    pub fn new(backend: &RedisBackend) -> CylonResult<Self> {
        Ok(Self {
            inner: RedisStorage::new(backend)?,
        })
    }

    /// Start async upload
    pub fn upload_object_async_mut(
        &mut self,
        data: Arc<ChannelData>,
        name: String,
        context: Option<Arc<FmiContext>>,
        callback: Option<NbxCallback>,
    ) -> CylonResult<()> {
        let op_id = self.inner.next_op_id;
        self.inner.next_op_id += 1;

        let deadline = Instant::now() + Duration::from_millis(self.inner.max_timeout as u64);

        let op = RedisAsyncOp {
            request: data,
            object_name: name,
            op_type: Operation::Send,
            context,
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
        context: Option<Arc<FmiContext>>,
        callback: Option<NbxCallback>,
    ) -> CylonResult<()> {
        let op_id = self.inner.next_op_id;
        self.inner.next_op_id += 1;

        let deadline = Instant::now() + Duration::from_millis(self.inner.max_timeout as u64);

        let op = RedisAsyncOp {
            request: buf,
            object_name: name,
            op_type: Operation::Receive,
            context,
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

/// Type alias for Redis-backed ClientServer channel
pub type RedisChannel = ClientServer<RedisStorage>;

/// Create a new Redis channel from backend configuration
pub fn new_redis_channel(backend: &RedisBackend) -> CylonResult<RedisChannel> {
    let storage = RedisStorage::new(backend)?;
    Ok(ClientServer::new(storage))
}