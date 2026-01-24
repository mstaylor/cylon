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

//! Cylon Node.js Native Addon
//!
//! This crate provides a Node.js native addon (via napi-rs) that exposes
//! Cylon's FMI communication layer to JavaScript. It serves as the host
//! runtime for cylon-wasm distributed operations.

#[macro_use]
extern crate napi_derive;

use napi::bindgen_prelude::*;
use std::sync::Arc;

use cylon::net::fmi::cylon_communicator::{FMIConfig, FMICommunicator};
use cylon::net::communicator::Communicator as CylonCommunicator;

/// Configuration options for creating an FMI communicator
#[napi(object)]
pub struct FmiConfigOptions {
    pub rank: i32,
    pub world_size: i32,
    pub host: Option<String>,
    pub port: Option<i32>,
    pub max_timeout: Option<i32>,
    pub comm_name: Option<String>,
    pub nonblocking: Option<bool>,
    pub redis_host: Option<String>,
    pub redis_port: Option<i32>,
    pub redis_namespace: Option<String>,
}

/// FMI Communicator wrapper for Node.js
#[napi]
pub struct Communicator {
    inner: Arc<FMICommunicator>,
}

#[napi]
impl Communicator {
    #[napi(factory)]
    pub fn create(options: FmiConfigOptions) -> Result<Self> {
        let config = FMIConfig::builder()
            .rank(options.rank)
            .world_size(options.world_size)
            .host(options.host.as_deref().unwrap_or("localhost"))
            .port(options.port.unwrap_or(8080))
            .max_timeout(options.max_timeout.unwrap_or(30000))
            .comm_name(options.comm_name.as_deref().unwrap_or("cylon"))
            .nonblocking(options.nonblocking.unwrap_or(true))
            .redis_host(options.redis_host.as_deref().unwrap_or("localhost"))
            .redis_port(options.redis_port.unwrap_or(6379))
            .redis_namespace(options.redis_namespace.as_deref().unwrap_or("cylon"))
            .build();

        let inner = FMICommunicator::make(&config)
            .map_err(|e| Error::from_reason(format!("Failed to create communicator: {}", e)))?;

        Ok(Self { inner })
    }

    #[napi]
    pub fn get_rank(&self) -> i32 {
        self.inner.get_rank()
    }

    #[napi]
    pub fn get_world_size(&self) -> i32 {
        self.inner.get_world_size()
    }

    #[napi]
    pub fn barrier(&self) -> Result<()> {
        self.inner
            .barrier()
            .map_err(|e| Error::from_reason(format!("Barrier failed: {}", e)))
    }

    /// All-to-all exchange: partitions[i] goes to worker i
    #[napi]
    pub fn all_to_all(&self, partitions: Vec<Buffer>) -> Result<Vec<Buffer>> {
        let send_data: Vec<Vec<u8>> = partitions.iter().map(|b| b.to_vec()).collect();

        let results = self
            .inner
            .all_to_all(send_data)
            .map_err(|e| Error::from_reason(format!("AllToAll failed: {}", e)))?;

        Ok(results.into_iter().map(Buffer::from).collect())
    }

    /// Allgather: each worker contributes, all receive all
    #[napi]
    pub fn all_gather(&self, data: Buffer) -> Result<Vec<Buffer>> {
        let results = self
            .inner
            .allgather(&data)
            .map_err(|e| Error::from_reason(format!("AllGather failed: {}", e)))?;

        Ok(results.into_iter().map(Buffer::from).collect())
    }

    /// Broadcast from root to all workers
    #[napi]
    pub fn broadcast(&self, data: Buffer, root: i32) -> Result<Buffer> {
        let mut buf = data.to_vec();
        self.inner
            .broadcast(&mut buf, root)
            .map_err(|e| Error::from_reason(format!("Broadcast failed: {}", e)))?;

        Ok(Buffer::from(buf))
    }
}

#[napi]
pub fn create_communicator(options: FmiConfigOptions) -> Result<Communicator> {
    Communicator::create(options)
}