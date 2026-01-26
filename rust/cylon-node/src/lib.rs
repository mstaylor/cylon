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
//! Cylon's communication layer to JavaScript. It serves as the host
//! runtime for cylon-wasm distributed operations.
//!
//! Supports multiple communication backends via the Communicator trait.

#[macro_use]
extern crate napi_derive;

use napi::bindgen_prelude::*;
use std::sync::Arc;

use cylon::net::communicator::Communicator as CylonCommunicator;
use cylon::net::CommType;

#[cfg(feature = "fmi")]
use cylon::net::fmi::cylon_communicator::{FMIConfig, FMICommunicator};

/// Communication backend type
#[napi(string_enum)]
pub enum CommunicatorType {
    /// FMI backend (Redis/S3 based)
    Fmi,
    /// MPI backend (requires MPI runtime)
    Mpi,
    /// Libfabric backend (high-performance fabric interface)
    Libfabric,
    /// UCX backend (Unified Communication X)
    Ucx,
    /// UCC backend (Unified Collective Communication)
    Ucc,
    /// Gloo backend (Facebook's collective communication library)
    Gloo,
}

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

/// Generic communicator configuration
#[napi(object)]
pub struct CommunicatorConfig {
    /// Backend type to use
    pub comm_type: CommunicatorType,
    /// FMI-specific options (required if comm_type is Fmi)
    pub fmi: Option<FmiConfigOptions>,
}

/// Communicator wrapper for Node.js
/// Uses the Communicator trait for backend-agnostic operations
#[napi]
pub struct Communicator {
    inner: Arc<dyn CylonCommunicator>,
}

#[napi]
impl Communicator {
    /// Create a communicator with the specified backend
    #[napi(factory)]
    pub fn create(config: CommunicatorConfig) -> Result<Self> {
        let inner: Arc<dyn CylonCommunicator> = match config.comm_type {
            CommunicatorType::Fmi => {
                #[cfg(feature = "fmi")]
                {
                    let fmi_opts = config.fmi.ok_or_else(|| {
                        Error::from_reason("FMI config required when using FMI backend")
                    })?;

                    let fmi_config = FMIConfig::builder()
                        .rank(fmi_opts.rank)
                        .world_size(fmi_opts.world_size)
                        .host(fmi_opts.host.as_deref().unwrap_or("localhost"))
                        .port(fmi_opts.port.unwrap_or(8080))
                        .max_timeout(fmi_opts.max_timeout.unwrap_or(30000))
                        .comm_name(fmi_opts.comm_name.as_deref().unwrap_or("cylon"))
                        .nonblocking(fmi_opts.nonblocking.unwrap_or(true))
                        .redis_host(fmi_opts.redis_host.as_deref().unwrap_or("localhost"))
                        .redis_port(fmi_opts.redis_port.unwrap_or(6379))
                        .redis_namespace(fmi_opts.redis_namespace.as_deref().unwrap_or("cylon"))
                        .build();

                    FMICommunicator::make(&fmi_config)
                        .map_err(|e| Error::from_reason(format!("Failed to create FMI communicator: {}", e)))?
                }
                #[cfg(not(feature = "fmi"))]
                {
                    return Err(Error::from_reason("FMI backend not enabled in build"));
                }
            }
            CommunicatorType::Mpi => {
                // MPI is typically not suitable for Node.js due to initialization requirements
                // but we include it for completeness
                return Err(Error::from_reason(
                    "MPI backend not supported in Node.js addon. Use FMI instead."
                ));
            }
            CommunicatorType::Libfabric => {
                // Libfabric (OFI) - high-performance fabric interface
                // TODO: Implement when cylon supports libfabric backend
                return Err(Error::from_reason(
                    "Libfabric backend not yet implemented. Use FMI instead."
                ));
            }
            CommunicatorType::Ucx => {
                // UCX - Unified Communication X
                // TODO: Implement when cylon supports UCX backend
                return Err(Error::from_reason(
                    "UCX backend not yet implemented. Use FMI instead."
                ));
            }
            CommunicatorType::Ucc => {
                // UCC - Unified Collective Communication
                // TODO: Implement when cylon supports UCC backend
                return Err(Error::from_reason(
                    "UCC backend not yet implemented. Use FMI instead."
                ));
            }
            CommunicatorType::Gloo => {
                // Gloo - Facebook's collective communication library
                // TODO: Implement when cylon supports Gloo backend
                return Err(Error::from_reason(
                    "Gloo backend not yet implemented. Use FMI instead."
                ));
            }
        };

        Ok(Self { inner })
    }

    /// Create an FMI communicator (convenience method)
    #[napi(factory)]
    pub fn create_fmi(options: FmiConfigOptions) -> Result<Self> {
        Self::create(CommunicatorConfig {
            comm_type: CommunicatorType::Fmi,
            fmi: Some(options),
        })
    }

    /// Get the communication backend type
    #[napi]
    pub fn get_comm_type(&self) -> String {
        match self.inner.get_comm_type() {
            CommType::Local => "local".to_string(),
            #[cfg(feature = "mpi")]
            CommType::Mpi => "mpi".to_string(),
            #[cfg(feature = "fmi")]
            CommType::Fmi => "fmi".to_string(),
            #[cfg(feature = "ucx")]
            CommType::Ucx => "ucx".to_string(),
            #[cfg(feature = "ucc")]
            CommType::Ucc => "ucc".to_string(),
            #[cfg(feature = "libfabric")]
            CommType::Libfabric => "libfabric".to_string(),
            #[cfg(feature = "gloo")]
            CommType::Gloo => "gloo".to_string(),
        }
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
    /// This is the key primitive for distributed shuffles.
    #[napi]
    pub fn all_to_all(&self, partitions: Vec<Buffer>) -> Result<Vec<Buffer>> {
        let send_data: Vec<Vec<u8>> = partitions.iter().map(|b| b.to_vec()).collect();

        let results = self
            .inner
            .all_to_all(send_data)
            .map_err(|e| Error::from_reason(format!("AllToAll failed: {}", e)))?;

        Ok(results.into_iter().map(Buffer::from).collect())
    }

    /// Allgather: each worker contributes data, all receive all data
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

    /// Gather: collect data from all workers to root
    /// Implemented using allgather (as per Communicator trait design)
    /// Returns results only on root, empty vector on other workers
    #[napi]
    pub fn gather(&self, data: Buffer, root: i32) -> Result<Vec<Buffer>> {
        // Use allgather since byte-level gather isn't in the Communicator trait
        let results = self
            .inner
            .allgather(&data)
            .map_err(|e| Error::from_reason(format!("Gather failed: {}", e)))?;

        if self.inner.get_rank() == root {
            Ok(results.into_iter().map(Buffer::from).collect())
        } else {
            Ok(vec![])
        }
    }

    /// Scatter: distribute partitions from root to all workers
    /// Implemented using all_to_all (as per Communicator trait design)
    /// Root sends partition[i] to worker i, returns this worker's partition
    #[napi]
    pub fn scatter(&self, partitions: Vec<Buffer>, root: i32) -> Result<Buffer> {
        let world_size = self.inner.get_world_size() as usize;
        let rank = self.inner.get_rank();

        // Build send data: root sends partitions, others send empty
        let send_data: Vec<Vec<u8>> = if rank == root {
            if partitions.len() != world_size {
                return Err(Error::from_reason(format!(
                    "Scatter requires {} partitions, got {}",
                    world_size,
                    partitions.len()
                )));
            }
            partitions.iter().map(|b| b.to_vec()).collect()
        } else {
            vec![vec![]; world_size]
        };

        let results = self
            .inner
            .all_to_all(send_data)
            .map_err(|e| Error::from_reason(format!("Scatter failed: {}", e)))?;

        // Each worker's result is what they received from root
        Ok(Buffer::from(results[root as usize].clone()))
    }

    /// Point-to-point send
    #[napi]
    pub fn send(&self, data: Buffer, dest: i32, tag: i32) -> Result<()> {
        self.inner
            .send(&data, dest, tag)
            .map_err(|e| Error::from_reason(format!("Send failed: {}", e)))
    }

    /// Point-to-point receive
    #[napi]
    pub fn recv(&self, source: i32, tag: i32) -> Result<Buffer> {
        let mut buffer = Vec::new();
        self.inner
            .recv(&mut buffer, source, tag)
            .map_err(|e| Error::from_reason(format!("Recv failed: {}", e)))?;
        Ok(Buffer::from(buffer))
    }
}

/// Create an FMI communicator (convenience function)
#[napi]
pub fn create_communicator(options: FmiConfigOptions) -> Result<Communicator> {
    Communicator::create_fmi(options)
}
