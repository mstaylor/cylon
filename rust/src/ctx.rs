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

//! Cylon context and configuration
//!
//! Ported from cpp/src/cylon/ctx/cylon_context.hpp

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

use crate::error::{CylonError, CylonResult, Status};
use crate::net::{CommConfig, CommType, Communicator};

/// Memory pool trait for custom memory management
pub trait MemoryPool: Send + Sync {
    fn allocate(&self, size: usize) -> CylonResult<*mut u8>;
    fn deallocate(&self, ptr: *mut u8, size: usize);
}

/// Default memory pool implementation using system allocator
pub struct DefaultMemoryPool;

impl MemoryPool for DefaultMemoryPool {
    fn allocate(&self, size: usize) -> CylonResult<*mut u8> {
        use std::alloc::{alloc, Layout};
        let layout = Layout::from_size_align(size, 8)
            .map_err(|_| CylonError::OutOfMemory)?;
        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            Err(CylonError::OutOfMemory)
        } else {
            Ok(ptr)
        }
    }

    fn deallocate(&self, ptr: *mut u8, size: usize) {
        use std::alloc::{dealloc, Layout};
        if let Ok(layout) = Layout::from_size_align(size, 8) {
            unsafe { dealloc(ptr, layout) };
        }
    }
}

/// The entry point to cylon operations
/// Corresponds to C++ CylonContext class from cpp/src/cylon/ctx/cylon_context.hpp
#[derive(Debug)]
pub struct CylonContext {
    config: RwLock<HashMap<String, String>>,
    is_distributed: bool,
    communicator: Option<Arc<dyn Communicator>>,
    memory_pool: Arc<dyn MemoryPool>,
    sequence_no: Mutex<i32>,
}

impl CylonContext {
    /// Constructor (equivalent to C++ constructor)
    pub fn new(distributed: bool) -> Self {
        Self {
            config: RwLock::new(HashMap::new()),
            is_distributed: distributed,
            communicator: None,
            memory_pool: Arc::new(DefaultMemoryPool),
            sequence_no: Mutex::new(0),
        }
    }

    /// Initializes context (equivalent to C++ Init())
    pub fn init() -> Arc<Self> {
        Arc::new(Self::new(false))
    }

    /// Initializes distributed context (equivalent to C++ InitDistributed())
    pub async fn init_distributed(config: Arc<dyn CommConfig>) -> CylonResult<Arc<Self>> {
        let mut ctx = Self::new(true);

        // Initialize communicator based on config
        let communicator = config.create_communicator().await?;
        ctx.communicator = Some(communicator);

        Ok(Arc::new(ctx))
    }

    /// Completes and closes all operations under the context (equivalent to C++ Finalize())
    pub async fn finalize(&self) -> CylonResult<()> {
        if let Some(ref comm) = self.communicator {
            comm.finalize().await?;
        }
        Ok(())
    }

    /// Adds a configuration (equivalent to C++ AddConfig())
    pub fn add_config(&self, key: impl Into<String>, value: impl Into<String>) {
        let mut config = self.config.write().unwrap();
        config.insert(key.into(), value.into());
    }

    /// Returns a configuration (equivalent to C++ GetConfig())
    pub fn get_config(&self, key: &str, default: &str) -> String {
        let config = self.config.read().unwrap();
        config.get(key).cloned().unwrap_or_else(|| default.to_string())
    }

    /// Returns the Communicator instance (equivalent to C++ GetCommunicator())
    pub fn get_communicator(&self) -> Option<Arc<dyn Communicator>> {
        self.communicator.clone()
    }

    /// Sets a Communicator (equivalent to C++ setCommunicator())
    pub fn set_communicator(&mut self, communicator: Arc<dyn Communicator>) {
        self.communicator = Some(communicator);
    }

    /// Sets if distributed (equivalent to C++ setDistributed())
    pub fn set_distributed(&mut self, distributed: bool) {
        self.is_distributed = distributed;
    }

    /// Check if context is distributed (equivalent to C++ IsDistributed())
    pub fn is_distributed(&self) -> bool {
        self.is_distributed
    }

    /// Returns the local rank (equivalent to C++ GetRank())
    pub fn get_rank(&self) -> i32 {
        if let Some(ref comm) = self.communicator {
            comm.get_rank()
        } else {
            0
        }
    }

    /// Returns the world size (equivalent to C++ GetWorldSize())
    pub fn get_world_size(&self) -> i32 {
        if let Some(ref comm) = self.communicator {
            comm.get_world_size()
        } else {
            1
        }
    }

    /// Returns the neighbors in the world (equivalent to C++ GetNeighbours())
    pub fn get_neighbours(&self, include_self: bool) -> Vec<i32> {
        let world_size = self.get_world_size();
        let rank = self.get_rank();

        let mut neighbours = Vec::new();
        for i in 0..world_size {
            if include_self || i != rank {
                neighbours.push(i);
            }
        }
        neighbours
    }

    /// Returns memory pool (equivalent to C++ GetMemoryPool())
    pub fn get_memory_pool(&self) -> Arc<dyn MemoryPool> {
        self.memory_pool.clone()
    }

    /// Sets a memory pool (equivalent to C++ SetMemoryPool())
    pub fn set_memory_pool(&mut self, mem_pool: Arc<dyn MemoryPool>) {
        self.memory_pool = mem_pool;
    }

    /// Returns the next sequence number (equivalent to C++ GetNextSequence())
    pub fn get_next_sequence(&self) -> i32 {
        let mut seq = self.sequence_no.lock().unwrap();
        *seq += 1;
        *seq
    }

    /// Get communication type (equivalent to C++ GetCommType())
    pub fn get_comm_type(&self) -> CommType {
        if let Some(ref comm) = self.communicator {
            comm.get_comm_type()
        } else {
            CommType::Local
        }
    }

    /// Performs a barrier operation (equivalent to C++ Barrier())
    pub async fn barrier(&self) -> CylonResult<()> {
        if self.is_distributed() {
            if let Some(ref comm) = self.communicator {
                comm.barrier().await?;
            }
        }
        Ok(())
    }
}