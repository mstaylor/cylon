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

//! Communication configuration
//!
//! Ported from cpp/src/cylon/net/comm_config.hpp

use async_trait::async_trait;
use std::sync::Arc;

use crate::error::CylonResult;
use super::{Communicator, CommType};

/// Communication configuration trait
/// Corresponds to C++ CommConfig interface
#[async_trait]
pub trait CommConfig: Send + Sync {
    fn get_type(&self) -> CommType;
    async fn create_communicator(&self) -> CylonResult<Arc<dyn Communicator>>;
}