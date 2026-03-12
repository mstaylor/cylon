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

//! Communication operation types
//!
//! Ported from cpp/src/cylon/net/comm_operations.hpp

/// Reduction operations for collective communication
/// Corresponds to ReduceOp enum in cpp/src/cylon/net/comm_operations.hpp
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    Min,
    Max,
    Prod,
    Land,
    Lor,
    Band,
    Bor,
}