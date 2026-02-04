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

//! UUID utilities
//!
//! Ported from cpp/src/cylon/util/uuid.hpp

use uuid::Uuid;

/// Generate a random UUID v4
pub fn generate_uuid() -> String {
    Uuid::new_v4().to_string()
}

/// Generate a random UUID v4 as bytes
pub fn generate_uuid_bytes() -> [u8; 16] {
    *Uuid::new_v4().as_bytes()
}

/// Parse a UUID string
pub fn parse_uuid(s: &str) -> Result<Uuid, uuid::Error> {
    Uuid::parse_str(s)
}

/// Convert UUID to string with hyphens
pub fn uuid_to_string(uuid: &Uuid) -> String {
    uuid.to_string()
}

/// Convert UUID to string without hyphens
pub fn uuid_to_simple_string(uuid: &Uuid) -> String {
    uuid.simple().to_string()
}