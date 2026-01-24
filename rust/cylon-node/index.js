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

/**
 * Cylon Node.js Native Addon
 *
 * This is a stub that loads the native addon built by napi-rs.
 * Only Linux is supported due to library dependencies.
 */

const { existsSync } = require('fs');
const { join } = require('path');

const { platform, arch } = process;

// Only Linux is supported
if (platform !== 'linux') {
  throw new Error(
    `Platform '${platform}' is not supported. Only Linux is supported due to library dependencies.`
  );
}

let nativeBinding = null;
let loadError = null;

// Platform-specific binding names for Linux
function getPossibleNames() {
  const names = [];

  if (arch === 'x64') {
    names.push('cylon-node.linux-x64-gnu.node');
    names.push('cylon-node.linux-x64-musl.node');
  } else if (arch === 'arm64') {
    names.push('cylon-node.linux-arm64-gnu.node');
    names.push('cylon-node.linux-arm64-musl.node');
  }

  // Generic names as fallback
  names.push('cylon_node.node');
  names.push('index.node');

  return names;
}

// Try to load from various locations
const possibleNames = getPossibleNames();
const searchPaths = [
  __dirname,
  join(__dirname, 'target', 'release'),
  join(__dirname, 'target', 'debug'),
];

for (const searchPath of searchPaths) {
  if (!existsSync(searchPath)) continue;

  for (const name of possibleNames) {
    const bindingPath = join(searchPath, name);
    if (existsSync(bindingPath)) {
      try {
        nativeBinding = require(bindingPath);
        break;
      } catch (e) {
        loadError = e;
      }
    }
  }

  if (nativeBinding) break;
}

if (!nativeBinding) {
  if (loadError) {
    throw loadError;
  }
  throw new Error(
    `Failed to load native binding for linux-${arch}. ` +
      'Please ensure the native addon is built. Run: npm run build'
  );
}

module.exports = nativeBinding;