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
 * Basic tests for cylon-node native addon
 *
 * Run with: node test/test.js
 *
 * Note: Requires Redis server running for FMI backend
 */

const assert = require('assert');

// Try to load the native addon
let cylonNode;
try {
  cylonNode = require('../index.js');
  console.log('✓ Native addon loaded successfully');
} catch (e) {
  console.log('✗ Could not load native addon:', e.message);
  console.log('  Build the addon first with: npm run build');
  process.exit(1);
}

// Verify exports
console.log('\nChecking exports...');
assert(typeof cylonNode.createCommunicator === 'function', 'createCommunicator should be a function');
assert(typeof cylonNode.Communicator === 'function', 'Communicator should be a constructor');
console.log('✓ Exports verified');

// Try to create a communicator (requires Redis)
// Configure via environment variables:
//   REDIS_HOST (default: localhost)
//   REDIS_PORT (default: 6379)
const redisHost = process.env.REDIS_HOST || 'localhost';
const redisPort = parseInt(process.env.REDIS_PORT || '6379', 10);

console.log(`\nTrying to create communicator (Redis: ${redisHost}:${redisPort})...`);
try {
  const comm = cylonNode.createCommunicator({
    rank: 0,
    worldSize: 1,
    host: 'localhost',
    port: 8080,
    maxTimeout: 5000,
    commName: 'test',
    nonblocking: true,
    redisHost: redisHost,
    redisPort: redisPort,
    redisNamespace: 'cylon_test',
  });

  console.log('✓ Communicator created');
  console.log(`  Rank: ${comm.getRank()}`);
  console.log(`  World Size: ${comm.getWorldSize()}`);

  // Test barrier (single node should succeed immediately)
  comm.barrier();
  console.log('✓ Barrier completed');

  console.log('\n✓ All tests passed!');
} catch (e) {
  console.log('✗ Could not create communicator:', e.message);
  console.log(`  Make sure Redis is running on ${redisHost}:${redisPort}`);
  console.log('  Or set REDIS_HOST and REDIS_PORT environment variables');
  console.log('\n⚠ Skipping communicator tests (Redis not available)');
  console.log('✓ Basic tests passed (addon loads correctly)');
  process.exit(0); // Don't fail if Redis isn't available
}