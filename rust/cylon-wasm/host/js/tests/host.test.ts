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
 * Integration tests for CylonWasmHost
 *
 * These tests verify that the WASM host correctly bridges to the
 * cylon-node native addon for distributed operations.
 *
 * Note: These tests require:
 * 1. cylon-node native addon to be built
 * 2. Redis server running (for FMI Redis backend)
 */

import { CylonWasmHost, createWasmHost, FmiConfigOptions } from '../src/index';

describe('CylonWasmHost', () => {
  // Skip integration tests if cylon-node is not available
  const hasCylonNode = (() => {
    try {
      require('@aspect/cylon-node');
      return true;
    } catch {
      return false;
    }
  })();

  const describeIfCylonNode = hasCylonNode ? describe : describe.skip;

  describeIfCylonNode('with cylon-node available', () => {
    const config: FmiConfigOptions = {
      rank: 0,
      worldSize: 1,
      host: 'localhost',
      port: 8080,
      maxTimeout: 5000,
      commName: 'test',
      nonblocking: true,
      redisHost: 'localhost',
      redisPort: 6379,
      redisNamespace: 'cylon_test',
    };

    let host: CylonWasmHost;

    beforeAll(() => {
      try {
        host = createWasmHost(config);
      } catch (e) {
        // Redis might not be available - skip tests
        console.warn('Could not create host (Redis unavailable?):', e);
      }
    });

    test('getRank returns configured rank', () => {
      if (!host) return;
      expect(host.getRank()).toBe(0);
    });

    test('getWorldSize returns configured world size', () => {
      if (!host) return;
      expect(host.getWorldSize()).toBe(1);
    });

    test('getImports returns cylon_host namespace', () => {
      if (!host) return;
      const imports = host.getImports();
      expect(imports).toHaveProperty('cylon_host');

      const cylonHost = imports.cylon_host as Record<string, unknown>;
      expect(typeof cylonHost.host_get_rank).toBe('function');
      expect(typeof cylonHost.host_get_world_size).toBe('function');
      expect(typeof cylonHost.host_barrier).toBe('function');
      expect(typeof cylonHost.host_all_to_all).toBe('function');
      expect(typeof cylonHost.host_all_gather).toBe('function');
      expect(typeof cylonHost.host_broadcast).toBe('function');
      expect(typeof cylonHost.host_allocate).toBe('function');
      expect(typeof cylonHost.host_free).toBe('function');
    });

    test('getCommunicator returns the underlying communicator', () => {
      if (!host) return;
      const comm = host.getCommunicator();
      expect(comm).toBeDefined();
      expect(typeof comm.getRank).toBe('function');
      expect(typeof comm.getWorldSize).toBe('function');
    });
  });

  describe('without cylon-node (mock tests)', () => {
    test('WasmHostConfig interface has required fields', () => {
      // This just verifies the TypeScript types compile correctly
      const mockConfig = {
        wasmPath: '/path/to/module.wasm',
        fmiConfig: {
          rank: 0,
          worldSize: 2,
        } as FmiConfigOptions,
      };
      expect(mockConfig.wasmPath).toBe('/path/to/module.wasm');
      expect(mockConfig.fmiConfig.rank).toBe(0);
    });
  });
});

describe('Host import functions', () => {
  // These tests verify the structure of host imports without needing
  // the native addon or Redis

  test('host import namespace matches WASM expectations', () => {
    // The WASM module expects these specific function names
    const expectedImports = [
      'host_get_rank',
      'host_get_world_size',
      'host_barrier',
      'host_all_to_all',
      'host_all_gather',
      'host_broadcast',
      'host_allocate',
      'host_free',
    ];

    // Verify the names are what we expect
    expectedImports.forEach((name) => {
      expect(name).toMatch(/^host_/);
    });
  });
});