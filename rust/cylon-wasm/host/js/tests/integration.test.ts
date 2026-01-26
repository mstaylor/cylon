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
 * End-to-end integration tests for WASM + Host
 *
 * These tests load the actual cylon-wasm module (built with wasm-pack)
 * and verify local operations work correctly.
 *
 * Note: FMI requires at least 2 workers for rendezvous, so distributed
 * operations are tested separately with multi-process tests.
 *
 * Requirements:
 * 1. cylon-wasm built with wasm-pack: wasm-pack build --target nodejs --release
 */

import * as fs from 'fs';
import * as path from 'path';

// Path to the wasm-pack generated package
const WASM_PKG_PATH = path.resolve(__dirname, '../../../pkg');
const WASM_JS_PATH = path.join(WASM_PKG_PATH, 'cylon_wasm.js');

// Check if WASM package is available
const hasWasmPkg = fs.existsSync(WASM_JS_PATH);

const describeLocal = hasWasmPkg ? describe : describe.skip;

/**
 * Local WASM operations - no FMI/Redis required
 * These test the pure WASM compute without distributed communication
 */
describeLocal('WASM Local Operations', () => {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let cylonWasm: any;

  beforeAll(async () => {
    try {
      // Create a mock cylon_host module for local-only operations
      // These stubs are only called for distributed operations
      const mockCylonHost = {
        host_get_rank: () => 0,
        host_get_world_size: () => 1,
        host_barrier: () => {},
        host_all_to_all: () => { throw new Error('FMI not available in local mode'); },
        host_all_gather: () => { throw new Error('FMI not available in local mode'); },
        host_broadcast: () => { throw new Error('FMI not available in local mode'); },
        host_gather: () => { throw new Error('FMI not available in local mode'); },
        host_scatter: () => { throw new Error('FMI not available in local mode'); },
      };

      // Register mock cylon_host module
      const Module = require('module');
      const originalResolve = Module._resolveFilename;
      Module._resolveFilename = function (request: string, ...args: unknown[]) {
        if (request === 'cylon_host') {
          return 'cylon_host';
        }
        return originalResolve.call(this, request, ...args);
      };

      require.cache['cylon_host'] = {
        id: 'cylon_host',
        filename: 'cylon_host',
        loaded: true,
        exports: mockCylonHost,
      } as NodeJS.Module;

      // Load the wasm-pack generated module
      cylonWasm = require(WASM_JS_PATH);

      console.log('WASM module loaded successfully');
      console.log('Available exports:', Object.keys(cylonWasm).length);
    } catch (e) {
      console.error('Failed to initialize WASM:', e);
      throw e;
    }
  });

  test('WASM module has expected exports', () => {
    expect(cylonWasm).toBeDefined();

    // Core operations
    expect(typeof cylonWasm.version).toBe('function');
    expect(typeof cylonWasm.init).toBe('function');

    // Table operations
    expect(typeof cylonWasm.json_to_ipc).toBe('function');
    expect(typeof cylonWasm.ipc_to_json).toBe('function');
    expect(typeof cylonWasm.table_info).toBe('function');
    expect(typeof cylonWasm.join_tables).toBe('function');
    expect(typeof cylonWasm.union_tables).toBe('function');
    expect(typeof cylonWasm.filter_table).toBe('function');
    expect(typeof cylonWasm.sort_table).toBe('function');
    expect(typeof cylonWasm.groupby_table).toBe('function');
  });

  test('version returns a string', () => {
    const version = cylonWasm.version();
    expect(typeof version).toBe('string');
    expect(version.length).toBeGreaterThan(0);
    console.log('cylon-wasm version:', version);
  });

  test('json_to_ipc and ipc_to_json roundtrip', () => {
    // TableData format: { columns: [names], data: [{ type, data }] }
    const testData = JSON.stringify({
      columns: ['id', 'name'],
      data: [
        { type: 'Int32', data: [1, 2, 3] },
        { type: 'String', data: ['Alice', 'Bob', 'Charlie'] },
      ],
    });

    // Convert JSON to Arrow IPC
    const ipcData = cylonWasm.json_to_ipc(testData);
    expect(ipcData).toBeInstanceOf(Uint8Array);
    expect(ipcData.length).toBeGreaterThan(0);

    // Get table info
    const info = JSON.parse(cylonWasm.table_info(ipcData));
    expect(info.num_rows).toBe(3);
    expect(info.num_columns).toBe(2);

    // Convert back to JSON
    const jsonResult = cylonWasm.ipc_to_json(ipcData);
    const parsed = JSON.parse(jsonResult);
    expect(parsed).toHaveProperty('columns');
  });

  test('local join operation works', () => {
    const left = JSON.stringify({
      columns: ['id', 'value'],
      data: [
        { type: 'Int32', data: [1, 2, 3] },
        { type: 'Int32', data: [10, 20, 30] },
      ],
    });

    const right = JSON.stringify({
      columns: ['id', 'data'],
      data: [
        { type: 'Int32', data: [2, 3, 4] },
        { type: 'String', data: ['b', 'c', 'd'] },
      ],
    });

    const leftIpc = cylonWasm.json_to_ipc(left);
    const rightIpc = cylonWasm.json_to_ipc(right);

    const joinConfig = JSON.stringify({
      join_type: 'inner',
      left_on: [0],
      right_on: [0],
    });

    const resultIpc = cylonWasm.join_tables(leftIpc, rightIpc, joinConfig);
    const info = JSON.parse(cylonWasm.table_info(resultIpc));

    // Inner join on id: 2, 3 match
    expect(info.num_rows).toBe(2);
    expect(info.num_columns).toBe(4); // left_id, value, right_id, data
  });

  test('filter operation works', () => {
    const data = JSON.stringify({
      columns: ['id', 'value'],
      data: [
        { type: 'Int32', data: [1, 2, 3, 4, 5] },
        { type: 'Int32', data: [10, 20, 30, 40, 50] },
      ],
    });

    const ipcData = cylonWasm.json_to_ipc(data);

    const filterConfig = JSON.stringify({
      predicates: [{ column: 1, op: 'gt', value: 25 }],
      logic: 'and',
    });

    const resultIpc = cylonWasm.filter_table(ipcData, filterConfig);
    const info = JSON.parse(cylonWasm.table_info(resultIpc));

    // Values > 25: 30, 40, 50
    expect(info.num_rows).toBe(3);
  });

  test('sort operation works', () => {
    const data = JSON.stringify({
      columns: ['id', 'value'],
      data: [
        { type: 'Int32', data: [3, 1, 4, 1, 5] },
        { type: 'String', data: ['c', 'a', 'd', 'b', 'e'] },
      ],
    });

    const ipcData = cylonWasm.json_to_ipc(data);
    const resultIpc = cylonWasm.sort_table(ipcData, 0, true);
    const info = JSON.parse(cylonWasm.table_info(resultIpc));

    expect(info.num_rows).toBe(5);
  });

  test('groupby operation works', () => {
    const data = JSON.stringify({
      columns: ['category', 'value'],
      data: [
        { type: 'String', data: ['a', 'b', 'a', 'b', 'a'] },
        { type: 'Int32', data: [10, 20, 30, 40, 50] },
      ],
    });

    const ipcData = cylonWasm.json_to_ipc(data);

    const groupbyConfig = JSON.stringify({
      keys: [0],
      aggregations: [{ column: 1, op: 'sum', alias: 'total' }],
    });

    const resultIpc = cylonWasm.groupby_table(ipcData, groupbyConfig);
    const info = JSON.parse(cylonWasm.table_info(resultIpc));

    // Two groups: 'a' and 'b'
    expect(info.num_rows).toBe(2);
  });

  test('union operation works', () => {
    const left = JSON.stringify({
      columns: ['id'],
      data: [{ type: 'Int32', data: [1, 2, 3] }],
    });

    const right = JSON.stringify({
      columns: ['id'],
      data: [{ type: 'Int32', data: [3, 4, 5] }],
    });

    const leftIpc = cylonWasm.json_to_ipc(left);
    const rightIpc = cylonWasm.json_to_ipc(right);

    const resultIpc = cylonWasm.union_tables(leftIpc, rightIpc);
    const info = JSON.parse(cylonWasm.table_info(resultIpc));

    // Union with dedup: 1, 2, 3, 4, 5
    expect(info.num_rows).toBe(5);
  });

  test('intersect operation works', () => {
    const left = JSON.stringify({
      columns: ['id'],
      data: [{ type: 'Int32', data: [1, 2, 3, 4] }],
    });

    const right = JSON.stringify({
      columns: ['id'],
      data: [{ type: 'Int32', data: [3, 4, 5, 6] }],
    });

    const leftIpc = cylonWasm.json_to_ipc(left);
    const rightIpc = cylonWasm.json_to_ipc(right);

    const resultIpc = cylonWasm.intersect_tables(leftIpc, rightIpc);
    const info = JSON.parse(cylonWasm.table_info(resultIpc));

    // Intersect: 3, 4
    expect(info.num_rows).toBe(2);
  });

  test('subtract operation works', () => {
    const left = JSON.stringify({
      columns: ['id'],
      data: [{ type: 'Int32', data: [1, 2, 3, 4] }],
    });

    const right = JSON.stringify({
      columns: ['id'],
      data: [{ type: 'Int32', data: [3, 4, 5, 6] }],
    });

    const leftIpc = cylonWasm.json_to_ipc(left);
    const rightIpc = cylonWasm.json_to_ipc(right);

    const resultIpc = cylonWasm.subtract_tables(leftIpc, rightIpc);
    const info = JSON.parse(cylonWasm.table_info(resultIpc));

    // Subtract: 1, 2 (in left but not in right)
    expect(info.num_rows).toBe(2);
  });

  test('compute aggregates work', () => {
    const data = JSON.stringify({
      columns: ['values'],
      data: [{ type: 'Int32', data: [1, 2, 3, 4, 5] }],
    });

    const ipcData = cylonWasm.json_to_ipc(data);

    expect(cylonWasm.compute_sum(ipcData, 0)).toBe(15);
    expect(cylonWasm.compute_min(ipcData, 0)).toBe(1);
    expect(cylonWasm.compute_max(ipcData, 0)).toBe(5);
    expect(cylonWasm.compute_mean(ipcData, 0)).toBe(3);
    expect(cylonWasm.compute_count(ipcData, 0)).toBe(BigInt(5));
  });
});

// Skip message when WASM package is missing
if (!hasWasmPkg) {
  console.log(`WASM package not found at: ${WASM_PKG_PATH}`);
  console.log('Build with: cd rust/cylon-wasm && wasm-pack build --target nodejs --release');
}