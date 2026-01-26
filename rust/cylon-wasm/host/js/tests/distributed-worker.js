#!/usr/bin/env node
/**
 * Distributed WASM Worker
 *
 * This worker is spawned by the distributed test runner.
 * It initializes FMI with the given rank and runs distributed operations.
 *
 * Usage: node distributed-worker.js <rank> <worldSize> <commName> <redisHost> <redisPort>
 */

const path = require('path');

// Parse command line arguments
const [,, rankStr, worldSizeStr, commName, redisHost, redisPort] = process.argv;
const rank = parseInt(rankStr, 10);
const worldSize = parseInt(worldSizeStr, 10);

console.log(`[Worker ${rank}] Starting with worldSize=${worldSize}, commName=${commName}`);

// Path to the wasm-pack generated package
const WASM_PKG_PATH = path.resolve(__dirname, '../../../pkg');
const WASM_JS_PATH = path.join(WASM_PKG_PATH, 'cylon_wasm.js');

async function main() {
  try {
    // Load cylon-node
    const { createCommunicator } = require('@aspect/cylon-node');

    // Create FMI communicator with the given rank
    // Use 127.0.0.1 for host since workers are on the same machine
    const config = {
      rank,
      worldSize,
      host: '127.0.0.1',
      port: 18080 + rank,  // Use high ports to avoid conflicts
      maxTimeout: 30000,
      commName,
      nonblocking: true,
      redisHost,
      redisPort: parseInt(redisPort, 10),
      redisNamespace: 'cylon_wasm_dist_test',
    };

    console.log(`[Worker ${rank}] Creating communicator...`);
    const communicator = createCommunicator(config);
    console.log(`[Worker ${rank}] Communicator created, actual rank=${communicator.getRank()}, worldSize=${communicator.getWorldSize()}`);

    // Create cylon_host module with real FMI functions
    const cylonHostModule = {
      host_get_rank: () => communicator.getRank(),
      host_get_world_size: () => communicator.getWorldSize(),
      host_barrier: () => communicator.barrier(),
      host_all_to_all: (partitions_ptr, partition_lens_ptr, num_partitions, results_ptr_out, num_results_out) => {
        throw new Error('host_all_to_all not implemented in worker test');
      },
      host_all_gather: (data_ptr, data_len, results_ptr_out, num_results_out) => {
        throw new Error('host_all_gather not implemented in worker test');
      },
      host_broadcast: (root, data_ptr, data_len, result_ptr_out, result_len_out) => {
        throw new Error('host_broadcast not implemented in worker test');
      },
      host_gather: (root, data_ptr, data_len, results_ptr_out, num_results_out) => {
        throw new Error('host_gather not implemented in worker test');
      },
      host_scatter: (root, partitions_ptr, partition_lens_ptr, num_partitions, result_ptr_out, result_len_out) => {
        throw new Error('host_scatter not implemented in worker test');
      },
    };

    // Register cylon_host module
    const Module = require('module');
    const originalResolve = Module._resolveFilename;
    Module._resolveFilename = function (request, ...args) {
      if (request === 'cylon_host') {
        return 'cylon_host';
      }
      return originalResolve.call(this, request, ...args);
    };

    require.cache['cylon_host'] = {
      id: 'cylon_host',
      filename: 'cylon_host',
      loaded: true,
      exports: cylonHostModule,
    };

    // Load WASM module
    console.log(`[Worker ${rank}] Loading WASM module...`);
    const cylonWasm = require(WASM_JS_PATH);
    console.log(`[Worker ${rank}] WASM module loaded, version=${cylonWasm.version()}`);

    // Test 1: Verify rank and world size through WASM
    console.log(`[Worker ${rank}] TEST 1: Verify rank/worldSize through WASM`);
    const wasmRank = cylonWasm.dist_get_rank();
    const wasmWorldSize = cylonWasm.dist_get_world_size();
    console.log(`[Worker ${rank}] WASM reports rank=${wasmRank}, worldSize=${wasmWorldSize}`);

    if (wasmRank !== rank) {
      throw new Error(`Rank mismatch: expected ${rank}, got ${wasmRank}`);
    }
    if (wasmWorldSize !== worldSize) {
      throw new Error(`WorldSize mismatch: expected ${worldSize}, got ${wasmWorldSize}`);
    }
    console.log(`[Worker ${rank}] TEST 1 PASSED`);

    // Test 2: Barrier synchronization through WASM
    console.log(`[Worker ${rank}] TEST 2: Barrier synchronization`);
    console.log(`[Worker ${rank}] Entering barrier...`);
    cylonWasm.dist_barrier();
    console.log(`[Worker ${rank}] Exited barrier`);
    console.log(`[Worker ${rank}] TEST 2 PASSED`);

    // Test 3: Local operations work on each worker
    console.log(`[Worker ${rank}] TEST 3: Local operations`);
    const testData = JSON.stringify({
      columns: ['id', 'value'],
      data: [
        { type: 'Int32', data: [rank * 10 + 1, rank * 10 + 2, rank * 10 + 3] },
        { type: 'Int32', data: [100, 200, 300] },
      ],
    });
    const ipcData = cylonWasm.json_to_ipc(testData);
    const info = JSON.parse(cylonWasm.table_info(ipcData));
    console.log(`[Worker ${rank}] Created table with ${info.num_rows} rows, ${info.num_columns} columns`);

    if (info.num_rows !== 3 || info.num_columns !== 2) {
      throw new Error(`Table info mismatch: expected 3 rows, 2 cols, got ${info.num_rows} rows, ${info.num_columns} cols`);
    }
    console.log(`[Worker ${rank}] TEST 3 PASSED`);

    // Final barrier before exit
    console.log(`[Worker ${rank}] Final barrier before exit...`);
    cylonWasm.dist_barrier();

    console.log(`[Worker ${rank}] ALL TESTS PASSED`);
    process.exit(0);
  } catch (error) {
    console.error(`[Worker ${rank}] ERROR:`, error.message);
    process.exit(1);
  }
}

main();