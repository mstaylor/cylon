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
 * WASM Host Loader for Cylon Distributed Operations
 *
 * This module provides the host-side implementation for cylon-wasm distributed
 * operations. It bridges WASM imports to the native cylon-node addon which
 * provides FMI communication primitives.
 */

import { Communicator, FmiConfigOptions, createCommunicator } from '@aspect/cylon-node';
import * as fs from 'fs';
import * as path from 'path';

/**
 * Configuration for the WASM host
 */
export interface WasmHostConfig {
  /** Path to the WASM module */
  wasmPath: string;
  /** FMI communicator configuration */
  fmiConfig: FmiConfigOptions;
}

/**
 * Memory management for WASM linear memory
 */
class WasmMemory {
  private memory: WebAssembly.Memory | null = null;
  private allocatedBuffers: Map<number, number> = new Map(); // ptr -> size
  private nextPtr: number = 1024; // Start after potential static data

  setMemory(memory: WebAssembly.Memory) {
    this.memory = memory;
  }

  /**
   * Allocate a buffer in WASM memory and return pointer
   */
  allocate(size: number): number {
    const ptr = this.nextPtr;
    this.nextPtr += size + (8 - (size % 8)); // 8-byte alignment
    this.allocatedBuffers.set(ptr, size);
    return ptr;
  }

  /**
   * Write data to WASM memory at given pointer
   */
  write(ptr: number, data: Uint8Array): void {
    if (!this.memory) throw new Error('WASM memory not initialized');
    const view = new Uint8Array(this.memory.buffer);
    view.set(data, ptr);
  }

  /**
   * Read data from WASM memory
   */
  read(ptr: number, len: number): Uint8Array {
    if (!this.memory) throw new Error('WASM memory not initialized');
    const view = new Uint8Array(this.memory.buffer);
    return view.slice(ptr, ptr + len);
  }

  /**
   * Write a 32-bit integer to WASM memory
   */
  writeI32(ptr: number, value: number): void {
    if (!this.memory) throw new Error('WASM memory not initialized');
    const view = new DataView(this.memory.buffer);
    view.setInt32(ptr, value, true); // little-endian
  }

  /**
   * Read a 32-bit integer from WASM memory
   */
  readI32(ptr: number): number {
    if (!this.memory) throw new Error('WASM memory not initialized');
    const view = new DataView(this.memory.buffer);
    return view.getInt32(ptr, true); // little-endian
  }

  /**
   * Free allocated buffer (no-op for simple allocator)
   */
  free(ptr: number): void {
    this.allocatedBuffers.delete(ptr);
  }
}

/**
 * WASM Host that provides distributed communication to WASM modules
 */
export class CylonWasmHost {
  private communicator: Communicator;
  private wasmMemory: WasmMemory;
  private wasmInstance: WebAssembly.Instance | null = null;

  constructor(fmiConfig: FmiConfigOptions) {
    this.communicator = createCommunicator(fmiConfig);
    this.wasmMemory = new WasmMemory();
  }

  /**
   * Get the rank of this worker
   */
  getRank(): number {
    return this.communicator.getRank();
  }

  /**
   * Get the total number of workers
   */
  getWorldSize(): number {
    return this.communicator.getWorldSize();
  }

  /**
   * Create host imports for WASM instantiation
   */
  getImports(): WebAssembly.Imports {
    return {
      cylon_host: {
        // Get rank of this worker
        host_get_rank: (): number => {
          return this.communicator.getRank();
        },

        // Get total number of workers
        host_get_world_size: (): number => {
          return this.communicator.getWorldSize();
        },

        // Synchronization barrier
        host_barrier: (): number => {
          try {
            this.communicator.barrier();
            return 0; // Success
          } catch (e) {
            console.error('Barrier failed:', e);
            return -1; // Error
          }
        },

        // All-to-all exchange
        // Input: ptr to array of (data_ptr, data_len) pairs, count
        // Output: writes result to output_ptr, returns status
        host_all_to_all: (
          partitions_ptr: number,
          partition_lens_ptr: number,
          count: number,
          output_ptr: number,
          output_lens_ptr: number
        ): number => {
          try {
            const worldSize = this.communicator.getWorldSize();
            if (count !== worldSize) {
              console.error(`AllToAll: partition count ${count} != world size ${worldSize}`);
              return -1;
            }

            // Read partition lengths
            const partitions: Buffer[] = [];
            for (let i = 0; i < count; i++) {
              const len = this.wasmMemory.readI32(partition_lens_ptr + i * 4);
              const dataPtr = this.wasmMemory.readI32(partitions_ptr + i * 4);
              const data = this.wasmMemory.read(dataPtr, len);
              partitions.push(Buffer.from(data));
            }

            // Perform all-to-all
            const results = this.communicator.allToAll(partitions);

            // Write results back to WASM memory
            for (let i = 0; i < results.length; i++) {
              const result = results[i];
              const resultPtr = this.wasmMemory.allocate(result.length);
              this.wasmMemory.write(resultPtr, new Uint8Array(result));
              this.wasmMemory.writeI32(output_ptr + i * 4, resultPtr);
              this.wasmMemory.writeI32(output_lens_ptr + i * 4, result.length);
            }

            return 0; // Success
          } catch (e) {
            console.error('AllToAll failed:', e);
            return -1;
          }
        },

        // Allgather: each worker contributes, all receive all
        host_all_gather: (
          data_ptr: number,
          data_len: number,
          output_ptr: number,
          output_lens_ptr: number
        ): number => {
          try {
            const data = this.wasmMemory.read(data_ptr, data_len);
            const results = this.communicator.allGather(Buffer.from(data));

            // Write results back
            for (let i = 0; i < results.length; i++) {
              const result = results[i];
              const resultPtr = this.wasmMemory.allocate(result.length);
              this.wasmMemory.write(resultPtr, new Uint8Array(result));
              this.wasmMemory.writeI32(output_ptr + i * 4, resultPtr);
              this.wasmMemory.writeI32(output_lens_ptr + i * 4, result.length);
            }

            return 0;
          } catch (e) {
            console.error('AllGather failed:', e);
            return -1;
          }
        },

        // Broadcast from root to all
        host_broadcast: (
          data_ptr: number,
          data_len: number,
          root: number,
          output_ptr: number,
          output_len_ptr: number
        ): number => {
          try {
            const data = this.wasmMemory.read(data_ptr, data_len);
            const result = this.communicator.broadcast(Buffer.from(data), root);

            // Write result back
            const resultPtr = this.wasmMemory.allocate(result.length);
            this.wasmMemory.write(resultPtr, new Uint8Array(result));
            this.wasmMemory.writeI32(output_ptr, resultPtr);
            this.wasmMemory.writeI32(output_len_ptr, result.length);

            return 0;
          } catch (e) {
            console.error('Broadcast failed:', e);
            return -1;
          }
        },

        // Memory allocation helper for WASM
        host_allocate: (size: number): number => {
          return this.wasmMemory.allocate(size);
        },

        // Memory deallocation helper
        host_free: (ptr: number): void => {
          this.wasmMemory.free(ptr);
        },
      },
    };
  }

  /**
   * Load and instantiate a WASM module with host imports
   */
  async loadWasm(wasmPath: string): Promise<WebAssembly.Instance> {
    const wasmBuffer = fs.readFileSync(wasmPath);
    const wasmModule = await WebAssembly.compile(wasmBuffer);

    const imports = this.getImports();
    this.wasmInstance = await WebAssembly.instantiate(wasmModule, imports);

    // Get and set the WASM memory
    const memory = this.wasmInstance.exports.memory as WebAssembly.Memory;
    if (memory) {
      this.wasmMemory.setMemory(memory);
    }

    return this.wasmInstance;
  }

  /**
   * Get the underlying communicator for direct access
   */
  getCommunicator(): Communicator {
    return this.communicator;
  }

  /**
   * Get the WASM instance after loading
   */
  getWasmInstance(): WebAssembly.Instance | null {
    return this.wasmInstance;
  }
}

/**
 * Create a WASM host with FMI configuration
 */
export function createWasmHost(config: FmiConfigOptions): CylonWasmHost {
  return new CylonWasmHost(config);
}

/**
 * Load and run a WASM module with distributed support
 */
export async function loadAndRunWasm(config: WasmHostConfig): Promise<WebAssembly.Instance> {
  const host = createWasmHost(config.fmiConfig);
  return host.loadWasm(config.wasmPath);
}

// Re-export types from cylon-node
export { FmiConfigOptions } from '@aspect/cylon-node';