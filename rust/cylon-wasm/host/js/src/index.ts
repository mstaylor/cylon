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
 * Uses WASM's exported wasm_alloc/wasm_free for proper memory management
 */
class WasmMemory {
  private memory: WebAssembly.Memory | null = null;
  private wasmAlloc: ((size: number) => number) | null = null;
  private wasmFree: ((ptr: number, size: number) => void) | null = null;
  private allocatedBuffers: Map<number, number> = new Map(); // ptr -> size

  setMemory(memory: WebAssembly.Memory) {
    this.memory = memory;
  }

  setAllocators(
    wasmAlloc: (size: number) => number,
    wasmFree: (ptr: number, size: number) => void
  ) {
    this.wasmAlloc = wasmAlloc;
    this.wasmFree = wasmFree;
  }

  /**
   * Allocate a buffer in WASM memory using wasm_alloc
   */
  allocate(size: number): number {
    if (!this.wasmAlloc) {
      throw new Error('WASM allocator not initialized');
    }
    const ptr = this.wasmAlloc(size);
    if (ptr !== 0) {
      this.allocatedBuffers.set(ptr, size);
    }
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
   * Write a 32-bit pointer/size (usize) to WASM memory
   * In wasm32, usize is 32 bits
   */
  writeUsize(ptr: number, value: number): void {
    if (!this.memory) throw new Error('WASM memory not initialized');
    const view = new DataView(this.memory.buffer);
    view.setUint32(ptr, value, true); // little-endian
  }

  /**
   * Read a 32-bit pointer/size (usize) from WASM memory
   */
  readUsize(ptr: number): number {
    if (!this.memory) throw new Error('WASM memory not initialized');
    const view = new DataView(this.memory.buffer);
    return view.getUint32(ptr, true); // little-endian
  }

  /**
   * Write a 32-bit signed integer to WASM memory
   */
  writeI32(ptr: number, value: number): void {
    if (!this.memory) throw new Error('WASM memory not initialized');
    const view = new DataView(this.memory.buffer);
    view.setInt32(ptr, value, true); // little-endian
  }

  /**
   * Read a 32-bit signed integer from WASM memory
   */
  readI32(ptr: number): number {
    if (!this.memory) throw new Error('WASM memory not initialized');
    const view = new DataView(this.memory.buffer);
    return view.getInt32(ptr, true); // little-endian
  }

  /**
   * Free allocated buffer using wasm_free
   */
  free(ptr: number, size?: number): void {
    if (!this.wasmFree) return;
    const actualSize = size ?? this.allocatedBuffers.get(ptr);
    if (actualSize !== undefined) {
      this.wasmFree(ptr, actualSize);
      this.allocatedBuffers.delete(ptr);
    }
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
   *
   * These signatures MUST match exactly what cylon-wasm/src/imports.rs declares.
   * The memory protocol:
   * - partitions_ptr: pointer to array of interleaved (ptr, len) pairs as usize
   * - result_ptr_out: pointer where host writes the allocated result pointer
   * - result_len_out: pointer where host writes the result length
   */
  getImports(): WebAssembly.Imports {
    const USIZE_SIZE = 4; // wasm32 uses 32-bit pointers

    return {
      cylon_host: {
        // Get rank of this worker
        // Signature: fn host_get_rank() -> i32
        host_get_rank: (): number => {
          return this.communicator.getRank();
        },

        // Get total number of workers
        // Signature: fn host_get_world_size() -> i32
        host_get_world_size: (): number => {
          return this.communicator.getWorldSize();
        },

        // Synchronization barrier
        // Signature: fn host_barrier()
        host_barrier: (): void => {
          this.communicator.barrier();
        },

        // Broadcast data from root to all workers
        // Signature: fn host_broadcast(
        //   data_ptr: *const u8,
        //   data_len: usize,
        //   root: i32,
        //   result_ptr_out: *mut *mut u8,
        //   result_len_out: *mut usize,
        // ) -> i32
        host_broadcast: (
          data_ptr: number,
          data_len: number,
          root: number,
          result_ptr_out: number,
          result_len_out: number
        ): number => {
          try {
            // Read input data (only valid on root, but read anyway)
            const data = data_len > 0 ? this.wasmMemory.read(data_ptr, data_len) : new Uint8Array(0);
            const result = this.communicator.broadcast(Buffer.from(data), root);

            // Allocate result in WASM memory and write pointer/length
            const resultPtr = this.wasmMemory.allocate(result.length);
            this.wasmMemory.write(resultPtr, new Uint8Array(result));
            this.wasmMemory.writeUsize(result_ptr_out, resultPtr);
            this.wasmMemory.writeUsize(result_len_out, result.length);

            return 0; // Success
          } catch (e) {
            console.error('Broadcast failed:', e);
            return -1;
          }
        },

        // All-to-all exchange
        // Signature: fn host_all_to_all(
        //   partitions_ptr: *const usize,  // array of (ptr, len) pairs
        //   num_partitions: usize,
        //   results_ptr_out: *mut *mut usize,  // out: array of (ptr, len) pairs
        //   num_results_out: *mut usize,
        // ) -> i32
        host_all_to_all: (
          partitions_ptr: number,
          num_partitions: number,
          results_ptr_out: number,
          num_results_out: number
        ): number => {
          try {
            const worldSize = this.communicator.getWorldSize();
            if (num_partitions !== worldSize) {
              console.error(`AllToAll: partition count ${num_partitions} != world size ${worldSize}`);
              return -1;
            }

            // Read partitions from interleaved (ptr, len) array
            const partitions: Buffer[] = [];
            for (let i = 0; i < num_partitions; i++) {
              const dataPtr = this.wasmMemory.readUsize(partitions_ptr + i * 2 * USIZE_SIZE);
              const dataLen = this.wasmMemory.readUsize(partitions_ptr + (i * 2 + 1) * USIZE_SIZE);
              const data = dataLen > 0 ? this.wasmMemory.read(dataPtr, dataLen) : new Uint8Array(0);
              partitions.push(Buffer.from(data));
            }

            // Perform all-to-all
            const results = this.communicator.allToAll(partitions);

            // Allocate results info array: (ptr, len) pairs for each result
            const numResults = results.length;
            const resultsInfoPtr = this.wasmMemory.allocate(numResults * 2 * USIZE_SIZE);

            for (let i = 0; i < numResults; i++) {
              const result = results[i];
              const resultPtr = this.wasmMemory.allocate(result.length);
              this.wasmMemory.write(resultPtr, new Uint8Array(result));
              this.wasmMemory.writeUsize(resultsInfoPtr + i * 2 * USIZE_SIZE, resultPtr);
              this.wasmMemory.writeUsize(resultsInfoPtr + (i * 2 + 1) * USIZE_SIZE, result.length);
            }

            // Write output pointers
            this.wasmMemory.writeUsize(results_ptr_out, resultsInfoPtr);
            this.wasmMemory.writeUsize(num_results_out, numResults);

            return 0; // Success
          } catch (e) {
            console.error('AllToAll failed:', e);
            return -1;
          }
        },

        // Gather data from all workers to root
        // Signature: fn host_gather(
        //   data_ptr: *const u8,
        //   data_len: usize,
        //   root: i32,
        //   results_ptr_out: *mut *mut usize,  // out: array of (ptr, len) pairs
        //   num_results_out: *mut usize,
        // ) -> i32
        host_gather: (
          data_ptr: number,
          data_len: number,
          root: number,
          results_ptr_out: number,
          num_results_out: number
        ): number => {
          try {
            const data = data_len > 0 ? this.wasmMemory.read(data_ptr, data_len) : new Uint8Array(0);

            // Use communicator.gather which handles the backend-agnostic implementation
            const results = this.communicator.gather(Buffer.from(data), root);

            if (results.length > 0) {
              // Root receives results
              const numResults = results.length;
              const resultsInfoPtr = this.wasmMemory.allocate(numResults * 2 * USIZE_SIZE);

              for (let i = 0; i < numResults; i++) {
                const result = results[i];
                const resultPtr = this.wasmMemory.allocate(result.length);
                this.wasmMemory.write(resultPtr, new Uint8Array(result));
                this.wasmMemory.writeUsize(resultsInfoPtr + i * 2 * USIZE_SIZE, resultPtr);
                this.wasmMemory.writeUsize(resultsInfoPtr + (i * 2 + 1) * USIZE_SIZE, result.length);
              }

              this.wasmMemory.writeUsize(results_ptr_out, resultsInfoPtr);
              this.wasmMemory.writeUsize(num_results_out, numResults);
            } else {
              // Non-root workers get empty result
              this.wasmMemory.writeUsize(results_ptr_out, 0);
              this.wasmMemory.writeUsize(num_results_out, 0);
            }

            return 0;
          } catch (e) {
            console.error('Gather failed:', e);
            return -1;
          }
        },

        // Scatter data from root to all workers
        // Signature: fn host_scatter(
        //   partitions_ptr: *const usize,  // array of (ptr, len) pairs (only on root)
        //   num_partitions: usize,
        //   root: i32,
        //   result_ptr_out: *mut *mut u8,
        //   result_len_out: *mut usize,
        // ) -> i32
        host_scatter: (
          partitions_ptr: number,
          num_partitions: number,
          root: number,
          result_ptr_out: number,
          result_len_out: number
        ): number => {
          try {
            const rank = this.communicator.getRank();

            // Read partitions on root, empty array on non-root
            const partitions: Buffer[] = [];
            if (rank === root) {
              for (let i = 0; i < num_partitions; i++) {
                const dataPtr = this.wasmMemory.readUsize(partitions_ptr + i * 2 * USIZE_SIZE);
                const dataLen = this.wasmMemory.readUsize(partitions_ptr + (i * 2 + 1) * USIZE_SIZE);
                const data = dataLen > 0 ? this.wasmMemory.read(dataPtr, dataLen) : new Uint8Array(0);
                partitions.push(Buffer.from(data));
              }
            }

            // Use communicator.scatter which handles the backend-agnostic implementation
            const myResult = this.communicator.scatter(partitions, root);

            // Write result
            const resultPtr = this.wasmMemory.allocate(myResult.length);
            this.wasmMemory.write(resultPtr, new Uint8Array(myResult));
            this.wasmMemory.writeUsize(result_ptr_out, resultPtr);
            this.wasmMemory.writeUsize(result_len_out, myResult.length);

            return 0;
          } catch (e) {
            console.error('Scatter failed:', e);
            return -1;
          }
        },

        // All-gather: each worker contributes, all receive all
        // Signature: fn host_all_gather(
        //   data_ptr: *const u8,
        //   data_len: usize,
        //   results_ptr_out: *mut *mut usize,  // out: array of (ptr, len) pairs
        //   num_results_out: *mut usize,
        // ) -> i32
        host_all_gather: (
          data_ptr: number,
          data_len: number,
          results_ptr_out: number,
          num_results_out: number
        ): number => {
          try {
            const data = data_len > 0 ? this.wasmMemory.read(data_ptr, data_len) : new Uint8Array(0);
            const results = this.communicator.allGather(Buffer.from(data));

            // Allocate results info array
            const numResults = results.length;
            const resultsInfoPtr = this.wasmMemory.allocate(numResults * 2 * USIZE_SIZE);

            for (let i = 0; i < numResults; i++) {
              const result = results[i];
              const resultPtr = this.wasmMemory.allocate(result.length);
              this.wasmMemory.write(resultPtr, new Uint8Array(result));
              this.wasmMemory.writeUsize(resultsInfoPtr + i * 2 * USIZE_SIZE, resultPtr);
              this.wasmMemory.writeUsize(resultsInfoPtr + (i * 2 + 1) * USIZE_SIZE, result.length);
            }

            // Write output pointers
            this.wasmMemory.writeUsize(results_ptr_out, resultsInfoPtr);
            this.wasmMemory.writeUsize(num_results_out, numResults);

            return 0;
          } catch (e) {
            console.error('AllGather failed:', e);
            return -1;
          }
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

    // Get and set the WASM allocators (exported by cylon-wasm/src/imports.rs)
    const wasmAlloc = this.wasmInstance.exports.wasm_alloc as ((size: number) => number) | undefined;
    const wasmFree = this.wasmInstance.exports.wasm_free as ((ptr: number, size: number) => void) | undefined;

    if (wasmAlloc && wasmFree) {
      this.wasmMemory.setAllocators(wasmAlloc, wasmFree);
    } else {
      console.warn('WASM module does not export wasm_alloc/wasm_free, using fallback allocator');
      // Provide a simple fallback allocator that uses the linear memory directly
      let nextPtr = 65536; // Start after first page
      this.wasmMemory.setAllocators(
        (size: number) => {
          const ptr = nextPtr;
          nextPtr += size + (8 - (size % 8)); // 8-byte alignment
          return ptr;
        },
        (_ptr: number, _size: number) => {
          // No-op for fallback
        }
      );
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