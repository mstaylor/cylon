/**
 * @anthropic/cylon-wasm - WebAssembly DataFrame operations for Node.js
 *
 * This module provides a Node.js interface to Cylon DataFrame operations
 * running in WebAssembly.
 *
 * @example
 * ```typescript
 * import { createRuntime } from '@anthropic/cylon-wasm';
 *
 * const runtime = await createRuntime();
 *
 * // Join tables (Arrow IPC format)
 * const result = runtime.joinTables(leftIpc, rightIpc, {
 *   joinType: 'inner',
 *   leftOn: [0],
 *   rightOn: [0]
 * });
 * ```
 *
 * @license Apache-2.0
 */

import { readFile } from 'fs/promises';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

// =============================================================================
// Types
// =============================================================================

export interface JoinConfig {
  joinType: 'inner' | 'left' | 'right' | 'full_outer';
  leftOn: number[];
  rightOn: number[];
}

export interface FilterPredicate {
  column: number;
  op: 'eq' | 'ne' | 'lt' | 'le' | 'gt' | 'ge';
  value: number | string | boolean;
}

export interface FilterConfig {
  predicates: FilterPredicate[];
  logic: 'and' | 'or';
}

export interface AggregationSpec {
  column: number;
  op: 'sum' | 'min' | 'max' | 'count' | 'mean' | 'var' | 'stddev' | 'nunique';
  alias?: string;
}

export interface GroupByConfig {
  keys: number[];
  aggregations: AggregationSpec[];
}

export interface SortSpec {
  column: number;
  ascending: boolean;
}

export interface SortConfig {
  columns: SortSpec[];
}

export interface TableInfo {
  num_rows: number;
  num_columns: number;
  columns: Array<{ name: string; type: string }>;
}

// =============================================================================
// WASM Runtime
// =============================================================================

/**
 * WASM Runtime for Cylon operations.
 *
 * Manages the WebAssembly module and provides methods for table operations.
 * Uses wasm-bindgen's memory management pattern:
 * 1. Allocate memory in WASM heap via __wbindgen_malloc
 * 2. Copy data from JS to WASM memory
 * 3. Call WASM function with pointer + length
 * 4. Read result from WASM memory
 * 5. Free memory via __wbindgen_free
 */
export class WasmRuntime {
  private instance: WebAssembly.Instance | null = null;
  private memory: WebAssembly.Memory | null = null;
  private malloc: ((size: number, align: number) => number) | null = null;
  private free: ((ptr: number, size: number, align: number) => void) | null = null;

  /**
   * Initialize the WASM runtime from a file path or buffer.
   */
  async initialize(wasmSource: string | ArrayBuffer): Promise<void> {
    let wasmBuffer: ArrayBuffer;

    if (typeof wasmSource === 'string') {
      const buffer = await readFile(wasmSource);
      wasmBuffer = buffer.buffer.slice(
        buffer.byteOffset,
        buffer.byteOffset + buffer.byteLength
      );
    } else {
      wasmBuffer = wasmSource;
    }

    // Create import object with wasm-bindgen stubs
    const imports = this.createImports();

    // Instantiate module
    const { instance } = await WebAssembly.instantiate(wasmBuffer, imports);
    this.instance = instance;

    // Cache exports
    const exports = instance.exports as Record<string, unknown>;
    this.memory = exports.memory as WebAssembly.Memory;
    this.malloc = exports.__wbindgen_malloc as (size: number, align: number) => number;
    this.free = exports.__wbindgen_free as (ptr: number, size: number, align: number) => void;

    // Call init if available
    const init = exports.init as (() => void) | undefined;
    if (init) {
      init();
    }
  }

  private createImports(): WebAssembly.Imports {
    return {
      wbg: {
        __wbg_new_8a6f238a6ece86ea: () => null,
        __wbg_stack_0ed75d68575b0f3c: () => {},
        __wbg_error_7534b8e9a36f1ab4: () => {},
        __wbindgen_init_externref_table: () => {},
        __wbindgen_cast_2241b6af4c4b2941: () => null,
      },
    };
  }

  private ensureInitialized(): void {
    if (!this.instance) {
      throw new Error('WASM runtime not initialized. Call initialize() first.');
    }
  }

  private getExport<T>(name: string): T {
    this.ensureInitialized();
    const exports = this.instance!.exports as Record<string, unknown>;
    const fn = exports[name];
    if (!fn) {
      throw new Error(`WASM export '${name}' not found`);
    }
    return fn as T;
  }

  // Allocate bytes in WASM memory, copy data, return [ptr, length]
  private allocBytes(data: Uint8Array): [number, number] {
    this.ensureInitialized();
    const ptr = this.malloc!(data.length, 1);
    const mem = new Uint8Array(this.memory!.buffer);
    mem.set(data, ptr);
    return [ptr, data.length];
  }

  // Allocate UTF-8 string in WASM memory
  private allocString(str: string): [number, number] {
    const encoder = new TextEncoder();
    const bytes = encoder.encode(str);
    return this.allocBytes(bytes);
  }

  // Read bytes from WASM memory
  private readBytes(ptr: number, length: number): Uint8Array {
    this.ensureInitialized();
    const mem = new Uint8Array(this.memory!.buffer);
    // Return a copy (not a view) since WASM memory can be resized
    return mem.slice(ptr, ptr + length);
  }

  // Read UTF-8 string from WASM memory
  private readString(ptr: number, length: number): string {
    const bytes = this.readBytes(ptr, length);
    const decoder = new TextDecoder();
    return decoder.decode(bytes);
  }

  // Free allocated memory
  private freeMemory(ptr: number, length: number): void {
    if (ptr && this.free) {
      this.free(ptr, length, 1);
    }
  }

  // ===========================================================================
  // Public API
  // ===========================================================================

  /**
   * Get the cylon-wasm version.
   */
  version(): string {
    const fn = this.getExport<(ptr: number) => void>('version');
    const retptr = this.malloc!(8, 4);
    try {
      fn(retptr);
      const mem = new DataView(this.memory!.buffer);
      const resultPtr = mem.getInt32(retptr, true);
      const resultLen = mem.getInt32(retptr + 4, true);
      return this.readString(resultPtr, resultLen);
    } finally {
      this.freeMemory(retptr, 8);
    }
  }

  /**
   * Check if SIMD is available.
   */
  simdAvailable(): boolean {
    const fn = this.getExport<() => number>('simd_available');
    return fn() !== 0;
  }

  /**
   * Get table info from Arrow IPC data.
   */
  tableInfo(data: Uint8Array): TableInfo {
    const fn = this.getExport<(retptr: number, ptr: number, len: number) => void>('table_info');
    const [ptr, len] = this.allocBytes(data);
    const retptr = this.malloc!(16, 4);

    try {
      fn(retptr, ptr, len);
      const mem = new DataView(this.memory!.buffer);
      const resultPtr = mem.getInt32(retptr, true);
      const resultLen = mem.getInt32(retptr + 4, true);
      const json = this.readString(resultPtr, resultLen);
      return JSON.parse(json);
    } finally {
      this.freeMemory(retptr, 16);
    }
  }

  /**
   * Convert JSON table to Arrow IPC format.
   */
  jsonToIpc(json: string): Uint8Array {
    const fn = this.getExport<(retptr: number, ptr: number, len: number) => void>('json_to_ipc');
    const [ptr, len] = this.allocString(json);
    const retptr = this.malloc!(16, 4);

    try {
      fn(retptr, ptr, len);
      const mem = new DataView(this.memory!.buffer);
      const resultPtr = mem.getInt32(retptr, true);
      const resultLen = mem.getInt32(retptr + 4, true);
      return this.readBytes(resultPtr, resultLen);
    } finally {
      this.freeMemory(retptr, 16);
    }
  }

  /**
   * Convert Arrow IPC to JSON.
   */
  ipcToJson(data: Uint8Array): string {
    const fn = this.getExport<(retptr: number, ptr: number, len: number) => void>('ipc_to_json');
    const [ptr, len] = this.allocBytes(data);
    const retptr = this.malloc!(16, 4);

    try {
      fn(retptr, ptr, len);
      const mem = new DataView(this.memory!.buffer);
      const resultPtr = mem.getInt32(retptr, true);
      const resultLen = mem.getInt32(retptr + 4, true);
      return this.readString(resultPtr, resultLen);
    } finally {
      this.freeMemory(retptr, 16);
    }
  }

  /**
   * Join two tables (Arrow IPC format).
   */
  joinTables(left: Uint8Array, right: Uint8Array, config: JoinConfig): Uint8Array {
    const fn = this.getExport<(
      retptr: number,
      lptr: number, llen: number,
      rptr: number, rlen: number,
      cptr: number, clen: number
    ) => void>('join_tables');

    const [lptr, llen] = this.allocBytes(left);
    const [rptr, rlen] = this.allocBytes(right);
    const configJson = JSON.stringify({
      join_type: config.joinType,
      left_on: config.leftOn,
      right_on: config.rightOn,
    });
    const [cptr, clen] = this.allocString(configJson);
    const retptr = this.malloc!(16, 4);

    try {
      fn(retptr, lptr, llen, rptr, rlen, cptr, clen);
      const mem = new DataView(this.memory!.buffer);
      const resultPtr = mem.getInt32(retptr, true);
      const resultLen = mem.getInt32(retptr + 4, true);
      return this.readBytes(resultPtr, resultLen);
    } finally {
      this.freeMemory(retptr, 16);
    }
  }

  /**
   * Filter table rows (Arrow IPC format).
   */
  filterTable(data: Uint8Array, config: FilterConfig): Uint8Array {
    const fn = this.getExport<(
      retptr: number,
      dptr: number, dlen: number,
      cptr: number, clen: number
    ) => void>('filter_table');

    const [dptr, dlen] = this.allocBytes(data);
    const configJson = JSON.stringify(config);
    const [cptr, clen] = this.allocString(configJson);
    const retptr = this.malloc!(16, 4);

    try {
      fn(retptr, dptr, dlen, cptr, clen);
      const mem = new DataView(this.memory!.buffer);
      const resultPtr = mem.getInt32(retptr, true);
      const resultLen = mem.getInt32(retptr + 4, true);
      return this.readBytes(resultPtr, resultLen);
    } finally {
      this.freeMemory(retptr, 16);
    }
  }

  /**
   * GroupBy with aggregations (Arrow IPC format).
   */
  groupByTable(data: Uint8Array, config: GroupByConfig): Uint8Array {
    const fn = this.getExport<(
      retptr: number,
      dptr: number, dlen: number,
      cptr: number, clen: number
    ) => void>('groupby_table');

    const [dptr, dlen] = this.allocBytes(data);
    const configJson = JSON.stringify(config);
    const [cptr, clen] = this.allocString(configJson);
    const retptr = this.malloc!(16, 4);

    try {
      fn(retptr, dptr, dlen, cptr, clen);
      const mem = new DataView(this.memory!.buffer);
      const resultPtr = mem.getInt32(retptr, true);
      const resultLen = mem.getInt32(retptr + 4, true);
      return this.readBytes(resultPtr, resultLen);
    } finally {
      this.freeMemory(retptr, 16);
    }
  }

  /**
   * Project (select) columns (Arrow IPC format).
   */
  projectTable(data: Uint8Array, columns: number[]): Uint8Array {
    const fn = this.getExport<(
      retptr: number,
      dptr: number, dlen: number,
      cptr: number, clen: number
    ) => void>('project_table');

    const [dptr, dlen] = this.allocBytes(data);
    const [cptr, clen] = this.allocString(JSON.stringify(columns));
    const retptr = this.malloc!(16, 4);

    try {
      fn(retptr, dptr, dlen, cptr, clen);
      const mem = new DataView(this.memory!.buffer);
      const resultPtr = mem.getInt32(retptr, true);
      const resultLen = mem.getInt32(retptr + 4, true);
      return this.readBytes(resultPtr, resultLen);
    } finally {
      this.freeMemory(retptr, 16);
    }
  }

  /**
   * Sort table by column (Arrow IPC format).
   */
  sortTable(data: Uint8Array, column: number, ascending: boolean): Uint8Array {
    const fn = this.getExport<(
      retptr: number,
      dptr: number, dlen: number,
      column: number, ascending: number
    ) => void>('sort_table');

    const [dptr, dlen] = this.allocBytes(data);
    const retptr = this.malloc!(16, 4);

    try {
      fn(retptr, dptr, dlen, column, ascending ? 1 : 0);
      const mem = new DataView(this.memory!.buffer);
      const resultPtr = mem.getInt32(retptr, true);
      const resultLen = mem.getInt32(retptr + 4, true);
      return this.readBytes(resultPtr, resultLen);
    } finally {
      this.freeMemory(retptr, 16);
    }
  }

  /**
   * Union two tables (Arrow IPC format).
   */
  unionTables(left: Uint8Array, right: Uint8Array): Uint8Array {
    const fn = this.getExport<(
      retptr: number,
      lptr: number, llen: number,
      rptr: number, rlen: number
    ) => void>('union_tables');

    const [lptr, llen] = this.allocBytes(left);
    const [rptr, rlen] = this.allocBytes(right);
    const retptr = this.malloc!(16, 4);

    try {
      fn(retptr, lptr, llen, rptr, rlen);
      const mem = new DataView(this.memory!.buffer);
      const resultPtr = mem.getInt32(retptr, true);
      const resultLen = mem.getInt32(retptr + 4, true);
      return this.readBytes(resultPtr, resultLen);
    } finally {
      this.freeMemory(retptr, 16);
    }
  }

  /**
   * Intersect two tables (Arrow IPC format).
   */
  intersectTables(left: Uint8Array, right: Uint8Array): Uint8Array {
    const fn = this.getExport<(
      retptr: number,
      lptr: number, llen: number,
      rptr: number, rlen: number
    ) => void>('intersect_tables');

    const [lptr, llen] = this.allocBytes(left);
    const [rptr, rlen] = this.allocBytes(right);
    const retptr = this.malloc!(16, 4);

    try {
      fn(retptr, lptr, llen, rptr, rlen);
      const mem = new DataView(this.memory!.buffer);
      const resultPtr = mem.getInt32(retptr, true);
      const resultLen = mem.getInt32(retptr + 4, true);
      return this.readBytes(resultPtr, resultLen);
    } finally {
      this.freeMemory(retptr, 16);
    }
  }

  /**
   * Subtract tables (Arrow IPC format).
   */
  subtractTables(left: Uint8Array, right: Uint8Array): Uint8Array {
    const fn = this.getExport<(
      retptr: number,
      lptr: number, llen: number,
      rptr: number, rlen: number
    ) => void>('subtract_tables');

    const [lptr, llen] = this.allocBytes(left);
    const [rptr, rlen] = this.allocBytes(right);
    const retptr = this.malloc!(16, 4);

    try {
      fn(retptr, lptr, llen, rptr, rlen);
      const mem = new DataView(this.memory!.buffer);
      const resultPtr = mem.getInt32(retptr, true);
      const resultLen = mem.getInt32(retptr + 4, true);
      return this.readBytes(resultPtr, resultLen);
    } finally {
      this.freeMemory(retptr, 16);
    }
  }

  /**
   * Remove duplicate rows (Arrow IPC format).
   */
  uniqueTable(data: Uint8Array, columns: number[], keepFirst: boolean): Uint8Array {
    const fn = this.getExport<(
      retptr: number,
      dptr: number, dlen: number,
      cptr: number, clen: number,
      keepFirst: number
    ) => void>('unique_table');

    const [dptr, dlen] = this.allocBytes(data);
    const [cptr, clen] = this.allocString(JSON.stringify(columns));
    const retptr = this.malloc!(16, 4);

    try {
      fn(retptr, dptr, dlen, cptr, clen, keepFirst ? 1 : 0);
      const mem = new DataView(this.memory!.buffer);
      const resultPtr = mem.getInt32(retptr, true);
      const resultLen = mem.getInt32(retptr + 4, true);
      return this.readBytes(resultPtr, resultLen);
    } finally {
      this.freeMemory(retptr, 16);
    }
  }

  /**
   * Compute sum of column.
   */
  computeSum(data: Uint8Array, column: number): number {
    const fn = this.getExport<(
      retptr: number,
      dptr: number, dlen: number,
      column: number
    ) => void>('compute_sum');

    const [dptr, dlen] = this.allocBytes(data);
    const retptr = this.malloc!(16, 8);

    try {
      fn(retptr, dptr, dlen, column);
      const mem = new DataView(this.memory!.buffer);
      return mem.getFloat64(retptr, true);
    } finally {
      this.freeMemory(retptr, 16);
    }
  }

  /**
   * Compute mean of column.
   */
  computeMean(data: Uint8Array, column: number): number {
    const fn = this.getExport<(
      retptr: number,
      dptr: number, dlen: number,
      column: number
    ) => void>('compute_mean');

    const [dptr, dlen] = this.allocBytes(data);
    const retptr = this.malloc!(16, 8);

    try {
      fn(retptr, dptr, dlen, column);
      const mem = new DataView(this.memory!.buffer);
      return mem.getFloat64(retptr, true);
    } finally {
      this.freeMemory(retptr, 16);
    }
  }

  /**
   * Compute min of column.
   */
  computeMin(data: Uint8Array, column: number): number {
    const fn = this.getExport<(
      retptr: number,
      dptr: number, dlen: number,
      column: number
    ) => void>('compute_min');

    const [dptr, dlen] = this.allocBytes(data);
    const retptr = this.malloc!(16, 8);

    try {
      fn(retptr, dptr, dlen, column);
      const mem = new DataView(this.memory!.buffer);
      return mem.getFloat64(retptr, true);
    } finally {
      this.freeMemory(retptr, 16);
    }
  }

  /**
   * Compute max of column.
   */
  computeMax(data: Uint8Array, column: number): number {
    const fn = this.getExport<(
      retptr: number,
      dptr: number, dlen: number,
      column: number
    ) => void>('compute_max');

    const [dptr, dlen] = this.allocBytes(data);
    const retptr = this.malloc!(16, 8);

    try {
      fn(retptr, dptr, dlen, column);
      const mem = new DataView(this.memory!.buffer);
      return mem.getFloat64(retptr, true);
    } finally {
      this.freeMemory(retptr, 16);
    }
  }

  /**
   * Compute count of column.
   */
  computeCount(data: Uint8Array, column: number): bigint {
    const fn = this.getExport<(
      retptr: number,
      dptr: number, dlen: number,
      column: number
    ) => void>('compute_count');

    const [dptr, dlen] = this.allocBytes(data);
    const retptr = this.malloc!(16, 8);

    try {
      fn(retptr, dptr, dlen, column);
      const mem = new DataView(this.memory!.buffer);
      return mem.getBigInt64(retptr, true);
    } finally {
      this.freeMemory(retptr, 16);
    }
  }
}

// =============================================================================
// Factory Functions
// =============================================================================

/**
 * Get default path to WASM binary.
 */
function getDefaultWasmPath(): string {
  const currentFile = fileURLToPath(import.meta.url);
  const currentDir = dirname(currentFile);
  return join(currentDir, '..', 'pkg', 'cylon_wasm_bg.wasm');
}

/**
 * Create a new WASM runtime instance.
 *
 * @param wasmPath - Path to the WASM binary (optional)
 * @returns Initialized WasmRuntime
 *
 * @example
 * ```typescript
 * const runtime = await createRuntime();
 * const result = runtime.joinTables(left, right, {
 *   joinType: 'inner',
 *   leftOn: [0],
 *   rightOn: [0]
 * });
 * ```
 */
export async function createRuntime(wasmPath?: string): Promise<WasmRuntime> {
  const runtime = new WasmRuntime();
  await runtime.initialize(wasmPath || getDefaultWasmPath());
  return runtime;
}

/**
 * Create a runtime from a WASM buffer.
 *
 * @param wasmBuffer - WASM binary as ArrayBuffer
 * @returns Initialized WasmRuntime
 */
export async function createRuntimeFromBuffer(wasmBuffer: ArrayBuffer): Promise<WasmRuntime> {
  const runtime = new WasmRuntime();
  await runtime.initialize(wasmBuffer);
  return runtime;
}

// =============================================================================
// Distributed Runtime (Approach B: WASM with Host Imports)
// =============================================================================

/**
 * Communication backend interface (synchronous).
 *
 * Implementations provide the actual communication (FMI, MPI, etc.)
 * that WASM calls via host imports. All methods are synchronous
 * to match WASM's synchronous host import model.
 */
export interface CommunicationBackend {
  getRank(): number;
  getWorldSize(): number;
  barrier(): void;
  allToAll(partitions: Uint8Array[]): Uint8Array[];
  broadcast(data: Uint8Array, root: number): Uint8Array;
  gather(data: Uint8Array, root: number): Uint8Array[] | null;
  scatter(partitions: Uint8Array[] | null, root: number): Uint8Array;
  allGather(data: Uint8Array): Uint8Array[];
}

/**
 * Single-node communication backend (no actual communication).
 */
export class LocalBackend implements CommunicationBackend {
  constructor(
    private readonly rank: number = 0,
    private readonly worldSize: number = 1
  ) {}

  getRank(): number {
    return this.rank;
  }

  getWorldSize(): number {
    return this.worldSize;
  }

  barrier(): void {
    // No-op for single worker
  }

  allToAll(partitions: Uint8Array[]): Uint8Array[] {
    // In local mode, just return the partition for this rank
    if (partitions.length > 0 && this.rank < partitions.length) {
      return [partitions[this.rank]];
    }
    return [];
  }

  broadcast(data: Uint8Array, _root: number): Uint8Array {
    return data;
  }

  gather(data: Uint8Array, root: number): Uint8Array[] | null {
    if (this.rank === root) {
      return [data];
    }
    return null;
  }

  scatter(partitions: Uint8Array[] | null, _root: number): Uint8Array {
    if (partitions && this.rank < partitions.length) {
      return partitions[this.rank];
    }
    return new Uint8Array(0);
  }

  allGather(data: Uint8Array): Uint8Array[] {
    return [data];
  }
}

/**
 * Distributed WASM Runtime with host imports for communication.
 *
 * Approach B: WASM contains distributed orchestration logic,
 * hosts provide communication primitives (all_to_all, barrier, etc.)
 *
 * All operations are synchronous to match WASM's host import model.
 */
export class DistributedWasmRuntime extends WasmRuntime {
  private comm: CommunicationBackend;

  constructor(comm?: CommunicationBackend) {
    super();
    this.comm = comm || new LocalBackend();
  }

  /**
   * Initialize with host imports for distributed operations.
   */
  async initialize(wasmSource: string | ArrayBuffer): Promise<void> {
    let wasmBuffer: ArrayBuffer;

    if (typeof wasmSource === 'string') {
      const buffer = await readFile(wasmSource);
      wasmBuffer = buffer.buffer.slice(
        buffer.byteOffset,
        buffer.byteOffset + buffer.byteLength
      );
    } else {
      wasmBuffer = wasmSource;
    }

    // Create import object with wasm-bindgen stubs + host imports
    const imports = this.createDistributedImports();

    // Instantiate module
    const { instance } = await WebAssembly.instantiate(wasmBuffer, imports);
    (this as any).instance = instance;

    // Cache exports
    const exports = instance.exports as Record<string, unknown>;
    (this as any).memory = exports.memory as WebAssembly.Memory;
    (this as any).malloc = exports.__wbindgen_malloc as (size: number, align: number) => number;
    (this as any).free = exports.__wbindgen_free as (ptr: number, size: number, align: number) => void;

    // Call init if available
    const init = exports.init as (() => void) | undefined;
    if (init) {
      init();
    }
  }

  private getMemory(): WebAssembly.Memory {
    return (this as any).memory;
  }

  private getMalloc(): (size: number, align: number) => number {
    return (this as any).malloc;
  }

  private readBytesInternal(ptr: number, length: number): Uint8Array {
    const mem = new Uint8Array(this.getMemory().buffer);
    return mem.slice(ptr, ptr + length);
  }

  private writeBytesInternal(ptr: number, data: Uint8Array): void {
    const mem = new Uint8Array(this.getMemory().buffer);
    mem.set(data, ptr);
  }

  private createDistributedImports(): WebAssembly.Imports {
    const runtime = this;

    return {
      wbg: {
        __wbg_new_8a6f238a6ece86ea: () => null,
        __wbg_stack_0ed75d68575b0f3c: () => {},
        __wbg_error_7534b8e9a36f1ab4: () => {},
        __wbindgen_init_externref_table: () => {},
        __wbindgen_cast_2241b6af4c4b2941: () => null,
      },
      cylon_host: {
        host_get_rank: (): number => {
          return runtime.comm.getRank();
        },

        host_get_world_size: (): number => {
          return runtime.comm.getWorldSize();
        },

        host_barrier: (): void => {
          runtime.comm.barrier();
        },

        host_broadcast: (
          dataPtr: number,
          dataLen: number,
          root: number,
          resultPtrOut: number,
          resultLenOut: number
        ): number => {
          try {
            const rank = runtime.comm.getRank();
            const data = rank === root
              ? runtime.readBytesInternal(dataPtr, dataLen)
              : new Uint8Array(0);

            const result = runtime.comm.broadcast(data, root);

            // Allocate and write result
            const mem = new DataView(runtime.getMemory().buffer);
            if (result.length > 0) {
              const resultPtr = runtime.getMalloc()(result.length, 1);
              runtime.writeBytesInternal(resultPtr, result);
              mem.setUint32(resultPtrOut, resultPtr, true);
              mem.setUint32(resultLenOut, result.length, true);
            } else {
              mem.setUint32(resultPtrOut, 0, true);
              mem.setUint32(resultLenOut, 0, true);
            }
            return 0;
          } catch (e) {
            console.error('host_broadcast error:', e);
            return 1;
          }
        },

        host_all_to_all: (
          partitionsPtr: number,
          numPartitions: number,
          resultsPtrOut: number,
          numResultsOut: number
        ): number => {
          try {
            // Read partition info from WASM memory
            const mem = new DataView(runtime.getMemory().buffer);
            const partitions: Uint8Array[] = [];

            for (let i = 0; i < numPartitions; i++) {
              const ptrOffset = partitionsPtr + i * 16;
              const pPtr = Number(mem.getBigUint64(ptrOffset, true));
              const pLen = Number(mem.getBigUint64(ptrOffset + 8, true));

              if (pPtr && pLen) {
                partitions.push(runtime.readBytesInternal(pPtr, pLen));
              } else {
                partitions.push(new Uint8Array(0));
              }
            }

            // Perform all-to-all
            const results = runtime.comm.allToAll(partitions);

            // Allocate results in WASM memory
            const numResults = results.length;
            const infoSize = numResults * 16;
            const infoPtr = runtime.getMalloc()(infoSize, 8);

            for (let i = 0; i < numResults; i++) {
              const result = results[i];
              if (result.length > 0) {
                const rPtr = runtime.getMalloc()(result.length, 1);
                runtime.writeBytesInternal(rPtr, result);
                mem.setBigUint64(infoPtr + i * 16, BigInt(rPtr), true);
                mem.setBigUint64(infoPtr + i * 16 + 8, BigInt(result.length), true);
              } else {
                mem.setBigUint64(infoPtr + i * 16, BigInt(0), true);
                mem.setBigUint64(infoPtr + i * 16 + 8, BigInt(0), true);
              }
            }

            // Write output pointers
            mem.setBigUint64(resultsPtrOut, BigInt(infoPtr), true);
            mem.setBigUint64(numResultsOut, BigInt(numResults), true);

            return 0;
          } catch (e) {
            console.error('host_all_to_all error:', e);
            return 1;
          }
        },

        host_gather: (
          dataPtr: number,
          dataLen: number,
          root: number,
          resultsPtrOut: number,
          numResultsOut: number
        ): number => {
          try {
            const data = runtime.readBytesInternal(dataPtr, dataLen);
            const results = runtime.comm.gather(data, root);
            const mem = new DataView(runtime.getMemory().buffer);

            if (results !== null) {
              const numResults = results.length;
              const infoSize = numResults * 16;
              const infoPtr = runtime.getMalloc()(infoSize, 8);

              for (let i = 0; i < numResults; i++) {
                const result = results[i];
                if (result.length > 0) {
                  const rPtr = runtime.getMalloc()(result.length, 1);
                  runtime.writeBytesInternal(rPtr, result);
                  mem.setBigUint64(infoPtr + i * 16, BigInt(rPtr), true);
                  mem.setBigUint64(infoPtr + i * 16 + 8, BigInt(result.length), true);
                } else {
                  mem.setBigUint64(infoPtr + i * 16, BigInt(0), true);
                  mem.setBigUint64(infoPtr + i * 16 + 8, BigInt(0), true);
                }
              }

              mem.setBigUint64(resultsPtrOut, BigInt(infoPtr), true);
              mem.setBigUint64(numResultsOut, BigInt(numResults), true);
            } else {
              mem.setBigUint64(resultsPtrOut, BigInt(0), true);
              mem.setBigUint64(numResultsOut, BigInt(0), true);
            }

            return 0;
          } catch (e) {
            console.error('host_gather error:', e);
            return 1;
          }
        },

        host_scatter: (
          partitionsPtr: number,
          numPartitions: number,
          root: number,
          resultPtrOut: number,
          resultLenOut: number
        ): number => {
          try {
            const rank = runtime.comm.getRank();
            let partitions: Uint8Array[] | null = null;
            const mem = new DataView(runtime.getMemory().buffer);

            if (rank === root && partitionsPtr && numPartitions > 0) {
              partitions = [];
              for (let i = 0; i < numPartitions; i++) {
                const pPtr = Number(mem.getBigUint64(partitionsPtr + i * 16, true));
                const pLen = Number(mem.getBigUint64(partitionsPtr + i * 16 + 8, true));
                if (pPtr && pLen) {
                  partitions.push(runtime.readBytesInternal(pPtr, pLen));
                } else {
                  partitions.push(new Uint8Array(0));
                }
              }
            }

            const result = runtime.comm.scatter(partitions, root);

            if (result.length > 0) {
              const rPtr = runtime.getMalloc()(result.length, 1);
              runtime.writeBytesInternal(rPtr, result);
              mem.setBigUint64(resultPtrOut, BigInt(rPtr), true);
              mem.setBigUint64(resultLenOut, BigInt(result.length), true);
            } else {
              mem.setBigUint64(resultPtrOut, BigInt(0), true);
              mem.setBigUint64(resultLenOut, BigInt(0), true);
            }

            return 0;
          } catch (e) {
            console.error('host_scatter error:', e);
            return 1;
          }
        },

        host_all_gather: (
          dataPtr: number,
          dataLen: number,
          resultsPtrOut: number,
          numResultsOut: number
        ): number => {
          try {
            const data = runtime.readBytesInternal(dataPtr, dataLen);
            const results = runtime.comm.allGather(data);
            const mem = new DataView(runtime.getMemory().buffer);

            const numResults = results.length;
            const infoSize = numResults * 16;
            const infoPtr = runtime.getMalloc()(infoSize, 8);

            for (let i = 0; i < numResults; i++) {
              const result = results[i];
              if (result.length > 0) {
                const rPtr = runtime.getMalloc()(result.length, 1);
                runtime.writeBytesInternal(rPtr, result);
                mem.setBigUint64(infoPtr + i * 16, BigInt(rPtr), true);
                mem.setBigUint64(infoPtr + i * 16 + 8, BigInt(result.length), true);
              } else {
                mem.setBigUint64(infoPtr + i * 16, BigInt(0), true);
                mem.setBigUint64(infoPtr + i * 16 + 8, BigInt(0), true);
              }
            }

            mem.setBigUint64(resultsPtrOut, BigInt(infoPtr), true);
            mem.setBigUint64(numResultsOut, BigInt(numResults), true);

            return 0;
          } catch (e) {
            console.error('host_all_gather error:', e);
            return 1;
          }
        },
      },
    };
  }

  // ===========================================================================
  // Distributed Operations (call WASM which orchestrates via host imports)
  // ===========================================================================

  /**
   * Distributed join operation.
   *
   * WASM handles orchestration:
   * 1. Hash partition both tables
   * 2. Call host_all_to_all for shuffle
   * 3. Local join
   */
  distributedJoin(left: Uint8Array, right: Uint8Array, config: JoinConfig): Uint8Array {
    const fn = this.getExportInternal<(
      retptr: number,
      lptr: number, llen: number,
      rptr: number, rlen: number,
      cptr: number, clen: number
    ) => void>('distributed_join');

    const [lptr, llen] = this.allocBytesInternal(left);
    const [rptr, rlen] = this.allocBytesInternal(right);
    const configJson = JSON.stringify({
      join_type: config.joinType,
      left_on: config.leftOn,
      right_on: config.rightOn,
    });
    const [cptr, clen] = this.allocStringInternal(configJson);
    const retptr = this.getMalloc()(16, 4);

    try {
      fn(retptr, lptr, llen, rptr, rlen, cptr, clen);
      const mem = new DataView(this.getMemory().buffer);
      const resultPtr = mem.getInt32(retptr, true);
      const resultLen = mem.getInt32(retptr + 4, true);
      return this.readBytesInternal(resultPtr, resultLen);
    } finally {
      this.freeMemoryInternal(retptr, 16);
    }
  }

  /**
   * Distributed union operation.
   */
  distributedUnion(left: Uint8Array, right: Uint8Array): Uint8Array {
    const fn = this.getExportInternal<(
      retptr: number,
      lptr: number, llen: number,
      rptr: number, rlen: number
    ) => void>('distributed_union');

    const [lptr, llen] = this.allocBytesInternal(left);
    const [rptr, rlen] = this.allocBytesInternal(right);
    const retptr = this.getMalloc()(16, 4);

    try {
      fn(retptr, lptr, llen, rptr, rlen);
      const mem = new DataView(this.getMemory().buffer);
      const resultPtr = mem.getInt32(retptr, true);
      const resultLen = mem.getInt32(retptr + 4, true);
      return this.readBytesInternal(resultPtr, resultLen);
    } finally {
      this.freeMemoryInternal(retptr, 16);
    }
  }

  /**
   * Distributed intersect operation.
   */
  distributedIntersect(left: Uint8Array, right: Uint8Array): Uint8Array {
    const fn = this.getExportInternal<(
      retptr: number,
      lptr: number, llen: number,
      rptr: number, rlen: number
    ) => void>('distributed_intersect');

    const [lptr, llen] = this.allocBytesInternal(left);
    const [rptr, rlen] = this.allocBytesInternal(right);
    const retptr = this.getMalloc()(16, 4);

    try {
      fn(retptr, lptr, llen, rptr, rlen);
      const mem = new DataView(this.getMemory().buffer);
      const resultPtr = mem.getInt32(retptr, true);
      const resultLen = mem.getInt32(retptr + 4, true);
      return this.readBytesInternal(resultPtr, resultLen);
    } finally {
      this.freeMemoryInternal(retptr, 16);
    }
  }

  /**
   * Distributed subtract operation.
   */
  distributedSubtract(left: Uint8Array, right: Uint8Array): Uint8Array {
    const fn = this.getExportInternal<(
      retptr: number,
      lptr: number, llen: number,
      rptr: number, rlen: number
    ) => void>('distributed_subtract');

    const [lptr, llen] = this.allocBytesInternal(left);
    const [rptr, rlen] = this.allocBytesInternal(right);
    const retptr = this.getMalloc()(16, 4);

    try {
      fn(retptr, lptr, llen, rptr, rlen);
      const mem = new DataView(this.getMemory().buffer);
      const resultPtr = mem.getInt32(retptr, true);
      const resultLen = mem.getInt32(retptr + 4, true);
      return this.readBytesInternal(resultPtr, resultLen);
    } finally {
      this.freeMemoryInternal(retptr, 16);
    }
  }

  /**
   * Distributed groupby operation.
   */
  distributedGroupBy(data: Uint8Array, config: GroupByConfig): Uint8Array {
    const fn = this.getExportInternal<(
      retptr: number,
      dptr: number, dlen: number,
      cptr: number, clen: number
    ) => void>('distributed_groupby');

    const [dptr, dlen] = this.allocBytesInternal(data);
    const configJson = JSON.stringify(config);
    const [cptr, clen] = this.allocStringInternal(configJson);
    const retptr = this.getMalloc()(16, 4);

    try {
      fn(retptr, dptr, dlen, cptr, clen);
      const mem = new DataView(this.getMemory().buffer);
      const resultPtr = mem.getInt32(retptr, true);
      const resultLen = mem.getInt32(retptr + 4, true);
      return this.readBytesInternal(resultPtr, resultLen);
    } finally {
      this.freeMemoryInternal(retptr, 16);
    }
  }

  // Internal helpers that access parent's private members
  private getExportInternal<T>(name: string): T {
    const instance = (this as any).instance;
    if (!instance) {
      throw new Error('WASM runtime not initialized');
    }
    const exports = instance.exports as Record<string, unknown>;
    const fn = exports[name];
    if (!fn) {
      throw new Error(`WASM export '${name}' not found`);
    }
    return fn as T;
  }

  private allocBytesInternal(data: Uint8Array): [number, number] {
    const ptr = this.getMalloc()(data.length, 1);
    this.writeBytesInternal(ptr, data);
    return [ptr, data.length];
  }

  private allocStringInternal(str: string): [number, number] {
    const encoder = new TextEncoder();
    const bytes = encoder.encode(str);
    return this.allocBytesInternal(bytes);
  }

  private freeMemoryInternal(ptr: number, length: number): void {
    const free = (this as any).free;
    if (ptr && free) {
      free(ptr, length, 1);
    }
  }
}

/**
 * Create a distributed WASM runtime.
 *
 * @param wasmPath - Path to WASM binary
 * @param comm - Communication backend (default: LocalBackend)
 * @returns Initialized DistributedWasmRuntime
 *
 * @example
 * ```typescript
 * // Local mode (single node)
 * const runtime = await createDistributedRuntime();
 * const result = runtime.distributedJoin(left, right, {
 *   joinType: 'inner',
 *   leftOn: [0],
 *   rightOn: [0]
 * });
 * ```
 */
export async function createDistributedRuntime(
  wasmPath?: string,
  comm?: CommunicationBackend
): Promise<DistributedWasmRuntime> {
  const runtime = new DistributedWasmRuntime(comm);
  await runtime.initialize(wasmPath || getDefaultWasmPath());
  return runtime;
}
