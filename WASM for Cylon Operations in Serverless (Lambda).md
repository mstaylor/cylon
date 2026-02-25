# WASM for Cylon Operations in Serverless (Lambda)

## Architecture Overview

Cylon's WASM deployment uses a **host import model**: the WASM module contains all compute operations (join, filter, sort, groupby, set operations, SIMD similarity) while the host (Node.js) provides communication primitives (all-to-all, broadcast, gather, barrier) via the native `cylon-node` addon.

```
┌─────────────────────────────────────────────────────────────────┐
│  Node.js Host Runtime                                          │
│  ├─ cylon-node (napi-rs native addon)                          │
│  │   └─ Communicator: FMI / UCX / Libfabric backends           │
│  ├─ CylonWasmHost (TypeScript)                                 │
│  │   └─ Bridges WASM imports → cylon-node communicator         │
│  └─ Application logic (Lambda handler, orchestration)          │
└────────────────────┬────────────────────────────────────────────┘
                     │ WebAssembly imports/exports
┌────────────────────▼────────────────────────────────────────────┐
│  cylon-wasm (WASM Module, ~4.3MB)                              │
│  ├─ Table operations: join, filter, sort, groupby              │
│  ├─ Set operations: union, intersect, subtract, unique         │
│  ├─ Aggregates: sum, min, max, count, mean, variance, stddev  │
│  ├─ SIMD: cosine_similarity, euclidean_distance, dot_product   │
│  ├─ Distributed ops: distributed_join, distributed_groupby,   │
│  │   distributed_union, distributed_intersect, etc.            │
│  └─ Data format: Arrow IPC + JSON two-layer table design       │
└─────────────────────────────────────────────────────────────────┘
```

### Why This Architecture?

WASM cannot access sockets, files, or networks directly. Instead of embedding a full WASM runtime (Wasmtime/Wasmer) inside a native Rust Lambda, Cylon uses **host imports** — the WASM module declares extern functions that the host must provide. This gives us:

- Compute kernel runs in portable WASM (same binary for x86_64 + ARM64)
- Communication stays in native code (FMI/TCPunch for Lambda, UCX for HPC)
- No WASM runtime overhead — Node.js V8 engine runs WASM natively
- Clean separation: WASM is stateless compute, host owns network state

---

## Project Structure

```
rust/
├── src/                        # Main Cylon Rust library (v0.7.0)
│   ├── table.rs                # Table struct and operations
│   ├── net/                    # Communication backends
│   │   ├── fmi/                # FMI (Redis + TCPunch)
│   │   ├── ucx/                # UCX
│   │   └── libfabric/          # Libfabric
│   └── ...
│
├── cylon-wasm/                 # WASM module (compute kernel)
│   ├── Cargo.toml              # cdylib + rlib, wasm-bindgen
│   ├── src/
│   │   ├── lib.rs              # Module root, init(), simd_available()
│   │   ├── table.rs            # Two-layer table: TableData (JSON) ↔ Table (Arrow)
│   │   ├── join.rs             # Hash join via cylon's hash_join_batches()
│   │   ├── groupby.rs          # Hash-based groupby with aggregations
│   │   ├── filter.rs           # Row filtering with predicates
│   │   ├── simd.rs             # SIMD: sum, dot_product, cosine_similarity, euclidean_distance
│   │   ├── ops.rs              # project, slice, sort, merge, set ops, aggregates, hash_partition
│   │   ├── api.rs              # wasm_bindgen exports (57 functions)
│   │   ├── imports.rs          # Host import declarations (extern "C" in "cylon_host")
│   │   ├── distributed.rs      # Distributed ops using host imports
│   │   └── error.rs            # WasmError → JsValue conversion
│   ├── tests/                  # Unit tests per module
│   ├── host/js/                # TypeScript host runtime
│   │   ├── src/index.ts        # CylonWasmHost class
│   │   └── tests/              # Integration + distributed tests
│   └── pkg/                    # wasm-pack output (after build)
│
└── cylon-node/                 # Node.js native addon (napi-rs)
    ├── Cargo.toml              # Features: fmi, ucx, libfabric
    ├── src/lib.rs              # Communicator class with all collectives
    ├── index.js                # Platform-aware loader
    ├── index.d.ts              # TypeScript type definitions
    └── test/test.js            # Smoke tests
```

---

## Build and Test

### Prerequisites

```bash
# Install wasm-pack (one-time)
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Install wasm32 target (one-time)
rustup target add wasm32-unknown-unknown
```

### Build WASM Module

```bash
cd rust/cylon-wasm

# Standard build (nodejs target for Lambda/server-side)
wasm-pack build --target nodejs --release

# With SIMD enabled (128-bit vectors)
RUSTFLAGS='-C target-feature=+simd128' wasm-pack build --target nodejs --release --features simd

# For browser deployment
wasm-pack build --target web --release
```

Output in `pkg/`:
- `cylon_wasm_bg.wasm` — WASM binary (~4.3MB)
- `cylon_wasm.js` — JavaScript bindings (auto-generated)
- `cylon_wasm.d.ts` — TypeScript types
- `cylon_host.js` — Mock host module for local testing

### Build Node.js Addon

```bash
cd rust/cylon-node
npm run build    # Release build
npm run build:debug  # Debug build
```

Output: `cylon-node.linux-arm64-gnu.node` (or platform-appropriate binary)

### Run Tests

```bash
# Local WASM integration tests (no FMI required)
cd rust/cylon-wasm/host/js
npm test -- --testPathPattern=integration

# cylon-node smoke tests (requires Redis)
cd rust/cylon-node
REDIS_HOST=10.211.55.2 npm test

# Rust unit tests (no special setup)
cd rust
cargo test
```

### Verified Test Results

```
PASS tests/integration.test.ts
  WASM Local Operations
    ✓ WASM module has expected exports (57 exports)
    ✓ version returns a string
    ✓ json_to_ipc and ipc_to_json roundtrip
    ✓ local join operation works
    ✓ filter operation works
    ✓ sort operation works
    ✓ groupby operation works
    ✓ union operation works
    ✓ intersect operation works
    ✓ subtract operation works
    ✓ compute aggregates work

Tests: 11 passed, 11 total
```

---

## WASM Exports (API)

### Table Operations

| Export | Description |
|--------|-------------|
| `join_tables(left, right, config)` | Hash join (inner, left, right, full_outer) |
| `filter_table(table, predicates)` | Row selection with AND/OR predicate logic |
| `groupby_table(table, config)` | Hash groupby with sum/mean/min/max/count |
| `cylon_groupby(table, config)` | Native cylon groupby implementation |
| `sort_table(table, col, asc)` | Single-column sort |
| `sort_table_multi(table, cols, ascs)` | Multi-column sort |
| `project_table(table, cols)` | Column selection/projection |
| `slice_table(table, offset, len)` | Row slicing |
| `head_table(table, n)` / `tail_table(table, n)` | First/last N rows |
| `merge_tables(left, right)` | Vertical concatenation |

### Set Operations

| Export | Description |
|--------|-------------|
| `union_tables(left, right)` | Rows from both tables (deduplicated) |
| `intersect_tables(left, right)` | Rows present in both tables |
| `subtract_tables(left, right)` | Rows in left not in right |
| `unique_table(table)` | Remove duplicate rows |

### Aggregates

| Export | Description |
|--------|-------------|
| `compute_sum(table, col)` | Column sum |
| `compute_min(table, col)` / `compute_max(table, col)` | Column min/max |
| `compute_count(table, col)` | Non-null count |
| `compute_mean(table, col)` | Column mean |
| `compute_variance(table, col)` / `compute_stddev(table, col)` | Statistics |

### SIMD Operations (feature `simd`)

| Export | Description |
|--------|-------------|
| `sum_f32(data)` / `sum_f64(data)` | Vectorized sum |
| `dot_product_f32(a, b)` | SIMD dot product |
| `cosine_similarity_f32(a, b)` | Cosine similarity |
| `euclidean_distance_f32(a, b)` | Euclidean distance |

### Distributed Operations (via host imports)

| Export | Description |
|--------|-------------|
| `distributed_join(left, right, config)` | Hash partition → all-to-all shuffle → local join |
| `distributed_groupby(table, config)` | Local aggregate → partition → shuffle → final aggregate |
| `distributed_union(left, right)` | Hash partition → merge → all-to-all → unique |
| `distributed_intersect(left, right)` | Distributed intersection |
| `distributed_subtract(left, right)` | Distributed subtraction |

### Utilities

| Export | Description |
|--------|-------------|
| `hash_partition(table, col, n)` | Split table into N partitions by column hash |
| `json_to_ipc(json)` | Convert JSON table data to Arrow IPC binary |
| `ipc_to_json(ipc)` | Convert Arrow IPC to JSON (debugging) |
| `table_info(table)` | Schema, row/column counts |
| `init()` | Initialize WASM module |
| `simd_available()` | Check SIMD support |

---

## Host Import Interface

WASM declares these extern functions in the `"cylon_host"` module. The host must provide implementations.

```rust
extern "C" {
    // Context
    fn host_get_rank() -> i32;
    fn host_get_world_size() -> i32;

    // Synchronization
    fn host_barrier();

    // Collectives (data passed as ptr+len, results via output pointers)
    fn host_all_to_all(partitions_ptr, num_partitions, results_ptr_out, num_results_out) -> i32;
    fn host_broadcast(data_ptr, data_len, root, result_ptr_out, result_len_out) -> i32;
    fn host_gather(data_ptr, data_len, root, result_ptr_out, num_results_out) -> i32;
    fn host_scatter(partitions_ptr, num_partitions, root, result_ptr_out, result_len_out) -> i32;
    fn host_all_gather(data_ptr, data_len, result_ptr_out, num_results_out) -> i32;

    // Memory management (WASM exports these for host to allocate in WASM memory)
    fn wasm_alloc(size) -> *mut u8;
    fn wasm_free(ptr, size);
}
```

### TypeScript Host Implementation

The `CylonWasmHost` class in `host/js/src/index.ts` bridges these imports to `cylon-node`:

```typescript
import { Communicator } from '@aspect/cylon-node';

class CylonWasmHost {
    private communicator: Communicator;
    private memory: WasmMemory;

    getImports(): WebAssembly.Imports {
        return {
            cylon_host: {
                host_get_rank: () => this.communicator.getRank(),
                host_get_world_size: () => this.communicator.getWorldSize(),
                host_barrier: () => this.communicator.barrier(),
                host_all_to_all: (partitions_ptr, num_partitions, results_ptr_out, num_results_out) => {
                    // Read partition data from WASM memory
                    const partitions = this.memory.readPartitions(partitions_ptr, num_partitions);
                    // Execute via native communicator
                    const results = this.communicator.allToAll(partitions);
                    // Write results back to WASM memory
                    this.memory.writePartitions(results, results_ptr_out, num_results_out);
                    return 0;
                },
                // ... similar for broadcast, gather, scatter, all_gather
            }
        };
    }
}
```

---

## Data Format: Two-Layer Table Design

```
JavaScript/Python (JSON) ↔ TableData (serializable) ↔ Table (Arrow RecordBatch)
```

### JSON Format (for API boundary)

```json
{
    "columns": [
        {"name": "id", "type": "Int32", "data": [1, 2, 3]},
        {"name": "value", "type": "Float64", "data": [1.5, 2.5, 3.5]}
    ]
}
```

### Arrow IPC (for distributed operations)

Distributed operations use Arrow IPC binary format for serialization — not JSON. This avoids the overhead of text serialization for large datasets being shuffled across workers.

---

## cylon-node: Communication Layer

The `@aspect/cylon-node` native addon exposes the full `Communicator` trait to JavaScript via napi-rs.

### Supported Backends

| Backend | Feature Flag | Use Case |
|---------|-------------|----------|
| FMI | `fmi` (default) | Lambda serverless (Redis OOB + TCPunch P2P) |
| UCX | `ucx` | HPC/cloud VMs with RDMA |
| UCC | `ucc` | Enhanced UCX collectives |
| Libfabric | `libfabric` | AWS EFA, other fabrics |
| MPI | `mpi` | Not supported in Node.js |

### API

```javascript
const { Communicator } = require('@aspect/cylon-node');

// Create FMI communicator for Lambda
const comm = Communicator.createFmi({
    rank: 0,
    worldSize: 4,
    host: 'tcpunch-server.example.com',
    port: 10000,
    redisHost: 'redis.example.com',
    redisPort: 6379,
    redisNamespace: 'my-experiment',
    nonblocking: true,
    maxTimeout: 60000,
});

// Collectives
comm.barrier();
const allData = comm.allGather(myBuffer);
const shuffled = comm.allToAll(partitionBuffers);
const result = comm.broadcast(data, rootRank);

// Point-to-point
comm.send(data, destRank, tag);
const received = comm.recv(sourceRank, tag);

// Cleanup
// (communicator is dropped when GC'd)
```

---

## Deployment: Lambda with WASM

### Option 1: Node.js Lambda (Current Approach)

The current deployment uses a Node.js Lambda function that loads the WASM module via `wasm-pack` output and uses `cylon-node` for FMI communication.

```
Lambda Function (Node.js 18+)
├─ @aspect/cylon-node      (native .node addon — FMI communicator)
├─ @aspect/cylon-wasm-host (TypeScript — bridges WASM ↔ cylon-node)
└─ cylon_wasm_bg.wasm      (WASM compute kernel)
```

This is the architecture used for the Frontiers paper experiments (join, groupby, microbenchmarks at 1–64 nodes on Lambda).

### Option 2: Native Rust Lambda (Alternative)

For maximum performance without the WASM layer:

```
Lambda Function (Rust, provided.al2 runtime)
└─ cylon library compiled natively with FMI feature
```

This eliminates the ~50-60% WASM overhead but loses portability. Use when:
- Only targeting Lambda x86_64 or ARM64 (not both)
- Need maximum join/groupby throughput
- Don't need browser/edge deployment

### Docker Build (Node.js Lambda)

```dockerfile
FROM public.ecr.aws/lambda/nodejs:18

# Copy pre-built artifacts
COPY cylon-node/cylon-node.linux-x64-gnu.node ./
COPY cylon-wasm/pkg/ ./wasm/
COPY host/js/dist/ ./host/
COPY handler.js ./

CMD ["handler.handler"]
```

---

## Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| **cylon-wasm** (compute kernel) | **Complete** | 57 exports, all table ops + SIMD + distributed |
| **cylon-node** (native addon) | **Complete** | FMI/UCX/Libfabric backends, all collectives |
| **Host runtime** (TypeScript) | **Complete** | CylonWasmHost bridges imports to cylon-node |
| **Local operations** | **Tested** | 11/11 integration tests passing |
| **WASM build** | **Working** | `wasm-pack build --target nodejs --release` succeeds |
| **Distributed operations** | **Implemented** | Requires FMI infrastructure (Redis + TCPunch) |
| **SIMD operations** | **Implemented** | Feature flag `simd`, 128-bit WASM SIMD |
| **Browser target** | **Implemented** | `wasm-pack build --target web` |
| **Lambda deployment** | **Validated** | Used in Frontiers paper experiments (1–64 nodes) |

---

## Environment Setup

See [rust/ENVIRONMENT_SETUP.md](./rust/ENVIRONMENT_SETUP.md) for:
- Redis connectivity in Parallels VM
- MPI test setup with conda OpenMPI
- WASM integration test commands
- Distributed WASM test requirements (Redis + TCPunch)