# Cylon WASM Distributed Architecture

## Overview

This document describes the architecture for running Cylon distributed operations across different environments:

- **Python**: Native pycylon (Cython/C++ with FMI/MPI/UCX/UCC)
- **Node.js/Browser**: cylon-wasm (Rust compiled to WASM)

## Design Goals

1. **Native Performance for Python**: Use native pycylon with full C++ performance
2. **Portable WASM for JavaScript**: Single WASM binary for Node.js and browser
3. **Direct Communication**: Support for FMI/MPI/UCX/UCC - not limited to S3/Redis shuffle
4. **Zero-Copy Data Transfer**: Arrow IPC format for efficient data exchange
5. **Pluggable Communication**: Same communication backends work across platforms

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              Python                                      │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                    pycylon (Cython/C++)                             │ │
│  │                                                                      │ │
│  │  Native implementation with full performance:                       │ │
│  │  • distributed_join()    • distributed_sort()                       │ │
│  │  • distributed_union()   • distributed_groupby()                    │ │
│  │                                                                      │ │
│  │  Communication backends:                                            │ │
│  │  • FMI (Redis + TCP)     • MPI                                      │ │
│  │  • UCX/UCC               • S3 (for large data)                      │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                         Node.js / Browser                                │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                       cylon-wasm (Rust)                             │ │
│  │                                                                      │ │
│  │  WASM compute kernel with host imports:                             │ │
│  │  • join_tables()         • groupby_table()                          │ │
│  │  • filter_table()        • hash_partition()                         │ │
│  │  • union_tables()        • distributed_* (via host imports)         │ │
│  │                                                                      │ │
│  │  Host imports (provided by Node.js):                                │ │
│  │  • host_all_to_all()     • host_broadcast()                         │ │
│  │  • host_barrier()        • host_gather/scatter()                    │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                    │                                     │
│  ┌─────────────────────────────────▼──────────────────────────────────┐ │
│  │              @cylon/wasm (TypeScript loader)                        │ │
│  │                                                                      │ │
│  │  Provides host imports using FMI backends:                          │ │
│  │  • LocalBackend (single-node)                                       │ │
│  │  • Redis (serverless coordination)                                  │ │
│  │  • S3 (data exchange for large payloads)                            │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                          Direct Communication
                          (FMI / MPI / UCX / UCC)
                                    │
                                    ▼
                         ┌───────────────────┐
                         │   Other Workers   │
                         │  (Lambda / EC2 /  │
                         │   Containers)     │
                         └───────────────────┘
```

**Key Design Decision: Native for Python, WASM for JavaScript**

- **Python**: Use native pycylon directly for best performance. No WASM overhead.
- **Node.js/Browser**: Use cylon-wasm with host imports for portable compute.

## Components

### 1. cylon-wasm (Rust → WASM)

**Location**: `rust/cylon-wasm/`

**Purpose**: Pure compute kernel compiled to WebAssembly

**Features**:
- Table operations (join, filter, project, sort, etc.)
- Set operations (union, intersect, subtract, unique)
- GroupBy and aggregations
- Hash partitioning for distributed operations
- Distributed operations that use host imports for communication

**Does NOT include**:
- Tokio runtime
- Socket/network code
- File system access
- Any OS-specific code

### 2. Host Imports

WASM cannot directly access network or call native libraries. Instead, it declares "imports" - functions that the host environment must provide.

**Declared in Rust**:
```rust
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = cylonHost)]
    fn all_to_all(data: &[u8], partition_sizes: &[u32]) -> Vec<u8>;

    #[wasm_bindgen(js_namespace = cylonHost)]
    fn broadcast(data: &[u8], root: i32) -> Vec<u8>;

    #[wasm_bindgen(js_namespace = cylonHost)]
    fn gather(data: &[u8], root: i32) -> Vec<u8>;

    #[wasm_bindgen(js_namespace = cylonHost)]
    fn scatter(data: &[u8], sizes: &[u32], root: i32) -> Vec<u8>;

    #[wasm_bindgen(js_namespace = cylonHost)]
    fn get_rank() -> i32;

    #[wasm_bindgen(js_namespace = cylonHost)]
    fn get_world_size() -> i32;

    #[wasm_bindgen(js_namespace = cylonHost)]
    fn barrier();
}
```

### 3. pycylon (Python)

**Location**: `python/pycylon/`

**pycylon**: Native Python extension (Cython/C++) with full cylon functionality including FMI/MPI/UCX communication.

Python does not use WASM - it uses the native C++ implementation directly for best performance.

### 4. @cylon/wasm (Node.js)

**Location**: `js/packages/native/` and `js/packages/wasm/`

**@cylon/native**: N-API native addon with full cylon functionality including FMI/MPI/UCX communication.

**@cylon/wasm**: WASM loader that:
- Loads the cylon-wasm module via WebAssembly API
- Provides host import implementations that bridge to @cylon/native

## Data Flow: Distributed Join Example

```
Step 1: User calls distributed_join()
        ┌─────────────────────────────────────────────────────┐
        │  cylon.distributed_join(left, right, config)        │
        └─────────────────────────────────────────────────────┘
                                    │
                                    ▼
Step 2: WASM partitions data locally
        ┌─────────────────────────────────────────────────────┐
        │  left_parts = hash_partition(left, keys, world_size)│
        │  right_parts = hash_partition(right, keys, world_size)
        └─────────────────────────────────────────────────────┘
                                    │
                                    ▼
Step 3: WASM calls host import for shuffle
        ┌─────────────────────────────────────────────────────┐
        │  left_received = cylonHost.all_to_all(left_parts)   │
        │  right_received = cylonHost.all_to_all(right_parts) │
        └─────────────────────────────────────────────────────┘
                                    │
                     ┌──────────────┴──────────────┐
                     ▼                             ▼
Step 4: Host performs network communication
        ┌─────────────────────┐    ┌─────────────────────┐
        │  Python: pycylon    │ OR │  Node: @cylon/native│
        │  fmi.all_to_all()   │    │  fmi.allToAll()     │
        └─────────────────────┘    └─────────────────────┘
                     │                             │
                     └──────────────┬──────────────┘
                                    │
                          FMI/MPI/UCX Network
                                    │
                                    ▼
Step 5: Data received from other workers
        ┌─────────────────────────────────────────────────────┐
        │  Returns shuffled data to WASM                      │
        └─────────────────────────────────────────────────────┘
                                    │
                                    ▼
Step 6: WASM performs local join
        ┌─────────────────────────────────────────────────────┐
        │  result = join_tables(left_received, right_received)│
        └─────────────────────────────────────────────────────┘
                                    │
                                    ▼
Step 7: Return result to user
        ┌─────────────────────────────────────────────────────┐
        │  return result (Arrow IPC bytes)                    │
        └─────────────────────────────────────────────────────┘
```

## Project Structure

```
cylon/
├── rust/
│   ├── Cargo.toml              # Main crate (with runtime feature)
│   ├── src/                    # Core cylon implementation
│   │   ├── lib.rs
│   │   ├── table.rs
│   │   ├── join/
│   │   ├── ops/
│   │   ├── net/
│   │   │   ├── fmi/            # FMI communication
│   │   │   ├── mpi/            # MPI communication
│   │   │   └── ucx/            # UCX/UCC communication
│   │   └── ...
│   │
│   ├── cylon-wasm/             # WASM module
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── api.rs          # WASM-bindgen exports
│   │   │   ├── imports.rs      # Host import declarations
│   │   │   ├── distributed.rs  # Distributed ops using imports
│   │   │   ├── table.rs
│   │   │   ├── join.rs
│   │   │   ├── ops.rs
│   │   │   └── ...
│   │   └── pkg/                # wasm-pack output
│   │
│   └── cylon-node/             # N-API addon for Node.js
│       ├── Cargo.toml
│       └── src/
│           └── lib.rs
│
├── python/
│   └── pycylon/                # Native Python extension (Cython/C++)
│       ├── pycylon/            # Python package
│       └── src/                # Cython sources
│
└── js/
    └── packages/
        ├── wasm/               # @cylon/wasm
        │   ├── package.json
        │   ├── src/
        │   │   ├── index.ts
        │   │   ├── loader.ts   # WASM loader + host imports
        │   │   └── types.ts
        │   └── pkg/            # cylon-wasm output copied here
        │
        └── native/             # @cylon/native (N-API)
            ├── package.json
            └── src/
                └── index.ts
```

## Build Outputs

| Build Command | Output | Usage |
|---------------|--------|-------|
| `wasm-pack build rust/cylon-wasm` | `cylon_wasm.wasm` + JS bindings | Node.js / Browser |
| `maturin build -m python/pycylon` | `pycylon-*.whl` | Python (native) |
| `napi build rust/cylon-node` | `cylon.node` | Node.js (native addon) |

## Feature Flags

### Main Cylon Crate (`rust/Cargo.toml`)

```toml
[features]
default = ["runtime", "parquet"]

# Runtime features (not available in WASM)
runtime = ["dep:tokio", "dep:socket2", "dep:libc"]

# Communication backends
mpi = ["dep:mpi", "dep:mpi-sys", "runtime"]
fmi = ["dep:redis", "runtime"]
ucx = ["dep:redis", "dep:bindgen", "runtime"]
ucc = ["ucx"]

# Optional features
parquet = ["dep:parquet"]
datafusion = ["dep:datafusion"]
```

### WASM Crate (`rust/cylon-wasm/Cargo.toml`)

```toml
[dependencies]
# Use cylon WITHOUT runtime feature
cylon = { path = "..", default-features = false }
```

## Communication Backends

### FMI (Fault-tolerant Messaging Interface)

- Custom protocol for Lambda-to-Lambda direct communication
- Uses Redis for coordination/discovery
- TCP for data transfer with NAT traversal

### MPI (Message Passing Interface)

- Standard HPC communication
- Works on EC2, ECS, Kubernetes with MPI runtime
- Not available on Lambda

### UCX/UCC

- High-performance communication for InfiniBand/RoCE
- Used in HPC environments
- Provides collective operations (all-to-all, broadcast, etc.)

## Usage Examples

### Python (Native pycylon)

```python
from pycylon import CylonContext, Table
from pycylon.net.fmi_config import FMIConfig

# Initialize FMI context
config = FMIConfig(
    rank=0, world_size=4,
    host="tcpunch-server", port=9999,
    redis_host="redis-host", redis_port=6379
)
ctx = CylonContext(config=config, distributed=True)

# Load tables
left = Table.from_arrow(ctx, left_arrow_table)
right = Table.from_arrow(ctx, right_arrow_table)

# Distributed join - native C++ with FMI communication
result = left.distributed_join(right, join_type="inner",
                                left_on=[0], right_on=[0])
```

### Node.js (WASM)

```typescript
import { loadCylonWasm } from '@cylon/wasm';
import { initFMI } from '@cylon/native';

// Initialize FMI
const ctx = await initFMI({ redisUrl: 'redis://localhost:6379' });

// Load WASM with host imports connected to FMI
const cylon = await loadCylonWasm(ctx);

// Distributed join - WASM computes, FMI communicates
const result = cylon.distributedJoin(leftData, rightData, {
    joinType: 'inner',
    leftOn: [0],
    rightOn: [0]
});
```

### Lambda Handler (Python - Native)

```python
# lambda_handler.py
from pycylon import CylonContext, Table
from pycylon.net.fmi_config import FMIConfig

# Initialize once per container
config = FMIConfig(...)
ctx = CylonContext(config=config, distributed=True)

def handler(event, context):
    operation = event['operation']

    if operation == 'distributed_join':
        # Get data from S3 or event
        left = Table.from_arrow(ctx, get_data(event['left']))
        right = Table.from_arrow(ctx, get_data(event['right']))

        # Execute distributed join - native C++ with FMI
        result = left.distributed_join(right, **event['config'])

        # Store result
        return store_result(result.to_arrow())
```

### Lambda Handler (Node.js - WASM)

```typescript
// lambda_handler.ts
import { loadCylonWasm, LocalBackend } from '@cylon/wasm';

// Initialize once per container
const cylon = await loadCylonWasm(new LocalBackend());

export async function handler(event: any) {
    if (event.operation === 'join') {
        const result = cylon.joinTables(
            event.left,
            event.right,
            event.config
        );
        return { statusCode: 200, body: result };
    }
}
```

## Performance Considerations

1. **Arrow IPC**: Near zero-copy serialization between WASM and host
2. **Host Import Overhead**: Each host call has ~microsecond overhead
3. **Batch Operations**: Minimize host calls by batching data
4. **Memory**: WASM has linear memory, Arrow buffers are contiguous

## Future Enhancements

1. **WASI Sockets**: When WASI networking stabilizes, WASM could do communication directly
2. **Shared Memory**: Zero-copy between WASM and native via shared ArrayBuffer
3. **SIMD**: WASM SIMD for vectorized compute operations
4. **Threads**: WASM threads for parallel compute within a worker