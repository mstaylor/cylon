# Table Architecture in Cylon WASM

## Overview

The table module implements a **two-layer design** that bridges high-performance Arrow computation with JSON-based JavaScript/Python interoperability.

## The Problem

### Arrow RecordBatch (Internal)
- Columnar memory layout optimized for computation
- Zero-copy operations and cache-friendly access
- Shared algorithms with native Cylon Rust code

### WASM Memory Boundary
- WASM linear memory is isolated from JavaScript/Python
- Cannot directly share Arrow memory across the boundary
- Need serialization for data exchange

## Solution: Two-Layer Table Design

```
┌─────────────────────────────────────────────────────────────────┐
│                     JavaScript / Python                          │
│                                                                  │
│   const data = {                                                 │
│     columns: ["id", "value"],                                    │
│     data: [                                                      │
│       { type: "Int32", data: [1, 2, 3] },                       │
│       { type: "Float64", data: [1.1, 2.2, 3.3] }                │
│     ]                                                            │
│   };                                                             │
└──────────────────────────┬──────────────────────────────────────┘
                           │ JSON.stringify() / JSON.parse()
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      TableData (Serializable)                    │
│                                                                  │
│   struct TableData {                                             │
│       columns: Vec<String>,                                      │
│       data: Vec<ColumnData>,                                     │
│   }                                                              │
│                                                                  │
│   Methods: from_json(), to_json(), to_json_pretty()              │
└──────────────────────────┬──────────────────────────────────────┘
                           │ to_record_batch() / from_record_batch()
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Table (Computation Wrapper)                 │
│                                                                  │
│   struct Table {                                                 │
│       batch: RecordBatch,  // Arrow columnar data               │
│   }                                                              │
│                                                                  │
│   Used by: join, groupby, filter, aggregation operations         │
└─────────────────────────────────────────────────────────────────┘
```

## Layer 1: TableData (Serialization Layer)

### Purpose
- JSON-serializable representation of tabular data
- Handles type information through tagged enums
- Manages null values with `Option<T>`

### Structure

```rust
#[derive(Serialize, Deserialize)]
pub struct TableData {
    pub columns: Vec<String>,      // Column names
    pub data: Vec<ColumnData>,     // Column values with type tags
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum ColumnData {
    Int32(Vec<Option<i32>>),
    Int64(Vec<Option<i64>>),
    Float32(Vec<Option<f32>>),
    Float64(Vec<Option<f64>>),
    String(Vec<Option<String>>),
    Boolean(Vec<Option<bool>>),
}
```

### JSON Format

```json
{
  "columns": ["id", "name", "value"],
  "data": [
    { "type": "Int32", "data": [1, 2, 3, null, 5] },
    { "type": "String", "data": ["alice", "bob", null, "dave", "eve"] },
    { "type": "Float64", "data": [1.1, 2.2, 3.3, 4.4, 5.5] }
  ]
}
```

### Key Features
- **Tagged enum serialization**: The `#[serde(tag = "type", content = "data")]` attribute creates self-describing JSON
- **Null handling**: `Option<T>` naturally serializes to JSON `null`
- **No Arrow dependency in JSON**: Clean interchange format

## Layer 2: Table (Computation Layer)

### Purpose
- Thin wrapper around Arrow `RecordBatch`
- Provides interface for Cylon operations
- Enables code sharing with native Cylon

### Structure

```rust
pub struct Table {
    batch: RecordBatch,
}

impl Table {
    pub fn from_table_data(data: &TableData) -> WasmResult<Self>;
    pub fn to_table_data(&self) -> WasmResult<TableData>;

    // Access methods
    pub fn batch(&self) -> &RecordBatch;
    pub fn num_rows(&self) -> usize;
    pub fn num_columns(&self) -> usize;
    pub fn column(&self, index: usize) -> Option<&ArrayRef>;
}
```

### Why Wrap RecordBatch?
1. **Abstraction**: Hide Arrow internals from API consumers
2. **Error handling**: Convert Arrow errors to WASM-friendly errors
3. **Future flexibility**: Can change internal representation without API changes
4. **Consistency**: Match native Cylon's Table interface

## Data Flow

### Input Flow (JS → WASM)

```
JavaScript Object
       │
       ▼ JSON.stringify()
JSON String
       │
       ▼ WASM boundary (string copy)
Rust &str
       │
       ▼ TableData::from_json()
TableData
       │
       ▼ Table::from_table_data()
Table (RecordBatch)
       │
       ▼ Operation (join, groupby, etc.)
Result Table
```

### Output Flow (WASM → JS)

```
Result Table
       │
       ▼ table.to_table_data()
TableData
       │
       ▼ table_data.to_json()
JSON String
       │
       ▼ WASM boundary (string copy)
JavaScript String
       │
       ▼ JSON.parse()
JavaScript Object
```

## Example Usage

### From JavaScript

```javascript
// Prepare input data
const leftTable = {
    columns: ["id", "name"],
    data: [
        { type: "Int32", data: [1, 2, 3] },
        { type: "String", data: ["alice", "bob", "charlie"] }
    ]
};

const rightTable = {
    columns: ["id", "department"],
    data: [
        { type: "Int32", data: [1, 2, 4] },
        { type: "String", data: ["engineering", "sales", "marketing"] }
    ]
};

// Call WASM function
const result = wasm.join_tables(
    JSON.stringify(leftTable),
    JSON.stringify(rightTable),
    JSON.stringify({ join_type: "inner", left_on: [0], right_on: [0] })
);

// Parse result
const outputTable = JSON.parse(result);
console.log(outputTable.columns);  // ["id", "name", "department"]
```

### Internal Rust Implementation

```rust
#[wasm_bindgen]
pub fn join_tables(left_json: &str, right_json: &str, config_json: &str) -> Result<String, JsValue> {
    // 1. Parse JSON to TableData
    let left_data = TableData::from_json(left_json)?;
    let right_data = TableData::from_json(right_json)?;

    // 2. Convert to Table (Arrow RecordBatch)
    let left_table = Table::from_table_data(&left_data)?;
    let right_table = Table::from_table_data(&right_data)?;

    // 3. Parse config
    let config: JoinConfig = serde_json::from_str(config_json)?;

    // 4. Perform join (uses Arrow operations internally)
    let result = hash_join(&left_table, &right_table, &config)?;

    // 5. Convert back to TableData
    let result_data = result.to_table_data()?;

    // 6. Serialize to JSON
    Ok(result_data.to_json()?)
}
```

## Performance Considerations

### Serialization Overhead
- JSON parsing/serialization adds overhead
- Acceptable for Lambda/serverless where data sizes are bounded
- For large datasets, consider Arrow IPC format (future enhancement)

### Memory Efficiency
- Data is copied at WASM boundary (unavoidable)
- Arrow operations are zero-copy within WASM
- Columnar format enables efficient aggregations

### Optimization Opportunities
1. **Arrow IPC**: For large datasets, use binary Arrow format instead of JSON
2. **Streaming**: Process data in chunks for memory-constrained environments
3. **Typed arrays**: Use JavaScript TypedArrays for numeric data (avoids JSON for numbers)

## Comparison with Native Cylon

| Aspect | Native Cylon | Cylon WASM |
|--------|--------------|------------|
| Table representation | RecordBatch | RecordBatch (wrapped) |
| Data interchange | Arrow IPC/Parquet | JSON (TableData) |
| Memory sharing | Direct pointers | Copy at boundary |
| Operations | Full Arrow compute | Subset (join, groupby, filter) |
| Distribution | MPI/UCX | Single-node (Lambda) |

## Supported Data Types

| ColumnData Variant | Arrow DataType | JSON Example |
|-------------------|----------------|--------------|
| Int32 | DataType::Int32 | `[1, 2, null, 4]` |
| Int64 | DataType::Int64 | `[100, 200, 300]` |
| Float32 | DataType::Float32 | `[1.5, 2.5, 3.5]` |
| Float64 | DataType::Float64 | `[1.123, 2.456]` |
| String | DataType::Utf8 | `["a", "b", null]` |
| Boolean | DataType::Boolean | `[true, false, null]` |

## Distributed Architecture (Approach B: WASM with Host Imports)

### Overview

Cylon WASM supports distributed operations across multiple workers (Lambda functions,
serverless instances, etc.) using a **WASM with Host Imports** architecture.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         WASM Module                                  │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  Distributed Operations (orchestration logic)                  │ │
│  │  - distributed_join()      - distributed_groupby()             │ │
│  │  - distributed_union()     - distributed_intersect()           │ │
│  │  - distributed_subtract()                                      │ │
│  │                                                                │ │
│  │  Algorithm: 1. Hash partition → 2. Shuffle → 3. Local compute  │ │
│  └─────────────────────────┬──────────────────────────────────────┘ │
│                            │ calls                                   │
│  ┌─────────────────────────▼──────────────────────────────────────┐ │
│  │  Host Imports (extern "C" in cylon_host namespace)             │ │
│  │  - host_get_rank() / host_get_world_size()                     │ │
│  │  - host_all_to_all()     ← key primitive for shuffles          │ │
│  │  - host_broadcast() / host_barrier()                           │ │
│  │  - host_gather() / host_scatter() / host_all_gather()          │ │
│  └─────────────────────────┬──────────────────────────────────────┘ │
└────────────────────────────┼────────────────────────────────────────┘
                             │ provided by host
┌────────────────────────────▼────────────────────────────────────────┐
│                      Host Runtime                                    │
│  Node.js  │  Browser                                                │
│                                                                      │
│  Communication via FMI backends:                                     │
│  - LocalBackend (single-node, no I/O)                               │
│  - Redis (serverless coordination)                                  │
│  - S3 (data exchange for large payloads)                            │
└─────────────────────────────────────────────────────────────────────┘

Note: For Python, use native pycylon directly (Cython/C++ with FMI/MPI/UCX).
```

### Why This Architecture?

**Problem**: Distributed algorithms (partition → shuffle → compute) are the same
regardless of host language. Without this architecture, we'd duplicate the logic
in Python, Node.js, and any other host.

**Solution**: Orchestration logic lives in WASM (written once in Rust). Hosts
only implement simple communication primitives.

| Approach | Host Implements | Code Duplication |
|----------|-----------------|------------------|
| Host Orchestration | Full algorithm | High (per language) |
| **WASM with Host Imports** | Just primitives | None |

### Host Import Interface

```rust
// In WASM (rust/cylon-wasm/src/imports.rs)
#[link(wasm_import_module = "cylon_host")]
extern "C" {
    fn host_get_rank() -> i32;
    fn host_get_world_size() -> i32;
    fn host_barrier();
    fn host_all_to_all(
        partitions_ptr: *const usize,
        num_partitions: usize,
        results_ptr_out: *mut *mut usize,
        num_results_out: *mut usize,
    ) -> i32;
    // ... other primitives
}
```

### Data Format: Arrow IPC

Distributed operations use Arrow IPC (binary) format for efficiency:

```
Worker 0                    Worker 1
┌──────────┐               ┌──────────┐
│ Table A  │               │ Table B  │
└────┬─────┘               └────┬─────┘
     │ to_arrow_ipc()           │ to_arrow_ipc()
     ▼                          ▼
┌──────────┐               ┌──────────┐
│ IPC bytes│───────────────│ IPC bytes│  (via host_all_to_all)
└──────────┘               └──────────┘
     │ from_arrow_ipc()         │ from_arrow_ipc()
     ▼                          ▼
┌──────────┐               ┌──────────┐
│ Result   │               │ Result   │
└──────────┘               └──────────┘
```

### Synchronous Design

Host imports are **synchronous** to match WASM's execution model:

```rust
// WASM blocks here until host returns
let results = all_to_all(partitions)?;
```

This works well for:
- **Python**: Can block on I/O naturally
- **Node.js local mode**: No actual I/O
- **Serverless**: FMI uses synchronous Redis operations

### Usage Example

**Python (use native pycylon, not WASM):**
```python
from pycylon import CylonContext, Table
from pycylon.net.fmi_config import FMIConfig

# Create distributed context with FMI
config = FMIConfig(rank=0, world_size=4, host="tcpunch-server", port=9999,
                   redis_host="redis-host", redis_port=6379, ...)
ctx = CylonContext(config=config, distributed=True)

# Distributed join - native C++ implementation
result = left_table.distributed_join(right_table, join_type="inner",
                                      left_on=[0], right_on=[0])
```

**Node.js (WASM with host imports):**
```typescript
import { createDistributedRuntime, LocalBackend } from '@anthropic/cylon-wasm';

const runtime = await createDistributedRuntime();
const result = runtime.distributedJoin(leftIpc, rightIpc, {
  joinType: 'inner',
  leftOn: [0],
  rightOn: [0]
});
```

### Performance Notes

**Copies at WASM boundary are unavoidable** for distributed operations:
1. Data must leave WASM to go over the network
2. Results must come back into WASM

The Arrow IPC format minimizes overhead (binary, not JSON), but copies happen
at the boundary regardless of architecture choice.

**For local operations** (no communication), data stays in WASM memory and
operations are zero-copy.

## Future Enhancements

1. **Additional types**: Date, Timestamp, Decimal, List, Struct
2. **JSPI support**: JavaScript Promise Integration for async host imports in browsers
3. **Streaming API**: Process large datasets in chunks
4. **Custom backends**: Redis, S3, or direct TCP for communication
