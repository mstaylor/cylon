# Cylon WASM

WebAssembly build of Cylon DataFrame operations with optional SIMD support.

## Overview

This crate provides WASM-compatible implementations of core Cylon operations:
- **Join**: Inner, left, right, and full outer hash joins
- **GroupBy**: Hash-based grouping with aggregations (sum, mean, min, max, count)
- **Filter**: Row selection with predicates (comparisons, AND/OR logic)
- **SIMD**: Vectorized operations for sum, dot product, cosine similarity, euclidean distance

## Architecture

The crate uses **Arrow-rs** internally for data representation, enabling code compatibility with native Cylon while exposing a JSON-based API for JavaScript and Python consumers.

```
JavaScript/Python (JSON) ←→ TableData (Serializable) ←→ Table (Arrow RecordBatch)
```

See [docs/TABLE_ARCHITECTURE.md](docs/TABLE_ARCHITECTURE.md) for details.

## Project Structure

```
cylon-wasm/
├── Cargo.toml
├── docs/
│   ├── SIMD_VECTORIZATION.md     # SIMD explanation
│   └── TABLE_ARCHITECTURE.md     # Table wrapper pattern
├── src/
│   ├── lib.rs                    # Main entry point
│   ├── error.rs                  # Error types
│   ├── simd.rs                   # SIMD-optimized operations
│   ├── table.rs                  # Table/DataFrame wrapper
│   ├── join.rs                   # Hash join implementation
│   ├── groupby.rs                # GroupBy with aggregations
│   ├── filter.rs                 # Row filtering
│   └── api.rs                    # WASM API (JSON-based)
└── tests/
    ├── test_simd.rs
    ├── test_table.rs
    ├── test_join.rs
    ├── test_groupby.rs
    └── test_filter.rs
```

## Build

Prerequisites:
```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-pack
```

Standard build:
```bash
wasm-pack build --target web
```

With SIMD support:
```bash
RUSTFLAGS='-C target-feature=+simd128' wasm-pack build --target web --features simd
```

Size-optimized:
```bash
wasm-pack build --target web --release
```

## Usage

### JavaScript (Browser/Node.js)

```javascript
import init, { join_tables, groupby_table, filter_table, aggregate } from './pkg/cylon_wasm.js';

await init();

// Define tables
const employees = {
    columns: ["id", "name", "dept_id"],
    data: [
        { type: "Int32", data: [1, 2, 3] },
        { type: "String", data: ["Alice", "Bob", "Carol"] },
        { type: "Int32", data: [10, 20, 10] }
    ]
};

const departments = {
    columns: ["dept_id", "dept_name"],
    data: [
        { type: "Int32", data: [10, 20] },
        { type: "String", data: ["Engineering", "Sales"] }
    ]
};

// Join
const joined = JSON.parse(join_tables(
    JSON.stringify(employees),
    JSON.stringify(departments),
    JSON.stringify({
        join_type: "inner",
        left_on: [2],
        right_on: [0]
    })
));

// GroupBy
const grouped = JSON.parse(groupby_table(
    JSON.stringify(employees),
    JSON.stringify({
        keys: [2],
        aggregations: [
            { column: 0, op: "count", alias: "employee_count" }
        ]
    })
));

// Filter
const filtered = JSON.parse(filter_table(
    JSON.stringify(employees),
    JSON.stringify({
        predicates: [
            { column: 2, op: "eq", value: 10 }
        ]
    })
));

// Single aggregation
const total = aggregate(JSON.stringify(employees), 0, "count");
```

### Python (with wasmtime)

```python
from wasmtime import Store, Module, Instance
import json

store = Store()
module = Module.from_file(store.engine, "pkg/cylon_wasm_bg.wasm")
instance = Instance(store, module, [])

# Call exported functions via wasmtime bindings
```

### AWS Lambda (Node.js)

```javascript
import init, { join_tables } from './cylon_wasm.js';

let initialized = false;

export async function handler(event) {
    if (!initialized) {
        await init();
        initialized = true;
    }

    const { left, right, config } = event;
    const result = join_tables(
        JSON.stringify(left),
        JSON.stringify(right),
        JSON.stringify(config)
    );

    return {
        statusCode: 200,
        body: result
    };
}
```

## API Reference

### join_tables(left_json, right_json, config_json)

Join two tables.

Config:
```json
{
    "join_type": "inner|left|right|full_outer",
    "left_on": [0],
    "right_on": [0],
    "left_suffix": "_l",
    "right_suffix": "_r"
}
```

### groupby_table(table_json, config_json)

Group by keys and compute aggregations.

Config:
```json
{
    "keys": [0],
    "aggregations": [
        { "column": 1, "op": "sum|mean|min|max|count", "alias": "optional_name" }
    ]
}
```

### filter_table(table_json, config_json)

Filter rows by predicates.

Config:
```json
{
    "predicates": [
        { "column": 0, "op": "eq|ne|lt|le|gt|ge", "value": 100 }
    ],
    "logic": "and|or"
}
```

### aggregate(table_json, column, op)

Compute single aggregation over column.

### SIMD Functions

Direct array operations (for vectors/embeddings):
- `sum_f32(data)` / `sum_f64(data)`
- `dot_product_f32(a, b)`
- `cosine_similarity_f32(a, b)`
- `euclidean_distance_f32(a, b)`

## Data Types

Supported column types:
- `Int32`, `Int64`
- `Float32`, `Float64`
- `String`
- `Boolean`

## Testing

```bash
cargo test
```

## Hot-Swappable Deployment

The WASM binary can be updated without rebuilding containers:

1. Build new `.wasm` file
2. Upload to S3/storage
3. Update Lambda environment variable pointing to WASM location
4. New invocations use updated code

No container rebuild required.

## References

- [SIMD Vectorization Details](docs/SIMD_VECTORIZATION.md)
- [Table Architecture](docs/TABLE_ARCHITECTURE.md)
- Polychroniou et al. "Rethinking SIMD Vectorization for In-Memory Databases" (SIGMOD 2015)
