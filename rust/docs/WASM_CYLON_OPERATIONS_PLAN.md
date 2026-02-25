# WASM Cylon Operations Implementation Plan

This document extends the existing "Adding WASM to Cylon Operations" guide to cover core DataFrame operations including join, aggregation, groupby, and filter operations.

## Design Principles

### Key Architectural Decision: WASM + Node.js for Hot-Swappable Logic

**Problem**: Traditional container-based deployments require rebuilding and redeploying containers for every code change, leading to slow iteration cycles and operational overhead.

**Solution**: Use WASM modules that can be hot-swapped without container rebuilds.

```
┌─────────────────────────────────────────────────────────────────────┐
│  Previous Architecture (Container-based)                            │
│  ───────────────────────────────────────                            │
│  • Python + Native Cylon compiled into Docker container             │
│  • Any logic change → rebuild container → redeploy (~5-15 min)      │
│  • Container size: 500MB+                                           │
│  • Tight coupling between runtime and business logic                │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  New Architecture (WASM-based, Hot-Swappable)                       │
│  ────────────────────────────────────────────                       │
│  • Node.js runtime (stable, rarely changes)                         │
│  • cylon_wasm.wasm loaded from S3 at runtime                        │
│  • Logic change → upload new .wasm to S3 → immediate effect         │
│  • WASM module size: 2-5MB                                          │
│  • Decoupled: runtime vs business logic                             │
└─────────────────────────────────────────────────────────────────────┘
```

### Dual Runtime Support: Node.js AND Python

Both runtimes are **production-viable**. Choose based on your ecosystem and requirements.

```
┌─────────────────────────────────────────────────────────────────────┐
│                     WASM Runtime Options                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────┐    ┌─────────────────────────┐        │
│  │  Node.js (V8)           │    │  Python (wasmtime-py)   │        │
│  │  ─────────────          │    │  ───────────────────    │        │
│  │  • Native WASM support  │    │  • Familiar ecosystem   │        │
│  │  • Faster cold start    │    │  • NumPy/Pandas interop │        │
│  │  • TypedArray zero-copy │    │  • Jupyter notebooks    │        │
│  │  • Lambda Node.js 20.x  │    │  • Lambda Python 3.12   │        │
│  └─────────────────────────┘    └─────────────────────────┘        │
│              │                              │                       │
│              └──────────┬───────────────────┘                       │
│                         ▼                                           │
│           ┌─────────────────────────┐                               │
│           │  cylon_wasm.wasm        │                               │
│           │  (Same binary for both) │                               │
│           └─────────────────────────┘                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

| Factor | Node.js | Python + wasmtime |
|--------|---------|-------------------|
| WASM Support | Native (V8 built-in) | wasmtime-py library |
| Cold Start | ~80-170ms | ~170-350ms |
| Warm Execution | Baseline | +10-15% overhead |
| Data Passing | TypedArrays (zero-copy) | NumPy arrays (copy) |
| Ecosystem | npm, TypeScript | NumPy, Pandas, Jupyter |
| Best For | Production APIs, low-latency | Data science, experiments |
| Production Ready | ✅ | ✅ |

### Reference

SIMD optimization techniques based on:
> Polychroniou, O., Raghavan, A., & Ross, K. A. (2015). "Rethinking SIMD Vectorization for In-Memory Databases." ACM SIGMOD 2015.
> https://dl.acm.org/doi/10.1145/2723372.2747645

## Overview: Complete WASM Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  AWS Lambda (Node.js Runtime)                                        │
│  ├─ index.js (orchestration, S3 fetch, benchmarking)                 │
│  ├─ WebAssembly.instantiate() - native, optimized                    │
│  └─ cylon_wasm.wasm loaded from S3 (hot-swappable)                   │
└──────────────────────────────────────────────────────────────────────┘
                                ↑
┌──────────────────────────────────────────────────────────────────────┐
│  cylon-wasm Crate (Compiled to WASM)                                 │
│  ├─ WasmDataFrame (Table wrapper)                                    │
│  ├─ WasmJoin (hash_join, sort_join)                                  │
│  ├─ WasmAggregate (sum, min, max, mean, count) - SIMD optimized      │
│  ├─ WasmGroupBy (group + aggregate)                                  │
│  ├─ WasmFilter (predicate-based filtering)                           │
│  ├─ WasmSimilarity (existing: cosine, dot_product)                   │
│  └─ Memory management + SIMD128 optimization                         │
└──────────────────────────────────────────────────────────────────────┘
                                ↑
┌──────────────────────────────────────────────────────────────────────┐
│  Cylon Rust Core (Design Reference)                                  │
│  ├─ join/hash_join.rs                                                │
│  ├─ compute/aggregates.rs                                            │
│  ├─ groupby/hash_groupby.rs                                          │
│  ├─ partition/hash_partition.rs                                      │
│  └─ table.rs (Table abstraction)                                     │
└──────────────────────────────────────────────────────────────────────┘
```

## Phase 1: Arrow-rs Hybrid Approach (Recommended)

### Key Insight: arrow-rs Compiles to WASM

The `arrow-rs` crate **does compile to WASM**. This enables:
- **Code sharing** between native Cylon and WASM Cylon
- **Single implementation** of core operations
- **Full Arrow ecosystem** (RecordBatch, compute kernels)

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  Host Runtime (Node.js OR Python)                                   │
│  • TypedArrays / NumPy arrays                                       │
│  • JSON for complex data                                            │
└─────────────────────────────────────────────────────────────────────┘
                    │
                    ▼  Thin API Layer (wasm-bindgen exports)
┌─────────────────────────────────────────────────────────────────────┐
│  cylon-wasm/src/api.rs                                              │
│  • table_from_arrays() → arrow::RecordBatch                         │
│  • table_from_json() → arrow::RecordBatch                           │
│  • to_arrays() / to_json() ← arrow::RecordBatch                     │
└─────────────────────────────────────────────────────────────────────┘
                    │
                    ▼  Arrow RecordBatch (internal)
┌─────────────────────────────────────────────────────────────────────┐
│  Shared Cylon Core (SAME CODE as native Cylon!)                     │
│  • join/hash_join.rs        ← No changes needed                     │
│  • compute/aggregates.rs    ← No changes needed                     │
│  • groupby/hash_groupby.rs  ← No changes needed                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Benefits

| Aspect | Value |
|--------|-------|
| Code Sharing | Core operations shared with native Cylon |
| Maintenance | Single implementation |
| Host API | Simple arrays/JSON (works with Node.js and Python) |
| Internal | Full Arrow ecosystem |
| WASM Size | ~5MB (acceptable) |

### Cargo.toml for WASM-Compatible Arrow

```toml
[package]
name = "cylon-wasm"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib", "rlib"]

[features]
default = ["console_error_panic_hook"]
simd = []

[dependencies]
# Arrow with WASM-compatible features only
arrow = { version = "53", default-features = false, features = [
    "ipc",           # Serialization
    # Excluded: "ffi", "pyarrow", filesystem features
]}
arrow-array = { version = "53", default-features = false }
arrow-schema = { version = "53", default-features = false }
arrow-select = { version = "53", default-features = false }
arrow-ord = { version = "53", default-features = false }
arrow-row = { version = "53", default-features = false }

# WASM bindings
wasm-bindgen = "0.2"
js-sys = "0.3"
web-sys = { version = "0.3", features = ["console"] }

# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# Error handling in WASM
console_error_panic_hook = { version = "0.1", optional = true }

# Hashmap for joins
hashbrown = "0.14"

[dev-dependencies]
wasm-bindgen-test = "0.3"

[profile.release]
opt-level = 3
lto = true
codegen-units = 1
```

### Core API Implementation

```rust
// cylon-wasm/src/api.rs
//
// Thin layer that converts between host types and Arrow.
// All core operations reuse existing Cylon Rust code.

use wasm_bindgen::prelude::*;
use std::sync::Arc;

use arrow::array::{ArrayRef, Float32Array, Float64Array, Int32Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

// Import existing Cylon modules (shared code!)
use crate::table::Table;
use crate::join::hash_join;
use crate::join::config::{JoinConfig, JoinType};

/// Opaque handle to a Table in WASM memory
#[wasm_bindgen]
pub struct TableHandle {
    #[wasm_bindgen(skip)]
    pub table: Table,
}

#[wasm_bindgen]
impl TableHandle {
    pub fn num_rows(&self) -> usize {
        self.table.num_rows()
    }

    pub fn num_columns(&self) -> usize {
        self.table.num_columns()
    }

    /// Export to JSON string
    pub fn to_json(&self) -> Result<String, JsValue> {
        self.table.to_json()
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Get column as f32 pointer + length (for TypedArray view)
    pub fn get_column_f32_ptr(&self, idx: usize) -> Result<Vec<f32>, JsValue> {
        let col = self.table.column(idx)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let arr = col.as_any()
            .downcast_ref::<Float32Array>()
            .ok_or_else(|| JsValue::from_str("Not Float32"))?;

        Ok(arr.values().to_vec())
    }
}

/// Create table from JSON
#[wasm_bindgen]
pub fn table_from_json(json: &str) -> Result<TableHandle, JsValue> {
    let table = Table::from_json(json)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(TableHandle { table })
}

/// Create table from flat arrays + schema
/// This is the primary interface for both Node.js and Python
#[wasm_bindgen]
pub fn table_from_f32_columns(
    data: &[f32],           // Flattened: [col0_row0, col0_row1, ..., col1_row0, ...]
    num_rows: usize,
    num_cols: usize,
    col_names_json: &str,   // JSON array of column names
) -> Result<TableHandle, JsValue> {
    let col_names: Vec<String> = serde_json::from_str(col_names_json)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let mut fields = Vec::new();
    let mut columns: Vec<ArrayRef> = Vec::new();

    for (i, name) in col_names.iter().enumerate() {
        let start = i * num_rows;
        let end = start + num_rows;
        let col_data = &data[start..end];

        fields.push(Field::new(name, DataType::Float32, false));
        columns.push(Arc::new(Float32Array::from(col_data.to_vec())));
    }

    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema, columns)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let table = Table::from_batch(batch);
    Ok(TableHandle { table })
}

// ============================================================================
// Core Operations - REUSE existing Cylon code!
// ============================================================================

/// Hash join (reuses cylon::join::hash_join)
#[wasm_bindgen]
pub fn wasm_hash_join(
    left: &TableHandle,
    right: &TableHandle,
    left_on: Vec<usize>,
    right_on: Vec<usize>,
    join_type: u8,
) -> Result<TableHandle, JsValue> {
    let jt = match join_type {
        0 => JoinType::Inner,
        1 => JoinType::Left,
        2 => JoinType::Right,
        3 => JoinType::FullOuter,
        _ => return Err(JsValue::from_str("Invalid join type")),
    };

    let config = JoinConfig::new(left_on, right_on, jt);

    // REUSE existing Cylon implementation!
    let result = hash_join::hash_join(&left.table, &right.table, &config)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(TableHandle { table: result })
}

/// GroupBy (reuses cylon::groupby::hash_groupby)
#[wasm_bindgen]
pub fn wasm_groupby(
    table: &TableHandle,
    group_cols: Vec<usize>,
    agg_cols: Vec<usize>,
    agg_ops: Vec<u8>,
) -> Result<TableHandle, JsValue> {
    use crate::groupby::hash_groupby;
    use crate::mapreduce::AggregationOpId;

    let ops: Vec<AggregationOpId> = agg_ops.iter()
        .map(|&op| match op {
            0 => AggregationOpId::Sum,
            1 => AggregationOpId::Min,
            2 => AggregationOpId::Max,
            3 => AggregationOpId::Mean,
            4 => AggregationOpId::Count,
            _ => AggregationOpId::Sum,
        })
        .collect();

    let result = hash_groupby::hash_groupby(&table.table, &group_cols, &agg_cols, &ops)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(TableHandle { table: result })
}

// ============================================================================
// SIMD Operations (direct, no Arrow overhead)
// ============================================================================

#[wasm_bindgen]
pub fn simd_sum_f32(data: &[f32]) -> f32 {
    crate::simd::sum_f32(data)
}

#[wasm_bindgen]
pub fn simd_dot_product(a: &[f32], b: &[f32]) -> f32 {
    crate::simd::dot_product(a, b)
}

#[wasm_bindgen]
pub fn simd_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    crate::simd::cosine_similarity(a, b)
}

#[wasm_bindgen]
pub fn simd_batch_similarity(query: &[f32], embeddings: &[f32], dim: usize) -> Vec<f32> {
    crate::simd::batch_cosine_similarity(query, embeddings, dim)
}
```

## Phase 2: WASM Hash Join Implementation

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     WasmJoin Operation                           │
├─────────────────────────────────────────────────────────────────┤
│  1. Build Phase:                                                 │
│     ├─ Select smaller table (build table)                        │
│     ├─ Hash join columns → build hash map                        │
│     └─ Store: hash → list of row indices                         │
│                                                                  │
│  2. Probe Phase:                                                 │
│     ├─ Iterate larger table (probe table)                        │
│     ├─ Hash join columns → lookup in hash map                    │
│     └─ Collect matching (left_idx, right_idx) pairs              │
│                                                                  │
│  3. Materialize Phase:                                           │
│     ├─ Gather rows from both tables using index pairs            │
│     └─ Build output WasmDataFrame                                │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation

```rust
// cylon-wasm/src/join.rs

use wasm_bindgen::prelude::*;
use std::collections::HashMap;
use crate::table::{WasmDataFrame, WasmColumn};

/// Join type
#[wasm_bindgen]
#[derive(Clone, Copy, Debug)]
pub enum JoinType {
    Inner,
    Left,
    Right,
    FullOuter,
}

/// Hash join configuration
#[wasm_bindgen]
pub struct JoinConfig {
    join_type: JoinType,
    left_on: Vec<usize>,
    right_on: Vec<usize>,
}

#[wasm_bindgen]
impl JoinConfig {
    #[wasm_bindgen(constructor)]
    pub fn new(join_type: JoinType, left_on: Vec<usize>, right_on: Vec<usize>) -> JoinConfig {
        JoinConfig { join_type, left_on, right_on }
    }
}

/// WASM-compatible hash join
#[wasm_bindgen]
pub fn hash_join(
    left: &WasmDataFrame,
    right: &WasmDataFrame,
    config: &JoinConfig,
) -> Result<WasmDataFrame, JsValue> {
    // Determine build/probe tables (smaller = build)
    let (build_table, probe_table, build_on, probe_on, build_from_right) =
        if left.num_rows() <= right.num_rows() {
            (left, right, &config.left_on, &config.right_on, false)
        } else {
            (right, left, &config.right_on, &config.left_on, true)
        };

    // Build phase: create hash map
    let hash_map = build_hash_map(build_table, build_on)?;

    // Probe phase: find matches
    let (build_indices, probe_indices) = probe_hash_map(
        &hash_map,
        probe_table,
        probe_on,
        build_table.num_rows(),
        config.join_type,
    )?;

    // Materialize: gather rows
    let (left_indices, right_indices) = if build_from_right {
        (probe_indices, build_indices)
    } else {
        (build_indices, probe_indices)
    };

    materialize_join(left, right, &left_indices, &right_indices)
}

/// Build hash map from join columns
fn build_hash_map(
    table: &WasmDataFrame,
    join_cols: &[usize],
) -> Result<HashMap<u64, Vec<usize>>, JsValue> {
    let mut map: HashMap<u64, Vec<usize>> = HashMap::new();

    for row_idx in 0..table.num_rows() {
        let hash = hash_row(table, row_idx, join_cols)?;
        map.entry(hash).or_insert_with(Vec::new).push(row_idx);
    }

    Ok(map)
}

/// Hash a row's join columns
fn hash_row(
    table: &WasmDataFrame,
    row_idx: usize,
    cols: &[usize],
) -> Result<u64, JsValue> {
    use std::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;

    let mut hasher = DefaultHasher::new();

    for &col_idx in cols {
        match table.column(col_idx) {
            Some(WasmColumn::Int32(data)) => data[row_idx].hash(&mut hasher),
            Some(WasmColumn::Int64(data)) => data[row_idx].hash(&mut hasher),
            Some(WasmColumn::Float32(data)) => data[row_idx].to_bits().hash(&mut hasher),
            Some(WasmColumn::Float64(data)) => data[row_idx].to_bits().hash(&mut hasher),
            Some(WasmColumn::String(data)) => data[row_idx].hash(&mut hasher),
            Some(WasmColumn::Boolean(data)) => data[row_idx].hash(&mut hasher),
            _ => return Err(JsValue::from_str("Invalid column index")),
        }
    }

    Ok(hasher.finish())
}

/// Probe hash map to find matches
fn probe_hash_map(
    hash_map: &HashMap<u64, Vec<usize>>,
    probe_table: &WasmDataFrame,
    probe_cols: &[usize],
    build_size: usize,
    join_type: JoinType,
) -> Result<(Vec<i64>, Vec<i64>), JsValue> {
    let probe_size = probe_table.num_rows();
    let mut build_indices = Vec::new();
    let mut probe_indices = Vec::new();

    match join_type {
        JoinType::Inner => {
            for probe_idx in 0..probe_size {
                let hash = hash_row(probe_table, probe_idx, probe_cols)?;
                if let Some(build_matches) = hash_map.get(&hash) {
                    for &build_idx in build_matches {
                        build_indices.push(build_idx as i64);
                        probe_indices.push(probe_idx as i64);
                    }
                }
            }
        }
        JoinType::Left => {
            for probe_idx in 0..probe_size {
                let hash = hash_row(probe_table, probe_idx, probe_cols)?;
                if let Some(build_matches) = hash_map.get(&hash) {
                    for &build_idx in build_matches {
                        build_indices.push(build_idx as i64);
                        probe_indices.push(probe_idx as i64);
                    }
                } else {
                    // No match - include probe row with null build row
                    build_indices.push(-1);
                    probe_indices.push(probe_idx as i64);
                }
            }
        }
        JoinType::Right => {
            // Similar to Left but roles reversed
            // Handled by build_from_right flag in caller
            for probe_idx in 0..probe_size {
                let hash = hash_row(probe_table, probe_idx, probe_cols)?;
                if let Some(build_matches) = hash_map.get(&hash) {
                    for &build_idx in build_matches {
                        build_indices.push(build_idx as i64);
                        probe_indices.push(probe_idx as i64);
                    }
                } else {
                    build_indices.push(-1);
                    probe_indices.push(probe_idx as i64);
                }
            }
        }
        JoinType::FullOuter => {
            let mut build_matched = vec![false; build_size];

            // Probe all rows
            for probe_idx in 0..probe_size {
                let hash = hash_row(probe_table, probe_idx, probe_cols)?;
                let mut matched = false;

                if let Some(build_matches) = hash_map.get(&hash) {
                    for &build_idx in build_matches {
                        build_indices.push(build_idx as i64);
                        probe_indices.push(probe_idx as i64);
                        build_matched[build_idx] = true;
                        matched = true;
                    }
                }

                if !matched {
                    build_indices.push(-1);
                    probe_indices.push(probe_idx as i64);
                }
            }

            // Add unmatched build rows
            for (build_idx, matched) in build_matched.iter().enumerate() {
                if !matched {
                    build_indices.push(build_idx as i64);
                    probe_indices.push(-1);
                }
            }
        }
    }

    Ok((build_indices, probe_indices))
}

/// Materialize joined rows into output DataFrame
fn materialize_join(
    left: &WasmDataFrame,
    right: &WasmDataFrame,
    left_indices: &[i64],
    right_indices: &[i64],
) -> Result<WasmDataFrame, JsValue> {
    // Implementation: gather rows from both tables using indices
    // -1 indicates null row (for outer joins)
    todo!("Implement row gathering")
}
```

## Phase 3: WASM Aggregation Operations

### SIMD-Optimized Aggregations

```rust
// cylon-wasm/src/aggregate.rs

use wasm_bindgen::prelude::*;

#[wasm_bindgen]
#[derive(Clone, Copy, Debug)]
pub enum AggregateOp {
    Sum,
    Min,
    Max,
    Mean,
    Count,
    Variance,
    StdDev,
}

/// SIMD-optimized sum for f32 arrays
#[wasm_bindgen]
pub fn sum_f32(data: &[f32]) -> f32 {
    #[cfg(target_arch = "wasm32")]
    {
        use std::arch::wasm32::*;

        let len = data.len();
        let chunks = len / 4;

        let mut sum_vec = f32x4_splat(0.0);

        for i in 0..chunks {
            let idx = i * 4;
            let vec = f32x4(data[idx], data[idx+1], data[idx+2], data[idx+3]);
            sum_vec = f32x4_add(sum_vec, vec);
        }

        // Horizontal sum
        let mut sum = f32x4_extract_lane::<0>(sum_vec)
                    + f32x4_extract_lane::<1>(sum_vec)
                    + f32x4_extract_lane::<2>(sum_vec)
                    + f32x4_extract_lane::<3>(sum_vec);

        // Handle remainder
        for i in (chunks * 4)..len {
            sum += data[i];
        }

        sum
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        data.iter().sum()
    }
}

/// SIMD-optimized min for f32 arrays
#[wasm_bindgen]
pub fn min_f32(data: &[f32]) -> f32 {
    #[cfg(target_arch = "wasm32")]
    {
        use std::arch::wasm32::*;

        if data.is_empty() {
            return f32::NAN;
        }

        let len = data.len();
        let chunks = len / 4;

        let mut min_vec = f32x4_splat(f32::MAX);

        for i in 0..chunks {
            let idx = i * 4;
            let vec = f32x4(data[idx], data[idx+1], data[idx+2], data[idx+3]);
            min_vec = f32x4_pmin(min_vec, vec);
        }

        let mut min = f32x4_extract_lane::<0>(min_vec)
            .min(f32x4_extract_lane::<1>(min_vec))
            .min(f32x4_extract_lane::<2>(min_vec))
            .min(f32x4_extract_lane::<3>(min_vec));

        for i in (chunks * 4)..len {
            min = min.min(data[i]);
        }

        min
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        data.iter().copied().fold(f32::MAX, f32::min)
    }
}

/// SIMD-optimized max for f32 arrays
#[wasm_bindgen]
pub fn max_f32(data: &[f32]) -> f32 {
    #[cfg(target_arch = "wasm32")]
    {
        use std::arch::wasm32::*;

        if data.is_empty() {
            return f32::NAN;
        }

        let len = data.len();
        let chunks = len / 4;

        let mut max_vec = f32x4_splat(f32::MIN);

        for i in 0..chunks {
            let idx = i * 4;
            let vec = f32x4(data[idx], data[idx+1], data[idx+2], data[idx+3]);
            max_vec = f32x4_pmax(max_vec, vec);
        }

        let mut max = f32x4_extract_lane::<0>(max_vec)
            .max(f32x4_extract_lane::<1>(max_vec))
            .max(f32x4_extract_lane::<2>(max_vec))
            .max(f32x4_extract_lane::<3>(max_vec));

        for i in (chunks * 4)..len {
            max = max.max(data[i]);
        }

        max
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        data.iter().copied().fold(f32::MIN, f32::max)
    }
}

/// Mean calculation (uses SIMD sum)
#[wasm_bindgen]
pub fn mean_f32(data: &[f32]) -> f32 {
    if data.is_empty() {
        return f32::NAN;
    }
    sum_f32(data) / data.len() as f32
}

/// Variance calculation (population variance, ddof=0)
#[wasm_bindgen]
pub fn variance_f32(data: &[f32]) -> f32 {
    if data.is_empty() {
        return f32::NAN;
    }

    let mean = mean_f32(data);
    let sum_sq: f32 = data.iter().map(|x| (x - mean).powi(2)).sum();
    sum_sq / data.len() as f32
}

/// Standard deviation
#[wasm_bindgen]
pub fn stddev_f32(data: &[f32]) -> f32 {
    variance_f32(data).sqrt()
}

/// DataFrame aggregate operations
#[wasm_bindgen]
impl WasmDataFrame {
    /// Aggregate a column
    pub fn aggregate(&self, col_idx: usize, op: AggregateOp) -> Result<f64, JsValue> {
        let col = self.columns.get(col_idx)
            .ok_or_else(|| JsValue::from_str("Column not found"))?;

        // Convert to f64 for aggregation
        let data: Vec<f64> = match col {
            WasmColumn::Int32(v) => v.iter().map(|x| *x as f64).collect(),
            WasmColumn::Int64(v) => v.iter().map(|x| *x as f64).collect(),
            WasmColumn::Float32(v) => v.iter().map(|x| *x as f64).collect(),
            WasmColumn::Float64(v) => v.clone(),
            _ => return Err(JsValue::from_str("Cannot aggregate non-numeric column")),
        };

        let result = match op {
            AggregateOp::Sum => data.iter().sum(),
            AggregateOp::Min => data.iter().copied().fold(f64::MAX, f64::min),
            AggregateOp::Max => data.iter().copied().fold(f64::MIN, f64::max),
            AggregateOp::Mean => data.iter().sum::<f64>() / data.len() as f64,
            AggregateOp::Count => data.len() as f64,
            AggregateOp::Variance => {
                let mean = data.iter().sum::<f64>() / data.len() as f64;
                let sum_sq: f64 = data.iter().map(|x| (x - mean).powi(2)).sum();
                sum_sq / data.len() as f64
            }
            AggregateOp::StdDev => {
                let mean = data.iter().sum::<f64>() / data.len() as f64;
                let sum_sq: f64 = data.iter().map(|x| (x - mean).powi(2)).sum();
                (sum_sq / data.len() as f64).sqrt()
            }
        };

        Ok(result)
    }
}
```

## Phase 4: WASM GroupBy Implementation

```rust
// cylon-wasm/src/groupby.rs

use wasm_bindgen::prelude::*;
use std::collections::HashMap;
use crate::table::{WasmDataFrame, WasmColumn};
use crate::aggregate::AggregateOp;

/// GroupBy configuration
#[wasm_bindgen]
pub struct GroupByConfig {
    group_cols: Vec<usize>,
    agg_cols: Vec<usize>,
    agg_ops: Vec<AggregateOp>,
}

#[wasm_bindgen]
impl GroupByConfig {
    #[wasm_bindgen(constructor)]
    pub fn new(
        group_cols: Vec<usize>,
        agg_cols: Vec<usize>,
        agg_ops: Vec<u8>, // Serialized AggregateOp
    ) -> GroupByConfig {
        let ops: Vec<AggregateOp> = agg_ops.iter()
            .map(|&op| match op {
                0 => AggregateOp::Sum,
                1 => AggregateOp::Min,
                2 => AggregateOp::Max,
                3 => AggregateOp::Mean,
                4 => AggregateOp::Count,
                _ => AggregateOp::Sum,
            })
            .collect();

        GroupByConfig { group_cols, agg_cols, agg_ops: ops }
    }
}

/// Hash-based GroupBy operation
#[wasm_bindgen]
pub fn groupby(
    df: &WasmDataFrame,
    config: &GroupByConfig,
) -> Result<WasmDataFrame, JsValue> {
    // Build groups: group_key_hash -> list of row indices
    let mut groups: HashMap<u64, Vec<usize>> = HashMap::new();

    for row_idx in 0..df.num_rows() {
        let hash = hash_group_key(df, row_idx, &config.group_cols)?;
        groups.entry(hash).or_insert_with(Vec::new).push(row_idx);
    }

    // Compute aggregations per group
    let num_groups = groups.len();

    // Initialize result columns
    // Group key columns + aggregation result columns
    let mut result_columns: Vec<WasmColumn> = Vec::new();

    // Add group key columns (take first row of each group)
    for &col_idx in &config.group_cols {
        let col = df.column(col_idx).ok_or_else(|| JsValue::from_str("Invalid column"))?;
        let result_col = extract_group_keys(col, &groups)?;
        result_columns.push(result_col);
    }

    // Add aggregation result columns
    for (i, &agg_col_idx) in config.agg_cols.iter().enumerate() {
        let col = df.column(agg_col_idx).ok_or_else(|| JsValue::from_str("Invalid column"))?;
        let op = config.agg_ops.get(i).copied().unwrap_or(AggregateOp::Sum);
        let result_col = compute_group_aggregate(col, &groups, op)?;
        result_columns.push(result_col);
    }

    // Build column names
    let mut names = Vec::new();
    for &col_idx in &config.group_cols {
        names.push(df.column_name(col_idx).unwrap_or_default().to_string());
    }
    for (i, &col_idx) in config.agg_cols.iter().enumerate() {
        let base_name = df.column_name(col_idx).unwrap_or_default();
        let op_name = match config.agg_ops.get(i) {
            Some(AggregateOp::Sum) => "sum",
            Some(AggregateOp::Min) => "min",
            Some(AggregateOp::Max) => "max",
            Some(AggregateOp::Mean) => "mean",
            Some(AggregateOp::Count) => "count",
            _ => "agg",
        };
        names.push(format!("{}_{}", base_name, op_name));
    }

    Ok(WasmDataFrame {
        columns: result_columns,
        column_names: names,
        num_rows: num_groups,
    })
}

/// Hash group key columns for a row
fn hash_group_key(
    df: &WasmDataFrame,
    row_idx: usize,
    group_cols: &[usize],
) -> Result<u64, JsValue> {
    use std::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;

    let mut hasher = DefaultHasher::new();

    for &col_idx in group_cols {
        match df.column(col_idx) {
            Some(WasmColumn::Int32(data)) => data[row_idx].hash(&mut hasher),
            Some(WasmColumn::Int64(data)) => data[row_idx].hash(&mut hasher),
            Some(WasmColumn::String(data)) => data[row_idx].hash(&mut hasher),
            Some(WasmColumn::Float32(data)) => data[row_idx].to_bits().hash(&mut hasher),
            Some(WasmColumn::Float64(data)) => data[row_idx].to_bits().hash(&mut hasher),
            Some(WasmColumn::Boolean(data)) => data[row_idx].hash(&mut hasher),
            _ => return Err(JsValue::from_str("Invalid group column")),
        }
    }

    Ok(hasher.finish())
}

/// Extract group key values (first row of each group)
fn extract_group_keys(
    col: &WasmColumn,
    groups: &HashMap<u64, Vec<usize>>,
) -> Result<WasmColumn, JsValue> {
    match col {
        WasmColumn::Int32(data) => {
            let values: Vec<i32> = groups.values()
                .map(|indices| data[indices[0]])
                .collect();
            Ok(WasmColumn::Int32(values))
        }
        WasmColumn::Int64(data) => {
            let values: Vec<i64> = groups.values()
                .map(|indices| data[indices[0]])
                .collect();
            Ok(WasmColumn::Int64(values))
        }
        WasmColumn::String(data) => {
            let values: Vec<String> = groups.values()
                .map(|indices| data[indices[0]].clone())
                .collect();
            Ok(WasmColumn::String(values))
        }
        WasmColumn::Float32(data) => {
            let values: Vec<f32> = groups.values()
                .map(|indices| data[indices[0]])
                .collect();
            Ok(WasmColumn::Float32(values))
        }
        WasmColumn::Float64(data) => {
            let values: Vec<f64> = groups.values()
                .map(|indices| data[indices[0]])
                .collect();
            Ok(WasmColumn::Float64(values))
        }
        _ => Err(JsValue::from_str("Unsupported column type for group key")),
    }
}

/// Compute aggregate for each group
fn compute_group_aggregate(
    col: &WasmColumn,
    groups: &HashMap<u64, Vec<usize>>,
    op: AggregateOp,
) -> Result<WasmColumn, JsValue> {
    // Convert to f64 for aggregation
    let data: Vec<f64> = match col {
        WasmColumn::Int32(v) => v.iter().map(|x| *x as f64).collect(),
        WasmColumn::Int64(v) => v.iter().map(|x| *x as f64).collect(),
        WasmColumn::Float32(v) => v.iter().map(|x| *x as f64).collect(),
        WasmColumn::Float64(v) => v.clone(),
        _ => return Err(JsValue::from_str("Cannot aggregate non-numeric column")),
    };

    let results: Vec<f64> = groups.values()
        .map(|indices| {
            let group_data: Vec<f64> = indices.iter().map(|&i| data[i]).collect();
            aggregate_slice(&group_data, op)
        })
        .collect();

    Ok(WasmColumn::Float64(results))
}

/// Aggregate a slice of data
fn aggregate_slice(data: &[f64], op: AggregateOp) -> f64 {
    match op {
        AggregateOp::Sum => data.iter().sum(),
        AggregateOp::Min => data.iter().copied().fold(f64::MAX, f64::min),
        AggregateOp::Max => data.iter().copied().fold(f64::MIN, f64::max),
        AggregateOp::Mean => data.iter().sum::<f64>() / data.len() as f64,
        AggregateOp::Count => data.len() as f64,
        AggregateOp::Variance => {
            let mean = data.iter().sum::<f64>() / data.len() as f64;
            data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64
        }
        AggregateOp::StdDev => {
            let mean = data.iter().sum::<f64>() / data.len() as f64;
            (data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64).sqrt()
        }
    }
}
```

## Phase 5: WASM Filter Operations

```rust
// cylon-wasm/src/filter.rs

use wasm_bindgen::prelude::*;
use crate::table::{WasmDataFrame, WasmColumn};

/// Filter comparison operators
#[wasm_bindgen]
#[derive(Clone, Copy, Debug)]
pub enum FilterOp {
    Eq,      // ==
    Ne,      // !=
    Lt,      // <
    Le,      // <=
    Gt,      // >
    Ge,      // >=
}

/// Filter predicate
#[wasm_bindgen]
pub struct FilterPredicate {
    col_idx: usize,
    op: FilterOp,
    value: f64,  // For numeric comparisons
    string_value: Option<String>,  // For string comparisons
}

#[wasm_bindgen]
impl FilterPredicate {
    /// Create numeric filter
    #[wasm_bindgen(constructor)]
    pub fn new_numeric(col_idx: usize, op: FilterOp, value: f64) -> FilterPredicate {
        FilterPredicate { col_idx, op, value, string_value: None }
    }

    /// Create string filter
    pub fn new_string(col_idx: usize, op: FilterOp, value: String) -> FilterPredicate {
        FilterPredicate { col_idx, op, value: 0.0, string_value: Some(value) }
    }
}

/// Filter DataFrame rows
#[wasm_bindgen]
pub fn filter(
    df: &WasmDataFrame,
    predicate: &FilterPredicate,
) -> Result<WasmDataFrame, JsValue> {
    // Evaluate predicate for each row
    let mask = evaluate_predicate(df, predicate)?;

    // Gather matching rows
    let matching_indices: Vec<usize> = mask.iter()
        .enumerate()
        .filter_map(|(i, &matched)| if matched { Some(i) } else { None })
        .collect();

    // Build result DataFrame
    let result_columns: Vec<WasmColumn> = df.columns.iter()
        .map(|col| gather_rows(col, &matching_indices))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(WasmDataFrame {
        columns: result_columns,
        column_names: df.column_names.clone(),
        num_rows: matching_indices.len(),
    })
}

/// Evaluate predicate for all rows
fn evaluate_predicate(
    df: &WasmDataFrame,
    predicate: &FilterPredicate,
) -> Result<Vec<bool>, JsValue> {
    let col = df.column(predicate.col_idx)
        .ok_or_else(|| JsValue::from_str("Column not found"))?;

    if let Some(ref string_val) = predicate.string_value {
        // String comparison
        match col {
            WasmColumn::String(data) => {
                Ok(data.iter()
                    .map(|v| match predicate.op {
                        FilterOp::Eq => v == string_val,
                        FilterOp::Ne => v != string_val,
                        FilterOp::Lt => v < string_val,
                        FilterOp::Le => v <= string_val,
                        FilterOp::Gt => v > string_val,
                        FilterOp::Ge => v >= string_val,
                    })
                    .collect())
            }
            _ => Err(JsValue::from_str("String predicate on non-string column")),
        }
    } else {
        // Numeric comparison
        let values: Vec<f64> = match col {
            WasmColumn::Int32(data) => data.iter().map(|x| *x as f64).collect(),
            WasmColumn::Int64(data) => data.iter().map(|x| *x as f64).collect(),
            WasmColumn::Float32(data) => data.iter().map(|x| *x as f64).collect(),
            WasmColumn::Float64(data) => data.clone(),
            _ => return Err(JsValue::from_str("Numeric predicate on non-numeric column")),
        };

        let threshold = predicate.value;
        Ok(values.iter()
            .map(|&v| match predicate.op {
                FilterOp::Eq => (v - threshold).abs() < f64::EPSILON,
                FilterOp::Ne => (v - threshold).abs() >= f64::EPSILON,
                FilterOp::Lt => v < threshold,
                FilterOp::Le => v <= threshold,
                FilterOp::Gt => v > threshold,
                FilterOp::Ge => v >= threshold,
            })
            .collect())
    }
}

/// Gather rows by indices
fn gather_rows(col: &WasmColumn, indices: &[usize]) -> Result<WasmColumn, JsValue> {
    match col {
        WasmColumn::Int32(data) => {
            Ok(WasmColumn::Int32(indices.iter().map(|&i| data[i]).collect()))
        }
        WasmColumn::Int64(data) => {
            Ok(WasmColumn::Int64(indices.iter().map(|&i| data[i]).collect()))
        }
        WasmColumn::Float32(data) => {
            Ok(WasmColumn::Float32(indices.iter().map(|&i| data[i]).collect()))
        }
        WasmColumn::Float64(data) => {
            Ok(WasmColumn::Float64(indices.iter().map(|&i| data[i]).collect()))
        }
        WasmColumn::String(data) => {
            Ok(WasmColumn::String(indices.iter().map(|&i| data[i].clone()).collect()))
        }
        WasmColumn::Boolean(data) => {
            Ok(WasmColumn::Boolean(indices.iter().map(|&i| data[i]).collect()))
        }
        _ => Err(JsValue::from_str("Unsupported column type")),
    }
}
```

## Phase 6: Memory Management (from ruv-swarm patterns)

```rust
// cylon-wasm/src/memory.rs

use wasm_bindgen::prelude::*;
use std::collections::VecDeque;

/// Memory pool for reusing allocations
#[wasm_bindgen]
pub struct MemoryPool {
    small_blocks: VecDeque<Vec<u8>>,   // 64KB
    medium_blocks: VecDeque<Vec<u8>>,  // 256KB
    large_blocks: VecDeque<Vec<u8>>,   // 1MB
    small_size: usize,
    medium_size: usize,
    large_size: usize,
    max_cached: usize,
}

#[wasm_bindgen]
impl MemoryPool {
    #[wasm_bindgen(constructor)]
    pub fn new() -> MemoryPool {
        MemoryPool {
            small_blocks: VecDeque::new(),
            medium_blocks: VecDeque::new(),
            large_blocks: VecDeque::new(),
            small_size: 64 * 1024,      // 64KB
            medium_size: 256 * 1024,    // 256KB
            large_size: 1024 * 1024,    // 1MB
            max_cached: 10,             // Max cached blocks per size class
        }
    }

    /// Allocate a block of at least `size` bytes
    pub fn allocate(&mut self, size: usize) -> Vec<u8> {
        if size <= self.small_size {
            self.small_blocks.pop_front()
                .unwrap_or_else(|| vec![0u8; self.small_size])
        } else if size <= self.medium_size {
            self.medium_blocks.pop_front()
                .unwrap_or_else(|| vec![0u8; self.medium_size])
        } else {
            self.large_blocks.pop_front()
                .unwrap_or_else(|| vec![0u8; self.large_size.max(size)])
        }
    }

    /// Return a block to the pool for reuse
    pub fn deallocate(&mut self, mut block: Vec<u8>) {
        let size = block.len();
        block.fill(0);  // Clear for security

        if size <= self.small_size && self.small_blocks.len() < self.max_cached {
            self.small_blocks.push_back(block);
        } else if size <= self.medium_size && self.medium_blocks.len() < self.max_cached {
            self.medium_blocks.push_back(block);
        } else if self.large_blocks.len() < self.max_cached {
            self.large_blocks.push_back(block);
        }
        // Otherwise drop the block
    }

    /// Get current memory usage
    pub fn memory_usage(&self) -> usize {
        self.small_blocks.len() * self.small_size
            + self.medium_blocks.len() * self.medium_size
            + self.large_blocks.len() * self.large_size
    }
}

/// Global memory pool (thread-local for WASM single-threaded model)
thread_local! {
    static POOL: std::cell::RefCell<MemoryPool> = std::cell::RefCell::new(MemoryPool::new());
}

/// Allocate from global pool
pub fn pool_allocate(size: usize) -> Vec<u8> {
    POOL.with(|p| p.borrow_mut().allocate(size))
}

/// Return to global pool
pub fn pool_deallocate(block: Vec<u8>) {
    POOL.with(|p| p.borrow_mut().deallocate(block));
}
```

## Phase 7: Project Structure

```
cylon-wasm/
├── Cargo.toml
├── src/
│   ├── lib.rs              # Main entry point, re-exports
│   ├── table.rs            # WasmDataFrame implementation
│   ├── join.rs             # Hash join implementation
│   ├── aggregate.rs        # SIMD-optimized aggregations
│   ├── groupby.rs          # GroupBy implementation
│   ├── filter.rs           # Filter operations
│   ├── similarity.rs       # Existing similarity ops
│   ├── memory.rs           # Memory pool management
│   └── utils/
│       ├── bridge.rs       # JS type conversions
│       └── simd.rs         # SIMD abstractions
├── tests/
│   └── wasm.rs             # WASM integration tests
├── benches/
│   └── operations.rs       # Performance benchmarks
└── examples/
    └── browser/            # Browser demo
        ├── index.html
        └── main.js
```

## Phase 8: Cargo.toml Configuration

```toml
[package]
name = "cylon-wasm"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib", "rlib"]

[features]
default = ["console_error_panic_hook", "simd"]
simd = []
simd128 = ["simd"]

[dependencies]
wasm-bindgen = "0.2"
wasm-bindgen-futures = "0.4"
js-sys = "0.3"
web-sys = { version = "0.3", features = ["console", "Performance"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
serde-wasm-bindgen = "0.6"
console_error_panic_hook = { version = "0.1", optional = true }

[dev-dependencies]
wasm-bindgen-test = "0.3"

[profile.release]
opt-level = 3
lto = true
codegen-units = 1

# For smaller WASM size (optional)
[profile.release-small]
inherits = "release"
opt-level = "z"
```

## Phase 9: Build and Test

```bash
# Build with SIMD
RUSTFLAGS="-C target-feature=+simd128" wasm-pack build \
    --target web \
    --release

# Run tests
wasm-pack test --headless --firefox

# Optimize for size
wasm-opt -O3 -o pkg/cylon_wasm_bg_opt.wasm pkg/cylon_wasm_bg.wasm
```

## Phase 10: AWS Lambda Deployment with Hot-Swap

### Architecture: Hot-Swappable WASM on Lambda

```
┌─────────────────────────────────────────────────────────────────────┐
│                        AWS Infrastructure                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐      ┌─────────────────────────────────────────┐  │
│  │   S3        │      │  Lambda (Node.js 20.x)                  │  │
│  │   Bucket    │      │  ┌─────────────────────────────────────┐│  │
│  │             │ ───► │  │  index.js (stable, rarely changes) ││  │
│  │ cylon_wasm/ │      │  │  • Fetch WASM from S3 on cold start ││  │
│  │  └─ v1.0.0/ │      │  │  • Cache in memory for warm starts  ││  │
│  │     └─ cylon│      │  │  • Native WebAssembly.instantiate() ││  │
│  │        .wasm│      │  │                                     ││  │
│  │  └─ v1.1.0/ │      │  │  ┌─────────────────────────────┐   ││  │
│  │     └─ ...  │      │  │  │  cylon_wasm.wasm (2-5MB)    │   ││  │
│  │             │      │  │  │  • join, groupby, filter    │   ││  │
│  └─────────────┘      │  │  │  • SIMD aggregations        │   ││  │
│                       │  │  └─────────────────────────────┘   ││  │
│                       │  └─────────────────────────────────────┘│  │
│                       └─────────────────────────────────────────┘  │
│                                                                     │
│  Update Flow:                                                       │
│  1. Build new cylon_wasm.wasm locally                              │
│  2. Upload to S3: aws s3 cp cylon_wasm.wasm s3://bucket/v1.1.0/    │
│  3. Update Lambda env var: WASM_VERSION=v1.1.0                     │
│  4. Next invocation loads new WASM (no container rebuild!)         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Lambda Handler Implementation

```javascript
// index.js - Lambda handler (Node.js 20.x)
import { S3Client, GetObjectCommand } from '@aws-sdk/client-s3';

const s3 = new S3Client({ region: process.env.AWS_REGION });

// Cache WASM instance for warm starts
let wasmInstance = null;
let wasmMemory = null;

/**
 * Load WASM module from S3 (cached for warm starts)
 */
async function loadWasmModule() {
    if (wasmInstance) {
        return wasmInstance;
    }

    const bucket = process.env.WASM_BUCKET;
    const version = process.env.WASM_VERSION || 'latest';
    const key = `cylon_wasm/${version}/cylon_wasm.wasm`;

    console.log(`Loading WASM from s3://${bucket}/${key}`);
    const startTime = Date.now();

    // Fetch from S3
    const response = await s3.send(new GetObjectCommand({ Bucket: bucket, Key: key }));
    const wasmBytes = await response.Body.transformToByteArray();

    // Instantiate WASM module
    wasmMemory = new WebAssembly.Memory({ initial: 256, maximum: 4096 }); // 16MB - 256MB

    const { instance } = await WebAssembly.instantiate(wasmBytes, {
        env: {
            memory: wasmMemory,
            // Import any required functions
            console_log: (ptr, len) => {
                const bytes = new Uint8Array(wasmMemory.buffer, ptr, len);
                console.log(new TextDecoder().decode(bytes));
            },
        },
    });

    wasmInstance = instance;
    console.log(`WASM loaded in ${Date.now() - startTime}ms`);

    return wasmInstance;
}

/**
 * Lambda handler
 */
export async function handler(event) {
    const wasm = await loadWasmModule();

    // Parse input data
    const { operation, left, right, config } = JSON.parse(event.body || '{}');

    // Allocate memory and copy data to WASM
    const leftPtr = allocateAndCopy(wasm, wasmMemory, left);
    const rightPtr = allocateAndCopy(wasm, wasmMemory, right);
    const configPtr = allocateAndCopy(wasm, wasmMemory, config);

    let resultPtr;
    const startTime = Date.now();

    // Execute operation
    switch (operation) {
        case 'hash_join':
            resultPtr = wasm.exports.hash_join(leftPtr, rightPtr, configPtr);
            break;
        case 'groupby':
            resultPtr = wasm.exports.groupby(leftPtr, configPtr);
            break;
        case 'filter':
            resultPtr = wasm.exports.filter(leftPtr, configPtr);
            break;
        case 'aggregate':
            resultPtr = wasm.exports.aggregate(leftPtr, configPtr);
            break;
        default:
            return { statusCode: 400, body: JSON.stringify({ error: 'Unknown operation' }) };
    }

    const executionTime = Date.now() - startTime;

    // Read result from WASM memory
    const result = readResult(wasm, wasmMemory, resultPtr);

    // Free allocated memory
    wasm.exports.free(leftPtr);
    wasm.exports.free(rightPtr);
    wasm.exports.free(configPtr);
    wasm.exports.free(resultPtr);

    return {
        statusCode: 200,
        body: JSON.stringify({
            result,
            metrics: {
                executionTimeMs: executionTime,
                wasmVersion: process.env.WASM_VERSION,
            },
        }),
    };
}

/**
 * Allocate WASM memory and copy data
 */
function allocateAndCopy(wasm, memory, data) {
    const json = JSON.stringify(data);
    const bytes = new TextEncoder().encode(json);

    // Allocate memory in WASM
    const ptr = wasm.exports.alloc(bytes.length);

    // Copy data to WASM memory
    const wasmBytes = new Uint8Array(memory.buffer, ptr, bytes.length);
    wasmBytes.set(bytes);

    return ptr;
}

/**
 * Read result from WASM memory
 */
function readResult(wasm, memory, ptr) {
    // First 4 bytes are length
    const lenBytes = new Uint32Array(memory.buffer, ptr, 1);
    const len = lenBytes[0];

    // Rest is JSON data
    const dataBytes = new Uint8Array(memory.buffer, ptr + 4, len);
    const json = new TextDecoder().decode(dataBytes);

    return JSON.parse(json);
}
```

### TypedArray Zero-Copy Data Passing (Optimized)

For numeric data, use TypedArrays to avoid JSON serialization overhead:

```javascript
// optimized_handler.js - Zero-copy for numeric data

/**
 * Pass Float32Array directly to WASM (near zero-copy)
 */
function passFloat32Array(wasm, memory, data) {
    // Allocate in WASM
    const ptr = wasm.exports.alloc_f32(data.length);

    // Create view into WASM memory
    const wasmArray = new Float32Array(memory.buffer, ptr, data.length);

    // Copy data (fast memcpy)
    wasmArray.set(data);

    return { ptr, len: data.length };
}

/**
 * Read Float32Array result from WASM
 */
function readFloat32Array(memory, ptr, len) {
    return new Float32Array(memory.buffer, ptr, len);
}

/**
 * Optimized handler for numeric operations
 */
export async function numericHandler(event) {
    const wasm = await loadWasmModule();

    // Input as TypedArrays (from binary body or base64)
    const leftData = new Float32Array(event.leftBuffer);
    const rightData = new Float32Array(event.rightBuffer);

    // Pass to WASM (fast)
    const left = passFloat32Array(wasm, wasmMemory, leftData);
    const right = passFloat32Array(wasm, wasmMemory, rightData);

    // Execute SIMD-optimized operation
    const startTime = performance.now();
    const resultPtr = wasm.exports.simd_dot_product_batch(
        left.ptr, left.len,
        right.ptr, right.len
    );
    const executionTime = performance.now() - startTime;

    // Read result
    const resultLen = wasm.exports.result_len(resultPtr);
    const result = readFloat32Array(wasmMemory, resultPtr, resultLen);

    return {
        statusCode: 200,
        body: JSON.stringify({
            result: Array.from(result),
            metrics: { executionTimeMs: executionTime.toFixed(2) },
        }),
    };
}
```

### Python Runtime Implementation (wasmtime-py)

Python is equally supported for production use. Use `wasmtime-py` for the WASM runtime.

#### Installation

```bash
pip install wasmtime numpy
```

#### Lambda Handler (Python)

```python
# handler.py - Lambda handler (Python 3.12)
import json
import os
import boto3
import numpy as np
from wasmtime import Store, Module, Instance, Memory, Func, FuncType, ValType

s3 = boto3.client('s3')

# Cache WASM instance for warm starts
_wasm_instance = None
_wasm_memory = None
_store = None

def load_wasm_module():
    """Load WASM module from S3 (cached for warm starts)"""
    global _wasm_instance, _wasm_memory, _store

    if _wasm_instance is not None:
        return _wasm_instance, _wasm_memory, _store

    bucket = os.environ['WASM_BUCKET']
    version = os.environ.get('WASM_VERSION', 'latest')
    key = f"cylon_wasm/{version}/cylon_wasm.wasm"

    print(f"Loading WASM from s3://{bucket}/{key}")

    # Fetch from S3
    response = s3.get_object(Bucket=bucket, Key=key)
    wasm_bytes = response['Body'].read()

    # Create store and compile module
    _store = Store()
    module = Module(_store.engine, wasm_bytes)

    # Create memory (16MB initial, 256MB max)
    _wasm_memory = Memory(_store, MemoryType(limits=Limits(min=256, max=4096)))

    # Define imports (if needed)
    def console_log(ptr: int, length: int) -> None:
        data = _wasm_memory.data_ptr(_store)[ptr:ptr + length]
        print(bytes(data).decode('utf-8'))

    imports = [
        _wasm_memory,
        Func(_store, FuncType([ValType.i32(), ValType.i32()], []), console_log),
    ]

    _wasm_instance = Instance(_store, module, imports)
    print("WASM loaded successfully")

    return _wasm_instance, _wasm_memory, _store


def numpy_to_wasm(store, instance, memory, arr: np.ndarray) -> tuple:
    """Copy NumPy array to WASM memory, return (ptr, len)"""
    # Ensure float32
    arr = arr.astype(np.float32)
    data = arr.tobytes()

    # Allocate in WASM
    alloc = instance.exports(store)["alloc"]
    ptr = alloc(store, len(data))

    # Copy to WASM memory
    mem_data = memory.data_ptr(store)
    mem_data[ptr:ptr + len(data)] = data

    return ptr, len(arr)


def wasm_to_numpy(store, memory, ptr: int, length: int) -> np.ndarray:
    """Read WASM memory into NumPy array"""
    mem_data = memory.data_ptr(store)
    byte_length = length * 4  # float32 = 4 bytes
    data = bytes(mem_data[ptr:ptr + byte_length])
    return np.frombuffer(data, dtype=np.float32)


def handler(event, context):
    """Lambda handler"""
    instance, memory, store = load_wasm_module()
    exports = instance.exports(store)

    # Parse input
    body = json.loads(event.get('body', '{}'))
    operation = body.get('operation')
    left_data = np.array(body.get('left', []), dtype=np.float32)
    right_data = np.array(body.get('right', []), dtype=np.float32)

    # Copy data to WASM
    left_ptr, left_len = numpy_to_wasm(store, instance, memory, left_data)
    right_ptr, right_len = numpy_to_wasm(store, instance, memory, right_data)

    import time
    start_time = time.time()

    # Execute operation
    if operation == 'dot_product':
        result = exports["simd_dot_product"](store, left_ptr, right_ptr)
        result_data = {'scalar': result}
    elif operation == 'cosine_similarity':
        result = exports["simd_cosine_similarity"](store, left_ptr, right_ptr)
        result_data = {'scalar': result}
    elif operation == 'sum':
        result = exports["simd_sum_f32"](store, left_ptr, left_len)
        result_data = {'scalar': result}
    else:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': f'Unknown operation: {operation}'})
        }

    execution_time = (time.time() - start_time) * 1000

    # Free memory
    exports["free"](store, left_ptr)
    exports["free"](store, right_ptr)

    return {
        'statusCode': 200,
        'body': json.dumps({
            'result': result_data,
            'metrics': {
                'executionTimeMs': round(execution_time, 2),
                'wasmVersion': os.environ.get('WASM_VERSION', 'unknown'),
            }
        })
    }
```

#### DataFrame Operations (Python)

```python
# cylon_wasm.py - High-level Python wrapper

import json
import numpy as np
from wasmtime import Store, Module, Instance
from pathlib import Path

class CylonWasm:
    """Python wrapper for Cylon WASM operations"""

    def __init__(self, wasm_path: str):
        self.store = Store()
        wasm_bytes = Path(wasm_path).read_bytes()
        module = Module(self.store.engine, wasm_bytes)
        self.instance = Instance(self.store, module, [])
        self.exports = self.instance.exports(self.store)

    def _to_wasm(self, arr: np.ndarray) -> tuple:
        """Copy array to WASM memory"""
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        ptr = self.exports["alloc"](self.store, arr.nbytes)
        # Copy via memory view
        mem = self.exports["memory"]
        np.copyto(
            np.frombuffer(mem.data_ptr(self.store)[ptr:ptr+arr.nbytes], dtype=np.float32),
            arr.ravel()
        )
        return ptr, len(arr)

    def _from_wasm(self, ptr: int, length: int) -> np.ndarray:
        """Read array from WASM memory"""
        mem = self.exports["memory"]
        byte_len = length * 4
        return np.frombuffer(
            bytes(mem.data_ptr(self.store)[ptr:ptr+byte_len]),
            dtype=np.float32
        ).copy()

    def create_table(self, df: 'pd.DataFrame') -> int:
        """Create table from pandas DataFrame, return handle"""
        json_str = df.to_json(orient='records')
        # Allocate and copy JSON string
        json_bytes = json_str.encode('utf-8')
        ptr = self.exports["alloc"](self.store, len(json_bytes))
        mem = self.exports["memory"]
        mem.data_ptr(self.store)[ptr:ptr+len(json_bytes)] = json_bytes

        # Call table_from_json
        handle = self.exports["table_from_json"](self.store, ptr, len(json_bytes))
        self.exports["free"](self.store, ptr)
        return handle

    def hash_join(self, left_handle: int, right_handle: int,
                  left_on: list, right_on: list,
                  join_type: str = 'inner') -> int:
        """Perform hash join, return result handle"""
        join_type_map = {'inner': 0, 'left': 1, 'right': 2, 'outer': 3}
        jt = join_type_map.get(join_type, 0)

        return self.exports["wasm_hash_join"](
            self.store,
            left_handle,
            right_handle,
            left_on[0] if left_on else 0,
            right_on[0] if right_on else 0,
            jt
        )

    def groupby(self, table_handle: int,
                group_cols: list, agg_cols: list, agg_ops: list) -> int:
        """Perform groupby aggregation"""
        op_map = {'sum': 0, 'min': 1, 'max': 2, 'mean': 3, 'count': 4}
        ops = [op_map.get(op, 0) for op in agg_ops]

        return self.exports["wasm_groupby"](
            self.store,
            table_handle,
            group_cols[0] if group_cols else 0,
            agg_cols[0] if agg_cols else 0,
            ops[0] if ops else 0
        )

    # SIMD operations (direct, no table overhead)
    def dot_product(self, a: np.ndarray, b: np.ndarray) -> float:
        ptr_a, len_a = self._to_wasm(a)
        ptr_b, len_b = self._to_wasm(b)

        result = self.exports["simd_dot_product"](self.store, ptr_a, ptr_b)

        self.exports["free"](self.store, ptr_a)
        self.exports["free"](self.store, ptr_b)
        return result

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        ptr_a, _ = self._to_wasm(a)
        ptr_b, _ = self._to_wasm(b)

        result = self.exports["simd_cosine_similarity"](self.store, ptr_a, ptr_b)

        self.exports["free"](self.store, ptr_a)
        self.exports["free"](self.store, ptr_b)
        return result

    def batch_similarity(self, query: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity of query against all embeddings"""
        dim = len(query)
        num_embeddings = len(embeddings) // dim

        ptr_q, _ = self._to_wasm(query)
        ptr_e, _ = self._to_wasm(embeddings.ravel())

        result_ptr = self.exports["simd_batch_similarity"](
            self.store, ptr_q, ptr_e, dim
        )

        result = self._from_wasm(result_ptr, num_embeddings)

        self.exports["free"](self.store, ptr_q)
        self.exports["free"](self.store, ptr_e)
        self.exports["free"](self.store, result_ptr)

        return result
```

#### Python Benchmarking

```python
# benchmark.py - Microbenchmarks

import time
import numpy as np
from cylon_wasm import CylonWasm

def benchmark(name, fn, iterations=100):
    """Run benchmark and report statistics"""
    # Warmup
    for _ in range(10):
        fn()

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)

    times = np.array(times)
    print(f"{name}: avg={times.mean():.2f}ms, min={times.min():.2f}ms, "
          f"max={times.max():.2f}ms, p99={np.percentile(times, 99):.2f}ms")

    return {'avg': times.mean(), 'min': times.min(), 'max': times.max(), 'p99': np.percentile(times, 99)}


def run_benchmarks(wasm_path: str):
    cylon = CylonWasm(wasm_path)
    results = {}

    sizes = [1000, 10000, 100000]

    for size in sizes:
        print(f"\n--- Benchmarking with {size} elements ---")

        a = np.random.randn(size).astype(np.float32)
        b = np.random.randn(size).astype(np.float32)

        # Dot product
        results[f'dot_product_{size}'] = benchmark(
            'dot_product',
            lambda: cylon.dot_product(a, b)
        )

        # Cosine similarity
        results[f'cosine_similarity_{size}'] = benchmark(
            'cosine_similarity',
            lambda: cylon.cosine_similarity(a, b)
        )

        # Batch similarity (1 query vs 100 embeddings)
        dim = 512
        query = np.random.randn(dim).astype(np.float32)
        embeddings = np.random.randn(100, dim).astype(np.float32)

        results[f'batch_similarity_{size}'] = benchmark(
            'batch_similarity',
            lambda: cylon.batch_similarity(query, embeddings.ravel())
        )

    print("\n=== Results ===")
    for name, metrics in results.items():
        print(f"{name}: {metrics}")

    return results


if __name__ == '__main__':
    run_benchmarks('cylon_wasm.wasm')
```

#### Jupyter Notebook Integration

```python
# In Jupyter notebook
import numpy as np
import pandas as pd
from cylon_wasm import CylonWasm

# Load WASM module
cylon = CylonWasm('./pkg/cylon_wasm.wasm')

# Create sample data
orders = pd.DataFrame({
    'customer_id': [1, 2, 1, 3, 2],
    'amount': [100.0, 200.0, 150.0, 300.0, 50.0],
    'product': ['A', 'B', 'A', 'C', 'B']
})

customers = pd.DataFrame({
    'customer_id': [1, 2, 3],
    'name': ['Alice', 'Bob', 'Charlie']
})

# SIMD similarity search
query_embedding = np.random.randn(512).astype(np.float32)
document_embeddings = np.random.randn(1000, 512).astype(np.float32)

%timeit similarities = cylon.batch_similarity(query_embedding, document_embeddings.ravel())
# Output: 1.2 ms ± 50 µs per loop

# Get top-k most similar
similarities = cylon.batch_similarity(query_embedding, document_embeddings.ravel())
top_k_indices = np.argsort(similarities)[-10:][::-1]
print(f"Top 10 similar documents: {top_k_indices}")
```

### Infrastructure as Code (CDK/Terraform)

```typescript
// cdk/lib/cylon-wasm-stack.ts
import * as cdk from 'aws-cdk-lib';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as s3 from 'aws-cdk-lib/aws-s3';

export class CylonWasmStack extends cdk.Stack {
    constructor(scope: cdk.App, id: string, props?: cdk.StackProps) {
        super(scope, id, props);

        // S3 bucket for WASM modules
        const wasmBucket = new s3.Bucket(this, 'WasmBucket', {
            bucketName: 'cylon-wasm-modules',
            versioned: true, // Keep versions for rollback
        });

        // Lambda function
        const cylonLambda = new lambda.Function(this, 'CylonWasmLambda', {
            runtime: lambda.Runtime.NODEJS_20_X,
            handler: 'index.handler',
            code: lambda.Code.fromAsset('lambda'),
            memorySize: 1024,      // More memory = more CPU
            timeout: cdk.Duration.seconds(30),
            environment: {
                WASM_BUCKET: wasmBucket.bucketName,
                WASM_VERSION: 'v1.0.0',  // Update this to deploy new WASM
            },
        });

        // Grant S3 read access
        wasmBucket.grantRead(cylonLambda);
    }
}
```

### Deployment Workflow

```bash
# 1. Build WASM module
cd cylon-wasm
RUSTFLAGS="-C target-feature=+simd128" wasm-pack build --target web --release

# 2. Optimize for size
wasm-opt -O3 -o pkg/cylon_wasm_opt.wasm pkg/cylon_wasm_bg.wasm

# 3. Upload to S3 with version
VERSION="v1.1.0"
aws s3 cp pkg/cylon_wasm_opt.wasm s3://cylon-wasm-modules/cylon_wasm/${VERSION}/cylon_wasm.wasm

# 4. Update Lambda to use new version (no rebuild needed!)
aws lambda update-function-configuration \
    --function-name CylonWasmLambda \
    --environment "Variables={WASM_BUCKET=cylon-wasm-modules,WASM_VERSION=${VERSION}}"

# 5. Test
curl -X POST https://xxxxx.execute-api.us-east-1.amazonaws.com/prod/cylon \
    -H "Content-Type: application/json" \
    -d '{"operation": "hash_join", "left": [...], "right": [...], "config": {...}}'
```

### Benchmarking Script (Node.js)

```javascript
// benchmark.js - Microbenchmarks for WASM operations
import { performance } from 'perf_hooks';

async function runBenchmarks(wasm, memory) {
    const results = {};

    // Generate test data
    const sizes = [1000, 10000, 100000];

    for (const size of sizes) {
        console.log(`\n--- Benchmarking with ${size} rows ---`);

        // Generate random data
        const leftData = generateTestData(size);
        const rightData = generateTestData(size);

        // Hash Join benchmark
        results[`hash_join_${size}`] = await benchmark('hash_join', () => {
            const leftPtr = allocateData(wasm, memory, leftData);
            const rightPtr = allocateData(wasm, memory, rightData);
            const configPtr = allocateConfig(wasm, memory, { type: 'inner', leftOn: [0], rightOn: [0] });

            const result = wasm.exports.hash_join(leftPtr, rightPtr, configPtr);

            wasm.exports.free(leftPtr);
            wasm.exports.free(rightPtr);
            wasm.exports.free(configPtr);
            wasm.exports.free(result);
        });

        // GroupBy benchmark
        results[`groupby_${size}`] = await benchmark('groupby', () => {
            const dataPtr = allocateData(wasm, memory, leftData);
            const configPtr = allocateConfig(wasm, memory, { groupCols: [0], aggCols: [1], aggOps: ['sum'] });

            const result = wasm.exports.groupby(dataPtr, configPtr);

            wasm.exports.free(dataPtr);
            wasm.exports.free(configPtr);
            wasm.exports.free(result);
        });

        // SIMD Sum benchmark
        const floatData = new Float32Array(size);
        for (let i = 0; i < size; i++) floatData[i] = Math.random();

        results[`simd_sum_${size}`] = await benchmark('simd_sum', () => {
            const ptr = allocateFloat32(wasm, memory, floatData);
            wasm.exports.sum_f32(ptr, size);
            wasm.exports.free(ptr);
        });
    }

    console.log('\n=== Results ===');
    console.table(results);

    return results;
}

async function benchmark(name, fn, iterations = 100) {
    // Warmup
    for (let i = 0; i < 10; i++) fn();

    // Timed runs
    const times = [];
    for (let i = 0; i < iterations; i++) {
        const start = performance.now();
        fn();
        times.push(performance.now() - start);
    }

    const avg = times.reduce((a, b) => a + b) / times.length;
    const min = Math.min(...times);
    const max = Math.max(...times);
    const p99 = times.sort((a, b) => a - b)[Math.floor(times.length * 0.99)];

    console.log(`${name}: avg=${avg.toFixed(2)}ms, min=${min.toFixed(2)}ms, max=${max.toFixed(2)}ms, p99=${p99.toFixed(2)}ms`);

    return { avg, min, max, p99 };
}
```

## Implementation Timeline

| Phase | Description | Effort |
|-------|-------------|--------|
| 1 | WASM Table representation | Medium |
| 2 | Hash Join implementation | High |
| 3 | SIMD Aggregations | Medium |
| 4 | GroupBy implementation | Medium |
| 5 | Filter operations | Low |
| 6 | Memory management | Low |
| 7 | Testing & benchmarks | Medium |
| 8 | Browser demo | Low |

## Performance Targets

- **Join**: 100K rows in <100ms
- **GroupBy**: 100K rows in <50ms
- **Aggregations**: 1M values in <10ms (with SIMD)
- **Filter**: 100K rows in <20ms
- **Memory**: <50MB for typical workloads

## Integration with Existing Cylon

The WASM operations are designed to:

1. **Standalone**: Work independently in browser/edge without full Cylon dependency
2. **Compatible**: Use same semantics as Cylon Rust (join types, aggregation ops)
3. **Complementary**: Can coexist with native Rust builds for server-side processing
4. **Portable**: Single WASM binary runs anywhere WASM is supported

## Next Steps

1. Create `cylon-wasm` crate with basic structure
2. Implement `WasmDataFrame` with JSON serialization
3. Port hash join with SIMD optimization
4. Add groupby and aggregations
5. Create browser demo
6. Benchmark against native Rust
