# Rust DataFrame Libraries: Architectural Decision

## Overview

This document explains Cylon Rust's choice of DataFrame library for providing pandas-like functionality to Rust users.

## Background

Cylon's Python implementation integrates with pandas via zero-copy Arrow DataFrames using Cython bindings. For the Rust implementation, we need a similar capability - a DataFrame API that works seamlessly with Cylon's distributed table operations.

## The Rust DataFrame Landscape

### Available Options

| Library | Arrow Implementation | Description |
|---------|---------------------|-------------|
| **DataFusion** | `arrow-rs` (canonical) | Apache's query engine with DataFrame API |
| **Polars** | `polars-arrow` (fork of arrow2) | High-performance DataFrame library |
| **arrow-rs** | N/A (is the implementation) | Low-level Arrow arrays and kernels |

### The Arrow Compatibility Problem

Cylon Rust uses `arrow-rs`, the canonical Apache Arrow implementation. This creates an important compatibility consideration:

**Polars** uses its own Arrow fork (`polars-arrow`, derived from `arrow2`). Despite having identical memory layouts, these are **different Rust types**:

```rust
// These are incompatible types in Rust's type system
arrow::array::Int64Array        // arrow-rs
polars_arrow::array::Int64Array // polars-arrow
```

Converting between them requires the **C Data Interface (FFI)**:

```
arrow-rs Array → [C FFI structs] → polars-arrow Array
```

While FFI doesn't copy the underlying data buffers, it incurs:
- Metadata marshalling overhead
- Reference counting setup
- Validation on import
- Function call overhead

**DataFusion** uses `arrow-rs` directly, enabling **true zero-copy**:

```rust
// Same types - just ownership transfer, no conversion
let batch: RecordBatch = cylon_table.to_record_batch();
let df = ctx.read_batch(batch)?;  // Zero overhead
```

## Decision: DataFusion

**Cylon Rust uses Apache DataFusion for DataFrame operations.**

### Rationale

1. **True Zero-Copy Integration**
   - DataFusion uses `arrow-rs` natively
   - No FFI overhead when converting Cylon Tables to DataFrames
   - Same Arrow implementation throughout the stack

2. **Apache Ecosystem Alignment**
   - Both Cylon and DataFusion are Apache projects
   - Shared community and development practices
   - Long-term compatibility assurance

3. **Distributed Computing Ready**
   - DataFusion powers Ballista (distributed query engine)
   - Natural path for Cylon's distributed operations
   - Consistent architecture from single-node to cluster

4. **Extensibility**
   - Custom UDFs, UDAFs, and window functions
   - Pluggable optimizer rules
   - Custom data sources and sinks

5. **SQL Support**
   - Full SQL query support when needed
   - DataFrame API for programmatic use
   - Best of both worlds

### Trade-offs Acknowledged

| Aspect | DataFusion | Polars |
|--------|------------|--------|
| Pandas-like API | Good | Better |
| Documentation | Good | Excellent |
| Community size | Growing | Large |
| arrow-rs compatibility | Native | Requires FFI |
| Performance | Excellent | Excellent |

Polars has a more pandas-like API and larger community, but the FFI overhead and type incompatibility make DataFusion the better architectural choice for Cylon.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Application                         │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DataFusion DataFrame API                     │
│         (filter, select, join, aggregate, window, etc.)         │
└─────────────────────────────────────────────────────────────────┘
                                │
                                │ zero-copy
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Cylon Table                             │
│              (distributed operations, shuffle, etc.)            │
└─────────────────────────────────────────────────────────────────┘
                                │
                                │ zero-copy
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    arrow-rs RecordBatch                         │
│                  (columnar memory format)                       │
└─────────────────────────────────────────────────────────────────┘
```

## Usage Example

```rust
use datafusion::prelude::*;
use cylon::Table;

// Cylon distributed operation
let table: Table = cylon_ctx.distributed_join(left, right, &join_config)?;

// Convert to DataFusion DataFrame (zero-copy)
let batch = table.to_record_batch()?;
let ctx = SessionContext::new();
let df = ctx.read_batch(batch)?;

// DataFrame operations
let result = df
    .filter(col("status").eq(lit("active")))?
    .select(vec![col("id"), col("name"), col("value")])?
    .aggregate(
        vec![col("name")],
        vec![sum(col("value")).alias("total")]
    )?
    .sort(vec![col("total").sort(false, true)])?
    .collect()
    .await?;
```

## Optional Polars Interoperability

For users who prefer Polars, we provide conversion functions using the Arrow C Data Interface.
This feature is enabled with the `polars` feature flag.

```rust
// Enable with: cargo build --features polars
use cylon::Table;
use polars::prelude::*;

// Convert Cylon Table to Polars DataFrame (zero-copy via FFI)
let cylon_table: Table = /* ... */;
let polars_df = cylon_table.to_polars()?;

// Use Polars DataFrame API
let filtered = polars_df
    .filter(&polars_df.column("value")?.as_materialized_series().gt(100)?)?;

// Convert back to Cylon Table for distributed operations
let result_table = Table::from_polars(ctx, &filtered)?;
```

This is not the recommended path (DataFusion is preferred for native `arrow-rs` compatibility),
but is available for users with existing Polars workflows or who prefer Polars' pandas-like API.

## References

- [Apache DataFusion](https://datafusion.apache.org/)
- [DataFusion GitHub](https://github.com/apache/datafusion)
- [arrow-rs GitHub](https://github.com/apache/arrow-rs)
- [Polars GitHub](https://github.com/pola-rs/polars)
- [Arrow C Data Interface](https://arrow.apache.org/docs/format/CDataInterface.html)
- [DataFusion vs Polars FAQ](https://datafusion.apache.org/user-guide/faq.html)

## Appendix: FFI Overhead Explained

The C Data Interface enables zero-copy data sharing between different Arrow implementations, but involves:

1. **Export Phase** (arrow-rs → C structs)
   ```rust
   // Metadata converted to C-compatible structures
   FFI_ArrowSchema { format, name, metadata, ... }
   FFI_ArrowArray { length, null_count, buffers, ... }
   ```

2. **Import Phase** (C structs → polars-arrow)
   ```rust
   // Validation and type reconstruction
   polars_arrow::ffi::import_array(...)
   ```

**Key point**: Buffer pointers are preserved (no data copy), but there's O(1) overhead for metadata handling. For large datasets, this is negligible. For the tightest integration, native `arrow-rs` compatibility (DataFusion) eliminates this entirely.
