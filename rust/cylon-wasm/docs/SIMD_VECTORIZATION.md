# SIMD Vectorization in Cylon WASM

## Overview

SIMD (Single Instruction, Multiple Data) is a form of **data-level parallelism** that allows a single CPU instruction to operate on multiple data elements simultaneously. This is distinct from thread-level parallelism and works perfectly within Cylon's single-threaded execution model.

## How SIMD Works

### Traditional Scalar Processing

In scalar processing, operations execute one element at a time:

```
Iteration 1: a[0] + b[0] → c[0]
Iteration 2: a[1] + b[1] → c[1]
Iteration 3: a[2] + b[2] → c[2]
Iteration 4: a[3] + b[3] → c[3]
```

**4 iterations, 4 CPU cycles**

### SIMD Vector Processing

With SIMD, multiple elements are processed in a single instruction:

```
Single instruction: [a[0], a[1], a[2], a[3]] + [b[0], b[1], b[2], b[3]] → [c[0], c[1], c[2], c[3]]
```

**1 iteration, 1 CPU cycle** (theoretically 4x speedup)

## WASM SIMD128

WebAssembly provides **SIMD128** - 128-bit wide vector operations. This enables:

| Data Type | Elements per Vector | Operations |
|-----------|---------------------|------------|
| f32 (float) | 4 elements | 4-wide parallel ops |
| f64 (double) | 2 elements | 2-wide parallel ops |
| i32 (int) | 4 elements | 4-wide parallel ops |
| i64 (long) | 2 elements | 2-wide parallel ops |

### WASM SIMD Instructions Used

```rust
// Load 4 f32 values into a 128-bit vector
f32x4(a, b, c, d)

// Vector arithmetic
f32x4_add(v1, v2)    // Add 4 pairs simultaneously
f32x4_mul(v1, v2)    // Multiply 4 pairs simultaneously
f32x4_sub(v1, v2)    // Subtract 4 pairs simultaneously

// Reductions
f32x4_min(v1, v2)    // Element-wise minimum
f32x4_max(v1, v2)    // Element-wise maximum

// Extract individual lanes
f32x4_extract_lane::<0>(v)  // Get first element
```

## SIMD in Cylon Operations

### 1. Aggregations (SUM, MIN, MAX, MEAN)

**Sum Operation:**
```rust
// Process 4 elements per iteration
for chunk in data.chunks_exact(4) {
    let v = f32x4(chunk[0], chunk[1], chunk[2], chunk[3]);
    accumulator = f32x4_add(accumulator, v);
}
// Final horizontal sum of 4 lanes
result = lane[0] + lane[1] + lane[2] + lane[3] + remainder
```

**Performance:** For 1M elements, ~250K iterations instead of 1M.

### 2. Similarity Operations (Cosine, Euclidean)

**Dot Product (core of cosine similarity):**
```rust
// dot(a, b) = Σ(a[i] * b[i])
for (chunk_a, chunk_b) in a.chunks(4).zip(b.chunks(4)) {
    let va = f32x4(chunk_a[0..4]);
    let vb = f32x4(chunk_b[0..4]);
    let product = f32x4_mul(va, vb);      // 4 multiplications
    accumulator = f32x4_add(acc, product); // 4 additions
}
```

**Cosine Similarity:**
```
cosine(a, b) = dot(a, b) / (||a|| * ||b||)
             = dot(a, b) / (sqrt(dot(a,a)) * sqrt(dot(b,b)))
```

All three dot products benefit from SIMD.

**Euclidean Distance:**
```rust
// ||a - b|| = sqrt(Σ(a[i] - b[i])²)
for (chunk_a, chunk_b) in a.chunks(4).zip(b.chunks(4)) {
    let diff = f32x4_sub(va, vb);     // 4 subtractions
    let sq = f32x4_mul(diff, diff);   // 4 squares
    accumulator = f32x4_add(acc, sq); // 4 additions
}
result = sqrt(horizontal_sum(accumulator))
```

### 3. Filter Operations

SIMD can accelerate predicate evaluation:

```rust
// Filter: value > threshold
let threshold_vec = f32x4_splat(threshold);  // [t, t, t, t]
let mask = f32x4_gt(values, threshold_vec);  // 4 comparisons → bitmask
```

### 4. Join Hash Computation

While hash joins are memory-bound, SIMD can help with:
- Batch hash computation
- Parallel key comparisons

## Why SIMD Works with Single-Threaded Cylon

```
┌─────────────────────────────────────────────────────────────┐
│                     Thread-Level Parallelism                 │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │ Thread 1│  │ Thread 2│  │ Thread 3│  │ Thread 4│        │
│  │  a + b  │  │  c + d  │  │  e + f  │  │  g + h  │        │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘        │
│         Multiple execution units, synchronization needed     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     Data-Level Parallelism (SIMD)            │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                    Single Thread                     │    │
│  │   ┌─────────────────────────────────────────────┐   │    │
│  │   │  [a, b, c, d] + [e, f, g, h] = [r1,r2,r3,r4]│   │    │
│  │   │         One instruction, 4 operations        │   │    │
│  │   └─────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────┘    │
│              Single execution unit, no synchronization       │
└─────────────────────────────────────────────────────────────┘
```

**Key insight:** SIMD operates within a single thread using wide registers. Cylon's communicator handles distribution across processes, while SIMD accelerates local computation within each process.

## Performance Expectations

### Theoretical Speedup

| Operation | Scalar | SIMD (f32) | Speedup |
|-----------|--------|------------|---------|
| Sum 1M elements | 1M ops | 250K ops | ~4x |
| Dot product 1M | 2M ops | 500K ops | ~4x |
| Euclidean 1M | 3M ops | 750K ops | ~4x |

### Real-World Factors

Actual speedup is typically **2-3x** due to:
- Memory bandwidth limitations
- Cache effects
- Remainder handling (non-aligned data)
- Instruction overhead

### Benchmark Reference

From Polychroniou et al. "Rethinking SIMD Vectorization for In-Memory Databases" (SIGMOD 2015):
- Selection scans: 2-4x speedup
- Hash table probing: 2-3x speedup
- Sorting: 3-5x speedup

## Implementation in Cylon WASM

### Feature Flag

SIMD is conditionally compiled:

```toml
[features]
simd = []
```

Build with SIMD:
```bash
RUSTFLAGS='-C target-feature=+simd128' wasm-pack build --features simd
```

### Graceful Fallback

All SIMD functions have scalar fallbacks:

```rust
#[cfg(feature = "simd")]
pub fn simd_sum_f32(data: &[f32]) -> f32 {
    // SIMD implementation
}

#[cfg(not(feature = "simd"))]
pub fn simd_sum_f32(data: &[f32]) -> f32 {
    data.iter().sum()  // Scalar fallback
}
```

### Browser Compatibility

SIMD128 is supported in:
- Chrome 91+ (May 2021)
- Firefox 89+ (June 2021)
- Safari 16.4+ (March 2023)
- Node.js 16.4+ (with --experimental-wasm-simd, default in 18+)

## Operations Benefiting from SIMD

| Cylon Operation | SIMD Benefit | Notes |
|-----------------|--------------|-------|
| GroupBy SUM | High | Direct vectorization |
| GroupBy MIN/MAX | High | Parallel comparisons |
| GroupBy MEAN | High | Sum + count vectorized |
| GroupBy COUNT | Low | Simple increment |
| Filter (numeric) | Medium | Predicate evaluation |
| Join (hash build) | Low | Memory-bound |
| Join (probe) | Medium | Batch comparisons |
| Similarity search | High | Dot product heavy |

## References

1. Polychroniou, O., Raghavan, A., & Ross, K. A. (2015). **Rethinking SIMD Vectorization for In-Memory Databases**. SIGMOD '15.

2. Kersten, T., Leis, V., Kemper, A., Neumann, T., Pavlo, A., & Boncz, P. (2018). **Everything You Always Wanted to Know About Compiled and Vectorized Queries But Were Afraid to Ask**. VLDB '18.

3. WebAssembly SIMD Proposal: https://github.com/WebAssembly/simd

4. Arrow Compute Kernels: https://arrow.apache.org/docs/cpp/compute.html
