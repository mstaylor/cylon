# Adding WASM to Cylon Operations: Step-by-Step Guide

## Why Start Here?

Starting with WASM for Cylon operations is the right move because:
1. **Immediate validation**: Prove the integration works with a focused scope
2. **Performance baseline**: Establish SIMD speedup benchmarks early
3. **Learning curve**: Understand WASM compilation before scaling up
4. **Reusable pattern**: Once working, apply to all similarity operations

## Overview: What You're Building

```
┌─────────────────────────────────────────────────────────────┐
│  Cylon Rust Port (Your existing code)                       │
│  ├─ DataFrame operations                                    │
│  ├─ Distributed processing                                  │
│  └─ Checkpointing                                           │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  WASM Wrapper Layer (What you're adding)                    │
│  ├─ Similarity calculations (SIMD-optimized)                │
│  ├─ Vector operations                                       │
│  └─ Embedding processing                                    │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  Deployment Targets                                         │
│  ├─ Browser (WASM)                                          │
│  ├─ Lambda (Native + WASM option)                           │
│  └─ Edge (CloudFlare Workers, etc.)                         │
└─────────────────────────────────────────────────────────────┘
```

## Phase 1: Setup and Validation (Week 1)

### Step 1: Install WASM Toolchain

```bash
# Install wasm-pack (WASM build tool)
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Add WASM target to Rust
rustup target add wasm32-unknown-unknown

# Install wasm-opt for optimization (optional but recommended)
# On macOS:
brew install binaryen

# On Linux:
sudo apt-get install binaryen

# Verify installation
wasm-pack --version
rustup target list | grep wasm32
```

### Step 2: Create WASM-Compatible Cylon Wrapper

Create a new crate specifically for WASM operations:

```bash
cd /path/to/your/project
mkdir cylon-wasm
cd cylon-wasm
cargo init --lib
```

**cylon-wasm/Cargo.toml**:
```toml
[package]
name = "cylon-wasm"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib", "rlib"]

[dependencies]
# Your Cylon Rust port
cylon = { git = "https://github.com/mstaylor/cylon", branch = "cylon-rust" }

# WASM bindings
wasm-bindgen = "0.2"
wasm-bindgen-futures = "0.4"

# Serialization for data transfer
serde = { version = "1.0", features = ["derive"] }
serde-wasm-bindgen = "0.6"

# Console logging for debugging
console_error_panic_hook = "0.1"
web-sys = { version = "0.3", features = ["console"] }

[dev-dependencies]
wasm-bindgen-test = "0.3"

[profile.release]
opt-level = 3
lto = true
```

### Step 3: Start with Simple Vector Operations

**cylon-wasm/src/lib.rs**:
```rust
use wasm_bindgen::prelude::*;

// Set up panic hook for better error messages in browser
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}

/// Simple vector addition (test WASM compilation)
#[wasm_bindgen]
pub fn vector_add(a: Vec<f32>, b: Vec<f32>) -> Vec<f32> {
    assert_eq!(a.len(), b.len(), "Vectors must have equal length");
    
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| x + y)
        .collect()
}

/// Dot product (foundation for similarity)
#[wasm_bindgen]
pub fn dot_product(a: Vec<f32>, b: Vec<f32>) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have equal length");
    
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| x * y)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_add() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let result = vector_add(a, b);
        assert_eq!(result, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let result = dot_product(a, b);
        assert_eq!(result, 32.0); // 1*4 + 2*5 + 3*6 = 32
    }
}
```

### Step 4: Build and Test

```bash
# Build for WASM
cd cylon-wasm
wasm-pack build --target web

# This creates pkg/ directory with:
# - cylon_wasm_bg.wasm (the compiled WASM)
# - cylon_wasm.js (JavaScript bindings)
# - cylon_wasm.d.ts (TypeScript definitions)

# Test in browser
wasm-pack test --headless --firefox
```

### Step 5: Create Simple HTML Test

**cylon-wasm/test.html**:
```html
<!DOCTYPE html>
<html>
<head>
    <title>Cylon WASM Test</title>
    <script type="module">
        import init, { vector_add, dot_product } from './pkg/cylon_wasm.js';

        async function run() {
            // Initialize WASM module
            await init();

            // Test vector addition
            const a = new Float32Array([1.0, 2.0, 3.0]);
            const b = new Float32Array([4.0, 5.0, 6.0]);
            const sum = vector_add(a, b);
            console.log('Vector add:', sum); // [5, 7, 9]

            // Test dot product
            const dot = dot_product(a, b);
            console.log('Dot product:', dot); // 32

            // Display results
            document.getElementById('results').innerHTML = `
                <p>Vector add: [${sum}]</p>
                <p>Dot product: ${dot}</p>
            `;
        }

        run();
    </script>
</head>
<body>
    <h1>Cylon WASM Test</h1>
    <div id="results">Loading...</div>
</body>
</html>
```

**Test it**:
```bash
# Serve the HTML file (WASM requires HTTP server)
python3 -m http.server 8000

# Open browser to http://localhost:8000/test.html
# Check browser console for results
```

**✅ Checkpoint**: If you see the results in the browser, WASM compilation is working!

## Phase 2: Add SIMD Optimization (Week 1-2)

### Step 6: Implement SIMD-Optimized Similarity

**cylon-wasm/src/similarity.rs**:
```rust
use wasm_bindgen::prelude::*;

/// Cosine similarity without SIMD (baseline)
#[wasm_bindgen]
pub fn cosine_similarity_scalar(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    
    dot / (norm_a * norm_b)
}

/// Cosine similarity with SIMD optimization
#[wasm_bindgen]
pub fn cosine_similarity_simd(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    
    #[cfg(target_arch = "wasm32")]
    {
        use std::arch::wasm32::*;
        
        let len = a.len();
        let chunks = len / 4;
        let remainder = len % 4;
        
        // Initialize SIMD accumulators
        let mut dot_vec = f32x4_splat(0.0);
        let mut norm_a_vec = f32x4_splat(0.0);
        let mut norm_b_vec = f32x4_splat(0.0);
        
        // Process 4 elements at a time
        unsafe {
            for i in 0..chunks {
                let idx = i * 4;
                
                // Load 4 elements from each vector
                let va = f32x4(a[idx], a[idx+1], a[idx+2], a[idx+3]);
                let vb = f32x4(b[idx], b[idx+1], b[idx+2], b[idx+3]);
                
                // Accumulate dot product
                dot_vec = f32x4_add(dot_vec, f32x4_mul(va, vb));
                
                // Accumulate norms
                norm_a_vec = f32x4_add(norm_a_vec, f32x4_mul(va, va));
                norm_b_vec = f32x4_add(norm_b_vec, f32x4_mul(vb, vb));
            }
        }
        
        // Reduce SIMD vectors to scalars
        let dot = f32x4_extract_lane::<0>(dot_vec) +
                  f32x4_extract_lane::<1>(dot_vec) +
                  f32x4_extract_lane::<2>(dot_vec) +
                  f32x4_extract_lane::<3>(dot_vec);
        
        let norm_a_sq = f32x4_extract_lane::<0>(norm_a_vec) +
                        f32x4_extract_lane::<1>(norm_a_vec) +
                        f32x4_extract_lane::<2>(norm_a_vec) +
                        f32x4_extract_lane::<3>(norm_a_vec);
        
        let norm_b_sq = f32x4_extract_lane::<0>(norm_b_vec) +
                        f32x4_extract_lane::<1>(norm_b_vec) +
                        f32x4_extract_lane::<2>(norm_b_vec) +
                        f32x4_extract_lane::<3>(norm_b_vec);
        
        // Handle remaining elements
        let mut dot_rem = 0.0;
        let mut norm_a_rem = 0.0;
        let mut norm_b_rem = 0.0;
        
        for i in (len - remainder)..len {
            dot_rem += a[i] * b[i];
            norm_a_rem += a[i] * a[i];
            norm_b_rem += b[i] * b[i];
        }
        
        let final_dot = dot + dot_rem;
        let final_norm_a = (norm_a_sq + norm_a_rem).sqrt();
        let final_norm_b = (norm_b_sq + norm_b_rem).sqrt();
        
        if final_norm_a == 0.0 || final_norm_b == 0.0 {
            return 0.0;
        }
        
        final_dot / (final_norm_a * final_norm_b)
    }
    
    #[cfg(not(target_arch = "wasm32"))]
    {
        // Fallback to scalar version for non-WASM targets
        cosine_similarity_scalar(a, b)
    }
}

/// Batch similarity computation (for multiple contexts)
#[wasm_bindgen]
pub fn batch_cosine_similarity(
    query: &[f32],
    embeddings: &[f32],
    embedding_dim: usize,
) -> Vec<f32> {
    assert_eq!(query.len(), embedding_dim);
    assert_eq!(embeddings.len() % embedding_dim, 0);
    
    let num_embeddings = embeddings.len() / embedding_dim;
    let mut results = Vec::with_capacity(num_embeddings);
    
    for i in 0..num_embeddings {
        let start = i * embedding_dim;
        let end = start + embedding_dim;
        let embedding = &embeddings[start..end];
        
        let similarity = cosine_similarity_simd(query, embedding);
        results.push(similarity);
    }
    
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        
        let scalar = cosine_similarity_scalar(&a, &b);
        let simd = cosine_similarity_simd(&a, &b);
        
        assert!((scalar - 1.0).abs() < 1e-6);
        assert!((simd - 1.0).abs() < 1e-6);
        assert!((scalar - simd).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        
        let scalar = cosine_similarity_scalar(&a, &b);
        let simd = cosine_similarity_simd(&a, &b);
        
        assert!((scalar - 0.0).abs() < 1e-6);
        assert!((simd - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_batch_similarity() {
        let query = vec![1.0, 0.0, 0.0];
        let embeddings = vec![
            1.0, 0.0, 0.0,  // Same as query
            0.0, 1.0, 0.0,  // Orthogonal
            0.5, 0.5, 0.0,  // Partial match
        ];
        
        let results = batch_cosine_similarity(&query, &embeddings, 3);
        
        assert_eq!(results.len(), 3);
        assert!((results[0] - 1.0).abs() < 1e-6);
        assert!((results[1] - 0.0).abs() < 1e-6);
    }
}
```

**Add to lib.rs**:
```rust
pub mod similarity;
pub use similarity::*;
```

### Step 7: Build with SIMD Enabled

```bash
# Build with SIMD feature enabled
RUSTFLAGS="-C target-feature=+simd128" wasm-pack build \
    --target web \
    --out-dir pkg-simd \
    --release

# Compare file sizes
ls -lh pkg/cylon_wasm_bg.wasm      # Without SIMD
ls -lh pkg-simd/cylon_wasm_bg.wasm # With SIMD
```

### Step 8: Benchmark SIMD vs Scalar

**cylon-wasm/benchmark.html**:
```html
<!DOCTYPE html>
<html>
<head>
    <title>Cylon WASM SIMD Benchmark</title>
    <script type="module">
        import init, { 
            cosine_similarity_scalar, 
            cosine_similarity_simd,
            batch_cosine_similarity 
        } from './pkg-simd/cylon_wasm.js';

        async function benchmark() {
            await init();

            // Generate random 512-dimensional embeddings (typical size)
            const dim = 512;
            const numContexts = 1000;
            
            const query = new Float32Array(dim);
            const embeddings = new Float32Array(numContexts * dim);
            
            for (let i = 0; i < dim; i++) {
                query[i] = Math.random();
            }
            
            for (let i = 0; i < numContexts * dim; i++) {
                embeddings[i] = Math.random();
            }

            // Benchmark scalar version
            console.time('Scalar');
            for (let i = 0; i < numContexts; i++) {
                const start = i * dim;
                const end = start + dim;
                const embedding = embeddings.slice(start, end);
                cosine_similarity_scalar(query, embedding);
            }
            console.timeEnd('Scalar');

            // Benchmark SIMD version
            console.time('SIMD');
            for (let i = 0; i < numContexts; i++) {
                const start = i * dim;
                const end = start + dim;
                const embedding = embeddings.slice(start, end);
                cosine_similarity_simd(query, embedding);
            }
            console.timeEnd('SIMD');

            // Benchmark batch processing
            console.time('Batch SIMD');
            const results = batch_cosine_similarity(query, embeddings, dim);
            console.timeEnd('Batch SIMD');

            document.getElementById('results').innerHTML = `
                <p>Processed ${numContexts} contexts with ${dim}-dimensional embeddings</p>
                <p>Check console for timing results</p>
                <p>Expected: SIMD should be 2-4x faster than scalar</p>
            `;
        }

        benchmark();
    </script>
</head>
<body>
    <h1>Cylon WASM SIMD Benchmark</h1>
    <div id="results">Running benchmark...</div>
</body>
</html>
```

**✅ Checkpoint**: You should see 2-4x speedup with SIMD in the browser console!

## Phase 3: Integrate with Cylon DataFrame Operations (Week 2)

### Step 9: Add Cylon DataFrame Support

**cylon-wasm/src/dataframe.rs**:
```rust
use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};

/// Simplified DataFrame representation for WASM
#[wasm_bindgen]
#[derive(Clone)]
pub struct WasmDataFrame {
    data: Vec<u8>, // Serialized data
}

#[wasm_bindgen]
impl WasmDataFrame {
    /// Create from JSON data
    #[wasm_bindgen(constructor)]
    pub fn new(json_data: &str) -> Result<WasmDataFrame, JsValue> {
        let data = json_data.as_bytes().to_vec();
        Ok(WasmDataFrame { data })
    }
    
    /// Convert to JSON
    pub fn to_json(&self) -> String {
        String::from_utf8_lossy(&self.data).to_string()
    }
    
    /// Compute similarities for all rows
    pub fn compute_similarities(
        &self,
        query_embedding: &[f32],
        embedding_column: &str,
    ) -> Result<Vec<f32>, JsValue> {
        // Parse JSON data
        let json_str = self.to_json();
        let rows: Vec<serde_json::Value> = serde_json::from_str(&json_str)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        
        let mut similarities = Vec::with_capacity(rows.len());
        
        for row in rows {
            // Extract embedding from row
            let embedding_value = &row[embedding_column];
            let embedding: Vec<f32> = serde_json::from_value(embedding_value.clone())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            
            // Compute similarity
            let similarity = crate::similarity::cosine_similarity_simd(
                query_embedding,
                &embedding,
            );
            
            similarities.push(similarity);
        }
        
        Ok(similarities)
    }
}
```

### Step 10: Integration Test with Real Cylon Data

**cylon-wasm/examples/cylon_integration.rs**:
```rust
use cylon_wasm::*;

fn main() {
    // Simulate Cylon DataFrame with embeddings
    let contexts = vec![
        serde_json::json!({
            "context_id": "ctx_001",
            "embedding": vec![1.0, 0.0, 0.0],
        }),
        serde_json::json!({
            "context_id": "ctx_002",
            "embedding": vec![0.0, 1.0, 0.0],
        }),
        serde_json::json!({
            "context_id": "ctx_003",
            "embedding": vec![0.5, 0.5, 0.0],
        }),
    ];
    
    let json_data = serde_json::to_string(&contexts).unwrap();
    let df = WasmDataFrame::new(&json_data).unwrap();
    
    let query = vec![1.0, 0.0, 0.0];
    let similarities = df.compute_similarities(&query, "embedding").unwrap();
    
    println!("Similarities: {:?}", similarities);
    // Expected: [1.0, 0.0, ~0.707]
}
```

## Phase 4: Deploy and Use (Week 2)

### Step 11: Build Production WASM

```bash
# Build optimized WASM for production
RUSTFLAGS="-C target-feature=+simd128" wasm-pack build \
    --target web \
    --out-dir pkg-production \
    --release

# Optimize with wasm-opt
wasm-opt -O3 -o pkg-production/cylon_wasm_bg_opt.wasm \
    pkg-production/cylon_wasm_bg.wasm

# Check size reduction
ls -lh pkg-production/cylon_wasm_bg*.wasm
```

### Step 12: Use in Your Context Router

**Example integration in context-router**:
```rust
// In your Lambda or server code
use cylon_wasm::batch_cosine_similarity;

pub async fn find_similar_contexts(
    query_embedding: Vec<f32>,
    contexts: Vec<Context>,
) -> Result<Vec<SimilarContext>> {
    // Flatten embeddings for batch processing
    let embedding_dim = query_embedding.len();
    let embeddings: Vec<f32> = contexts.iter()
        .flat_map(|ctx| ctx.metadata.embedding.clone())
        .collect();
    
    // Use WASM SIMD for fast similarity computation
    let similarities = batch_cosine_similarity(
        &query_embedding,
        &embeddings,
        embedding_dim,
    );
    
    // Combine with context IDs and sort
    let mut results: Vec<_> = contexts.iter()
        .zip(similarities.iter())
        .map(|(ctx, &sim)| SimilarContext {
            context_id: ctx.context_id.clone(),
            similarity: sim,
        })
        .collect();
    
    results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
    
    Ok(results)
}
```

## Next Steps

### Week 3: Expand to More Operations
1. Add more Cylon operations to WASM (joins, aggregations)
2. Benchmark against native Rust
3. Optimize memory usage

### Week 4: Browser Research Tools
1. Build context explorer using WASM
2. Add visualization of similarity scores
3. Deploy to S3 + CloudFlare

### Month 2: Production Integration
1. Integrate with Lambda functions
2. Add caching layer
3. Monitor performance in production

## Success Criteria

✅ **Week 1**: WASM compiles and runs in browser
✅ **Week 2**: SIMD provides 2-4x speedup
✅ **Week 3**: Integrated with Cylon DataFrame operations
✅ **Week 4**: Deployed browser research tool

## Troubleshooting

### WASM doesn't load in browser
- Check browser console for errors
- Ensure you're serving via HTTP (not file://)
- Verify WASM SIMD is supported (Chrome 91+, Firefox 89+)

### SIMD not working
- Check RUSTFLAGS includes `+simd128`
- Verify browser supports WASM SIMD
- Test in Chrome/Firefox (Safari SIMD support is limited)

### Performance not improved
- Ensure release build (`--release`)
- Check embedding dimensions (SIMD helps more with larger vectors)
- Profile with browser DevTools

## Resources

- [wasm-pack documentation](https://rustwasm.github.io/docs/wasm-pack/)
- [WASM SIMD proposal](https://github.com/WebAssembly/simd)
- [ruv-swarm WASM examples](https://github.com/mstaylor/ruv-FANN/tree/main/ruv-swarm/wasm-modules)

---

**Start with Step 1 today and you'll have working WASM by end of week!** 🚀
