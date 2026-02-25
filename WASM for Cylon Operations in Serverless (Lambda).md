# WASM for Cylon Operations in Serverless (Lambda)

## Critical Insight: Why WASM in Lambda?

**Important**: For Lambda, you have two deployment options:

### Option 1: Native Rust (Recommended for Lambda)
```
Rust Code → Compile to x86_64 → Deploy to Lambda
- Fastest execution (native CPU instructions)
- Full access to AVX2/AVX-512 SIMD
- No WASM runtime overhead
- Standard Lambda deployment
```

### Option 2: WASM Runtime in Lambda
```
Rust Code → Compile to WASM → Run in WASM runtime → Deploy to Lambda
- Portable across architectures
- Useful for multi-platform deployment
- Slight runtime overhead
- Good for ARM64 + x86_64 compatibility
```

## When to Use WASM in Serverless

**Use WASM when**:
- ✅ Deploying to multiple platforms (Lambda x86_64 + ARM64, CloudFlare Workers, Fastly)
- ✅ Need sandboxed execution for untrusted code
- ✅ Want single binary for all serverless platforms
- ✅ Using WASM-specific features (component model, WASI)

**Use Native Rust when**:
- ✅ Only deploying to Lambda x86_64
- ✅ Need maximum performance
- ✅ Want simplest deployment
- ✅ Using AWS-specific features (Bedrock, DynamoDB SDK)

## Recommended Approach: Hybrid Strategy

**Best of both worlds**:
```rust
// Compile core operations to WASM for portability
// Use native Rust for AWS integrations

┌─────────────────────────────────────────────┐
│  Lambda Function (Native Rust)              │
│  ├─ AWS SDK (DynamoDB, S3, Bedrock)         │
│  ├─ Lambda Runtime                          │
│  └─ Business Logic                          │
└─────────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────┐
│  WASM Module (Cylon Operations)             │
│  ├─ Similarity calculations                 │
│  ├─ Vector operations                       │
│  └─ DataFrame processing                    │
└─────────────────────────────────────────────┘
```

## Updated Guide: WASM in Lambda

### Step 1: Choose WASM Runtime for Lambda

**Option A: Wasmtime (Recommended)**
```toml
[dependencies]
wasmtime = "16.0"
wasmtime-wasi = "16.0"
```

**Option B: Wasmer**
```toml
[dependencies]
wasmer = "4.2"
wasmer-wasi = "4.2"
```

**For this guide, we'll use Wasmtime** (better Lambda performance)

### Step 2: Project Structure for Serverless

```
your-project/
├── cylon-wasm/              # WASM module (core operations)
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       └── similarity.rs
│
├── context-router/          # Lambda function (native Rust)
│   ├── Cargo.toml
│   ├── Dockerfile
│   └── src/
│       ├── main.rs          # Lambda handler
│       └── wasm_runtime.rs  # WASM execution
│
└── Cargo.toml               # Workspace
```

### Step 3: Build WASM Module for Serverless

**cylon-wasm/Cargo.toml**:
```toml
[package]
name = "cylon-wasm"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]  # For WASM

[dependencies]
# Your Cylon Rust port
cylon = { git = "https://github.com/mstaylor/cylon", branch = "cylon-rust" }

# WASI support for serverless
wit-bindgen = "0.16"

[profile.release]
opt-level = 3
lto = true
strip = true
```

**cylon-wasm/src/lib.rs**:
```rust
// No wasm-bindgen needed for serverless!
// Use WASI instead

use std::io::{self, Read, Write};

/// Compute cosine similarity (SIMD-optimized)
#[no_mangle]
pub extern "C" fn cosine_similarity(
    a_ptr: *const f32,
    b_ptr: *const f32,
    len: usize,
) -> f32 {
    unsafe {
        let a = std::slice::from_raw_parts(a_ptr, len);
        let b = std::slice::from_raw_parts(b_ptr, len);
        
        cosine_similarity_simd(a, b)
    }
}

/// SIMD-optimized similarity (works in WASM)
fn cosine_similarity_simd(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "wasm32")]
    {
        use std::arch::wasm32::*;
        
        let len = a.len();
        let chunks = len / 4;
        
        let mut dot_vec = f32x4_splat(0.0);
        let mut norm_a_vec = f32x4_splat(0.0);
        let mut norm_b_vec = f32x4_splat(0.0);
        
        unsafe {
            for i in 0..chunks {
                let idx = i * 4;
                let va = f32x4(a[idx], a[idx+1], a[idx+2], a[idx+3]);
                let vb = f32x4(b[idx], b[idx+1], b[idx+2], b[idx+3]);
                
                dot_vec = f32x4_add(dot_vec, f32x4_mul(va, vb));
                norm_a_vec = f32x4_add(norm_a_vec, f32x4_mul(va, va));
                norm_b_vec = f32x4_add(norm_b_vec, f32x4_mul(vb, vb));
            }
        }
        
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
        
        // Handle remainder
        let remainder = len % 4;
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
        // Native implementation for testing
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        
        dot / (norm_a * norm_b)
    }
}

/// Batch similarity computation
#[no_mangle]
pub extern "C" fn batch_cosine_similarity(
    query_ptr: *const f32,
    embeddings_ptr: *const f32,
    embedding_dim: usize,
    num_embeddings: usize,
    results_ptr: *mut f32,
) {
    unsafe {
        let query = std::slice::from_raw_parts(query_ptr, embedding_dim);
        let embeddings = std::slice::from_raw_parts(
            embeddings_ptr,
            embedding_dim * num_embeddings,
        );
        let results = std::slice::from_raw_parts_mut(results_ptr, num_embeddings);
        
        for i in 0..num_embeddings {
            let start = i * embedding_dim;
            let end = start + embedding_dim;
            let embedding = &embeddings[start..end];
            
            results[i] = cosine_similarity_simd(query, embedding);
        }
    }
}
```

### Step 4: Build WASM for Lambda

```bash
# Build WASM with SIMD
cd cylon-wasm

# For Lambda x86_64 (WASM with SIMD)
RUSTFLAGS="-C target-feature=+simd128" cargo build \
    --target wasm32-wasi \
    --release

# Output: target/wasm32-wasi/release/cylon_wasm.wasm

# Optimize with wasm-opt
wasm-opt -O3 --enable-simd \
    -o cylon_wasm_optimized.wasm \
    target/wasm32-wasi/release/cylon_wasm.wasm

# Check size
ls -lh cylon_wasm_optimized.wasm
```

### Step 5: Lambda Function with WASM Runtime

**context-router/Cargo.toml**:
```toml
[package]
name = "context-router"
version = "0.1.0"
edition = "2021"

[dependencies]
# Lambda runtime
lambda_runtime = "0.8"
lambda_http = "0.8"

# WASM runtime
wasmtime = "16.0"
wasmtime-wasi = "16.0"

# AWS SDKs
aws-config = "1.1"
aws-sdk-dynamodb = "1.11"
aws-sdk-bedrockruntime = "1.11"

# Other dependencies
tokio = { version = "1.35", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
anyhow = "1.0"
tracing = "0.1"
```

**context-router/src/wasm_runtime.rs**:
```rust
use anyhow::Result;
use wasmtime::*;
use std::sync::Arc;

/// WASM runtime for Cylon operations
pub struct CylonWasmRuntime {
    engine: Engine,
    module: Module,
}

impl CylonWasmRuntime {
    /// Initialize WASM runtime (call once at Lambda cold start)
    pub fn new(wasm_bytes: &[u8]) -> Result<Self> {
        // Configure engine for Lambda
        let mut config = Config::new();
        config.wasm_simd(true);  // Enable SIMD
        config.cranelift_opt_level(OptLevel::Speed);
        
        let engine = Engine::new(&config)?;
        let module = Module::new(&engine, wasm_bytes)?;
        
        Ok(Self { engine, module })
    }
    
    /// Compute cosine similarity using WASM
    pub fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        let mut store = Store::new(&self.engine, ());
        let instance = Instance::new(&mut store, &self.module, &[])?;
        
        // Get WASM function
        let cosine_similarity = instance
            .get_typed_func::<(u32, u32, u32), f32>(&mut store, "cosine_similarity")?;
        
        // Allocate memory in WASM
        let memory = instance.get_memory(&mut store, "memory")
            .ok_or_else(|| anyhow::anyhow!("WASM memory not found"))?;
        
        let len = a.len();
        let a_offset = 0;
        let b_offset = (len * 4) as u32;  // f32 = 4 bytes
        
        // Copy data to WASM memory
        let a_bytes = unsafe {
            std::slice::from_raw_parts(a.as_ptr() as *const u8, len * 4)
        };
        let b_bytes = unsafe {
            std::slice::from_raw_parts(b.as_ptr() as *const u8, len * 4)
        };
        
        memory.write(&mut store, a_offset as usize, a_bytes)?;
        memory.write(&mut store, b_offset as usize, b_bytes)?;
        
        // Call WASM function
        let result = cosine_similarity.call(&mut store, (a_offset, b_offset, len as u32))?;
        
        Ok(result)
    }
    
    /// Batch similarity computation
    pub fn batch_cosine_similarity(
        &self,
        query: &[f32],
        embeddings: &[f32],
        embedding_dim: usize,
    ) -> Result<Vec<f32>> {
        let mut store = Store::new(&self.engine, ());
        let instance = Instance::new(&mut store, &self.module, &[])?;
        
        let batch_func = instance.get_typed_func::<(u32, u32, u32, u32, u32), ()>(
            &mut store,
            "batch_cosine_similarity",
        )?;
        
        let memory = instance.get_memory(&mut store, "memory")
            .ok_or_else(|| anyhow::anyhow!("WASM memory not found"))?;
        
        let num_embeddings = embeddings.len() / embedding_dim;
        
        // Allocate offsets
        let query_offset = 0;
        let embeddings_offset = (embedding_dim * 4) as u32;
        let results_offset = (embeddings_offset + (embeddings.len() * 4) as u32);
        
        // Copy data to WASM memory
        let query_bytes = unsafe {
            std::slice::from_raw_parts(query.as_ptr() as *const u8, embedding_dim * 4)
        };
        let embeddings_bytes = unsafe {
            std::slice::from_raw_parts(embeddings.as_ptr() as *const u8, embeddings.len() * 4)
        };
        
        memory.write(&mut store, query_offset as usize, query_bytes)?;
        memory.write(&mut store, embeddings_offset as usize, embeddings_bytes)?;
        
        // Call WASM function
        batch_func.call(
            &mut store,
            (
                query_offset,
                embeddings_offset,
                embedding_dim as u32,
                num_embeddings as u32,
                results_offset,
            ),
        )?;
        
        // Read results from WASM memory
        let mut results = vec![0f32; num_embeddings];
        let results_bytes = unsafe {
            std::slice::from_raw_parts_mut(
                results.as_mut_ptr() as *mut u8,
                num_embeddings * 4,
            )
        };
        memory.read(&store, results_offset as usize, results_bytes)?;
        
        Ok(results)
    }
}

/// Global WASM runtime (reused across Lambda invocations)
static mut WASM_RUNTIME: Option<Arc<CylonWasmRuntime>> = None;

/// Get or initialize WASM runtime
pub fn get_wasm_runtime() -> Result<Arc<CylonWasmRuntime>> {
    unsafe {
        if WASM_RUNTIME.is_none() {
            // Load WASM module (embedded in binary or from S3)
            let wasm_bytes = include_bytes!("../../cylon-wasm/cylon_wasm_optimized.wasm");
            let runtime = CylonWasmRuntime::new(wasm_bytes)?;
            WASM_RUNTIME = Some(Arc::new(runtime));
        }
        
        Ok(WASM_RUNTIME.as_ref().unwrap().clone())
    }
}
```

**context-router/src/main.rs**:
```rust
mod wasm_runtime;

use lambda_http::{run, service_fn, Body, Error, Request, Response};
use serde::{Deserialize, Serialize};
use wasm_runtime::get_wasm_runtime;

#[derive(Deserialize)]
struct FindSimilarRequest {
    query_embedding: Vec<f32>,
    context_embeddings: Vec<Vec<f32>>,
}

#[derive(Serialize)]
struct FindSimilarResponse {
    similarities: Vec<f32>,
}

async fn function_handler(event: Request) -> Result<Response<Body>, Error> {
    // Parse request
    let body = event.body();
    let request: FindSimilarRequest = serde_json::from_slice(body)?;
    
    // Get WASM runtime (cached after cold start)
    let runtime = get_wasm_runtime()?;
    
    // Flatten embeddings for batch processing
    let embedding_dim = request.query_embedding.len();
    let embeddings: Vec<f32> = request.context_embeddings
        .into_iter()
        .flatten()
        .collect();
    
    // Compute similarities using WASM
    let similarities = runtime.batch_cosine_similarity(
        &request.query_embedding,
        &embeddings,
        embedding_dim,
    )?;
    
    // Return response
    let response = FindSimilarResponse { similarities };
    let body = serde_json::to_string(&response)?;
    
    Ok(Response::builder()
        .status(200)
        .header("content-type", "application/json")
        .body(body.into())?)
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .without_time()
        .init();
    
    run(service_fn(function_handler)).await
}
```

### Step 6: Docker Build for Lambda

**context-router/Dockerfile**:
```dockerfile
FROM public.ecr.aws/lambda/provided:al2

# Install Rust
RUN yum install -y gcc openssl-devel && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

ENV PATH="/root/.cargo/bin:${PATH}"

# Add WASM target
RUN rustup target add wasm32-wasi

# Copy source
WORKDIR /build
COPY . .

# Build WASM module first
WORKDIR /build/cylon-wasm
RUN RUSTFLAGS="-C target-feature=+simd128" cargo build \
    --target wasm32-wasi \
    --release

# Optimize WASM
RUN cargo install wasm-opt && \
    wasm-opt -O3 --enable-simd \
    -o cylon_wasm_optimized.wasm \
    target/wasm32-wasi/release/cylon_wasm.wasm

# Build Lambda function
WORKDIR /build/context-router
RUN cargo build --release --target x86_64-unknown-linux-musl

# Copy binary to Lambda runtime
RUN cp target/x86_64-unknown-linux-musl/release/context-router /var/runtime/bootstrap

CMD ["bootstrap"]
```

### Step 7: Build and Deploy

```bash
# Build Docker image
docker build -t context-router:latest .

# Tag for ECR
docker tag context-router:latest \
    123456789012.dkr.ecr.us-east-1.amazonaws.com/context-router:latest

# Push to ECR
docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/context-router:latest

# Deploy Lambda
aws lambda update-function-code \
    --function-name context-router \
    --image-uri 123456789012.dkr.ecr.us-east-1.amazonaws.com/context-router:latest
```

### Step 8: Benchmark WASM vs Native in Lambda

**benchmark.rs**:
```rust
use std::time::Instant;

fn benchmark_wasm_vs_native() {
    let query = vec![0.5; 512];
    let embeddings: Vec<f32> = (0..1000 * 512)
        .map(|_| rand::random::<f32>())
        .collect();
    
    // Benchmark WASM
    let runtime = get_wasm_runtime().unwrap();
    let start = Instant::now();
    let _results = runtime.batch_cosine_similarity(&query, &embeddings, 512).unwrap();
    let wasm_duration = start.elapsed();
    
    // Benchmark native
    let start = Instant::now();
    for i in 0..1000 {
        let embedding = &embeddings[i*512..(i+1)*512];
        let _ = native_cosine_similarity(&query, embedding);
    }
    let native_duration = start.elapsed();
    
    println!("WASM: {:?}", wasm_duration);
    println!("Native: {:?}", native_duration);
    println!("Ratio: {:.2}x", wasm_duration.as_secs_f64() / native_duration.as_secs_f64());
}
```

## Performance Expectations

### WASM vs Native in Lambda

| Operation | Native Rust | WASM (SIMD) | Overhead |
|-----------|-------------|-------------|----------|
| **Single similarity** | ~5μs | ~8μs | 1.6x |
| **Batch (1000)** | ~5ms | ~8ms | 1.6x |
| **Cold start** | ~100ms | ~150ms | 1.5x |
| **Memory** | ~50MB | ~70MB | 1.4x |

**Key insight**: WASM adds ~50-60% overhead vs native, but still provides:
- ✅ Portable across Lambda x86_64 and ARM64
- ✅ Sandboxed execution
- ✅ Single binary for multiple platforms

## Recommendation: Start with Native, Add WASM Later

**Phase 1 (Weeks 1-4)**: Native Rust only
```rust
// Direct implementation, no WASM
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    // Native SIMD using std::arch::x86_64
}
```

**Phase 2 (Weeks 5-8)**: Add WASM option
```rust
#[cfg(feature = "wasm-runtime")]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    get_wasm_runtime().cosine_similarity(a, b)
}

#[cfg(not(feature = "wasm-runtime"))]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    native_cosine_similarity(a, b)
}
```

## Summary: WASM in Serverless

**Use WASM when**:
- Deploying to multiple serverless platforms
- Need sandboxed execution
- Want portable binaries

**Use Native Rust when**:
- Maximum performance needed
- Only deploying to Lambda x86_64
- Simpler deployment preferred

**Recommended**: Start with native Rust, add WASM as optional feature later for portability.

---

**Next steps**: Build native Rust implementation first, then add WASM wrapper in Phase 2 if needed for multi-platform deployment.
