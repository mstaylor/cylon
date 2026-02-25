# gcylon Chunked Memory Architecture

This document describes the architecture of memory-efficient chunked operations in gcylon (GPU Cylon) and the Rust FFI layer that exposes them.

## Problem Statement

Standard gcylon distributed operations (Shuffle, AllGather) can exhaust GPU memory on large datasets. The memory pressure comes from four stages that all coexist simultaneously:

1. **Input table** remains allocated throughout the operation
2. **Hash partitioning** creates a full copy of the input (Shuffle only)
3. **Serialization/send/receive buffers** scale with world size
4. **Final concatenation** creates yet another copy of the output

This results in peak GPU memory of **~4-5x input size** for Shuffle and **~(W+2)x input size** for AllGather (where W = world size). A 4-worker Shuffle of 4GB per worker needs ~18GB peak; AllGather needs ~32GB.

## Solution Overview

The chunked operations system addresses this through two mechanisms:

1. **Row-based chunking** — Split the input table into smaller row slices and process each chunk independently, freeing intermediate buffers between chunks
2. **Smart auto-selection** — At runtime, estimate memory requirements and automatically choose between the fast direct path (single-pass) and the chunked path

```
Peak Memory Comparison (4 workers, 4GB input per worker):

Operation      | Direct (current) | Chunked
---------------|------------------|--------
Shuffle        | 18GB (OOM)       | ~8GB
AllGather      | 32GB (OOM)       | ~10GB
```

### GPU-to-CPU Memory Spillover (RMM Managed Memory)

For scenarios where even chunked operations exceed GPU memory, use RMM's `managed_memory_resource` rather than custom allocators. This enables CUDA Unified Memory (UVM) which automatically migrates pages between GPU and CPU as needed:

```cpp
auto mr = rmm::mr::managed_memory_resource{};
rmm::mr::set_current_device_resource(&mr);
```

This is a runtime-level setting that affects all cuDF allocations transparently — no changes to gcylon operation code are needed. UVM handles page faulting at the hardware level, migrating pages on demand between GPU and CPU memory.

---

## Architecture

### Layer Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                        Rust Application                          │
│   GpuTable::shuffle(), GpuTable::allgather(), GpuTable::join()  │
├──────────────────────────────────────────────────────────────────┤
│                     Rust FFI Bindings                             │
│   gpu/config.rs, gpu/context.rs, gpu/table.rs, gpu/ffi.rs       │
├──────────────────────────────────────────────────────────────────┤
│                        C API Layer                                │
│   gcylon_c.h / gcylon_c.cpp                                      │
│   Thin wrapper: opaque types, error codes, config conversion     │
├──────────────────────────────────────────────────────────────────┤
│                     C++ Smart Operations                          │
│   SmartShuffle() / SmartAllGather()                              │
│   Memory estimation → route to direct or chunked path            │
├──────────────────────────────────────────────────────────────────┤
│                    C++ Chunked Operations                         │
│   ChunkedShuffle() / ChunkedAllGather()                          │
│   split_table() → per-chunk hash_partition + AllToAll → concat   │
├──────────────────────────────────────────────────────────────────┤
│                    Memory Utilities                               │
│   GcylonConfig, memory_utils (estimation)                        │
├──────────────────────────────────────────────────────────────────┤
│                   Existing gcylon Operations                      │
│   Shuffle(), AllGather(), Gather(), Bcast(), DistributedSort()   │
│   (cudf + MPI/UCX collective transport)                          │
└──────────────────────────────────────────────────────────────────┘
```

### Component Descriptions

#### 1. Configuration (`gcylon_config.hpp`)

`GcylonConfig` controls chunking behavior with these knobs:

| Field | Default | Purpose |
|-------|---------|---------|
| `gpu_memory_limit` | 0 (auto) | Explicit byte limit; 0 = use fraction of free memory |
| `gpu_memory_fraction` | 0.8 | Fraction of free GPU memory to budget for operations |
| `chunk_size_bytes` | 0 (auto) | Explicit chunk size; 0 = auto-calculate from memory |
| `min_chunk_rows` | 1024 | Floor to prevent tiny inefficient chunks |

Two presets are provided:
- **`Default()`** — 80% GPU memory with auto chunk sizing
- **`LowMemory()`** — 60% GPU memory for high memory pressure scenarios

#### 2. Memory Utilities (`staging/memory_utils.hpp`)

Provides runtime GPU memory introspection and cost estimation:

- **`get_gpu_memory_info()`** — Queries `cudaMemGetInfo` for free/total/used
- **`estimate_table_size()`** — Sums column data + null bitmasks for a `cudf::table_view`
- **`estimate_shuffle_memory()`** — Conservative estimate: `input_size * (world_size + 3)`
- **`estimate_allgather_memory()`** — Estimate: `input_size * (world_size + 2)`
- **`calculate_chunk_rows()`** — Divides available memory by per-row cost to find optimal chunk size, clamped to `[min_chunk_rows, total_rows]`
- **`fits_in_gpu_memory()`** — Boolean check against safety-fractioned free memory

#### 3. Chunked Operations (`gtable_api_chunked.cpp`)

**ChunkedShuffle** processes the input in row-based chunks:

```
For each chunk:
  1. cudf::slice(input, [start, end])     → chunk_view (zero-copy)
  2. cudf::hash_partition(chunk, cols, W)  → partitioned table + offsets
  3. gcylon::net::AllToAll(partitioned)    → exchanged chunk result
  4. partitioned goes out of scope        → GPU memory freed
Final: cudf::concatenate(all_results)     → output table
```

**ChunkedAllGather** follows the same pattern but gathers from all workers per chunk instead of hash-partitioning.

**SmartShuffle / SmartAllGather** check estimated memory against 50% of free GPU memory. If the operation fits comfortably, they use the fast single-pass direct path. Otherwise they route to the chunked path.

#### 4. C API (`c_api/gcylon_c.h`, `c_api/gcylon_c.cpp`)

A C-compatible wrapper layer that enables FFI from Rust and other languages:

- **Opaque types**: `GcylonContext*`, `GcylonTable*` hide C++ objects behind pointers
- **Status codes**: `GCYLON_OK` (0), `GCYLON_ERROR` (-1), `GCYLON_OOM` (-2), `GCYLON_INVALID_ARG` (-3)
- **Thread-local error**: `gcylon_get_last_error()` returns the last error message
- **Config conversion**: C `GcylonConfig` struct maps 1:1 to C++ `gcylon::GcylonConfig`
- **Operations exposed**: `gcylon_shuffle`, `gcylon_allgather`, `gcylon_gather`, `gcylon_broadcast`, `gcylon_distributed_join`, `gcylon_distributed_sort`, `gcylon_repartition`
- **Device management**: `gcylon_set_device`, `gcylon_get_device`, `gcylon_get_device_count`

The C API routes through Smart* variants (auto-selecting direct vs chunked) when available.

#### 5. Rust FFI Bindings (`rust/src/gpu/`)

The Rust module mirrors the C API with safe wrappers:

| Rust Module | Purpose |
|-------------|---------|
| `ffi.rs` | Raw `extern "C"` declarations matching `gcylon_c.h` |
| `config.rs` | `GpuConfig` with builder pattern (`with_gpu_memory_fraction()`, `with_chunk_size()`, etc.) |
| `context.rs` | `GpuContext` (RAII wrapper, `Drop` calls `gcylon_context_free`) with `rank()`, `world_size()`, `memory_info()` |
| `table.rs` | `GpuTable` with `shuffle()`, `allgather()`, `distributed_join()` — all accept `Option<GpuConfig>` |
| `mod.rs` | Public re-exports + device management functions (`set_device`, `get_device`, `get_device_count`) |

The `gpu` feature flag gates the entire module. Build with `cargo build --features gpu`.

---

## File Reference

### C++ Files

| File | Purpose |
|------|---------|
| `cpp/src/gcylon/gcylon_config.hpp` | `GcylonConfig` struct with presets |
| `cpp/src/gcylon/staging/memory_utils.hpp` | GPU memory queries and cost estimation |
| `cpp/src/gcylon/gtable_api.hpp` | Declarations for `ChunkedShuffle`, `ChunkedAllGather`, `SmartShuffle`, `SmartAllGather` |
| `cpp/src/gcylon/gtable_api_chunked.cpp` | Implementation of chunked and smart operations |
| `cpp/src/gcylon/c_api/gcylon_c.h` | C API header |
| `cpp/src/gcylon/c_api/gcylon_c.cpp` | C API implementation |

### Rust Files

| File | Purpose |
|------|---------|
| `rust/src/gpu/mod.rs` | Module root, device management, public exports |
| `rust/src/gpu/ffi.rs` | Raw FFI bindings to C API |
| `rust/src/gpu/config.rs` | `GpuConfig` safe wrapper |
| `rust/src/gpu/context.rs` | `GpuContext` RAII wrapper |
| `rust/src/gpu/table.rs` | `GpuTable` operations |
| `rust/examples/gpu_shuffle.rs` | Example: distributed shuffle with config |
| `rust/examples/gpu_join.rs` | Example: distributed join |
| `rust/examples/gpu_sort.rs` | Example: distributed sort |

---

## Testing

### Existing Tests (Direct Operations)

The existing test suite in `cpp/test/gcylon/` validates the direct (non-chunked) operations using MPI with 4 workers. All tests use the Catch2 framework and CSV-based input/output comparison.

| Test File | Operation | Semantics |
|-----------|-----------|-----------|
| `shuffle_gcylon_table_test.cpp` | Shuffle | Loads per-rank CSV files, shuffles by int and string columns, compares output against pre-computed expected CSVs |
| `allgather_gcylon_table_test.cpp` | AllGather | Slices tables per-rank, gathers across workers, verifies concatenated result matches the union of all input slices |
| `gather_gcylon_table_test.cpp` | Gather | Gathers to root rank (with/without root participation), verifies root's result matches concatenated inputs |
| `bcast_gcylon_table_test.cpp` | Broadcast | Broadcasts from root, verifies all receivers got exact copies of the original |
| `sort_gcylon_table_test.cpp` | DistributedSort | Sorts by specified columns, compares against pre-sorted expected files; also tests sliced inputs |
| `repartition_gcylon_table_test.cpp` | Repartition | Redistributes rows across workers (even or custom partition sizes), verifies row counts and data integrity via gather-compare |
| `create_cudf_table_test.cpp` | Table Creation | Basic sanity: creates cudf tables and verifies structure |

Test data lives in `data/input/` and `data/output/` (CSV files, copied to build dir by CMake). The `test_gutils.hpp` header provides shared helpers (`readCSV`, `writeCSV`, `PerformShuffleTest`, `PerformGatherTest`, etc.).

### Chunked Operation Tests

The `chunked_shuffle_test.cpp` test file validates the chunked memory system:

| Test | Semantics |
|------|-----------|
| **SmartShuffle small table** | Calls `SmartShuffle` on small CSV input; verifies it routes to the direct fast path and produces correct results by comparing against expected output |
| **ChunkedShuffle forced** | Sets `gpu_memory_fraction = 0.1` and `min_chunk_rows = 2` to force the chunked path even on GPUs with ample memory; compares output against direct `Shuffle` result |
| **ChunkedShuffle empty table** | Creates an empty (0-row) table slice and verifies `ChunkedShuffle` returns an empty table without error |
| **SmartAllGather small table** | Calls `SmartAllGather` on small data; verifies output matches direct `AllGather` |
| **ChunkedAllGather forced** | Forces chunking via low memory fraction; compares output against direct `AllGather` result |

### Running Tests on GPU Docker

The gcylon Docker image (`docker/gcylon/Dockerfile`) provides a self-contained GPU build environment with CUDA 12.8, cuDF/RMM (RAPIDS 24.10), MPI, UCX, UCC, and Redis.

#### Prerequisites

- Docker with NVIDIA Container Toolkit (`nvidia-docker2` or `nvidia-container-toolkit`)
- An NVIDIA GPU accessible to the host

#### Build the Docker Image

```bash
cd /path/to/cylon
docker build -t gcylon:latest -f docker/gcylon/Dockerfile .
```

This takes ~30-60 minutes. It builds:
1. Conda environment with cuDF/RMM + OpenMPI
2. UCX and UCC from source
3. Redis (hiredis + redis++)
4. Cylon C++ library with UCX/UCC/Redis
5. gcylon C++ GPU library (installed to conda prefix)
6. pygcylon Python bindings
7. Rust crate with `--features gpu`

#### Run C++ Tests

```bash
# Launch container with GPU access
docker run --gpus all -it gcylon:latest /bin/bash

# Inside the container:
source /opt/conda/etc/profile.d/conda.sh
conda activate cylon_dev

# Run all gcylon tests (4 MPI processes)
cd /cylon/cpp/build_gcylon
ctest --output-on-failure

# Or run individual tests directly with mpirun
mpirun --allow-run-as-root --oversubscribe \
    --mca opal_cuda_support 1 --mca pml ucx --mca btl_openib_allow_ib true \
    -np 4 ./bin/chunked_shuffle_test
```

The CMake test configuration (`cpp/test/gcylon/CMakeLists.txt`) sets up MPI parameters including `--mca opal_cuda_support 1` for GPU-aware MPI and runs each test with 4 processes.

#### Run Rust GPU Examples

```bash
# Inside the container:
cd /cylon/rust

# Run examples with MPI (requires mpirun)
mpirun --allow-run-as-root --oversubscribe -np 4 \
    ./target/release/examples/gpu_shuffle

mpirun --allow-run-as-root --oversubscribe -np 4 \
    ./target/release/examples/gpu_join

mpirun --allow-run-as-root --oversubscribe -np 4 \
    ./target/release/examples/gpu_sort
```

---

## Build Summary

| Step | Command | Output |
|------|---------|--------|
| Docker image | `docker build -t gcylon:latest -f docker/gcylon/Dockerfile .` | Image with all dependencies |
| C++ gcylon (in Docker) | `cd build_gcylon && cmake ... && make` | `libgcylon.so` in conda prefix |
| Rust (in Docker) | `cargo build --features gpu --examples --release` | Rust binaries in `target/release/` |
| C++ tests | `cd build_gcylon && ctest` | MPI-based tests with 4 workers |
| Rust examples | `mpirun -np 4 ./target/release/examples/gpu_shuffle` | Distributed GPU shuffle |

### Environment Variables (inside Docker)

| Variable | Value | Purpose |
|----------|-------|---------|
| `CYLON_HOME` | `/cylon` | Cylon source root |
| `CYLON_PREFIX` | `/cylon/install` | CPU Cylon install prefix |
| `GCYLON_BUILD` | `/cylon/cpp/build_gcylon` | gcylon build directory |
| `CONDA_PREFIX` | `/opt/conda/envs/cylon_dev` | Conda env with cuDF/RMM |
| `UCX_HOME` | `/ucx` | UCX source + install |
| `UCC_HOME` | `/ucc` | UCC source + install |
| `LD_LIBRARY_PATH` | `$GCYLON_BUILD/lib:$CYLON_PREFIX/lib:...` | Library search paths |