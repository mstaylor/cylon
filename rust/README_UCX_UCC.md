# UCX/UCC Integration for Cylon Rust

This document describes how to build Cylon Rust with UCX and UCC support.

## Prerequisites

- UCX installed (e.g., at `$HOME/ucx`)
- UCC installed (e.g., at `$HOME/ucc`)
- Redis server for OOB communication

## Environment Variables

### Option 1: Using Install Prefix

```bash
export UCX_INSTALL_PREFIX=$HOME/ucx
export UCC_INSTALL_PREFIX=$HOME/ucc
```

### Option 2: Using Separate Include/Lib Paths

```bash
export UCX_INCLUDEDIR=$HOME/ucx/include
export UCX_LIBDIR=$HOME/ucx/lib
export UCC_INCLUDEDIR=$HOME/ucc/include
export UCC_LIBDIR=$HOME/ucc/lib
```

### Option 3: Using Conda Environment

```bash
conda activate cylon_dev
# UCX/UCC will be automatically detected from CONDA_PREFIX
```

## Building

### Build with UCX only

```bash
cargo build --features ucx
```

### Build with UCX and UCC

```bash
cargo build --features ucc
```

Note: UCC feature automatically enables UCX feature.

## Running Examples

```bash
# Start Redis server (if using Redis OOB)
redis-server

# Run with environment variables
UCX_INSTALL_PREFIX=$HOME/ucx cargo run --features ucx --example ucx_example
```

## Architecture

- `build.rs` - Generates FFI bindings using bindgen at compile time
- `src/net/ucx/ucx_sys.rs` - UCX FFI bindings
- `src/net/ucc/ucc_sys.rs` - UCC FFI bindings
- `src/net/ucx/redis_oob.rs` - Redis-based out-of-band communication
- `src/net/ucx/communicator.rs` - UCX communicator implementation
- `src/net/ucc/communicator.rs` - UCC communicator implementation
