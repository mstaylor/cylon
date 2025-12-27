# Libfabric Communicator Design

## Overview

This document describes the design for integrating libfabric as a communication backend for Cylon Rust. Libfabric (Open Fabrics Interfaces - OFI) provides a framework-agnostic API for high-performance networking, supporting multiple fabric providers including TCP, shared memory, InfiniBand, and AWS Elastic Fabric Adapter (EFA).

### Goals

1. **Non-blocking Operations**: All communication operations return immediately; completion is checked via progress/wait mechanisms
2. **Provider Auto-Selection**: Libfabric automatically selects the best available provider based on the environment
3. **Full Collective Support**: Implement all collective operations (barrier, broadcast, allreduce, allgather, alltoall, reduce, scatter, gather, reduce_scatter)
4. **Consistent API**: Follow the same patterns as existing MPI, UCX, and FMI communicators
5. **AWS EFA Support**: Leverage high-performance EFA networking when available

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Cylon Application                         │
├─────────────────────────────────────────────────────────────────┤
│                      Communicator Trait                          │
├─────────────────────────────────────────────────────────────────┤
│                  LibfabricCommunicator                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Context   │  │  Endpoint   │  │   Completion Queue      │  │
│  │  (fid_fabric│  │  (fid_ep)   │  │      (fid_cq)           │  │
│  │   fid_domain│  │             │  │                         │  │
│  │   fid_av)   │  │             │  │                         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                    Libfabric FFI Bindings                        │
├─────────────────────────────────────────────────────────────────┤
│                      Provider Layer                              │
│  ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐   │
│  │  efa  │ │  shm  │ │  tcp  │ │ verbs │ │ psm2  │ │sockets│   │
│  └───────┘ └───────┘ └───────┘ └───────┘ └───────┘ └───────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Provider Selection

### Auto-Selection Mechanism

Libfabric's `fi_getinfo()` function automatically scans the environment and returns a list of available providers ordered by preference. The selection considers:

1. **Hardware availability**: EFA adapters, InfiniBand HCAs, etc.
2. **Requested capabilities**: Collectives, RMA, atomics
3. **Performance characteristics**: Bandwidth, latency

```c
// fi_getinfo() returns providers in preference order
struct fi_info *hints, *info;
hints = fi_allocinfo();
hints->caps = FI_COLLECTIVE | FI_MSG;
hints->ep_attr->type = FI_EP_RDM;

// Returns list of matching providers, best first
fi_getinfo(FI_VERSION(1,9), NULL, NULL, 0, hints, &info);
```

### Environment Variable Override

Users can override auto-selection via `FI_PROVIDER`:

```bash
# Force specific provider
export FI_PROVIDER=tcp

# Provider with specific options
export FI_PROVIDER=tcp;ofi_rxm

# Multiple providers (tried in order)
export FI_PROVIDER=efa,tcp
```

### Provider Capabilities

| Provider | Collectives | Use Case |
|----------|-------------|----------|
| efa | Yes | AWS EC2 with EFA |
| shm | Yes | Single-node (shared memory) |
| tcp | Yes (via rxm) | General TCP/IP networks |
| verbs | Yes | InfiniBand/RoCE |
| psm2 | Yes | Intel Omni-Path |
| sockets | Limited | Fallback TCP |

## Non-Blocking Operation Pattern

### Completion Queue Model

All operations are non-blocking. Completion is signaled via completion queues:

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Initiate   │────▶│   In Flight  │────▶│  Completed   │
│  Operation   │     │  (CQ Entry)  │     │  (CQ Read)   │
└──────────────┘     └──────────────┘     └──────────────┘
      │                     │                    │
      ▼                     ▼                    ▼
 fi_allreduce()      fi_cq_read()         Process result
 returns immediately  polls/waits
```

### Progress and Wait Functions

```rust
impl LibfabricCommunicator {
    /// Non-blocking progress check
    /// Returns true if any operations completed
    pub fn progress(&self) -> CylonResult<bool> {
        let mut cq_entry: fi_cq_tagged_entry = unsafe { std::mem::zeroed() };
        let ret = unsafe { fi_cq_read(self.cq, &mut cq_entry, 1) };

        match ret {
            1 => Ok(true),   // Completion available
            0 => Ok(false),  // No completions yet
            n if n == -FI_EAGAIN => Ok(false),
            n => Err(/* error handling */)
        }
    }

    /// Blocking wait for specific operation
    pub fn wait(&self, context: &OperationContext) -> CylonResult<()> {
        loop {
            let mut cq_entry: fi_cq_tagged_entry = unsafe { std::mem::zeroed() };
            let ret = unsafe { fi_cq_sread(self.cq, &mut cq_entry, 1, timeout) };

            if ret > 0 && cq_entry.op_context == context.as_ptr() {
                return Ok(());
            }
            // Handle errors, continue waiting
        }
    }

    /// Wait for all pending operations
    pub fn wait_all(&self) -> CylonResult<()> {
        while self.pending_ops.load(Ordering::SeqCst) > 0 {
            self.progress()?;
        }
        Ok(())
    }
}
```

### Operation Context Tracking

Each operation gets a unique context for completion matching:

```rust
pub struct OperationContext {
    id: u64,
    op_type: OperationType,
    status: AtomicU8,  // Pending, Completed, Error
}

pub enum OperationType {
    Send,
    Recv,
    Barrier,
    Broadcast,
    Allreduce,
    Allgather,
    Alltoall,
    Reduce,
    Scatter,
    Gather,
    ReduceScatter,
}
```

## Collective Operations

### Supported Operations

Libfabric provides native collective operations via `fi_collective.h`:

| Operation | Function | Description |
|-----------|----------|-------------|
| Barrier | `fi_barrier()` | Synchronize all processes |
| Broadcast | `fi_broadcast()` | Root sends to all |
| Allreduce | `fi_allreduce()` | Reduce + broadcast result |
| Allgather | `fi_allgather()` | Gather data to all |
| Alltoall | `fi_alltoall()` | All-to-all exchange |
| Reduce | `fi_reduce()` | Reduce to root |
| Scatter | `fi_scatter()` | Root distributes to all |
| Gather | `fi_gather()` | All send to root |
| Reduce-Scatter | `fi_reduce_scatter()` | Reduce + scatter |

### AV Set (Address Vector Set) for Collectives

Collectives require an AV set to define participating endpoints:

```rust
pub struct CollectiveGroup {
    av_set: *mut fid_av_set,
    coll_addr: fi_addr_t,
    members: Vec<fi_addr_t>,
}

impl CollectiveGroup {
    pub fn new(av: &AddressVector, members: &[fi_addr_t]) -> CylonResult<Self> {
        let mut av_set: *mut fid_av_set = std::ptr::null_mut();
        let attr = fi_av_set_attr {
            count: members.len(),
            ..Default::default()
        };

        unsafe {
            fi_av_set(av.as_ptr(), &attr, &mut av_set, std::ptr::null_mut())?;

            for &addr in members {
                fi_av_set_insert(av_set, addr)?;
            }

            let mut coll_addr: fi_addr_t = 0;
            fi_av_set_addr(av_set, &mut coll_addr)?;

            // Join the collective
            fi_join_collective(ep, coll_addr, av_set, 0, &mut mc, context)?;
        }

        Ok(Self { av_set, coll_addr, members })
    }
}
```

### Reduction Operations

Supported reduction operations map to `fi_op`:

```rust
pub fn to_fi_op(op: ReduceOp) -> fi_op {
    match op {
        ReduceOp::Sum => FI_SUM,
        ReduceOp::Min => FI_MIN,
        ReduceOp::Max => FI_MAX,
        ReduceOp::Prod => FI_PROD,
        ReduceOp::Land => FI_LAND,  // Logical AND
        ReduceOp::Lor => FI_LOR,    // Logical OR
        ReduceOp::Band => FI_BAND,  // Bitwise AND
        ReduceOp::Bor => FI_BOR,    // Bitwise OR
        ReduceOp::Lxor => FI_LXOR,  // Logical XOR
        ReduceOp::Bxor => FI_BXOR,  // Bitwise XOR
    }
}
```

### Data Types

Supported data types map to `fi_datatype`:

```rust
pub fn to_fi_datatype<T>() -> fi_datatype {
    match std::any::TypeId::of::<T>() {
        t if t == TypeId::of::<i8>() => FI_INT8,
        t if t == TypeId::of::<i16>() => FI_INT16,
        t if t == TypeId::of::<i32>() => FI_INT32,
        t if t == TypeId::of::<i64>() => FI_INT64,
        t if t == TypeId::of::<u8>() => FI_UINT8,
        t if t == TypeId::of::<u16>() => FI_UINT16,
        t if t == TypeId::of::<u32>() => FI_UINT32,
        t if t == TypeId::of::<u64>() => FI_UINT64,
        t if t == TypeId::of::<f32>() => FI_FLOAT,
        t if t == TypeId::of::<f64>() => FI_DOUBLE,
        _ => panic!("Unsupported data type"),
    }
}
```

## Out-of-Band (OOB) Communication

### Address Exchange Protocol

Before libfabric endpoints can communicate, they must exchange addresses. We use Redis (consistent with UCX/UCC implementation):

```
┌─────────────────────────────────────────────────────────────────┐
│                    Address Exchange Flow                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Worker 0              Redis              Worker 1               │
│     │                    │                    │                  │
│     │  1. Get local addr │                    │                  │
│     │◄──fi_getname()─────│                    │                  │
│     │                    │                    │                  │
│     │  2. Publish addr   │                    │                  │
│     │───SET worker:0────▶│                    │                  │
│     │                    │                    │                  │
│     │                    │◄───SET worker:1────│  3. Publish addr │
│     │                    │                    │                  │
│     │  4. Get peer addr  │                    │                  │
│     │───GET worker:1────▶│                    │                  │
│     │◄──────────────────│                    │                  │
│     │                    │                    │                  │
│     │  5. Insert to AV   │                    │                  │
│     │◄──fi_av_insert()───│                    │                  │
│     │                    │                    │                  │
└─────────────────────────────────────────────────────────────────┘
```

### Redis Keys Structure

```
cylon:fmi:{session_id}:worker:{rank}:addr     # Worker's libfabric address
cylon:fmi:{session_id}:worker:{rank}:ready    # Ready flag
cylon:fmi:{session_id}:barrier:{barrier_id}   # Barrier counter
cylon:fmi:{session_id}:world_size             # Total number of workers
```

## API Design

### Configuration

```rust
/// Libfabric communicator configuration
#[derive(Clone, Debug)]
pub struct LibfabricConfig {
    /// Force specific provider (None = auto-select)
    pub provider: Option<String>,

    /// Endpoint type
    pub endpoint_type: EndpointType,

    /// Completion queue size
    pub cq_size: usize,

    /// Address vector size (max peers)
    pub av_size: usize,

    /// Progress mode
    pub progress_mode: ProgressMode,

    /// Redis URL for OOB communication
    pub redis_url: String,

    /// Session ID for namespace isolation
    pub session_id: String,
}

#[derive(Clone, Debug, Default)]
pub enum EndpointType {
    #[default]
    ReliableDatagram,  // FI_EP_RDM - most common
    Message,           // FI_EP_MSG - connection-oriented
}

#[derive(Clone, Debug, Default)]
pub enum ProgressMode {
    #[default]
    Auto,      // FI_PROGRESS_AUTO
    Manual,    // FI_PROGRESS_MANUAL
}

impl Default for LibfabricConfig {
    fn default() -> Self {
        Self {
            provider: None,  // Auto-select
            endpoint_type: EndpointType::ReliableDatagram,
            cq_size: 1024,
            av_size: 256,
            progress_mode: ProgressMode::Auto,
            redis_url: "redis://127.0.0.1:6379".to_string(),
            session_id: uuid::Uuid::new_v4().to_string(),
        }
    }
}
```

### Communicator Trait Implementation

```rust
impl Communicator for LibfabricCommunicator {
    fn rank(&self) -> i32 { self.rank }
    fn world_size(&self) -> i32 { self.world_size }

    fn send(&self, buf: &[u8], dest: i32, tag: i32) -> CylonResult<()>;
    fn recv(&self, buf: &mut [u8], source: i32, tag: i32) -> CylonResult<()>;

    fn barrier(&self) -> CylonResult<()>;
    fn allgather(&self, send: &[u8], recv: &mut [u8]) -> CylonResult<()>;
    fn alltoall(&self, send: &[u8], recv: &mut [u8]) -> CylonResult<()>;
    fn allreduce<T: Copy>(&self, send: &[T], recv: &mut [T], op: ReduceOp) -> CylonResult<()>;
    fn broadcast<T: Copy>(&self, buf: &mut [T], root: i32) -> CylonResult<()>;
    fn gather(&self, send: &[u8], recv: &mut [u8], root: i32) -> CylonResult<()>;
    fn scatter(&self, send: &[u8], recv: &mut [u8], root: i32) -> CylonResult<()>;
    fn reduce<T: Copy>(&self, send: &[T], recv: &mut [T], op: ReduceOp, root: i32) -> CylonResult<()>;
    fn reduce_scatter<T: Copy>(&self, send: &[T], recv: &mut [T], op: ReduceOp) -> CylonResult<()>;
}
```

### Channel Implementation

```rust
pub struct LibfabricChannel {
    comm: Arc<LibfabricCommunicator>,
    pending_sends: Vec<OperationContext>,
    pending_recvs: Vec<OperationContext>,
}

impl Channel for LibfabricChannel {
    fn send(&mut self, buf: &ArrowBuffer, dest: i32, tag: i32) -> CylonResult<()>;
    fn recv(&mut self, source: i32, tag: i32) -> CylonResult<ArrowBuffer>;

    fn isend(&mut self, buf: &ArrowBuffer, dest: i32, tag: i32) -> CylonResult<RequestHandle>;
    fn irecv(&mut self, source: i32, tag: i32) -> CylonResult<RequestHandle>;

    fn progress(&mut self) -> CylonResult<bool>;
    fn wait(&mut self, handle: RequestHandle) -> CylonResult<()>;
    fn wait_all(&mut self) -> CylonResult<()>;
}
```

## Implementation Phases

### Phase 1: FFI Bindings & Build Setup

**Files:**
- `rust/build.rs` - Add libfabric detection and bindgen
- `rust/Cargo.toml` - Add `fmi` feature (libfabric)
- `rust/src/net/fmi/libfabric_sys.rs` - Raw FFI bindings

**Tasks:**
1. Detect libfabric installation via pkg-config or environment variable
2. Generate bindings for core headers:
   - `rdma/fabric.h`
   - `rdma/fi_endpoint.h`
   - `rdma/fi_domain.h`
   - `rdma/fi_cm.h`
   - `rdma/fi_collective.h`
3. Link against libfabric library

### Phase 2: Core Infrastructure

**Files:**
- `rust/src/net/fmi/mod.rs` - Module exports and config
- `rust/src/net/fmi/context.rs` - Fabric context management
- `rust/src/net/fmi/error.rs` - Error handling

**Tasks:**
1. Initialize fabric, domain, completion queue
2. Provider enumeration and selection
3. Error code translation

### Phase 3: Non-Blocking Communicator

**Files:**
- `rust/src/net/fmi/communicator.rs` - LibfabricCommunicator

**Tasks:**
1. Implement endpoint creation
2. Address vector setup
3. OOB address exchange via Redis
4. Progress/wait mechanisms
5. Basic send/recv operations

### Phase 4: Collective Operations

**Files:**
- `rust/src/net/fmi/operations.rs` - Collective implementations

**Tasks:**
1. AV set creation for collectives
2. Implement all collective operations:
   - barrier
   - broadcast
   - allreduce
   - allgather
   - alltoall
   - reduce
   - scatter
   - gather
   - reduce_scatter

### Phase 5: Channel Implementation

**Files:**
- `rust/src/net/fmi/channel.rs` - LibfabricChannel

**Tasks:**
1. Arrow buffer serialization
2. Non-blocking isend/irecv
3. Request tracking and completion

### Phase 6: Integration & Testing

**Files:**
- `rust/examples/fmi_libfabric_example.rs`
- `rust/tests/fmi_tests.rs`

**Tasks:**
1. Integration with CylonContext
2. Example programs
3. Unit and integration tests
4. Performance benchmarks

## AWS Lambda Considerations

### Limitations

| Constraint | Impact |
|------------|--------|
| No EFA access | Limited to tcp/sockets providers |
| Ephemeral instances | Collectives may fail if instance recycled |
| 15-minute timeout | Long operations may be interrupted |
| VPC required | Must configure security groups for inter-Lambda traffic |
| Cold starts | Latency before joining collectives |
| No shared memory | shm provider unavailable |

### Recommendations

For production distributed workloads on AWS:

1. **EC2 with EFA**: Best performance, full provider support
2. **EKS/ECS**: Persistent containers, stable networking
3. **AWS Batch**: Batch-style distributed workloads
4. **ParallelCluster**: HPC-optimized with EFA

Lambda may work for:
- Small-scale experiments
- Proof-of-concept testing
- Operations tolerant of failures

## File Structure

```
rust/src/net/fmi/
├── mod.rs                 # Module exports, LibfabricConfig
├── libfabric_sys.rs       # FFI bindings (generated)
├── error.rs               # Error handling
├── context.rs             # FabricContext (fabric, domain, cq)
├── endpoint.rs            # Endpoint management
├── address_vector.rs      # AV and AV set management
├── communicator.rs        # LibfabricCommunicator
├── channel.rs             # LibfabricChannel
├── operations.rs          # Collective operations
└── oob.rs                 # Out-of-band via Redis
```

## Dependencies

```toml
[dependencies]
# Existing
redis = { version = "0.25", optional = true, features = ["tokio-comp"] }

[build-dependencies]
bindgen = { version = "0.69", optional = true }

[features]
fmi = ["redis", "dep:bindgen"]  # Already defined, will use for libfabric
```

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `FI_PROVIDER` | Force specific provider | `tcp`, `efa`, `shm` |
| `FI_LOG_LEVEL` | Logging verbosity | `warn`, `info`, `debug` |
| `LIBFABRIC_PATH` | Custom installation path | `/home/user/libfabric/install` |
| `CYLON_FMI_REDIS_URL` | Redis URL for OOB | `redis://localhost:6379` |
| `CYLON_FMI_SESSION_ID` | Session identifier | `my-session-123` |

## References

- [Libfabric Documentation](https://ofiwg.github.io/libfabric/)
- [Libfabric Programmer's Manual](https://ofiwg.github.io/libfabric/main/man/)
- [AWS EFA Documentation](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html)
- [OFI Collective Specification](https://ofiwg.github.io/libfabric/main/man/fi_collective.3.html)
