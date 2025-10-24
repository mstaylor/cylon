# Distributed Operations Roadmap for Cylon Rust

## Overview

This document outlines the architecture and implementation plan for adding distributed operations to Cylon Rust. Distributed operations enable table operations across multiple processes using MPI for communication.

## Current Status

### Completed ✓
- All local table operations (280 tests passing)
- Basic CylonContext with stub `get_rank()` and `get_world_size()` methods
- Arrow-based table representation
- Comprehensive join, set operations, aggregations, and I/O

### Not Yet Implemented
- MPI communicator backend
- Communication primitives (all-to-all, gather, etc.)
- Distributed table operations (shuffle, distributed join, etc.)

## Architecture

### 1. MPI Backend Layer

The foundation for distributed operations is the MPI communication layer.

#### Components Needed:

**1.1 Communicator Trait** (`src/net/communicator.rs`)
```rust
pub trait Communicator: Send + Sync {
    fn get_rank(&self) -> i32;
    fn get_world_size(&self) -> i32;
    fn barrier(&self) -> CylonResult<()>;

    // Point-to-point communication
    fn send(&self, data: &[u8], dest: i32, tag: i32) -> CylonResult<()>;
    fn recv(&self, dest: &mut Vec<u8>, source: i32, tag: i32) -> CylonResult<()>;

    // Collective operations
    fn all_to_all(&self, send_data: Vec<Vec<u8>>) -> CylonResult<Vec<Vec<u8>>>;
    fn gather(&self, data: &[u8], root: i32) -> CylonResult<Vec<u8>>;
    fn allgather(&self, data: &[u8]) -> CylonResult<Vec<Vec<u8>>>;
    fn broadcast(&self, data: &mut Vec<u8>, root: i32) -> CylonResult<()>;
}
```

**1.2 MPI Implementation** (`src/net/mpi/mod.rs`)
```rust
pub struct MPICommunicator {
    universe: mpi::Universe,
    world: mpi::SystemCommunicator,
}

impl MPICommunicator {
    pub fn new() -> CylonResult<Self> {
        // Initialize MPI using rsmpi crate
    }
}

impl Communicator for MPICommunicator {
    // Implement all trait methods using rsmpi
}
```

**Dependencies:**
- Add `rsmpi` crate for Rust MPI bindings
- Requires MPI installation (OpenMPI or MPICH)

### 2. Arrow Table Serialization

Distributed operations require serializing Arrow tables for network transmission.

#### Components Needed:

**2.1 Table Serialization** (`src/net/serialize.rs`)
```rust
/// Serialize Arrow RecordBatch to bytes using Arrow IPC format
pub fn serialize_record_batch(batch: &RecordBatch) -> CylonResult<Vec<u8>> {
    // Use arrow::ipc::writer::StreamWriter
}

/// Deserialize bytes to Arrow RecordBatch
pub fn deserialize_record_batch(data: &[u8]) -> CylonResult<RecordBatch> {
    // Use arrow::ipc::reader::StreamReader
}

/// Serialize multiple batches with metadata
pub fn serialize_table(table: &Table) -> CylonResult<Vec<u8>> {
    // Serialize schema + all batches
}

/// Deserialize to Table
pub fn deserialize_table(ctx: Arc<CylonContext>, data: &[u8]) -> CylonResult<Table> {
    // Deserialize and reconstruct table
}
```

**Key Design Decision:**
- Use Arrow IPC (Inter-Process Communication) format for zero-copy serialization
- Maintain Arrow's columnar format during transmission
- Support chunked tables (multiple RecordBatches)

### 3. Shuffle Operation

Shuffle is the foundation for all distributed operations. It redistributes table rows across processes based on hash partitioning.

#### Components Needed:

**3.1 Hash Partitioning** (`src/ops/partition.rs`)
```rust
/// Partition table rows by hash of specified columns
/// Returns: (partitions, row_counts_per_partition)
pub fn hash_partition_table(
    table: &Table,
    hash_columns: &[i32],
    num_partitions: i32,
) -> CylonResult<(Vec<RecordBatch>, Vec<usize>)> {
    // 1. Compute hash for each row based on hash_columns
    // 2. Assign each row to partition: hash % num_partitions
    // 3. Split table into num_partitions batches
    // 4. Return partitioned batches + counts
}
```

**3.2 Shuffle Implementation** (`src/ops/shuffle.rs`)
```rust
/// Shuffle table across all processes using all-to-all communication
/// C++ reference: table.cpp:194-215 (shuffle_table_by_hashing)
pub fn shuffle(
    ctx: &Arc<CylonContext>,
    table: &Table,
    hash_columns: &[i32],
) -> CylonResult<Table> {
    let world_size = ctx.get_world_size();

    // 1. Hash partition local table into world_size partitions
    let (partitions, counts) = hash_partition_table(table, hash_columns, world_size)?;

    // 2. Serialize each partition
    let serialized: Vec<Vec<u8>> = partitions
        .iter()
        .map(|batch| serialize_record_batch(batch))
        .collect::<CylonResult<_>>()?;

    // 3. All-to-all exchange: send partition i to process i
    let received = ctx.communicator()?.all_to_all(serialized)?;

    // 4. Deserialize received data
    let received_batches: Vec<RecordBatch> = received
        .iter()
        .map(|data| deserialize_record_batch(data))
        .collect::<CylonResult<_>>()?;

    // 5. Combine into new table
    Table::from_record_batches(ctx.clone(), received_batches)
}
```

**C++ Reference:**
- `cpp/src/cylon/table.cpp:194-215` - `shuffle_table_by_hashing`
- `cpp/src/cylon/net/ops/all_to_all.hpp` - `all_to_all_arrow_tables`

### 4. Distributed Join

Distributed join shuffles both tables by join keys, then performs local joins.

#### Components Needed:

**4.1 Distributed Join** (`src/ops/join.rs` - add distributed variants)
```rust
/// Distributed hash join
/// C++ reference: table.cpp:290-318 (DistributedJoin)
pub fn distributed_join(
    ctx: &Arc<CylonContext>,
    left: &Table,
    right: &Table,
    join_type: JoinType,
    left_on: &[i32],
    right_on: &[i32],
) -> CylonResult<Table> {
    // 1. Shuffle left table by left_on columns
    let left_shuffled = shuffle(ctx, left, left_on)?;

    // 2. Shuffle right table by right_on columns
    let right_shuffled = shuffle(ctx, right, right_on)?;

    // 3. Perform local join (rows with same key are co-located)
    join(&left_shuffled, &right_shuffled, join_type, left_on, right_on)
}
```

**Algorithm:**
1. Hash partition both tables by join keys
2. Shuffle so matching keys are on same process
3. Perform local join (all matching rows are co-located)
4. Result is distributed across all processes

**C++ Reference:**
- `cpp/src/cylon/table.cpp:290-318` - `DistributedJoin`

### 5. Distributed Set Operations

Union, intersection, difference, and subtract operations on distributed tables.

#### Components Needed:

**5.1 Distributed Union** (`src/ops/set_ops.rs` - add distributed variants)
```rust
/// Distributed union of tables
/// C++ reference: table.cpp:436-466 (DistributedUnion)
pub fn distributed_union(
    ctx: &Arc<CylonContext>,
    left: &Table,
    right: &Table,
) -> CylonResult<Table> {
    // Approach 1: Gather to root, union, scatter
    // Approach 2: Local union, then shuffle for deduplication

    // For now, use simple approach:
    // 1. Gather both tables to root
    // 2. Perform local union at root
    // 3. Redistribute result
}
```

**5.2 Distributed Intersection** (`src/ops/set_ops.rs`)
```rust
/// Distributed intersection
/// C++ reference: table.cpp:500-532 (DistributedIntersect)
pub fn distributed_intersect(
    ctx: &Arc<CylonContext>,
    left: &Table,
    right: &Table,
) -> CylonResult<Table> {
    // 1. Shuffle both tables by all columns (for co-location)
    // 2. Perform local intersection
    // 3. Result is already distributed
}
```

**5.3 Distributed Difference** (`src/ops/set_ops.rs`)
```rust
/// Distributed difference (left - right)
/// C++ reference: table.cpp:562-600 (DistributedDifference)
pub fn distributed_difference(
    ctx: &Arc<CylonContext>,
    left: &Table,
    right: &Table,
) -> CylonResult<Table> {
    // Similar to intersect but keep rows only in left
}
```

**C++ Reference:**
- `cpp/src/cylon/table.cpp:436-466` - `DistributedUnion`
- `cpp/src/cylon/table.cpp:500-532` - `DistributedIntersect`
- `cpp/src/cylon/table.cpp:562-600` - `DistributedDifference`
- `cpp/src/cylon/table.cpp:630-660` - `DistributedSubtract`

### 6. Distributed Sort

Sort a distributed table globally.

#### Components Needed:

**6.1 Distributed Sort** (`src/ops/sort.rs` - new file)
```rust
/// Distributed sort using sample sort algorithm
/// C++ reference: table.cpp:698-726 (DistributedSort)
pub fn distributed_sort(
    ctx: &Arc<CylonContext>,
    table: &Table,
    sort_columns: &[i32],
) -> CylonResult<Table> {
    // Sample sort algorithm:
    // 1. Local sort
    // 2. Sample p-1 splitters from each process
    // 3. Gather samples, sort, select global splitters
    // 4. Partition by splitters
    // 5. All-to-all exchange
    // 6. Local sort of received data
}
```

**Algorithm (Sample Sort):**
1. Each process sorts its local data
2. Each process selects p-1 sample elements (splitters)
3. Root gathers all samples, sorts, selects global splitters
4. Broadcast global splitters to all processes
5. Each process partitions its data using splitters
6. All-to-all exchange
7. Each process sorts its received data
8. Result: globally sorted, distributed data

**C++ Reference:**
- `cpp/src/cylon/table.cpp:698-726` - `DistributedSort`

### 7. Gather and Broadcast Operations

Operations to collect distributed tables to one process or replicate from one.

#### Components Needed:

**7.1 Gather** (`src/ops/gather.rs` - new file)
```rust
/// Gather distributed table to root process
pub fn gather_table(
    ctx: &Arc<CylonContext>,
    table: &Table,
    root: i32,
) -> CylonResult<Option<Table>> {
    let rank = ctx.get_rank();

    // 1. Serialize local table
    let serialized = serialize_table(table)?;

    // 2. Gather to root
    let gathered = ctx.communicator()?.gather(&serialized, root)?;

    // 3. If root, deserialize and combine all tables
    if rank == root {
        let batches: Vec<RecordBatch> = /* deserialize gathered */;
        Ok(Some(Table::from_record_batches(ctx.clone(), batches)?))
    } else {
        Ok(None)
    }
}
```

**7.2 Broadcast** (`src/ops/gather.rs`)
```rust
/// Broadcast table from root to all processes
pub fn broadcast_table(
    ctx: &Arc<CylonContext>,
    table: Option<&Table>,
    root: i32,
) -> CylonResult<Table> {
    let rank = ctx.get_rank();

    let mut data = if rank == root {
        serialize_table(table.unwrap())?
    } else {
        Vec::new()
    };

    ctx.communicator()?.broadcast(&mut data, root)?;

    deserialize_table(ctx.clone(), &data)
}
```

## Implementation Phases

### Phase 1: MPI Foundation (Week 1-2)
**Goal:** Working MPI backend with basic communication

**Tasks:**
1. Add `rsmpi` dependency to Cargo.toml
2. Create `src/net/` module structure
3. Implement `Communicator` trait
4. Implement `MPICommunicator` with rsmpi
5. Update `CylonContext` to use real MPI communicator
6. Write basic MPI tests (send/recv, barrier)

**Deliverables:**
- `src/net/communicator.rs` - Trait definition
- `src/net/mpi/mod.rs` - MPI implementation
- `tests/mpi_basic_test.rs` - Basic MPI tests
- Working `mpirun -n 4 cargo test --test mpi_basic_test`

**Dependencies:**
```toml
[dependencies]
mpi = "0.7"  # rsmpi crate
```

### Phase 2: Serialization (Week 2)
**Goal:** Serialize/deserialize Arrow tables for network transmission

**Tasks:**
1. Implement `serialize_record_batch` using Arrow IPC
2. Implement `deserialize_record_batch`
3. Implement `serialize_table` for multi-batch tables
4. Implement `deserialize_table`
5. Write serialization tests

**Deliverables:**
- `src/net/serialize.rs` - Serialization functions
- `tests/serialize_test.rs` - Round-trip tests
- Performance benchmarks

### Phase 3: Collective Operations (Week 3)
**Goal:** Implement all-to-all and other collective operations

**Tasks:**
1. Implement `all_to_all` for Vec<Vec<u8>>
2. Implement `gather` and `allgather`
3. Implement `broadcast`
4. Write tests for each collective operation
5. Test with Arrow table data

**Deliverables:**
- Complete `Communicator` implementation
- `tests/collective_ops_test.rs`
- Working all-to-all with serialized tables

### Phase 4: Shuffle Operation (Week 4)
**Goal:** Core shuffle operation for redistributing tables

**Tasks:**
1. Implement `hash_partition_table` in `src/ops/partition.rs`
2. Implement `shuffle` in `src/ops/shuffle.rs`
3. Write comprehensive shuffle tests
4. Test with various data types and table sizes
5. Benchmark shuffle performance

**Deliverables:**
- `src/ops/partition.rs` - Hash partitioning
- `src/ops/shuffle.rs` - Shuffle implementation
- `tests/shuffle_test.rs`
- Performance analysis document

**Test Cases:**
- Shuffle by single column
- Shuffle by multiple columns
- Shuffle with different data types
- Shuffle with null values
- Shuffle with large tables (1M+ rows)

### Phase 5: Distributed Join (Week 5-6)
**Goal:** Distributed hash join operation

**Tasks:**
1. Add `distributed_join` to `src/ops/join.rs`
2. Implement for all join types (inner, left, right, outer)
3. Test correctness against local join
4. Test with different data distributions
5. Performance benchmarking

**Deliverables:**
- Distributed join functions in `src/ops/join.rs`
- `tests/distributed_join_test.rs`
- Correctness validation (compare with local join on gathered data)
- Performance benchmarks (speedup vs local)

**Test Cases:**
- Distributed inner join
- Distributed left/right/outer join
- Join with multiple key columns
- Join with skewed data distribution
- Join with different table sizes

### Phase 6: Distributed Set Operations (Week 7)
**Goal:** Union, intersection, difference operations

**Tasks:**
1. Implement `distributed_union`
2. Implement `distributed_intersect`
3. Implement `distributed_difference`
4. Implement `distributed_subtract`
5. Write tests for each operation

**Deliverables:**
- Distributed set ops in `src/ops/set_ops.rs`
- `tests/distributed_set_ops_test.rs`
- Correctness validation

### Phase 7: Distributed Sort (Week 8)
**Goal:** Global distributed sort using sample sort

**Tasks:**
1. Research sample sort algorithm
2. Implement local sort first
3. Implement splitter selection
4. Implement distributed sample sort
5. Test and benchmark

**Deliverables:**
- `src/ops/sort.rs` - Distributed sort
- `tests/distributed_sort_test.rs`
- Performance comparison with local sort

### Phase 8: Gather/Broadcast (Week 9)
**Goal:** Data collection and distribution primitives

**Tasks:**
1. Implement `gather_table`
2. Implement `broadcast_table`
3. Implement `allgather_table`
4. Write tests

**Deliverables:**
- `src/ops/gather.rs`
- `tests/gather_test.rs`

### Phase 9: Integration and Testing (Week 10)
**Goal:** End-to-end integration and comprehensive testing

**Tasks:**
1. Write integration tests combining multiple operations
2. Test with real-world data and workloads
3. Performance profiling and optimization
4. Documentation and examples
5. Compare performance with C++ implementation

**Deliverables:**
- `examples/distributed_word_count.rs`
- `examples/distributed_join_example.rs`
- Performance comparison report
- User documentation for distributed operations

## Testing Strategy

### Unit Tests
- Test each component in isolation
- Mock communicator for single-process testing
- Test edge cases (empty tables, single row, etc.)

### Integration Tests
- Require MPI environment
- Run with `mpirun -n 4 cargo test --test distributed_*`
- Test realistic workflows

### Correctness Validation
For each distributed operation:
1. Run distributed version with N processes
2. Gather result to root
3. Compare with local version on gathered input
4. Assert exact equality (deterministic operations)

### Performance Benchmarks
- Measure speedup vs local operations
- Weak scaling: fixed data per process
- Strong scaling: fixed total data, vary processes
- Compare with C++ Cylon performance

### Test Data
- Small tables (100s of rows) for quick tests
- Medium tables (10K-100K rows) for integration tests
- Large tables (1M+ rows) for performance tests
- Various data types and distributions

## Directory Structure

```
rust/
├── src/
│   ├── net/
│   │   ├── mod.rs           # Module exports
│   │   ├── communicator.rs  # Communicator trait
│   │   ├── serialize.rs     # Table serialization
│   │   └── mpi/
│   │       └── mod.rs       # MPI implementation
│   ├── ops/
│   │   ├── partition.rs     # Hash partitioning
│   │   ├── shuffle.rs       # Shuffle operation
│   │   ├── sort.rs          # Distributed sort
│   │   └── gather.rs        # Gather/broadcast
│   └── ...
├── tests/
│   ├── mpi_basic_test.rs
│   ├── serialize_test.rs
│   ├── collective_ops_test.rs
│   ├── shuffle_test.rs
│   ├── distributed_join_test.rs
│   ├── distributed_set_ops_test.rs
│   ├── distributed_sort_test.rs
│   └── gather_test.rs
├── examples/
│   ├── distributed_word_count.rs
│   └── distributed_join_example.rs
└── benches/
    └── distributed_ops.rs
```

## Dependencies

### Required Crates
```toml
[dependencies]
mpi = "0.7"              # rsmpi - Rust MPI bindings
arrow = "53.3.0"         # Already present
serde = "1.0"            # For metadata serialization
bincode = "1.3"          # For compact binary serialization

[dev-dependencies]
criterion = "0.5"        # Benchmarking
```

### System Requirements
- MPI implementation: OpenMPI 4.0+ or MPICH 3.3+
- Rust 1.70+
- CMake (for building rsmpi)

### Installation
```bash
# Ubuntu/Debian
sudo apt-get install libopenmpi-dev openmpi-bin

# macOS
brew install open-mpi

# Set environment for rsmpi
export OMPI_CC=gcc
export OMPI_CXX=g++
```

## Key Design Decisions

### 1. MPI Backend Only (Initially)
**Decision:** Start with MPI only, not gRPC/UCX
**Rationale:**
- C++ Cylon's distributed ops primarily use MPI
- MPI is standard for HPC environments
- rsmpi provides good Rust bindings
- Can add other backends later

### 2. Arrow IPC for Serialization
**Decision:** Use Arrow IPC format for table serialization
**Rationale:**
- Zero-copy serialization
- Maintains columnar format
- Native Arrow support
- Efficient for large tables

### 3. Hash-Based Partitioning
**Decision:** Use hash partitioning for shuffle
**Rationale:**
- Matches C++ implementation
- Even distribution for uniform data
- Simple and efficient
- Works well for joins and aggregations

### 4. Sample Sort for Global Sort
**Decision:** Use sample sort algorithm
**Rationale:**
- Standard distributed sorting algorithm
- Good load balancing
- Matches C++ implementation
- Efficient for large datasets

### 5. Blocking Communication (Initially)
**Decision:** Use blocking MPI calls initially
**Rationale:**
- Simpler to implement and debug
- Matches C++ synchronous API
- Can optimize with non-blocking later
- Sufficient for initial implementation

## Performance Considerations

### Optimization Opportunities
1. **Zero-copy**: Use Arrow IPC for serialization
2. **Overlap communication**: Non-blocking MPI calls
3. **Compression**: Optional compression for large tables
4. **Pipelining**: Process data as it arrives
5. **SIMD**: Use Arrow compute kernels

### Expected Performance
- **Speedup**: Near-linear for CPU-bound operations (join, sort)
- **Overhead**: 10-20% for communication and serialization
- **Scalability**: Up to 100s of processes for large datasets

### Benchmarking Metrics
- Throughput (rows/second)
- Latency (time to first result)
- Speedup (T_1 / T_n)
- Weak scaling efficiency
- Strong scaling efficiency

## Risks and Mitigation

### Risk 1: MPI Environment Complexity
**Issue:** Setting up MPI for testing is complex
**Mitigation:**
- Document setup clearly
- Provide Docker container with MPI
- CI/CD with MPI pre-installed

### Risk 2: Debugging Distributed Code
**Issue:** Debugging multi-process code is hard
**Mitigation:**
- Comprehensive logging
- Single-process test mode
- Deterministic test data
- Correctness validation against local ops

### Risk 3: Performance Gap with C++
**Issue:** Rust implementation might be slower
**Mitigation:**
- Profile and optimize hot paths
- Use Arrow compute kernels
- Benchmark incrementally
- Accept initial performance gap

### Risk 4: Data Skew
**Issue:** Hash partitioning can create imbalanced loads
**Mitigation:**
- Test with skewed data
- Document limitations
- Future: implement range partitioning

## Success Criteria

### Functional Requirements
- [ ] All distributed operations match C++ semantics
- [ ] All operations produce correct results
- [ ] Support all join types and set operations
- [ ] Handle edge cases (empty tables, nulls, etc.)

### Performance Requirements
- [ ] Near-linear speedup for CPU-bound operations
- [ ] <20% overhead vs C++ implementation
- [ ] Support 100M+ row tables
- [ ] Scale to 100+ processes

### Quality Requirements
- [ ] 90%+ test coverage for distributed code
- [ ] All tests pass with 1-16 processes
- [ ] Comprehensive documentation
- [ ] Example programs demonstrating usage

## Future Enhancements

### Phase 10+: Advanced Features
1. **Non-blocking Communication**: Overlap compute and communication
2. **Compression**: Optional compression for large tables
3. **UCX Backend**: Alternative to MPI for RDMA
4. **Distributed Aggregations**: GroupBy across processes
5. **Fault Tolerance**: Checkpointing and recovery
6. **Dynamic Load Balancing**: Adaptive partitioning
7. **Range Partitioning**: Better for sorted data
8. **Distributed I/O**: Parallel CSV/Parquet reading

## References

### C++ Cylon Code
- `cpp/src/cylon/table.cpp` - Main distributed operation implementations
- `cpp/src/cylon/net/` - Communication layer
- `cpp/src/cylon/ctx/cylon_context.hpp` - Distributed context

### Documentation
- [MPI Standard](https://www.mpi-forum.org/docs/)
- [rsmpi Documentation](https://docs.rs/mpi/)
- [Arrow IPC Format](https://arrow.apache.org/docs/format/Columnar.html#ipc-streaming-format)
- [Sample Sort Algorithm](https://en.wikipedia.org/wiki/Samplesort)

### Papers
- "Twisterx: Distributed data processing framework" - Original Cylon paper
- "Sample Sort on Apache Spark" - Distributed sorting techniques

---

## Getting Started

Once Phase 1 is complete, developers can start using distributed operations:

```rust
use cylon::ctx::CylonContext;
use cylon::table::Table;
use cylon::ops::join::distributed_join;
use std::sync::Arc;

fn main() {
    // Initialize MPI context
    let ctx = Arc::new(CylonContext::distributed().unwrap());

    let rank = ctx.get_rank();
    let world_size = ctx.get_world_size();

    println!("Process {}/{}", rank, world_size);

    // Load local partition of data
    let table1 = Table::from_csv(&ctx, "data_part.csv", None).unwrap();
    let table2 = Table::from_csv(&ctx, "other_part.csv", None).unwrap();

    // Distributed join
    let result = distributed_join(
        &ctx,
        &table1,
        &table2,
        JoinType::Inner,
        &[0],  // left key column
        &[0],  // right key column
    ).unwrap();

    println!("Rank {} has {} result rows", rank, result.rows());
}
```

Run with:
```bash
mpirun -n 4 cargo run --release --example distributed_join
```

---

**Document Version:** 1.0
**Last Updated:** 2025-10-24
**Status:** Ready for implementation
