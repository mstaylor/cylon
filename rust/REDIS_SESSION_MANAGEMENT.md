# Redis Session Management for UCX/UCC

## Problem

When running UCX/UCC programs that use Redis for out-of-band communication, **running multiple times without clearing Redis causes segfaults**. This happens because:

1. Old UCX worker addresses remain in Redis from previous runs
2. New processes try to use these stale addresses
3. UCX attempts to connect to invalid memory addresses → **SEGFAULT**

## Solution: Session ID

All processes in a run must use the same `CYLON_SESSION_ID` to namespace their Redis keys. This isolates each run from stale data.

## Usage

### Option 1: Using the Helper Script (Recommended)

```bash
# The script automatically generates a unique session ID
./run_with_session.sh mpirun -n 4 ./my_program

# For tests
./run_with_session.sh cargo test --features ucx -- --ignored
```

### Option 2: Manual Session ID

```bash
# Generate a unique session ID
export CYLON_SESSION_ID=$(uuidgen)  # or use timestamp, process ID, etc.
export CYLON_REDIS_URL=redis://localhost:6379

# Run your program - all processes will use the same session ID
mpirun -n 4 ./my_program

# Run tests
cargo test --features ucx test_ucx_communicator_init_with_redis -- --ignored
```

### Option 3: In Your Launcher Script

```python
# Python launcher example
import uuid
import os
import subprocess

session_id = str(uuid.uuid4())
os.environ['CYLON_SESSION_ID'] = session_id
os.environ['CYLON_REDIS_URL'] = 'redis://localhost:6379'

# Launch all processes with same environment
subprocess.run(['mpirun', '-n', '4', './my_program'])
```

## Redis Key Structure

All Redis keys are prefixed with the session ID:

```
{session_id}:num_cur_processes       # Rank assignment counter
{session_id}:ucp_worker_addr_mp      # UCX worker addresses (hash)
{session_id}:ucx_helper{rank}        # UCX synchronization queues
{session_id}:ucc_oob_mp{n}           # UCC collective data
{session_id}:ucc_helper{n}:{rank}    # UCC synchronization queues
```

## Benefits

✓ **No segfaults** - Old addresses don't interfere with new runs
✓ **Concurrent runs** - Multiple jobs can use same Redis instance
✓ **No manual cleanup** - Each run gets isolated namespace
✓ **Explicit control** - Launcher decides session lifetime

## Cleanup (Optional)

Session data remains in Redis after the job completes. To clean up:

```bash
# Clean up a specific session
redis-cli --scan --pattern "session_123:*" | xargs redis-cli DEL

# Clean up all Cylon sessions
redis-cli --scan --pattern "*:num_cur_processes" | sed 's/:num_cur_processes//' | xargs -I {} redis-cli --scan --pattern "{}:*" | xargs redis-cli DEL
```

Or set Redis key expiration (requires code modification).

## Environment Variables

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `CYLON_SESSION_ID` | **YES** | Unique session identifier | `e4b2c9d8-1234-5678-90ab-cdef12345678` |
| `CYLON_REDIS_URL` | No | Redis server address | `redis://localhost:6379` (default) |

## Error Messages

If you forget to set `CYLON_SESSION_ID`, you'll see:

```
Error: CYLON_SESSION_ID environment variable not set.
The launcher must set this to prevent conflicts with stale Redis data.
Example: export CYLON_SESSION_ID=$(uuidgen)
```

## Implementation Details

- Session ID is read during `UCXRedisOOBContext::new()` and `UCCRedisOOBContext::new()`
- All Redis operations use prefixed keys
- First process to call `get_world_size_and_rank()` starts rank counter at 0
- Each session has independent rank assignment
