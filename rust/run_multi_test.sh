#!/bin/bash
set -e

export LIBFABRIC_INSTALL_PREFIX=/home/parallels/libfabric/install
export LIBFABRIC_LIBDIR=/home/parallels/libfabric/install/lib
export LD_LIBRARY_PATH=/home/parallels/libfabric/install/lib:$LD_LIBRARY_PATH
export CYLON_REDIS_URL="redis://10.211.55.2:6379"

SESSION_ID="multi-$(date +%s)"
export CYLON_SESSION_ID="$SESSION_ID"

echo "========================================"
echo "Starting 2-process libfabric test"
echo "Session ID: $SESSION_ID"
echo "========================================"

# Run rank 0 in background
echo "[$(date +%T)] Starting Rank 0..."
CYLON_RANK=0 cargo test --features libfabric --test libfabric_communicator_test test_libfabric_communicator_init_with_redis -- --include-ignored --nocapture > /tmp/rank0.log 2>&1 &
PID0=$!

# Wait for rank 0 to register
sleep 3

# Run rank 1
echo "[$(date +%T)] Starting Rank 1..."
CYLON_RANK=1 cargo test --features libfabric --test libfabric_communicator_test test_libfabric_communicator_init_with_redis -- --include-ignored --nocapture > /tmp/rank1.log 2>&1 &
PID1=$!

echo "[$(date +%T)] Waiting for processes (PIDs: $PID0, $PID1)..."

wait $PID0
RC0=$?
echo "[$(date +%T)] Rank 0 exited with code: $RC0"

wait $PID1
RC1=$?
echo "[$(date +%T)] Rank 1 exited with code: $RC1"

echo ""
echo "========================================"
echo "RANK 0 OUTPUT (relevant lines):"
echo "========================================"
grep -E "(running|test |Creating|Rank|world_size|initialized|PASSED|FAILED|ok|error|Error|Failed)" /tmp/rank0.log || echo "(no matches)"

echo ""
echo "========================================"
echo "RANK 1 OUTPUT (relevant lines):"
echo "========================================"
grep -E "(running|test |Creating|Rank|world_size|initialized|PASSED|FAILED|ok|error|Error|Failed)" /tmp/rank1.log || echo "(no matches)"

echo ""
echo "========================================"
echo "Exit codes: Rank0=$RC0, Rank1=$RC1"
if [ $RC0 -eq 0 ] && [ $RC1 -eq 0 ]; then
    echo "MULTI-PROCESS TEST PASSED"
else
    echo "MULTI-PROCESS TEST FAILED"
    echo ""
    echo "Full Rank 0 log:"
    tail -50 /tmp/rank0.log
    echo ""
    echo "Full Rank 1 log:"
    tail -50 /tmp/rank1.log
fi
echo "========================================"
