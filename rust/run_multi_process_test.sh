#!/bin/bash
set -e

export LIBFABRIC_INSTALL_PREFIX=/home/parallels/libfabric/install
export LIBFABRIC_LIBDIR=/home/parallels/libfabric/install/lib
export LD_LIBRARY_PATH=/home/parallels/libfabric/install/lib:$LD_LIBRARY_PATH
export CYLON_REDIS_URL="redis://10.211.55.2:6379"

SESSION_ID="multi-$(date +%s)"
export CYLON_SESSION_ID="$SESSION_ID"

echo "========================================"
echo "Starting 2-process test"
echo "Session ID: $SESSION_ID"
echo "========================================"

# Run rank 0 in background
echo "Starting Rank 0..."
CYLON_RANK=0 cargo test --features libfabric test_libfabric_communicator_init_with_redis -- --ignored --nocapture 2>&1 > /tmp/rank0.log &
PID0=$!

sleep 3

# Run rank 1 in background
echo "Starting Rank 1..."
CYLON_RANK=1 cargo test --features libfabric test_libfabric_communicator_init_with_redis -- --ignored --nocapture 2>&1 > /tmp/rank1.log &
PID1=$!

echo "Waiting for both processes (PIDs: $PID0, $PID1)..."
wait $PID0
RC0=$?
wait $PID1
RC1=$?

echo "========================================"
echo "RANK 0 OUTPUT:"
echo "========================================"
cat /tmp/rank0.log | grep -E "(test |Running|Creating|Rank|world_size|initialized|PASSED|FAILED|ok|error|Error)"

echo ""
echo "========================================"
echo "RANK 1 OUTPUT:"
echo "========================================"
cat /tmp/rank1.log | grep -E "(test |Running|Creating|Rank|world_size|initialized|PASSED|FAILED|ok|error|Error)"

echo ""
echo "========================================"
echo "Exit codes: Rank0=$RC0, Rank1=$RC1"
echo "========================================"
