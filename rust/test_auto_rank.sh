#!/bin/bash
set -e

export LIBFABRIC_INSTALL_PREFIX=/home/parallels/libfabric/install
export LIBFABRIC_INCLUDEDIR=/home/parallels/libfabric/install/include
export LIBFABRIC_LIBDIR=/home/parallels/libfabric/install/lib
export LD_LIBRARY_PATH=/home/parallels/libfabric/install/lib:$LD_LIBRARY_PATH
export CYLON_REDIS_URL="redis://10.211.55.2:6379"

SESSION_ID="auto-rank-$(date +%s)"
export CYLON_SESSION_ID="$SESSION_ID"

echo "========================================"
echo "Testing AUTO-RANK assignment"
echo "(NO CYLON_RANK environment variable set)"
echo "Session ID: $SESSION_ID"
echo "========================================"

# Unset CYLON_RANK to prove it's not needed
unset CYLON_RANK

# Run two processes
cargo test --features libfabric --test libfabric_communicator_test test_libfabric_communicator_init_with_redis -- --include-ignored --nocapture > /tmp/auto_p1.log 2>&1 &
PID1=$!

sleep 2

cargo test --features libfabric --test libfabric_communicator_init_with_redis -- --include-ignored --nocapture > /tmp/auto_p2.log 2>&1 &
PID2=$!

wait $PID1
RC1=$?
wait $PID2
RC2=$?

echo ""
echo "=== Process 1 (started first) ==="
grep -E "(Rank:|test .*ok|test .*FAILED)" /tmp/auto_p1.log || echo "No matches"

echo ""
echo "=== Process 2 (started second) ==="
grep -E "(Rank:|test .*ok|test .*FAILED)" /tmp/auto_p2.log || echo "No matches"

echo ""
echo "Exit codes: P1=$RC1, P2=$RC2"
if [ $RC1 -eq 0 ] && [ $RC2 -eq 0 ]; then
    echo "AUTO-RANK TEST PASSED - ranks assigned automatically!"
else
    echo "TEST FAILED"
fi
