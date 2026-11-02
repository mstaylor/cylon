#!/bin/bash
##
# Launch N ranks of fmi_async_exchange_test over the direct-redis channel, each
# as its own OS process on this host. The last rank is the slow one.
#
# Usage:   run_fmi_async_exchange.sh <world_size> [rounds] [fast_ms] [slow_ms]
# Env:     REDIS_HOST (default 10.211.55.2), REDIS_PORT (default 6379),
#          PORT_BASE (default 18900), BIN (default ./fmi_async_exchange_test)
##
set -u

WS="${1:-4}"
ROUNDS="${2:-5}"
FAST_MS="${3:-200}"
SLOW_MS="${4:-2000}"
REDIS_HOST="${REDIS_HOST:-10.211.55.2}"
REDIS_PORT="${REDIS_PORT:-6379}"
PORT_BASE="${PORT_BASE:-18900}"
BIN="${BIN:-./fmi_async_exchange_test}"
COMM="fmiasync_${WS}_$(date +%s%N)_$$"
SLOW_RANK=$((WS - 1))

pids=()
for ((r = 0; r < WS; r++)); do
  "$BIN" "$r" "$WS" "$COMM" "$REDIS_HOST" "$REDIS_PORT" "$PORT_BASE" \
    "$ROUNDS" "$FAST_MS" "$SLOW_MS" "$SLOW_RANK" &
  pids+=($!)
done

rc=0
for p in "${pids[@]}"; do
  wait "$p" || rc=1
done

if [ "$rc" -eq 0 ]; then
  echo "fmi_async_exchange_test: all $WS ranks passed"
else
  echo "fmi_async_exchange_test: FAILED"
fi
exit "$rc"