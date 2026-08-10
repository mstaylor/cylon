#!/bin/bash
##
# Launch N ranks of fmi_collectives_test over the redis channel, each as its own
# OS process (redis coordinates ranks). comm_name is unique per run so concurrent
# runs never collide on redis keys.
#
# Usage:   run_fmi_collectives.sh <world_size>
# Env:     REDIS_HOST (default 10.211.55.2), REDIS_PORT (default 6379),
#          BIN (path to fmi_collectives_test; default ./fmi_collectives_test)
##
set -u

WS="${1:-4}"
REDIS_HOST="${REDIS_HOST:-10.211.55.2}"
REDIS_PORT="${REDIS_PORT:-6379}"
BIN="${BIN:-./fmi_collectives_test}"
COMM="fmicoll_${WS}_$(date +%s%N)_$$"

pids=()
for ((r = 0; r < WS; r++)); do
  "$BIN" "$r" "$WS" "$COMM" "$REDIS_HOST" "$REDIS_PORT" &
  pids+=($!)
done

rc=0
for p in "${pids[@]}"; do
  wait "$p" || rc=1
done

if [ "$rc" -eq 0 ]; then
  echo "PASS ws=$WS comm=$COMM"
else
  echo "FAIL ws=$WS comm=$COMM"
fi
exit "$rc"