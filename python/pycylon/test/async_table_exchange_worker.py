##
 # Licensed under the Apache License, Version 2.0 (the "License");
 # you may not use this file except in compliance with the License.
 # You may obtain a copy of the License at
 #
 # http://www.apache.org/licenses/LICENSE-2.0
 #
 # Unless required by applicable law or agreed to in writing, software
 # distributed under the License is distributed on an "AS IS" BASIS,
 # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 # See the License for the specific language governing permissions and
 # limitations under the License.
 ##

'''
One rank of test_async_table_exchange.py, launched as its own process.

Usage: async_table_exchange_worker.py <world_size> <comm_name> <redis_host> <redis_port>
       <listen_port> <rounds> <fast_ms> <slow_ms> <slow_rank> <finish_timeout_s>
       <slow_finish_delay_s>
A slow_rank of -1 makes every rank fast and skips the early-arrival check. The slow
rank waits slow_finish_delay_s before it finishes. Prints one JSON line with this
rank's failures and, when finishing raised, the error.
'''

import json
import sys
import time

import pyarrow as pa

from pycylon.frame import CylonEnv
from pycylon.net.async_table_exchange import AsyncTableExchange
from pycylon.net.fmi_config import FMIConfig

EMBEDDING_WIDTH = 4

SCHEMA = pa.schema([
    pa.field("src", pa.int32()),
    pa.field("round", pa.int32()),
    pa.field("text", pa.string()),
    pa.field("flag", pa.bool_()),
    pa.field("embedding", pa.list_(pa.float32(), EMBEDDING_WIDTH)),
])


def round_table(src, rnd):
    rows = rnd % 5
    return pa.table({
        "src": pa.array([src] * rows, pa.int32()),
        "round": pa.array([rnd] * rows, pa.int32()),
        "text": pa.array([f"r{src}-{rnd}-{i}" * (i + 1) for i in range(rows)], pa.string()),
        "flag": pa.array([(src + rnd + i) % 3 == 0 for i in range(rows)], pa.bool_()),
        "embedding": pa.array(
            [[float(src * 100 + rnd * 10 + i + k / 10) for k in range(EMBEDDING_WIDTH)]
             for i in range(rows)],
            pa.list_(pa.float32(), EMBEDDING_WIDTH)),
    }, schema=SCHEMA)


def main(argv):
    world_size, comm_name, redis_host = int(argv[0]), argv[1], argv[2]
    redis_port, listen_port, rounds = int(argv[3]), int(argv[4]), int(argv[5])
    fast_ms, slow_ms, slow_rank = int(argv[6]), int(argv[7]), int(argv[8])
    finish_timeout_s, slow_finish_delay_s = float(argv[9]), float(argv[10])

    config = FMIConfig(0, world_size, "", listen_port, 60000, True, comm_name, True,
                       redis_host, redis_port, comm_name, channel_type="direct-redis",
                       advertise_host="127.0.0.1")
    env = CylonEnv(config=config, distributed=True)
    ctx = env.context
    rank = ctx.get_rank()
    ctx.barrier()

    failures = []
    arrivals = {}
    t0 = time.monotonic()

    def collect(received):
        for source, reference, table in received:
            key = f"{source}:{reference}"
            arrivals[key] = arrivals.get(key, 0) + 1
            arrivals.setdefault(f"ms:{key}", (time.monotonic() - t0) * 1000)
            if not table.equals(round_table(source, reference)):
                failures.append(f"table from {source} round {reference} differs")

    exchange = AsyncTableExchange(ctx, SCHEMA)
    work_ms = slow_ms if rank == slow_rank else fast_ms
    for rnd in range(rounds):
        work_end = time.monotonic() + work_ms / 1000
        while time.monotonic() < work_end:
            collect(exchange.drain())
        exchange.publish(round_table(rank, rnd), rnd)
    settle_end = time.monotonic() + 5
    while not exchange.idle() and time.monotonic() < settle_end:
        collect(exchange.drain())
    if not exchange.idle():
        failures.append("never idle after its last publish")
    if rank == slow_rank:
        time.sleep(slow_finish_delay_s)
    try:
        collect(exchange.finish_and_drain(finish_timeout_s))
    except RuntimeError as exc:
        print(json.dumps({"rank": rank, "failures": failures, "finish_error": str(exc)}),
              flush=True)
        return 1

    peers = [p for p in range(world_size) if p != rank]
    for peer in peers:
        for rnd in range(rounds):
            if arrivals.get(f"{peer}:{rnd}") != 1:
                failures.append(f"table from {peer} round {rnd} arrived "
                                f"{arrivals.get(f'{peer}:{rnd}', 0)} times")
    if slow_rank >= 0 and rank != slow_rank:
        for peer in peers:
            if peer != slow_rank and arrivals.get(f"ms:{peer}:0", slow_ms) >= slow_ms:
                failures.append(f"round 0 from fast rank {peer} did not arrive while "
                                f"the slow rank was still working")

    env.finalize()
    print(json.dumps({"rank": rank, "failures": failures}), flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
