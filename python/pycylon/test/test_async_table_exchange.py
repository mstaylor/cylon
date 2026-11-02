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
Run test:
>> REDIS_HOST=<host> pytest -q python/pycylon/test/test_async_table_exchange.py

The multi-rank cases launch one process per rank over the FMI direct-redis
channel and are skipped when REDIS_HOST is unset.
'''

import json
import os
import socket
import subprocess
import sys
import uuid

import pyarrow as pa
import pytest

from pycylon import CylonContext
from pycylon.net.async_table_exchange import AsyncTableExchange

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "async_table_exchange_worker.py")

SCHEMA = pa.schema([pa.field("a", pa.int32())])

needs_redis = pytest.mark.skipif(not os.environ.get("REDIS_HOST"), reason="needs REDIS_HOST")


def free_ports(count):
    sockets = [socket.socket() for _ in range(count)]
    for s in sockets:
        s.bind(("127.0.0.1", 0))
    ports = [s.getsockname()[1] for s in sockets]
    for s in sockets:
        s.close()
    return ports


def run_world(world_size, rounds, fast_ms, slow_ms, slow_rank, finish_timeout_s=60.0,
              slow_finish_delay_s=0.0):
    comm_name = f"async_exchange_{uuid.uuid4().hex[:12]}"
    redis_host = os.environ["REDIS_HOST"]
    redis_port = os.environ.get("REDIS_PORT", "6379")
    procs = [
        subprocess.Popen(
            [sys.executable, WORKER, str(world_size), comm_name, redis_host, redis_port,
             str(port), str(rounds), str(fast_ms), str(slow_ms), str(slow_rank),
             str(finish_timeout_s), str(slow_finish_delay_s)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for port in free_ports(world_size)
    ]
    results = []
    for proc in procs:
        out, err = proc.communicate(timeout=180)
        lines = [line for line in out.splitlines() if line.startswith("{")]
        assert lines, f"worker exited {proc.returncode} without a result\n{err[-2000:]}"
        results.append(json.loads(lines[-1]))
    return results


def test_single_rank_has_no_peers_to_exchange_with():
    exchange = AsyncTableExchange(CylonContext(config=None, distributed=False), SCHEMA)
    exchange.publish(pa.table({"a": pa.array([1], pa.int32())}), 0)
    assert exchange.drain() == []
    assert exchange.idle()
    assert exchange.finish_and_drain(1.0) == []


def test_publish_rejects_a_table_with_another_schema():
    exchange = AsyncTableExchange(CylonContext(config=None, distributed=False), SCHEMA)
    with pytest.raises(ValueError):
        exchange.publish(pa.table({"b": pa.array([1.0], pa.float64())}), 0)


@needs_redis
@pytest.mark.parametrize("world_size", [2, 4])
def test_peers_receive_every_table_once_without_waiting_for_the_slow_rank(world_size):
    results = run_world(world_size, rounds=4, fast_ms=150, slow_ms=1500,
                        slow_rank=world_size - 1)

    assert sorted(r["rank"] for r in results) == list(range(world_size))
    assert [r["failures"] for r in results] == [[]] * world_size
    assert not any("finish_error" in r for r in results)


@needs_redis
def test_a_burst_beyond_the_channel_queue_limit_still_delivers_every_table():
    results = run_world(2, rounds=300, fast_ms=0, slow_ms=0, slow_rank=-1)

    assert [r["failures"] for r in results] == [[], []]
    assert not any("finish_error" in r for r in results)


@needs_redis
def test_finish_times_out_naming_the_peer_that_has_not_finished():
    results = run_world(2, rounds=1, fast_ms=0, slow_ms=0, slow_rank=1, finish_timeout_s=1.0,
                        slow_finish_delay_s=4.0)

    fast = next(r for r in results if r["rank"] == 0)
    assert "finish_error" in fast
    assert "1" in fast["finish_error"].split("waiting for")[-1]
