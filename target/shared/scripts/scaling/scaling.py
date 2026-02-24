import time
import argparse


import pandas as pd
from numpy.random import default_rng

import mpi4py
mpi4py.rc.initialize = False
mpi4py.rc.finalize = False

from cloudmesh.common.StopWatch import StopWatch
from cloudmesh.common.dotdict import dotdict
from cloudmesh.common.Shell import Shell
from cloudmesh.common.util import writefile

import boto3
from botocore.exceptions import ClientError

import os
import requests
import json

import logging
import socket
import time
import numpy as np

# Cost tracking module
from costlib import AWSPricing, CostTracker

def upload_file(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True

def environ_or_required(key, default = None, required=True):

    if default is None:
        return (
            {'default': os.environ.get(key)} if os.environ.get(key)
            else {'required': required}

        )
    else:
        return (
            {'default': os.environ.get(key)} if os.environ.get(key)
            else {'default': default}

        )

def get_service_ips(cluster, tasks):
    client = boto3.client("ecs", region_name="us-east-1")

    tasks_detail = client.describe_tasks(
        cluster=cluster,
        tasks=tasks
    )

    # first get the ENIs
    enis = []
    for task in tasks_detail.get("tasks", []):
        for attachment in task.get("attachments", []):
            enis.extend(
                detail.get("value")
                for detail in attachment.get("details", [])
                if detail.get("name") == "networkInterfaceId"
            )
    # now the ips

    print("eni: ", enis)
    ips = []
    for eni in enis:
        eni_resource = boto3.resource("ec2").NetworkInterface(eni)
        print("eni_resource", eni_resource)
        ips.append(eni_resource.private_ip_address)

    return ips

def get_ecs_task_arn_cluster(host):
    path = "/task"
    url = host + path
    headers = {"Content-Type": "application/json"}
    r = requests.get(url, headers=headers)
    print(f"r: {r}")
    d_r = json.loads(r.text)
    print(d_r)
    cluster = d_r["TaskARN"]
    taskArn = d_r["Cluster"]
    return {"TaskARN": cluster, "Cluster": taskArn}

def barrier(obj = None):
    return obj.barrier()


def floating_point_operations(size):
    # Generate two random arrays of the specified size
    array1 = np.random.random(size)
    array2 = np.random.random(size)

    # Perform floating-point operations
    result = (array1 * array2) + np.sin(array1) - np.log(array2 + 1e-8)
    return result

def floatingPointPerf(data = None, ipAddress = None):
    global ucc_config
    StopWatch.start(f"join_total_{data['env']}_{data['it']}")

    np.set_printoptions(suppress=True, precision=13)

    if ipAddress is not None:
        print("setting UCX_TCP_REMOTE_ADDRESS_OVERRIDE", ipAddress)
        os.environ['UCX_TCP_REMOTE_ADDRESS_OVERRIDE'] = ipAddress

    init_start = time.time()

    if data['env'] == 'fmi':
        communicator = fmi_communicator(data)
        rank = int(data["rank"])
        world_size = int(data["world_size"])
    else:
        communicator, env = cylon_communicator(data)
        rank = env.rank
        world_size = env.world_size

    init_end = time.time()

    com_init = (init_end - init_start) * 1000



    timing = {'rank': [], 'avg_t': [],
              'tot_l': [], 'time1': [],
              'time2': [], 'time3': [],
              'time4': [],'time5': [],
              'time6': [], 'time7': [],
              'elapsed_t': [], 'max_t': [], 'com_init_t': [], 'barrier_t': []}

    max_time = 0
    print("iterating over range")
    sizes = [10 ** i for i in range(1, 8)]
    for i in range(data['it']):

        barrier_start1 = time.time()

        if data['env'] == 'fmi':
            barrier(communicator)
        else:
            barrier(env)
        barrier_time1 = (time.time() - barrier_start1) * 1000
        StopWatch.start(f"join_{i}_{data['env']}_{data['it']}")
        t1 = time.time()

        times = []

        for size in sizes:
            start_time = time.time()
            floating_point_operations(size)
            end_time = time.time()
            elapsed_time = end_time - start_time
            times.append(elapsed_time)


        barrier_start2 = time.time()
        if data['env'] == 'fmi':
            barrier(communicator)
        else:
            barrier(env)
        t2 = time.time()
        barrier_time2 = (t2 - barrier_start2) * 1000
        t = (t2 - t1) * 1000

        if data['env'] == 'fmi':
            sum_t = communicator.allreduce(t, fmi.func(fmi.op.sum), fmi.types(fmi.datatypes.double))

            tot_l = len(communicator.allreduce(times, fmi.func(fmi.op.sum),
                                               fmi.types(fmi.datatypes.double_list, len(times))))

        else:
            sum_t = communicator.allreduce(t, ReduceOp.SUM)
            tot_l = communicator.allreduce(len(times), ReduceOp.SUM)

        if rank == 0:
            end_time = time.time()
            elapsed_time = (end_time - t2) * 1000
            avg_t = sum_t / world_size
            max_time = max(max_time, sum_t)
            print("### ", i, avg_t, tot_l, times[0], times[1], times[2],
                  times[3], times[4], times[5], times[6], elapsed_time, max_time,
                  com_init)

            timing['rank'].append(i)
            timing['avg_t'].append(avg_t)
            timing['tot_l'].append(tot_l)
            timing['time1'].append(times[0])
            timing['time2'].append(times[1])
            timing['time3'].append(times[2])
            timing['time4'].append(times[3])
            timing['time5'].append(times[4])
            timing['time6'].append(times[5])
            timing['time7'].append(times[6])
            timing['elapsed_t'].append(elapsed_time)
            timing['max_t'].append(max_time)
            timing['com_init_t'].append(com_init)
            timing['barrier_t'].append(barrier_time1 + barrier_time2)
            StopWatch.stop(f"join_{i}_{data['env']}_{data['it']}")

    StopWatch.stop(f"join_total_{data['env']}_{data['it']}")

    if rank == 0:
        StopWatch.benchmark(tag=str(data), filename=data['output_scaling_filename'])

        if data['env'] != 'rivanna':
            upload_file(file_name=data['output_scaling_filename'], bucket=data['s3_bucket'],
                        object_name=data['s3_stopwatch_object_name'])

        if os.path.exists(data['output_summary_filename']):
            os.remove(data['output_summary_filename'])
            # pd.DataFrame(timing).to_csv(data['output_summary_filename'], mode='a', index=False, header=False)
        # else:
        pd.DataFrame(timing).to_csv(data['output_summary_filename'], mode='w', index=False, header=True)

        if data['env'] != 'rivanna':
            upload_file(file_name=data['output_summary_filename'], bucket=data['s3_bucket'],
                        object_name=data['s3_summary_object_name'])

    if data['env'] != 'fmi':
        env.finalize()
def join(data=None, ipAddress = None):
    global ucc_config
    StopWatch.start(f"join_total_{data['env']}_{data['rows']}_{data['it']}")
    if ipAddress is not None:
        print("setting UCX_TCP_REMOTE_ADDRESS_OVERRIDE", ipAddress)
        os.environ['UCX_TCP_REMOTE_ADDRESS_OVERRIDE'] = ipAddress

    # Initialize cost tracker for serverless environments
    cost_tracker = None
    if data.get('enable_cost_tracking', True) and data['env'] in ('fmi', 'fmi-cylon'):
        cost_tracker = CostTracker(pricing_config=data.get('pricing_config'))
        cost_tracker.set_lambda_memory(data.get('lambda_memory_mb', 10240))
        cost_tracker.set_world_size(int(data['world_size']))

    init_start = time.time()

    if data['env'] == 'fmi':
        communicator = fmi_communicator(data)
        rank = int(data["rank"])
        world_size = int(data["world_size"])
    elif data['env'] == 'fmi-cylon':
        communicator, env = cylon_communicator(data)
        rank = env.rank
        world_size = int(data["world_size"])
    else:
        communicator, env = cylon_communicator(data)
        rank = env.rank
        world_size = env.world_size

    init_end = time.time()

    com_init = (init_end - init_start) * 1000

    u = data['unique']

    if data['scaling'] == 'w':  # weak
        num_rows = data['rows']
        max_val = num_rows * world_size
    else:  # 's' strong
        max_val = data['rows']
        num_rows = int(data['rows'] / world_size)

    # Track data generation time (C2: compute vs communication breakdown)
    data_gen_start = time.time()
    rng = default_rng(seed=rank)
    data1 = rng.integers(0, int(max_val * u), size=(num_rows, 2))
    data2 = rng.integers(0, int(max_val * u), size=(num_rows, 2))

    if data['env'] == 'fmi':
        df1 = pd.DataFrame(data1).add_prefix("col")
        df2 = pd.DataFrame(data2).add_prefix("col")
    else:
        df1 = DataFrame(pd.DataFrame(data1).add_prefix("col"))
        df2 = DataFrame(pd.DataFrame(data2).add_prefix("col"))
    data_gen_time = (time.time() - data_gen_start) * 1000

    timing = {'scaling': [], 'world': [], 'rows': [], 'max_value': [], 'rank': [], 'avg_t':[],
              'tot_l':[], 'elapsed_t': [], 'max_t': [], 'com_init_t': [], 'barrier_t': [],
              # Timing breakdown fields (C2: compute vs communication)
              'data_gen_t': [], 'compute_t': [], 'comm_t': [],
              # Cost tracking fields
              'lambda_memory_mb': [], 'lambda_duration_ms': [], 'lambda_gb_seconds': [],
              'lambda_cost_usd': [], 'step_fn_cost_usd': [], 'total_cost_usd': []}

    max_time = 0
    print("iterating over range")
    for i in range(data['it']):

        barrier_start1 = time.time()

        if data['env'] == 'fmi':
            barrier(communicator)
        else:
            barrier(env)
        #print("passed barrier")
        barrier_time1 = (time.time() - barrier_start1) * 1000
        StopWatch.start(f"join_{i}_{data['env']}_{data['rows']}_{data['it']}")
        t1 = time.time()

        # Track compute time (local join/merge operation)
        compute_start = time.time()
        if data['env'] == 'fmi':
            df3 = pd.concat([df1, df2], axis=1)
            result_array = df3.to_numpy().flatten().tolist()

        else:
            df3 = df1.merge(df2, on=[0], algorithm='sort', env=env)
        compute_time = (time.time() - compute_start) * 1000

        barrier_start2 = time.time()
        if data['env'] == 'fmi':
            barrier(communicator)
        else:
            barrier(env)
        t2 = time.time()
        barrier_time2 = (t2 - barrier_start2) * 1000
        t = (t2 - t1) * 1000

        # Track allreduce communication time
        allreduce_start = time.time()
        if data['env'] == 'fmi':
            sum_t = communicator.allreduce(t, fmi.func(fmi.op.sum), fmi.types(fmi.datatypes.double))
            tot_l = len(communicator.allreduce(result_array, fmi.func(fmi.op.sum),
                                           fmi.types(fmi.datatypes.int_list, len(result_array))))

        else:
            sum_t = communicator.allreduce(t, ReduceOp.SUM)
            tot_l = communicator.allreduce(len(df3), ReduceOp.SUM)
        allreduce_time = (time.time() - allreduce_start) * 1000

        # Total communication time = barriers + allreduce
        comm_time = barrier_time1 + barrier_time2 + allreduce_time

        if rank == 0:
            end_time = time.time()
            elapsed_time = (end_time - t2) * 1000
            avg_t = sum_t / world_size
            max_time = max(max_time, sum_t)
            print("### ", data['scaling'], world_size, num_rows, max_val, i, avg_t, tot_l, elapsed_time, max_time,com_init)
            timing['scaling'].append(data['scaling'])
            timing['world'].append(world_size)
            timing['rows'].append(num_rows)
            timing['max_value'].append(max_val)
            timing['rank'].append(i)
            timing['avg_t'].append(avg_t)
            timing['tot_l'].append(tot_l)
            timing['elapsed_t'].append(elapsed_time)
            timing['max_t'].append(max_time)
            timing['com_init_t'].append(com_init)
            timing['barrier_t'].append(barrier_time1 + barrier_time2)
            # Timing breakdown (C2: compute vs communication)
            timing['data_gen_t'].append(data_gen_time)
            timing['compute_t'].append(compute_time)
            timing['comm_t'].append(comm_time)

            # Cost tracking
            if cost_tracker:
                cost_tracker.record_duration(max_time)  # Use max time across all workers
                cost_summary = cost_tracker.get_summary_dict()
                timing['lambda_memory_mb'].append(cost_summary['lambda_memory_mb'])
                timing['lambda_duration_ms'].append(cost_summary['lambda_duration_ms'])
                timing['lambda_gb_seconds'].append(cost_summary['lambda_gb_seconds'])
                timing['lambda_cost_usd'].append(cost_summary['lambda_cost_usd'])
                timing['step_fn_cost_usd'].append(cost_summary['step_fn_cost_usd'])
                timing['total_cost_usd'].append(cost_summary['total_cost_usd'])
            else:
                timing['lambda_memory_mb'].append(0)
                timing['lambda_duration_ms'].append(0)
                timing['lambda_gb_seconds'].append(0)
                timing['lambda_cost_usd'].append(0)
                timing['step_fn_cost_usd'].append(0)
                timing['total_cost_usd'].append(0)

            StopWatch.stop(f"join_{i}_{data['env']}_{data['rows']}_{data['it']}")

    StopWatch.stop(f"join_total_{data['env']}_{data['rows']}_{data['it']}")

    if rank == 0:
        StopWatch.benchmark(tag=str(data), filename=data['output_scaling_filename'])

        if data['env'] != 'rivanna':
            upload_file(file_name=data['output_scaling_filename'], bucket=data['s3_bucket'],
                    object_name=data['s3_stopwatch_object_name'])
            if cost_tracker:
                cost_tracker.record_s3_put()

        if os.path.exists(data['output_summary_filename']):
            os.remove(data['output_summary_filename'])
        pd.DataFrame(timing).to_csv(data['output_summary_filename'], mode='w', index=False, header=True)

        if data['env'] != 'rivanna':
            upload_file(file_name=data['output_summary_filename'], bucket=data['s3_bucket'],
                    object_name=data['s3_summary_object_name'])
            if cost_tracker:
                cost_tracker.record_s3_put()

    if data['env'] != 'fmi':
        env.finalize()


def groupby_agg(data=None, ipAddress=None):
    """GroupBy with aggregation operation for scaling experiments"""
    global ucc_config
    StopWatch.start(f"groupby_total_{data['env']}_{data['rows']}_{data['it']}")
    if ipAddress is not None:
        os.environ['UCX_TCP_REMOTE_ADDRESS_OVERRIDE'] = ipAddress

    cost_tracker = None
    if data.get('enable_cost_tracking', True) and data['env'] in ('fmi', 'fmi-cylon'):
        cost_tracker = CostTracker(pricing_config=data.get('pricing_config'))
        cost_tracker.set_lambda_memory(data.get('lambda_memory_mb', 10240))
        cost_tracker.set_world_size(int(data['world_size']))

    init_start = time.time()

    if data['env'] == 'fmi':
        communicator = fmi_communicator(data)
        rank = int(data["rank"])
        world_size = int(data["world_size"])
    elif data['env'] == 'fmi-cylon':
        communicator, env = cylon_communicator(data)
        rank = env.rank
        world_size = int(data["world_size"])
    else:
        communicator, env = cylon_communicator(data)
        rank = env.rank
        world_size = env.world_size

    init_end = time.time()
    com_init = (init_end - init_start) * 1000

    u = data['unique']
    if data['scaling'] == 'w':
        num_rows = data['rows']
        max_val = num_rows * world_size
    else:
        max_val = data['rows']
        num_rows = int(data['rows'] / world_size)

    # Track data generation time (C2: compute vs communication breakdown)
    data_gen_start = time.time()
    rng = default_rng(seed=rank)
    num_groups = min(1000, int(max_val * u))
    data1 = np.column_stack([
        rng.integers(0, num_groups, size=num_rows),
        rng.integers(0, int(max_val * u), size=num_rows),
        rng.integers(0, int(max_val * u), size=num_rows),
    ])

    if data['env'] == 'fmi':
        df1 = pd.DataFrame(data1, columns=['key', 'val1', 'val2'])
    else:
        df1 = DataFrame(pd.DataFrame(data1, columns=['key', 'val1', 'val2']))
    data_gen_time = (time.time() - data_gen_start) * 1000

    timing = {'scaling': [], 'world': [], 'rows': [], 'max_value': [], 'rank': [], 'avg_t': [],
              'tot_l': [], 'elapsed_t': [], 'max_t': [], 'com_init_t': [], 'barrier_t': [],
              # Timing breakdown fields (C2: compute vs communication)
              'data_gen_t': [], 'compute_t': [], 'comm_t': [],
              'lambda_memory_mb': [], 'lambda_duration_ms': [], 'lambda_gb_seconds': [],
              'lambda_cost_usd': [], 'step_fn_cost_usd': [], 'total_cost_usd': []}

    max_time = 0
    print("iterating over range for groupby")
    for i in range(data['it']):
        barrier_start1 = time.time()
        if data['env'] == 'fmi':
            barrier(communicator)
        else:
            barrier(env)
        barrier_time1 = (time.time() - barrier_start1) * 1000

        StopWatch.start(f"groupby_{i}_{data['env']}_{data['rows']}_{data['it']}")
        t1 = time.time()

        # Track compute time (local groupby/aggregation operation)
        compute_start = time.time()
        if data['env'] == 'fmi':
            df3 = df1.groupby('key').agg({'val1': 'sum', 'val2': 'max'}).reset_index()
            result_len = len(df3)
        else:
            df3 = df1.groupby(by='key', env=env).agg({'val1': 'sum', 'val2': 'max'})
            result_len = len(df3)
        compute_time = (time.time() - compute_start) * 1000

        barrier_start2 = time.time()
        if data['env'] == 'fmi':
            barrier(communicator)
        else:
            barrier(env)
        t2 = time.time()
        barrier_time2 = (t2 - barrier_start2) * 1000
        t = (t2 - t1) * 1000

        # Track allreduce communication time
        allreduce_start = time.time()
        if data['env'] == 'fmi':
            sum_t = communicator.allreduce(t, fmi.func(fmi.op.sum), fmi.types(fmi.datatypes.double))
            tot_l = communicator.allreduce(result_len, fmi.func(fmi.op.sum), fmi.types(fmi.datatypes.int32))
        else:
            sum_t = communicator.allreduce(t, ReduceOp.SUM)
            tot_l = communicator.allreduce(result_len, ReduceOp.SUM)
        allreduce_time = (time.time() - allreduce_start) * 1000

        # Total communication time = barriers + allreduce
        comm_time = barrier_time1 + barrier_time2 + allreduce_time

        if rank == 0:
            end_time = time.time()
            elapsed_time = (end_time - t2) * 1000
            avg_t = sum_t / world_size
            max_time = max(max_time, sum_t)
            print("### groupby", data['scaling'], world_size, num_rows, max_val, i, avg_t, tot_l, elapsed_time, max_time, com_init)
            timing['scaling'].append(data['scaling'])
            timing['world'].append(world_size)
            timing['rows'].append(num_rows)
            timing['max_value'].append(max_val)
            timing['rank'].append(i)
            timing['avg_t'].append(avg_t)
            timing['tot_l'].append(tot_l)
            timing['elapsed_t'].append(elapsed_time)
            timing['max_t'].append(max_time)
            timing['com_init_t'].append(com_init)
            timing['barrier_t'].append(barrier_time1 + barrier_time2)
            # Timing breakdown (C2: compute vs communication)
            timing['data_gen_t'].append(data_gen_time)
            timing['compute_t'].append(compute_time)
            timing['comm_t'].append(comm_time)

            if cost_tracker:
                cost_tracker.record_duration(max_time)
                cost_summary = cost_tracker.get_summary_dict()
                timing['lambda_memory_mb'].append(cost_summary['lambda_memory_mb'])
                timing['lambda_duration_ms'].append(cost_summary['lambda_duration_ms'])
                timing['lambda_gb_seconds'].append(cost_summary['lambda_gb_seconds'])
                timing['lambda_cost_usd'].append(cost_summary['lambda_cost_usd'])
                timing['step_fn_cost_usd'].append(cost_summary['step_fn_cost_usd'])
                timing['total_cost_usd'].append(cost_summary['total_cost_usd'])
            else:
                for key in ['lambda_memory_mb', 'lambda_duration_ms', 'lambda_gb_seconds',
                            'lambda_cost_usd', 'step_fn_cost_usd', 'total_cost_usd']:
                    timing[key].append(0)

            StopWatch.stop(f"groupby_{i}_{data['env']}_{data['rows']}_{data['it']}")

    StopWatch.stop(f"groupby_total_{data['env']}_{data['rows']}_{data['it']}")

    if rank == 0:
        StopWatch.benchmark(tag=str(data), filename=data['output_scaling_filename'])
        if data['env'] != 'rivanna':
            upload_file(file_name=data['output_scaling_filename'], bucket=data['s3_bucket'],
                        object_name=data['s3_stopwatch_object_name'])
            if cost_tracker:
                cost_tracker.record_s3_put()

        if os.path.exists(data['output_summary_filename']):
            os.remove(data['output_summary_filename'])
        pd.DataFrame(timing).to_csv(data['output_summary_filename'], mode='w', index=False, header=True)

        if data['env'] != 'rivanna':
            upload_file(file_name=data['output_summary_filename'], bucket=data['s3_bucket'],
                        object_name=data['s3_summary_object_name'])
            if cost_tracker:
                cost_tracker.record_s3_put()

    if data['env'] != 'fmi':
        env.finalize()


def comm_microbenchmark(data=None, ipAddress=None):
    """
    Communication microbenchmark to measure pure collective operation latencies.

    Measures:
    - Barrier latency (synchronization overhead)
    - AllReduce latency at various message sizes
    - AllReduce bandwidth (derived from latency and message size)

    This addresses reviewer comments:
    - L2: Demonstrates MPI-style collectives work in serverless (not just data shuffle)
    - C2: Isolates communication latency from compute for breakdown analysis
    - Contribution (iv): Provides data for communication substrate comparison
    """
    global ucc_config
    StopWatch.start(f"microbenchmark_total_{data['env']}_{data['it']}")
    if ipAddress is not None:
        os.environ['UCX_TCP_REMOTE_ADDRESS_OVERRIDE'] = ipAddress

    cost_tracker = None
    if data.get('enable_cost_tracking', True) and data['env'] in ('fmi', 'fmi-cylon'):
        cost_tracker = CostTracker(pricing_config=data.get('pricing_config'))
        cost_tracker.set_lambda_memory(data.get('lambda_memory_mb', 10240))
        cost_tracker.set_world_size(int(data['world_size']))

    init_start = time.time()

    if data['env'] == 'fmi':
        communicator = fmi_communicator(data)
        rank = int(data["rank"])
        world_size = int(data["world_size"])
    elif data['env'] == 'fmi-cylon':
        communicator, env = cylon_communicator(data)
        rank = env.rank
        world_size = int(data["world_size"])
    else:
        communicator, env = cylon_communicator(data)
        rank = env.rank
        world_size = env.world_size

    init_end = time.time()
    com_init = (init_end - init_start) * 1000

    # Message sizes for AllReduce bandwidth measurement (bytes)
    # Powers of 2 from 8 bytes to 1MB
    message_sizes = [8, 64, 512, 4096, 32768, 262144, 1048576]

    timing = {
        'world': [], 'iteration': [], 'com_init_t': [],
        'barrier_latency_ms': [],
        'msg_size_bytes': [], 'allreduce_latency_ms': [], 'allreduce_bandwidth_mbps': [],
        'lambda_memory_mb': [], 'lambda_duration_ms': [], 'lambda_gb_seconds': [],
        'lambda_cost_usd': [], 'step_fn_cost_usd': [], 'total_cost_usd': []
    }

    print(f"Running communication microbenchmark with {data['it']} iterations")

    for i in range(data['it']):
        # Warmup barrier
        if data['env'] == 'fmi':
            barrier(communicator)
        else:
            barrier(env)

        # Measure barrier latency
        barrier_start = time.time()
        for _ in range(10):  # Average over 10 barriers
            if data['env'] == 'fmi':
                barrier(communicator)
            else:
                barrier(env)
        barrier_end = time.time()
        barrier_latency_ms = ((barrier_end - barrier_start) * 1000) / 10

        # Measure AllReduce latency at each message size
        for msg_size in message_sizes:
            # Create data buffer of appropriate size (as doubles)
            num_elements = max(1, msg_size // 8)  # 8 bytes per double
            send_data = [1.0] * num_elements

            # Warmup
            if data['env'] == 'fmi':
                _ = communicator.allreduce(send_data, fmi.func(fmi.op.sum),
                                           fmi.types(fmi.datatypes.double_list, num_elements))
            else:
                _ = communicator.allreduce(sum(send_data), ReduceOp.SUM)

            # Synchronize before timing
            if data['env'] == 'fmi':
                barrier(communicator)
            else:
                barrier(env)

            # Measure AllReduce latency (average over 10 iterations)
            ar_start = time.time()
            for _ in range(10):
                if data['env'] == 'fmi':
                    _ = communicator.allreduce(send_data, fmi.func(fmi.op.sum),
                                               fmi.types(fmi.datatypes.double_list, num_elements))
                else:
                    _ = communicator.allreduce(sum(send_data), ReduceOp.SUM)
            ar_end = time.time()
            allreduce_latency_ms = ((ar_end - ar_start) * 1000) / 10

            # Calculate bandwidth in MB/s (total data moved = msg_size * 2 for reduce + broadcast)
            # In allreduce, each process sends and receives msg_size bytes
            if allreduce_latency_ms > 0:
                bandwidth_mbps = (msg_size / (1024 * 1024)) / (allreduce_latency_ms / 1000)
            else:
                bandwidth_mbps = 0

            if rank == 0:
                print(f"### microbench iter={i} world={world_size} msg_size={msg_size} "
                      f"barrier={barrier_latency_ms:.3f}ms allreduce={allreduce_latency_ms:.3f}ms "
                      f"bw={bandwidth_mbps:.2f}MB/s")

                timing['world'].append(world_size)
                timing['iteration'].append(i)
                timing['com_init_t'].append(com_init)
                timing['barrier_latency_ms'].append(barrier_latency_ms)
                timing['msg_size_bytes'].append(msg_size)
                timing['allreduce_latency_ms'].append(allreduce_latency_ms)
                timing['allreduce_bandwidth_mbps'].append(bandwidth_mbps)

                if cost_tracker:
                    total_time = com_init + barrier_latency_ms + (allreduce_latency_ms * len(message_sizes))
                    cost_tracker.record_duration(total_time)
                    cost_summary = cost_tracker.get_summary_dict()
                    timing['lambda_memory_mb'].append(cost_summary['lambda_memory_mb'])
                    timing['lambda_duration_ms'].append(cost_summary['lambda_duration_ms'])
                    timing['lambda_gb_seconds'].append(cost_summary['lambda_gb_seconds'])
                    timing['lambda_cost_usd'].append(cost_summary['lambda_cost_usd'])
                    timing['step_fn_cost_usd'].append(cost_summary['step_fn_cost_usd'])
                    timing['total_cost_usd'].append(cost_summary['total_cost_usd'])
                else:
                    for key in ['lambda_memory_mb', 'lambda_duration_ms', 'lambda_gb_seconds',
                                'lambda_cost_usd', 'step_fn_cost_usd', 'total_cost_usd']:
                        timing[key].append(0)

    StopWatch.stop(f"microbenchmark_total_{data['env']}_{data['it']}")

    if rank == 0:
        StopWatch.benchmark(tag=str(data), filename=data['output_scaling_filename'])
        if data['env'] != 'rivanna':
            upload_file(file_name=data['output_scaling_filename'], bucket=data['s3_bucket'],
                        object_name=data['s3_stopwatch_object_name'])
            if cost_tracker:
                cost_tracker.record_s3_put()

        if os.path.exists(data['output_summary_filename']):
            os.remove(data['output_summary_filename'])
        pd.DataFrame(timing).to_csv(data['output_summary_filename'], mode='w', index=False, header=True)

        if data['env'] != 'rivanna':
            upload_file(file_name=data['output_summary_filename'], bucket=data['s3_bucket'],
                        object_name=data['s3_summary_object_name'])
            if cost_tracker:
                cost_tracker.record_s3_put()

    if data['env'] != 'fmi':
        env.finalize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="cylon scaling")

    parser.add_argument("-loglevel", dest='loglevel', type=int, help="Cylon Log Level",
                        **environ_or_required('CYLON_LOG_LEVEL', default=100, required=False))

    parser.add_argument('-operation', dest='operation', type=str, **environ_or_required('CYLON_OPERATION'),
                        choices=['join', 'sort', 'slice', 'floatPerf', 'groupby', 'microbenchmark'],
                        help="Operation to run: join, sort, slice, floatPerf, groupby, microbenchmark")

    parser.add_argument("-enablefmiping", dest='enablefmiping', type=bool, help="Enable ping host for fmi",
                        **environ_or_required('ENABLE_FMI_PING', default=False, required=False))

    parser.add_argument('-env', dest='env', type=str, **environ_or_required('ENV'))

    parser.add_argument("-fmioptions", dest='fmioptions', type=str,
                        **environ_or_required('FMI_OPTIONS', required=False),
                        choices=['blocking', 'nonblocking'],
                        help="blocking, nonblocking")  # w

    parser.add_argument("-maxtimeout", dest='maxtimeout', type=int, help="FMI Max Timeout",
                        **environ_or_required('FMI_MAX_TIMEOUT', default=120000, required=False))

    parser.add_argument('-iterations', dest='it', type=int, **environ_or_required('ITERATIONS'))  # 10 (iterations)

    parser.add_argument('-outputf1', dest='output_scaling_filename', type=str,
                        help="Output filename for scaling results",
                        **environ_or_required('OUTPUT_SCALING_FILENAME'))

    parser.add_argument('-outputf2', dest='output_summary_filename', type=str,
                        help="Output filename for scaling summary results",
                        **environ_or_required('OUTPUT_SUMMARY_FILENAME'))

    parser.add_argument("-rendezvous", dest='rendezvous_host', type=str, help="rendezvous host name",
                        **environ_or_required('RENDEZVOUS_HOST', required=False))

    parser.add_argument('-rendport', dest='rendezvous_port', type=int, help="rendezvous port",
                        **environ_or_required('RENDEZVOUS_PORT', required=False))

    parser.add_argument("-resolverendip", dest='resolverendip', type=bool, help="resolve rendezvous ip address",
                        **environ_or_required('RESOLVE_RENDEZVOUS_HOST', default=False, required=False))

    parser.add_argument("-redishost", dest='redis_host', type=str, help="redis address, default to 127.0.0.1",
                        **environ_or_required('REDIS_HOST', default="127.0.0.1", required=False))  # 127.0.0.1

    parser.add_argument("-redisport", dest='redis_port', type=int, help="name of redis port",
                        **environ_or_required('REDIS_PORT', default=-1, required=False))  # 6379

    parser.add_argument("-redisnamespace", dest='redis_namespace', type=str, help="redis namespace prefix",
                        **environ_or_required('REDIS_NAMESPACE', default="cylon", required=False))  # 127.0.0.1

    parser.add_argument("-sessionid", dest='session_id', type=str, help="session ID for Redis key namespace isolation",
                        **environ_or_required('CYLON_SESSION_ID', required=False))

    parser.add_argument('-rank', dest='rank', type=int, help="rank",
                        **environ_or_required('RANK', default=-1, required=False))

    parser.add_argument('-rows', dest='rows', type=int, **environ_or_required('ROWS'))

    parser.add_argument('-scaling', dest='scaling', type=str, **environ_or_required('SCALING'), choices=['s', 'w'],
                        help="s=strong w=weak")  # w

    parser.add_argument('-s3bucket', dest='s3_bucket', type=str, help="S3 Bucket Name",
                        **environ_or_required('S3_BUCKET', required=False))

    parser.add_argument('-stopwatcho1', dest='s3_stopwatch_object_name', type=str, help="S3 Object Name",
                        **environ_or_required('S3_STOPWATCH_OBJECT_NAME', required=False))

    parser.add_argument('-stopwatcho2', dest='s3_summary_object_name', type=str, help="S3 Object Name",
                        **environ_or_required('S3_SUMMARY_OBJECT_NAME', required=False))

    parser.add_argument('-unique', dest='unique', type=float, **environ_or_required('UNIQUENESS'),
                        help="unique factor")  # 0.9

    parser.add_argument('-worldsize', dest='world_size', type=int, help="world size",
                        **environ_or_required('WORLD_SIZE'))

    # Cost tracking arguments
    parser.add_argument('-lambdamemory', dest='lambda_memory_mb', type=int,
                        help="Lambda memory size in MB for cost calculation",
                        **environ_or_required('AWS_LAMBDA_FUNCTION_MEMORY_SIZE', default=10240, required=False))

    parser.add_argument('-pricingconfig', dest='pricing_config', type=str,
                        help="Path to AWS pricing config JSON file",
                        **environ_or_required('AWS_PRICING_CONFIG', required=False))

    parser.add_argument('-enablecosttracking', dest='enable_cost_tracking', type=bool,
                        help="Enable cost tracking in output",
                        **environ_or_required('ENABLE_COST_TRACKING', default=True, required=False))

    parser.add_argument('-channeltype', dest='channel_type', type=str,
                        help="FMI channel type: direct, redis, s3",
                        **environ_or_required('FMI_CHANNEL_TYPE', default='direct', required=False),
                        choices=['direct', 'redis', 's3'])

    parser.add_argument('-s3region', dest='s3_region', type=str,
                        help="AWS S3 region for S3 channel backend",
                        **environ_or_required('FMI_S3_REGION', default='us-east-1', required=False))

    parser.add_argument('-keyttl', dest='key_ttl', type=int,
                        help="TTL in seconds for Redis/FMI keys (default 3600 = 1 hour)",
                        **environ_or_required('KEY_TTL', default=3600, required=False))

    parser.add_argument('-s3retryinitms', dest='s3_retry_initial_ms', type=int,
                        help="Initial backoff in ms for S3 GET retries on NoSuchKey (default 100)",
                        **environ_or_required('S3_RETRY_INITIAL_MS', default=100, required=False))

    parser.add_argument('-s3retrymaxms', dest='s3_retry_max_ms', type=int,
                        help="Max backoff cap in ms for S3 GET retries (default 5000)",
                        **environ_or_required('S3_RETRY_MAX_MS', default=5000, required=False))

    args = vars(parser.parse_args())

    ipaddress = None

    if args["env"] == "fmi":
        import fmi
        from fmilib.fmi_operations import fmi_communicator
    elif args["env"] == "fmi-cylon":
        from pycylon.frame import CylonEnv, DataFrame
        from pycylon.net.reduce_op import ReduceOp
        from cylonfmilib.cylon_fmi import cylon_communicator
        from pycylon.util.logging import log_level
        log_level(args['loglevel'])

    else:
        if args.get('session_id'):
            os.environ['CYLON_SESSION_ID'] = args['session_id']
        from pycylon.frame import CylonEnv, DataFrame
        from pycylon.net.ucc_config import UCCConfig
        from pycylon.net.redis_ucc_oob_context import UCCRedisOOBContext
        from pycylon.net.reduce_op import ReduceOp
        from cylonlib.cylon import cylon_communicator


    if args['env'] == 'fargate':
        host = os.environ["ECS_CONTAINER_METADATA_URI_V4"]
        data = get_ecs_task_arn_cluster(host)
        # This print statement passes the string back to the bash wrapper, don't remove
        print("taskARN/Cluster: ", data)

        ips = get_service_ips(data['Cluster'], [data["TaskARN"]])

        if ips is not None:
            ipaddress = ips[0]

        #os.environ['UCX_TLS'] = 'tcp,shm'

        os.environ['UCX_LOG_LEVEL'] = 'debug'
        #os.environ['UCX_NET_DEVICES'] = 'eth0'
        #os.environ['UCX_TLS'] = 'tcp'
        #os.environ['UCX_UNIFIED_MODE'] = 'n'
        #os.environ['UCX_NUM_EPS'] = '256'

        #os.environ['UCX_NET_DEVICES'] = 'eth0'

    if args['env'] == 'rivanna':
        # Get the hostname of the local machine
        hostname = socket.gethostname()

        # Get the private IP address associated with the hostname
        private_ip = socket.gethostbyname(hostname)

        print("Rivanna Private IP Address:", private_ip)
        ipaddress = private_ip


    if args['operation'] == 'join':
        print("executing join operation")
        join(args, ipaddress)
    elif args['operation'] == 'groupby':
        print("executing groupby operation")
        groupby_agg(args, ipaddress)
    elif args['operation'] == 'microbenchmark':
        print("executing communication microbenchmark")
        comm_microbenchmark(args, ipaddress)
    elif args['operation'] == 'floatPerf':
        print("executing floatingPointPerf")
        floatingPointPerf(args, ipaddress)
