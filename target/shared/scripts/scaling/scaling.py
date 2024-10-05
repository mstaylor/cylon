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

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
)
from opentelemetry.trace import (
    SpanKind,
    get_tracer_provider,
    set_tracer_provider,
)

set_tracer_provider(TracerProvider())
tracer = get_tracer_provider().get_tracer(__name__)

get_tracer_provider().add_span_processor(
    BatchSpanProcessor(ConsoleSpanExporter())
)


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


def environ_or_required(key, default=None, required=True):
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


def barrier(obj=None):
    return obj.barrier()


def join(data=None, ipAddress=None):
    global ucc_config
    StopWatch.start(f"join_total_{data['env']}_{data['rows']}_{data['it']}")
    if ipAddress is not None:
        print("setting UCX_TCP_REMOTE_ADDRESS_OVERRIDE", ipAddress)
        os.environ['UCX_TCP_REMOTE_ADDRESS_OVERRIDE'] = ipAddress

    with tracer.start_as_current_span("initialize-communicator", kind=SpanKind.SERVER,
                                      attributes={"my_list": data}):
        if data['env'] == 'fmi':
            communicator = fmi_communicator(data)
            rank = int(data["rank"])
            world_size = int(data["world_size"])
        else:
            communicator, env = cylon_communicator(data)
            rank = env.rank
            world_size = env.world_size

        u = data['unique']

        if data['scaling'] == 'w':  # weak
            num_rows = data['rows']
            max_val = num_rows * world_size
        else:  # 's' strong
            max_val = data['rows']
            num_rows = int(data['rows'] / world_size)

        rng = default_rng(seed=rank)
        data1 = rng.integers(0, int(max_val * u), size=(num_rows, 2))
        data2 = rng.integers(0, int(max_val * u), size=(num_rows, 2))

        if data['env'] == 'fmi':
            df1 = pd.DataFrame(data1).add_prefix("col")
            df2 = pd.DataFrame(data2).add_prefix("col")
        else:
            df1 = DataFrame(pd.DataFrame(data1).add_prefix("col"))
            df2 = DataFrame(pd.DataFrame(data2).add_prefix("col"))

        timing = {'scaling': [], 'world': [], 'rows': [], 'max_value': [], 'rank': [], 'avg_t': [],
                  'tot_l': [], 'avg_l': [], 'max_t': []}

        max_time = 0
        print("iterating over range")
        for i in range(data['it']):

            if data['env'] == 'fmi':
                barrier(communicator)
            else:
                barrier(env)
            StopWatch.start(f"join_{i}_{data['env']}_{data['rows']}_{data['it']}")
            t1 = time.time()

            if data['env'] == 'fmi':
                df3 = pd.concat([df1, df2], axis=1)
                result_array = df3.to_numpy().flatten().tolist()

            else:
                df3 = df1.merge(df2, on=[0], algorithm='sort', env=env)

            if data['env'] == 'fmi':
                barrier(communicator)
            else:
                barrier(env)
            t2 = time.time()
            t = (t2 - t1) * 1000

            max_time = max(max_time, t)
            if data['env'] == 'fmi':
                sum_t = communicator.allreduce(t, fmi.func(fmi.op.sum), fmi.types(fmi.datatypes.double))
                # tot_l = comm.reduce(len(df3))
                tot_l = len(communicator.allreduce(result_array, fmi.func(fmi.op.sum),
                                                   fmi.types(fmi.datatypes.int_list, len(result_array))))

            else:
                sum_t = communicator.allreduce(t, ReduceOp.SUM)
                tot_l = communicator.allreduce(len(df3), ReduceOp.SUM)

            if rank == 0:
                end_time = time.time()
                elapsed_time = (end_time - t1) / data['it']
                avg_t = sum_t / world_size

                print("### ", data['scaling'], world_size, num_rows, max_val, i, avg_t, tot_l, elapsed_time, max_time)
                timing['scaling'].append(data['scaling'])
                timing['world'].append(world_size)
                timing['rows'].append(num_rows)
                timing['max_value'].append(max_val)
                timing['rank'].append(i)
                timing['avg_t'].append(avg_t)
                timing['tot_l'].append(tot_l)
                timing['avg_l'].append(elapsed_time)
                timing['max_t'].append(max_time)
                StopWatch.stop(f"join_{i}_{data['env']}_{data['rows']}_{data['it']}")

        StopWatch.stop(f"join_total_{data['env']}_{data['rows']}_{data['it']}")

        if rank == 0:
            StopWatch.benchmark(tag=str(data), filename=data['output_scaling_filename'])

            if data['env'] != 'rivanna':
                upload_file(file_name=data['output_scaling_filename'], bucket=data['s3_bucket'],
                            object_name=data['s3_stopwatch_object_name'])

            if os.path.exists(data['output_summary_filename']):
                os.remove(data['output_summary_filename'])
                #pd.DataFrame(timing).to_csv(data['output_summary_filename'], mode='a', index=False, header=False)
            #else:
            pd.DataFrame(timing).to_csv(data['output_summary_filename'], mode='w', index=False, header=True)

            if data['env'] != 'rivanna':
                upload_file(file_name=data['output_summary_filename'], bucket=data['s3_bucket'],
                            object_name=data['s3_summary_object_name'])

        if data['env'] != 'fmi':
            env.finalize()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="cylon scaling")

    parser.add_argument('-e', dest='env', type=str, **environ_or_required('ENV'))

    parser.add_argument('-n', dest='rows', type=int, **environ_or_required('ROWS'))

    parser.add_argument('-i', dest='it', type=int, **environ_or_required('PARTITIONS'))  #10

    parser.add_argument('-u', dest='unique', type=float, **environ_or_required('UNIQUENESS'),
                        help="unique factor")  #0.9

    parser.add_argument('-s', dest='scaling', type=str, **environ_or_required('SCALING'), choices=['s', 'w'],
                        help="s=strong w=weak")  #w

    parser.add_argument('-o', dest='operation', type=str, **environ_or_required('CYLON_OPERATION'),
                        choices=['join', 'sort', 'slice'],
                        help="s=strong w=weak")  # w

    parser.add_argument('-w', dest='world_size', type=int, help="world size", **environ_or_required('WORLD_SIZE'))

    parser.add_argument('-r2', dest='rank', type=int, help="world size", **environ_or_required('RANK', required=False))

    parser.add_argument("-r", dest='redis_host', type=str, help="redis address, default to 127.0.0.1",
                        **environ_or_required('REDIS_HOST', required=False))  #127.0.0.1

    parser.add_argument("-p1", dest='redis_port', type=int, help="name of redis port",
                        **environ_or_required('REDIS_PORT', required=False))  #6379

    parser.add_argument('-f1', dest='output_scaling_filename', type=str, help="Output filename for scaling results",
                        **environ_or_required('OUTPUT_SCALING_FILENAME'))

    parser.add_argument('-f2', dest='output_summary_filename', type=str,
                        help="Output filename for scaling summary results",
                        **environ_or_required('OUTPUT_SUMMARY_FILENAME'))

    parser.add_argument('-b', dest='s3_bucket', type=str, help="S3 Bucket Name",
                        **environ_or_required('S3_BUCKET', required=False))

    parser.add_argument('-o1', dest='s3_stopwatch_object_name', type=str, help="S3 Object Name",
                        **environ_or_required('S3_STOPWATCH_OBJECT_NAME', required=False))

    parser.add_argument('-o2', dest='s3_summary_object_name', type=str, help="S3 Object Name",
                        **environ_or_required('S3_SUMMARY_OBJECT_NAME', required=False))

    args = vars(parser.parse_args())

    ipaddress = None

    if args["env"] == "fmi":
        import fmi
        from fmilib.fmi_operations import fmi_communicator
    else:
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

        os.environ['UCX_LOG_LEVEL'] = 'diag'
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
