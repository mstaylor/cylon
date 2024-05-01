import time
import argparse
import socket

import pandas as pd
from numpy.random import default_rng
import mpi4py
mpi4py.rc.initialize = False
mpi4py.rc.finalize = False
from pycylon.frame import CylonEnv, DataFrame
from cloudmesh.common.StopWatch import StopWatch
from cloudmesh.common.dotdict import dotdict
from cloudmesh.common.Shell import Shell
from cloudmesh.common.util import writefile
from pycylon.net.ucc_config import UCCConfig
from pycylon.net.redis_ucc_oob_context import UCCRedisOOBContext
from pycylon.net.reduce_op import ReduceOp
import boto3
from botocore.exceptions import ClientError
import os
import requests
import json

import logging

def environ_or_required(key):
    return (
        {'default': os.environ.get(key)} if os.environ.get(key)
        else {'required': True}
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


def cylon_join(data=None, ipAddress = None):
    global ucc_config
    StopWatch.start(f"join_total_{data['host']}_{data['rows']}_{data['it']}")


    #if publicAddress is not None:
    #    print("setting UCX_TCP_PUBLIC_REMOTE_ADDRESS_OVERRIDE ", publicAddress )
    #    os.environ['UCX_TCP_PUBLIC_REMOTE_ADDRESS_OVERRIDE'] = publicAddress
    #    os.environ['UCX_TCP_PUBLIC_IP_PORT'] = f"{public_port}"

    #os.environ['UCX_TCP_CONN_NB'] = "y" #set to noblocking
    #os.environ['UCX_TCP_ENABLE_REDIS'] = "y" #enable redis for lambda hole punch
    #os.environ['UCX_TCP_ENABLE_TCPUNCH'] = "y" #enable holepunching via ucx
    os.environ['UCX_TCP_REDIS_IP'] = data['redis_host']
    os.environ['UCX_TCP_REDIS_PORT'] = f"{data['redis_port']}"
    #os.environ['UCX_TCP_REUSE_SOCK_ADDR'] = '1'
    #os.environ['UCX_TCP_ENABLE_NAT_TRAVERSAL'] = "y"


    #if ipAddress is not None:
    #    print("setting UCX_TCP_REMOTE_ADDRESS_OVERRIDE", ipAddress)
    #    os.environ['UCX_TCP_REMOTE_ADDRESS_OVERRIDE'] = ipAddress

    os.environ['UCX_TCP_IGNORE_IFNAME'] = 'y'



    redis_context = UCCRedisOOBContext(data['world_size'], f"tcp://{data['redis_host']}:{data['redis_port']}")

    if redis_context is not None:
        ucc_config = UCCConfig(redis_context)
    else:
        print("configured redis context")

    if ucc_config is None:
        print("unable to initialize uccconfig")
    else:
        print("initialized uccconfig")


    env = CylonEnv(config=ucc_config, distributed=True)

    print("retrieved cylon env")

    context = env.context

    if context is None:
        print("unable to retrieve cylon context")
    else:
        print("received cylon context")

    communicator = context.get_communicator()

    u = data['unique']

    if data['scaling'] == 'w':  # weak
        num_rows = data['rows']
        max_val = num_rows * env.world_size
    else:  # 's' strong
        max_val = data['rows']
        num_rows = int(data['rows'] / env.world_size)

    rng = default_rng(seed=env.rank)
    data1 = rng.integers(0, int(max_val * u), size=(num_rows, 2))
    data2 = rng.integers(0, int(max_val * u), size=(num_rows, 2))

    df1 = DataFrame(pd.DataFrame(data1).add_prefix("col"))
    df2 = DataFrame(pd.DataFrame(data2).add_prefix("col"))

    timing = {'scaling': [], 'world': [], 'rows': [], 'max_value': [], 'rank': [], 'avg_t': [], 'tot_l': []}

    print("iterating over range")
    for i in range(data['it']):
        env.barrier()
        StopWatch.start(f"join_{i}_{data['host']}_{data['rows']}_{data['it']}")
        t1 = time.time()
        df3 = df1.merge(df2, on=[0], algorithm='sort', env=env)
        env.barrier()
        t2 = time.time()
        t = (t2 - t1) * 1000
        # sum_t = comm.reduce(t)
        sum_t = communicator.allreduce(t, ReduceOp.SUM)
        # tot_l = comm.reduce(len(df3))
        tot_l = communicator.allreduce(len(df3), ReduceOp.SUM)

        if env.rank == 0:
            avg_t = sum_t / env.world_size
            print("### ", data['scaling'], env.world_size, num_rows, max_val, i, avg_t, tot_l)
            timing['scaling'].append(data['scaling'])
            timing['world'].append(env.world_size)
            timing['rows'].append(num_rows)
            timing['max_value'].append(max_val)
            timing['rank'].append(i)
            timing['avg_t'].append(avg_t)
            timing['tot_l'].append(tot_l)
            #print("### ", data['scaling'], env.world_size, num_rows, max_val, i, avg_t, tot_l, file=open(data['output_summary_filename'], 'a'))
            StopWatch.stop(f"join_{i}_{data['host']}_{data['rows']}_{data['it']}")

    StopWatch.stop(f"join_total_{data['host']}_{data['rows']}_{data['it']}")

    if env.rank == 0:
        StopWatch.benchmark(tag=str(data), filename=data['output_scaling_filename'])
        upload_file(file_name=data['output_scaling_filename'], bucket=data['s3_bucket'],
                    object_name=data['s3_stopwatch_object_name'])

        if os.path.exists(data['output_summary_filename']):
            pd.DataFrame(timing).to_csv(data['output_summary_filename'], mode='a', index=False, header=False)
        else:
            pd.DataFrame(timing).to_csv(data['output_summary_filename'], mode='w', index=False, header=True)

        upload_file(file_name=data['output_summary_filename'], bucket=data['s3_bucket'],
                    object_name=data['s3_summary_object_name'])

    env.finalize()


def cylon_sort(data=None):
    StopWatch.start(f"sort_total_{data['host']}_{data['rows']}_{data['it']}")

    redis_context = UCCRedisOOBContext(data['world_size'], f"tcp://{data['redis_host']}:{data['redis_port']}")

    if redis_context is not None:
        ucc_config = UCCConfig(redis_context)

    if ucc_config is None:
        print("unable to initialize uccconfig")

    env = CylonEnv(config=ucc_config, distributed=True)

    context = env.context

    if context is None:
        print("unable to retrieve cylon context")

    communicator = context.get_communicator()

    u = data['unique']

    if data['scaling'] == 'w':  # weak
        num_rows = data['rows']
        max_val = num_rows * env.world_size
    else:  # 's' strong
        max_val = data['rows']
        num_rows = int(data['rows'] / env.world_size)

    rng = default_rng(seed=env.rank)
    data1 = rng.integers(0, int(max_val * u), size=(num_rows, 2))

    df1 = DataFrame(pd.DataFrame(data1).add_prefix("col"))

    if env.rank == 0:
        print("Task# ", data['task'])

    for i in range(data['it']):
        env.barrier()
        StopWatch.start(f"sort_{i}_{data['host']}_{data['rows']}_{data['it']}")
        t1 = time.time()
        df3 = df1.sort_values(by=[0], env=env)
        env.barrier()
        t2 = time.time()
        t = (t2 - t1)
        sum_t = communicator.allreduce(t, ReduceOp.SUM)
        # tot_l = comm.reduce(len(df3))
        tot_l = communicator.allreduce(len(df3), ReduceOp.SUM)

        if env.rank == 0:
            avg_t = sum_t / env.world_size
            print("### ", data['scaling'], env.world_size, num_rows, max_val, i, avg_t, tot_l)
            print("### ", data['scaling'], env.world_size, num_rows, max_val, i, avg_t, tot_l,
                  file=open(data['output_summary_filename'], 'a'))


            StopWatch.stop(f"sort_{i}_{data['host']}_{data['rows']}_{data['it']}")

    StopWatch.stop(f"sort_total_{data['host']}_{data['rows']}_{data['it']}")

    if env.rank == 0:
        StopWatch.benchmark(tag=str(data), filename=data['output_scaling_filename'])
        upload_file(file_name=data['output_scaling_filename'], bucket=data['s3_bucket'],
                    object_name=data['s3_stopwatch_object_name'])
        upload_file(file_name=data['output_summary_filename'], bucket=data['s3_bucket'],
                    object_name=data['s3_summary_object_name'])
        redis_context.clearDB()


def cylon_slice(data=None):
    StopWatch.start(f"slice_total_{data['host']}_{data['rows']}_{data['it']}")

    redis_context = UCCRedisOOBContext(data['world_size'], f"tcp://{data['redis_host']}:{data['redis_port']}")

    if redis_context is not None:
        ucc_config = UCCConfig(redis_context)

    if ucc_config is None:
        print("unable to initialize uccconfig")

    env = CylonEnv(config=ucc_config, distributed=True)

    context = env.context

    if context is None:
        print("unable to retrieve cylon context")

    communicator = context.get_communicator()

    u = data['unique']

    if data['scaling'] == 'w':  # weak
        num_rows = data['rows']
        max_val = num_rows * env.world_size
    else:  # 's' strong
        max_val = data['rows']
        num_rows = int(data['rows'] / env.world_size)

    rng = default_rng(seed=env.rank)
    data1 = rng.integers(0, int(max_val * u), size=(num_rows, 2))
    data2 = rng.integers(0, int(max_val * u), size=(num_rows, 2))

    df1 = DataFrame(pd.DataFrame(data1).add_prefix("col"))
    df2 = DataFrame(pd.DataFrame(data2).add_prefix("col"))

    if env.rank == 0:
        print("Task# ", data['task'])

    for i in range(data['it']):
        env.barrier()
        StopWatch.start(f"slice_{i}_{data['host']}_{data['rows']}_{data['it']}")
        t1 = time.time()
        df3 = df1[0:20000000, env]  # distributed slice
        # print(df3)
        # df3 = df1.merge(df2, on=[0], algorithm='sort', env=env)
        env.barrier()
        t2 = time.time()
        t = (t2 - t1)
        sum_t = communicator.allreduce(t, ReduceOp.SUM)
        # tot_l = comm.reduce(len(df3))
        tot_l = communicator.allreduce(len(df3), ReduceOp.SUM)

        if env.rank == 0:
            avg_t = sum_t / env.world_size
            print("### ", data['scaling'], env.world_size, num_rows, max_val, i, avg_t, tot_l)
            print("### ", data['scaling'], env.world_size, num_rows, max_val, i, avg_t, tot_l,
                  file=open(data['output_summary_filename'], 'a'))
            StopWatch.stop(f"slice_{i}_{data['host']}_{data['rows']}_{data['it']}")

    StopWatch.stop(f"slice_total_{data['host']}_{data['rows']}_{data['it']}")

    if env.rank == 0:
        StopWatch.benchmark(tag=str(data), filename=data['output_scaling_filename'])
        upload_file(file_name=data['output_scaling_filename'], bucket=data['s3_bucket'],
                    object_name=data['s3_stopwatch_object_name'])
        upload_file(file_name=data['output_summary_filename'], bucket=data['s3_bucket'],
                    object_name=data['s3_summary_object_name'])

    env.finalize()

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
            for detail in attachment.get("details", []):
                if detail.get("name") == "networkInterfaceId":
                    enis.append(detail.get("value"))

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
    dict = {
        "TaskARN": cluster,
        "Cluster": taskArn
    }
    return dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="cylon scaling")

    parser.add_argument('-n', dest='rows', type=int, **environ_or_required('ROWS'))

    parser.add_argument('-i', dest='it', type=int, **environ_or_required('PARTITIONS')) #10

    parser.add_argument('-u', dest='unique', type=float, **environ_or_required('UNIQUENESS'), help="unique factor") #0.9

    parser.add_argument('-s', dest='scaling', type=str, **environ_or_required('SCALING'), choices=['s', 'w'],
                        help="s=strong w=weak") #w

    parser.add_argument('-o', dest='operation', type=str, **environ_or_required('CYLON_OPERATION'), choices=['join', 'sort', 'slice'],
                        help="s=strong w=weak")  # w

    parser.add_argument('-w', dest='world_size', type=int, help="world size", **environ_or_required('WORLD_SIZE'))

    parser.add_argument("-r", dest='redis_host', type=str, help="redis address, default to 127.0.0.1",
                        **environ_or_required('REDIS_HOST')) #127.0.0.1

    parser.add_argument("-r2", dest='rendezvous_host', type=str, help="redis address, default to 127.0.0.1",
                        **environ_or_required('RENDEVOUS_HOST'))

    parser.add_argument("-p1", dest='redis_port', type=int, help="name of redis port", **environ_or_required('REDIS_PORT')) #6379

    parser.add_argument('-f1', dest='output_scaling_filename', type=str, help="Output filename for scaling results",
                        **environ_or_required('OUTPUT_SCALING_FILENAME'))

    parser.add_argument('-f2', dest='output_summary_filename', type=str, help="Output filename for scaling summary results",
                        **environ_or_required('OUTPUT_SUMMARY_FILENAME'))

    parser.add_argument('-b', dest='s3_bucket', type=str, help="S3 Bucket Name", **environ_or_required('S3_BUCKET'))

    parser.add_argument('-o1', dest='s3_stopwatch_object_name', type=str, help="S3 Object Name", **environ_or_required('S3_STOPWATCH_OBJECT_NAME'))

    parser.add_argument('-o2', dest='s3_summary_object_name', type=str, help="S3 Object Name",
                        **environ_or_required('S3_SUMMARY_OBJECT_NAME'))

    args = vars(parser.parse_args())


    os.environ['EXPOSE_ENV'] = "1-65535"
    os.environ['UCX_LOG_LEVEL'] = "TRACE"
    os.environ['UCX_LOG_LEVEL_TRIGGER'] = "TRACE"
    os.environ['UCX_TCP_RENDEZVOUS_IP'] = socket.gethostbyname(args['rendezvous_host'])
    os.environ['UCX_POSIX_DIR'] = '/tmp/'

    # Get the hostname of the local machine
    hostname = socket.gethostname()

    # Get the private IP address associated with the hostname
    private_ip = socket.gethostbyname(hostname)

    print("Private IP Address:", private_ip)

    print(f"configuring rendezvous ip to be {os.environ['UCX_TCP_RENDEZVOUS_IP']}")


    args['host'] = "aws"

    if args['operation'] == 'join':
        print("executing cylon join operation")
        cylon_join(args, private_ip)
    elif args['operation'] == 'sort':
        print("executing cylon sort operation")
        cylon_sort(args)
    else:
        print ("executing cylon slice operation")
        cylon_slice(args)


