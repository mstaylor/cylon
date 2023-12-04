import sys
import time
import argparse
import subprocess
import os

import logging

import time
import argparse

import pandas as pd
from numpy.random import default_rng
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




def environ_or_required(key, required: bool = True):

    return  (
        {'default': os.environ.get(key)} if os.environ.get(key)
        else {'required': required}
    )


def get_file(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # download the file
    s3_client = boto3.client('s3')
    try:
        print("downloading from S3")
        with open(file_name, 'wb') as f:
            s3_client.download_fileobj(bucket, object_name, f)

        return f
    except ClientError as e:
        print(f"error {e}")
        logging.error(e)
        return None

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
def cylon_join(data=None):
    global ucc_config
    StopWatch.start(f"join_total_{data['host']}_{data['rows']}_{data['it']}")

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

    timing = {'scaling': [], 'world': [], 'rows': [], 'max_value': [], 'rank': [], 'avg_t':[], 'tot_l':[]}

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
            StopWatch.stop(f"join_{i}_{data['host']}_{data['rows']}_{data['it']}")

    StopWatch.stop(f"join_total_{data['host']}_{data['rows']}_{data['it']}")

    if env.rank == 0:
        StopWatch.benchmark(tag=str(data), filename=data['output_scaling_filename'])
        upload_file(file_name=data['output_scaling_filename'], bucket=data['s3_bucket'], object_name=data['s3_stopwatch_object_name'])


        if os.path.exists(data['output_summary_filename']):
            pd.DataFrame(timing).to_csv(data['output_summary_filename'], mode='a', index=False, header=False)
        else:
            pd.DataFrame(timing).to_csv(data['output_summary_filename'], mode='w', index=False, header=True)


        upload_file(file_name=data['output_summary_filename'], bucket=data['s3_bucket'],
                    object_name=data['s3_summary_object_name'])

    env.finalize()
def join(data=None):
    print(f"executing join {data['output_filename']} {data['s3_bucket']} {data['s3_object_name']}")
    script = get_file(file_name=data['output_filename'], bucket=data['s3_bucket'], object_name=data['s3_object_name'])
    print("received data")
    if script is None:
        print(f"unable to retrieve file {data['output_filename']} from AWS S3")

    print("Retrieved script and now executing scripts")
    scriptargs = data['args']
    if scriptargs is not None:
        cmd = scriptargs.split()
        subprocess.call(['python'] + [data['output_filename']] + cmd, shell=False)
    else:
        subprocess.call(['python'] + [data['output_filename']], shell=False)

def handler(event, context):


    os.environ["S3_BUCKET"] = event.get("S3_BUCKET")
    os.environ["S3_OBJECT_NAME"] = event.get("S3_OBJECT_NAME")
    os.environ["OUTPUT_FILENAME"] = event.get("OUTPUT_FILENAME")
    os.environ['S3_STOPWATCH_OBJECT_NAME'] = event['S3_STOPWATCH_OBJECT_NAME']
    os.environ['OUTPUT_SCALING_FILENAME'] = event['OUTPUT_SCALING_FILENAME']
    os.environ['OUTPUT_SUMMARY_FILENAME'] = event['OUTPUT_SUMMARY_FILENAME']
    os.environ['S3_SUMMARY_OBJECT_NAME'] = event['S3_SUMMARY_OBJECT_NAME']
    os.environ['REDIS_HOST'] = event['REDIS_HOST']
    os.environ['EXPOSE_ENV'] = event['EXPOSE_ENV']
    os.environ['SCALING'] = event['SCALING']
    os.environ['UCX_TCP_PORT_RANGE'] = event['UCX_TCP_PORT_RANGE']
    os.environ['WORLD_SIZE'] = event['WORLD_SIZE']
    os.environ['PARTITIONS'] = event['PARTITIONS']
    os.environ['CYLON_OPERATION'] = event['CYLON_OPERATION']
    os.environ['ROWS'] = event['ROWS']

    #parser = argparse.ArgumentParser(description="run S3 script")

    #parser.add_argument('-b', dest='s3_bucket', type=str, help="S3 Bucket Name", **environ_or_required('S3_BUCKET'))
    #parser.add_argument('-o', dest='s3_object_name', type=str, help="S3 Object Name",
    #                    **environ_or_required('S3_OBJECT_NAME'))
    #parser.add_argument('-f', dest='output_filename', type=str, help="Output filename",
    #                    **environ_or_required('OUTPUT_FILENAME'))
    #parser.add_argument('-a', dest='args', type=str, help="script exec arguments",
    #                    **environ_or_required('EXEC_ARGS', required=False))

    parser = argparse.ArgumentParser(description="cylon scaling")

    parser.add_argument('-n', dest='rows', type=int, **environ_or_required('ROWS'))

    parser.add_argument('-i', dest='it', type=int, **environ_or_required('PARTITIONS'))  # 10

    parser.add_argument('-u', dest='unique', type=float, **environ_or_required('UNIQUENESS'),
                        help="unique factor")  # 0.9

    parser.add_argument('-s', dest='scaling', type=str, **environ_or_required('SCALING'), choices=['s', 'w'],
                        help="s=strong w=weak")  # w

    parser.add_argument('-o', dest='operation', type=str, **environ_or_required('CYLON_OPERATION'),
                        choices=['join', 'sort', 'slice'],
                        help="s=strong w=weak")  # w

    parser.add_argument('-w', dest='world_size', type=int, help="world size", **environ_or_required('WORLD_SIZE'))

    parser.add_argument("-r", dest='redis_host', type=str, help="redis address, default to 127.0.0.1",
                        **environ_or_required('REDIS_HOST'))  # 127.0.0.1

    parser.add_argument("-p1", dest='redis_port', type=int, help="name of redis port",
                        **environ_or_required('REDIS_PORT'))  # 6379

    parser.add_argument('-f1', dest='output_scaling_filename', type=str, help="Output filename for scaling results",
                        **environ_or_required('OUTPUT_SCALING_FILENAME'))

    parser.add_argument('-f2', dest='output_summary_filename', type=str,
                        help="Output filename for scaling summary results",
                        **environ_or_required('OUTPUT_SUMMARY_FILENAME'))

    parser.add_argument('-b', dest='s3_bucket', type=str, help="S3 Bucket Name", **environ_or_required('S3_BUCKET'))

    parser.add_argument('-o1', dest='s3_stopwatch_object_name', type=str, help="S3 Object Name",
                        **environ_or_required('S3_STOPWATCH_OBJECT_NAME'))

    parser.add_argument('-o2', dest='s3_summary_object_name', type=str, help="S3 Object Name",
                        **environ_or_required('S3_SUMMARY_OBJECT_NAME'))

    print ("parsing args")
    args, unknown = parser.parse_known_args()

    print("executing join")
    #cylon_join(vars(args))
    print("executed join")


    return f'Executed Serverless Cylon using Python{sys.version}! environment: {os.environ["S3_BUCKET"]}'