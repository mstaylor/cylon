import fmi
import sys

from cloudmesh.common.StopWatch import StopWatch
from cloudmesh.common.dotdict import dotdict
from cloudmesh.common.Shell import Shell
from cloudmesh.common.util import writefile

import boto3
from botocore.exceptions import ClientError

import time
import argparse
import logging
import os

def environ_or_required(key, required: bool = True):

    return (
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
        with open(file_name, 'wb') as f:
            s3_client.download_fileobj(bucket, object_name, f)
        return f
    except ClientError as e:
        print(f"error {e}")
        logging.error(e)
        return None

def fmi_send_receive(data=None):

    world_size = int(data["world_size"])
    rank = int(data["rank"])

    print("### ", world_size, rank)
    
    comm = fmi.Communicator(rank, world_size, "fmi.json", "fmi_direct_test", 512)
    comm.hint(fmi.hints.fast)

    comm.barrier()

    if rank == 0:
        comm.send(42, 1, fmi.types(fmi.datatypes.int))
        comm.send(14.2, 1, fmi.types(fmi.datatypes.double))
        comm.send([1, 2], 1, fmi.types(fmi.datatypes.int_list, 2))
        comm.send([1.32, 2.34], 1, fmi.types(fmi.datatypes.double_list, 2))
    elif rank == 1:
        # send / recv
        print(comm.recv(0, fmi.types(fmi.datatypes.int)))
        print(comm.recv(0, fmi.types(fmi.datatypes.double)))
        print(comm.recv(0, fmi.types(fmi.datatypes.int_list, 2)))
        print(comm.recv(0, fmi.types(fmi.datatypes.double_list, 2)))



def handler(event, context):
    os.environ["OPERATION"] = event.get("OPERATION")
    os.environ["WORLD_SIZE"] = event.get("WORLD_SIZE")
    os.environ["RANK"] = event.get("RANK")


    parser = argparse.ArgumentParser(description="fmi tests")

    parser.add_argument('-o', dest='operation', type=str, **environ_or_required('OPERATION'),
                        choices=['send-receive'])  # w


    parser.add_argument('-w', dest='world_size', type=int, help="world size", 
                        **environ_or_required('WORLD_SIZE'))

    parser.add_argument('-r', dest='rank', type=int, help="rank",
                        **environ_or_required('RANK'))

    print("parsing args")

    args, unknown = parser.parse_known_args()
    
    data = vars(args)

    if data['operation'] == 'send-receive':
        print("executing fmi send/receive operation")
        fmi_send_receive(data)


    return f'Executed Serverless Cylon using Python{sys.version}!'