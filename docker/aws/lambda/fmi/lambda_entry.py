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

def fmi_send_receive(data=None):
    
    comm = fmi.Communicator(int(data["rank"]), int(data["world_size"]), "fmi.json", "fmi_direct_test", 512)
    comm.hint(fmi.hints.fast)

    comm.barrier()

    if node_id == 0:
        comm.send(42, 1, fmi.types(fmi.datatypes.int))
        comm.send(14.2, 1, fmi.types(fmi.datatypes.double))
        comm.send([1, 2], 1, fmi.types(fmi.datatypes.int_list, 2))
        comm.send([1.32, 2.34], 1, fmi.types(fmi.datatypes.double_list, 2))
    elif node_id == 1:
        # send / recv
        print(comm.recv(0, fmi.types(fmi.datatypes.int)))
        print(comm.recv(0, fmi.types(fmi.datatypes.double)))
        print(comm.recv(0, fmi.types(fmi.datatypes.int_list, 2)))
        print(comm.recv(0, fmi.types(fmi.datatypes.double_list, 2)))



def handler(event, context):
    parser = argparse.ArgumentParser(description="fmi tests")

    args = vars(parser.parse_args())

    parser.add_argument('-o', dest='operation', type=str, **environ_or_required('OPERATION'),
                        choices=['send-receive'])  # w


    parser.add_argument('-w', dest='world_size', type=int, help="world size", 
                        **environ_or_required('WORLD_SIZE'))

    parser.add_argument('-w', dest='rank', type=int, help="rank",
                        **environ_or_required('RANK'))
    
    

    args, unknown = parser.parse_known_args()

    if args['operation'] == 'send-receive':
        print("executing fmi send/receive operation")
        fmi_send_receive(args)


    return f'Executed Serverless Cylon using Python{sys.version}!'