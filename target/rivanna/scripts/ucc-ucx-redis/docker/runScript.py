import subprocess
import time
import argparse

import boto3
from botocore.exceptions import ClientError
import os

import logging
from boto3.exceptions import S3TransferFailedError
import uuid


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



def execute_script(data=None):

    scriptargs = data['args']
    if scriptargs is not None:
        cmd = scriptargs.split()
        subprocess.call(['python'] + [data['script']] + cmd, shell=False)
    else:
        subprocess.call(['python'] + [data['script']], shell=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run S3 script")

    parser.add_argument('-s', dest='script', type=str, help="script to execute",
                        **environ_or_required('SCRIPT'))
    parser.add_argument('-a', dest='args', type=str, help="script exec arguments",
                        **environ_or_required('EXEC_ARGS', required=False))

    args = vars(parser.parse_args())
    execute_script(args)
