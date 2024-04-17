import sys
import time
import argparse
import subprocess

import boto3
from botocore.exceptions import ClientError
import os

import logging

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
        with open(file_name, 'wb') as f:
            s3_client.download_fileobj(bucket, object_name, f)
        return f
    except ClientError as e:
        print(f"error {e}")
        logging.error(e)
        return None

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
    os.environ['RENDEVOUS_HOST'] = event['RENDEVOUS_HOST']
    os.environ['SCALING'] = event['SCALING']
    os.environ['WORLD_SIZE'] = event['WORLD_SIZE']
    os.environ['PARTITIONS'] = event['PARTITIONS']
    os.environ['CYLON_OPERATION'] = event['CYLON_OPERATION']
    os.environ['ROWS'] = event['ROWS']
    os.environ["REDIS_PORT"] = event["REDIS_PORT"]
    os.environ["UNIQUENESS"] = event["UNIQUENESS"]

    parser = argparse.ArgumentParser(description="run S3 script")

    parser.add_argument('-b', dest='s3_bucket', type=str, help="S3 Bucket Name", **environ_or_required('S3_BUCKET'))
    parser.add_argument('-o', dest='s3_object_name', type=str, help="S3 Object Name",
                        **environ_or_required('S3_OBJECT_NAME'))
    parser.add_argument('-f', dest='output_filename', type=str, help="Output filename",
                        **environ_or_required('OUTPUT_FILENAME'))
    parser.add_argument('-a', dest='args', type=str, help="script exec arguments",
                        **environ_or_required('EXEC_ARGS', required=False))

    print ("parsing args")
    args, unknown = parser.parse_known_args()

    #execute

    #print("executing join")
    #join(vars(args))
    #print("executed join")

    p = subprocess.Popen('./natchecker -v', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in p.stdout.readlines():
        print(line),
    retval = p.wait()


    return f'Executed Serverless Cylon using Python{sys.version}! environment: {os.environ["S3_BUCKET"]}'