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


def get_file(file_name, bucket, prefix=None, use_folder=False):
    # If S3 object_name was not specified, use file_name
    if prefix is None:
        prefix = os.path.basename(file_name)

    # download the file
    s3_client = boto3.client('s3')
    try:

        if use_folder:
            list_kwargs = {
                'Bucket': bucket,
                'Prefix': prefix
            }

            response = s3_client.list_objects_v2(**list_kwargs)

            if 'Contents' not in response:
                print("No files found in the specified directory.")
                return

            for s3_object in response['Contents']:
                s3_key = s3_object["Key"]
                path, filename = os.path.split(s3_key)
                # bucket folders root does not include a /
                path = f"/tmp/{path}"
                print(f's3key: {s3_key} path: {path} filename: {filename}')
                if len(path) != 0 and not os.path.exists(path):
                    print(f'creating os path: {path}')
                    os.makedirs(path)
                if not s3_key.endswith("/"):
                    download_to = f'{path}/{filename}' if path else filename
                    print(f'downloading key: {s3_key} to {download_to}')
                    s3_client.download_file(bucket, s3_key, download_to)

        else:
            with open(file_name, 'wb') as f:
                s3_client.download_fileobj(bucket, prefix, f)
            return f
    except ClientError as e:
        logging.error(e)
        return None

def execute_script(data=None):
    usefolder = data['s3_object_type'] == 'folder'
    script = get_file(file_name=data['script'], bucket=data['s3_bucket'],
                      prefix=data['s3_object_name'], use_folder=usefolder)

    if script is None and not usefolder:
        print(f"unable to retrieve file {data['script']} from AWS S3")

    scriptargs = data['args']
    if scriptargs is not None:
        cmd = scriptargs.split()
        subprocess.call(['python'] + [data['script']] + cmd, shell=False)
    else:
        subprocess.call(['python'] + [data['script']], shell=False)


def handler(event, context):
    s3_bucket = event['s3_bucket']
    if s3_bucket is not None:
        os.environ["S3_BUCKET"] = s3_bucket
    s3_object_name = event["S3_OBJECT_NAME"]
    if s3_object_name is not None:
        os.environ["S3_OBJECT_NAME"] = s3_object_name
    script = event["script"]
    if script is not None:
        os.environ["SCRIPT"] = script
    s3_object_type = event["S3_OBJECT_TYPE"]
    if s3_object_type is not None:
        os.environ["S3_OBJECT_TYPE"] = s3_object_type
    s3_stopwatch_object_name = event["S3_STOPWATCH_OBJECT_NAME"]
    if s3_stopwatch_object_name is not None:
        os.environ["S3_STOPWATCH_OBJECT_NAME"] = s3_stopwatch_object_name
    output_scaling_filename = event["OUTPUT_SCALING_FILENAME"]
    if output_scaling_filename is not None:
        os.environ["OUTPUT_SCALING_FILENAME"] = output_scaling_filename
    output_summary_filename = event["OUTPUT_SUMMARY_FILENAME"]
    if output_summary_filename is not None:
        os.environ["OUTPUT_SUMMARY_FILENAME"] = output_summary_filename
    s3_summary_object_name = event["S3_SUMMARY_OBJECT_NAME"]
    if s3_summary_object_name is not None:
        os.environ["S3_SUMMARY_OBJECT_NAME"] = s3_summary_object_name
    redis_host = event["REDIS_HOST"]
    if redis_host is not None:
        os.environ["REDIS_HOST"] = redis_host
    redis_port = event["REDIS_PORT"]
    if redis_port is not None:
        os.environ["REDIS_PORT"] = redis_port
    redis_namespace = event["REDIS_NAMESPACE"]
    if redis_namespace is not None:
        os.environ["REDIS_NAMESPACE"] = redis_namespace
    rendezvous_host = event["RENDEZVOUS_HOST"]
    if rendezvous_host is not None:
        os.environ["RENDEZVOUS_HOST"] = rendezvous_host
    rendezvous_port = event["RENDEZVOUS_PORT"]
    if rendezvous_port is not None:
        os.environ["RENDEZVOUS_PORT"] = rendezvous_port
    resolve_rendezvous_host = event["RESOLVE_RENDEZVOUS_HOST"]
    if resolve_rendezvous_host is not None:
        os.environ["RESOLVE_RENDEZVOUS_HOST"] = resolve_rendezvous_host
    scaling = event["SCALING"]
    if scaling is not None:
        os.environ["SCALING"] = scaling
    world_size = event["WORLD_SIZE"]
    if world_size is not None:
        os.environ["WORLD_SIZE"] = world_size
    iterations = event["ITERATIONS"]
    if iterations is not None:
        os.environ["ITERATIONS"] = iterations
    cylon_operation = event["CYLON_OPERATION"]
    if cylon_operation is not None:
        os.environ["CYLON_OPERATION"] = cylon_operation
    rows = event["ROWS"]
    if rows is not None:
        os.environ["ROWS"] = rows
    uniqueness = event["UNIQUENESS"]
    if uniqueness is not None:
        os.environ["UNIQUENESS"] = uniqueness
    rank = event["RANK"]
    if rank is not None:
        os.environ["RANK"] = rank
    os.environ["ENV"] = "fmi-cylon"
    cylon_log_level = event["CYLON_LOG_LEVEL"]
    if cylon_log_level is not None:
        os.environ["CYLON_LOG_LEVEL"] = cylon_log_level
    fmi_options = event["FMI_OPTIONS"]
    if fmi_options is not None:
        os.environ["FMI_OPTIONS"] = fmi_options
    fmi_max_timeout = event["FMI_MAX_TIMEOUT"]
    if fmi_max_timeout is not None:
        os.environ["FMI_MAX_TIMEOUT"] = fmi_max_timeout

    parser = argparse.ArgumentParser(description="run S3 script")

    parser.add_argument('-b', dest='s3_bucket', type=str, help="S3 Bucket Name", **environ_or_required('S3_BUCKET'))
    parser.add_argument('-o', dest='s3_object_name', type=str, help="S3 Object Name",
                        **environ_or_required('S3_OBJECT_NAME'))
    parser.add_argument('-s', dest='script', type=str, help="script to execute",
                        **environ_or_required('SCRIPT'))
    parser.add_argument('-t', dest='s3_object_type', type=str, **environ_or_required('S3_OBJECT_TYPE', "File"),
                        choices=['file', 'folder'],
                        help="file or folder for S3 pull")  # w
    parser.add_argument('-a', dest='args', type=str, help="script exec arguments",
                        **environ_or_required('EXEC_ARGS', required=False))

    print("parsing args")
    args, unknown = parser.parse_known_args()

    # execute

    data = vars(args)

    print(f"executing script: {data['s3_object_name']}")
    execute_script(data)
    print("executed script")

    return f'Executed Serverless Cylon FMI using Python{sys.version}! environment: {os.environ["S3_BUCKET"]}'