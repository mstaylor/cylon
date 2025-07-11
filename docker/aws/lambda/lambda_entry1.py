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
    os.environ["S3_BUCKET"] = event.get("S3_BUCKET", None)
    os.environ["S3_OBJECT_NAME"] = event.get("S3_OBJECT_NAME", None)
    os.environ["SCRIPT"] = event.get("SCRIPT", None)
    os.environ["S3_OBJECT_TYPE"] = event.get("S3_OBJECT_TYPE", None)
    os.environ['S3_STOPWATCH_OBJECT_NAME'] = event.get("S3_STOPWATCH_OBJECT_NAME", None)
    os.environ["OUTPUT_SCALING_FILENAME"] = event.get("OUTPUT_SCALING_FILENAME", None)
    os.environ["OUTPUT_SUMMARY_FILENAME"] = event.get("OUTPUT_SUMMARY_FILENAME", None)
    os.environ["S3_SUMMARY_OBJECT_NAME"] = event.get("S3_SUMMARY_OBJECT_NAME", None)
    os.environ["REDIS_HOST"] = event.get("REDIS_HOST", None)
    os.environ["REDIS_PORT"] = event.get("REDIS_PORT", None)
    os.environ["REDIS_NAMESPACE"] =event.get("REDIS_NAMESPACE", None)
    os.environ["RENDEZVOUS_HOST"] = event.get("RENDEZVOUS_HOST", None)
    os.environ["RENDEZVOUS_PORT"] = event.get("RENDEZVOUS_PORT", None)
    os.environ["RESOLVE_RENDEZVOUS_HOST"] = event.get("RESOLVE_RENDEZVOUS_HOST", None)
    os.environ["SCALING"] = event.get("SCALING", None)
    os.environ["WORLD_SIZE"] = event.get("WORLD_SIZE", None)
    os.environ["ITERATIONS"] = event.get("ITERATIONS", None)
    os.environ["CYLON_OPERATION"] = event.get("CYLON_OPERATION", None)
    os.environ["ROWS"] = event.get("ROWS", None)
    os.environ["UNIQUENESS"] = event.get("UNIQUENESS", None)
    os.environ["RANK"] = event.get("RANK", None)
    os.environ["ENV"] = "fmi-cylon"
    os.environ["CYLON_LOG_LEVEL"] = event.get("CYLON_LOG_LEVEL", None)
    os.environ["FMI_OPTIONS"] = event.get("FMI_OPTIONS", None)
    os.environ["FMI_MAX_TIMEOUT"] = event.get("FMI_MAX_TIMEOUT", None)

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