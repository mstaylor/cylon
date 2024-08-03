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


def get_all_s3_objects(s3, **base_kwargs):
    continuation_token = None
    while True:
        list_kwargs = dict(MaxKeys=1000, **base_kwargs)
        if continuation_token:
            list_kwargs['ContinuationToken'] = continuation_token
        response = s3.list_objects_v2(**list_kwargs)
        yield from response.get('Contents', [])
        if not response.get('IsTruncated'):
            break
        continuation_token = response.get('NextContinuationToken')


def get_file(file_name, bucket, prefix=None, use_folder=False):
    # If S3 object_name was not specified, use file_name
    if prefix is None:
        prefix = os.path.basename(file_name)

    # download the file
    s3_client = boto3.client('s3')
    try:

        if use_folder:
            all_s3_objects_gen = get_all_s3_objects(s3_client, Bucket=bucket)

            for obj in all_s3_objects_gen:
                source = obj['Key']
                if source.startswith(prefix):
                    destination = os.path.join('/', source)
                    if not os.path.exists(os.path.dirname(destination)):
                        os.makedirs(os.path.dirname(destination))
                    try:
                        print(f'[DEBUG] Downloading: {source} --> {destination}')
                        s3_client.download_file(bucket, source, destination)
                    except (ClientError, S3TransferFailedError) as e:
                        print(f'[ERROR] Could not download file "{source}": {e}')

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

    if script is None:
        print(f"unable to retrieve file {data['script']} from AWS S3")

    scriptargs = data['args']
    if scriptargs is not None:
        cmd = scriptargs.split()
        subprocess.call(['python'] + [data['script']] + cmd, shell=False)
    else:
        subprocess.call(['python'] + [data['script']], shell=False)


if __name__ == "__main__":
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

    args = vars(parser.parse_args())
    execute_script(args)
