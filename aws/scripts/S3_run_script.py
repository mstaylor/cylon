import subprocess
import time
import argparse

import boto3
from botocore.exceptions import ClientError
import os

import logging
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

def get_file(file_name, bucket, prefix=None, use_folder=False):

    # If S3 object_name was not specified, use file_name
    if prefix is None:
        prefix = os.path.basename(file_name)

    # download the file
    s3_client = boto3.client('s3')
    try:

        if use_folder:
            response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
            if 'Contents' not in response:
                print("No files found in the specified directory.")
                return None

            current_dir = os.getcwd()

            for obj in response['Contents']:
                s3_file_path = obj['Key']
                print(f"retrieving key: {s3_file_path}")
                relative_path = os.path.relpath(s3_file_path, prefix)
                local_file_path = os.path.join(current_dir, relative_path)

                # Ensure local directory structure exists
                local_file_dir = os.path.dirname(local_file_path)
                if not os.path.exists(local_file_dir):
                    os.makedirs(local_file_dir)

                # Download the file
                print(f"Downloading {s3_file_path} to {local_file_path}")
                s3_client.download_file(bucket, s3_file_path, local_file_path)

                
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
    parser.add_argument('-o', dest='s3_object_name', type=str, help="S3 Object Name", **environ_or_required('S3_OBJECT_NAME'))
    parser.add_argument('-s', dest='script', type=str, help="script to execute",
                        **environ_or_required('SCRIPT'))
    parser.add_argument('-t', dest='s3_object_type', type=str, **environ_or_required('S3_OBJECT_TYPE', "File"), choices=['file', 'folder'],
                        help="file or folder for S3 pull")  # w
    parser.add_argument('-a', dest='args', type=str, help="script exec arguments",
                        **environ_or_required('EXEC_ARGS', required=False))



    args = vars(parser.parse_args())
    execute_script(args)
