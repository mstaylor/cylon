import sys
import time
import argparse

import boto3
from botocore.exceptions import ClientError
import os

import logging



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
        logging.error(e)
        return None
def handler(event, context):
    return f'Hello from AWS Lambda using Python{sys.version}!'