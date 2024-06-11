import sys
import time
import argparse
import subprocess

import boto3
from botocore.exceptions import ClientError
import os

import logging




def handler(event, context):
    print("running")