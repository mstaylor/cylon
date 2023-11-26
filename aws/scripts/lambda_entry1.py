import sys
def handler(event, context):
    return f'Hello from AWS Lambda using Python{sys.version}!'