import json
import ctypes
import socket
import os

class PeerConnectionData(ctypes.Structure):
    _fields_ = [
        ("ip", ctypes.c_uint32),  # IPv4 address
        ("port", ctypes.c_uint16)
    ]

def lambda_handler(event, context):

    bucket = event.get("bucket", "")
    scaling = event.get("scaling", "w")
    object_type = event.get("object_type", "py")
    world_size = event.get("world_size", 1)
    rows = event.get("rows", 100000)
    uniqueness = event.get("uniqueness", 0.9)
    script = event.get("script", "")
    S3_object_name = event.get("S3_object_name", "")
    S3Path = event.get("S3Path", "")
    stopwatch_object_name = event.get("stopwatch_object_name", "")
    summary_object_name = event.get("summary_object_name", "")
    iterations = event.get("iterations", 1)
    operation = event.get("operation", "join")
    output_scaling_filename = event.get("output_scaling_filename", "/tmp/scaling.csv")
    output_summary_filename = event.get("output_summary_filename", "/tmp/summary.csv")
    rendezvous_host = event.get("rendezvous_host", "")
    rendezvous_port = event.get("rendezvous_port", "15000")
    resolve_rendezvous_host = event.get("resolve_rendezvous_host", "False")
    redis_namespace = event.get("redis_namespace", "")
    fmi_options = event.get("fmi_options", "nonblocking")
    fmi_channel_type = event.get("fmi_channel_type", "direct")
    redis_host = event.get("redis_host", "")
    redis_port = event.get("redis_port", "6379")
    cylon_log_level = event.get("cylon_log_level", "100")
    fmi_max_timeout = event.get("fmi_max_timeout", "300000")
    fmi_enable_ping = event.get("fmi_enable_ping", "False")
    cylon_session_id = event.get("cylon_session_id", "")
    enable_cost_tracking = event.get("enable_cost_tracking", "True")
    fmi_s3_region = event.get("fmi_s3_region", "us-east-1")
    fmi_s3_bucket = event.get("fmi_s3_bucket", "")
    key_ttl = event.get("key_ttl", "3600")


    if fmi_channel_type == "direct":
        comSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        comSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        comSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        comSocket.connect(("cylon-rendezvous.aws-cylondata.com", 10000))

        #print("initializing rendezvous to remove all client pairs that may exist")

        #byteString = bytes("delete_pairs", 'utf-8')
        #comSocket.sendall(byteString)

        comSocket.close()


    try:
        os.remove(output_scaling_filename)
    except:
        print("could not remove output_scaling_filename")

    try:
        os.remove(output_summary_filename)
    except:
        print("could not remove output_summary_filename")

    scaling_str = "weak"

    if scaling == "s":
        scaling_str = "strong"


    S3Path = S3Path.format(scaling = scaling_str)
    stopwatch_object_name = stopwatch_object_name.format(scaling = scaling_str, world_size = world_size)
    summary_object_name = summary_object_name.format(scaling = scaling_str, world_size = world_size)
    output_scaling_filename = output_scaling_filename.format(scaling = scaling_str, world_size = world_size)
    output_summary_filename = output_summary_filename.format(scaling = scaling_str, world_size = world_size)



    result = []


    for i in range(0, int(world_size)):
        payload = {
            "S3_BUCKET": bucket,
            "S3_OBJECT_NAME": S3_object_name,
            "SCRIPT": script,
            "S3_OBJECT_TYPE": object_type,
            "OUTPUT_SCALING_FILENAME": output_scaling_filename,
            "OUTPUT_SUMMARY_FILENAME": output_summary_filename,
            "S3_STOPWATCH_OBJECT_NAME": f"{S3Path}{stopwatch_object_name}",
            "S3_SUMMARY_OBJECT_NAME":f"{S3Path}{summary_object_name}",
            "SCALING": scaling,
            "WORLD_SIZE": world_size,
            "RANK": str(i),
            "ITERATIONS": iterations,
            "CYLON_OPERATION": operation,
            "ROWS": rows,
            "UNIQUENESS": uniqueness,
            "RENDEZVOUS_HOST": rendezvous_host,
            "RENDEZVOUS_PORT": rendezvous_port,
            "RESOLVE_RENDEZVOUS_HOST": resolve_rendezvous_host,
            "REDIS_NAMESPACE": redis_namespace,
            "FMI_OPTIONS": fmi_options,
            "FMI_CHANNEL_TYPE": fmi_channel_type,
            "REDIS_HOST": redis_host,
            "REDIS_PORT": redis_port,
            "CYLON_LOG_LEVEL": cylon_log_level,
            "FMI_MAX_TIMEOUT": fmi_max_timeout,
            "ENABLE_FMI_PING": fmi_enable_ping,
            "CYLON_SESSION_ID": cylon_session_id,
            "ENABLE_COST_TRACKING": enable_cost_tracking,
            "FMI_S3_REGION": fmi_s3_region,
            "FMI_S3_BUCKET": fmi_s3_bucket,
            "KEY_TTL": key_ttl
        }
        result.append(payload)



    return {
        'statusCode': 200,
        'body': result
    }
