import os
import sys
import argparse
from textwrap import dedent
from cloudmesh.common.util import writefile
from cloudmesh.common.util import readfile
from cloudmesh.common.util import banner
from cloudmesh.common.console import Console
import argparse


counter = 0

debug = False

partition = "bii-gpu"

partition = "parallel"

parser = argparse.ArgumentParser(description="cylon rivanna scaling test")
parser.add_argument('-r', dest='rows', type=int, required=True)

parser.add_argument('-n', dest='nodes', type=int, required=True)
parser.add_argument('-t', dest='threads', type=int, required=True)
parser.add_argument('-c', dest='cpus', type=int, required=True)

parser.add_argument('-s', dest='scaling', type=str, default='w', choices=['s', 'w'],
                    help="s=strong w=weak")

parser.add_argument('-w', dest='world_size', type=int, help="world size", required=True)

parser.add_argument("-r1", dest='redis_host', type=str, help="redis address",
                        required=True) #127.0.0.1

parser.add_argument("-p1", dest='redis_port', type=int, help="name of redis port",
                    default=6379)  # 6379

parser.add_argument('-e', dest='env', type=str, default="rivanna")

parser.add_argument('-i', dest='it', type=int, default=10) #10

parser.add_argument('-u', dest='unique', type=float, help="unique factor", default=0.9)

parser.add_argument('-o', dest='operation', type=str, choices=['join', 'sort', 'slice', 'floatPerf'],
                        default="join")

parser.add_argument('-p2', dest='partition', type=str, default="standard")
parser.add_argument('-m', dest='memory', type=str, default="DefMemPerNode")

#parser.add_argument('-p2', dest='ucx_port_range', type=str,
#                   help="Range of ports to use for UCX",
#                       required=True)

parser.add_argument('-f1', dest='output_scaling_filename', type=str,
                    help="Output filename for scaling results",
                       required=True)

parser.add_argument('-f2', dest='output_summary_filename', type=str,
                    help="Output filename for scaling summary results",
                        required=True)

parser.add_argument('-d', dest='docker_image', type=str, help="docker image or sif",
                    default="qad5gv/cylon-rivanna", required=True)

parser.add_argument('-l1', dest='log_bind_host', type=str, help="directory for output on the rivanna host",
                    required=True)
parser.add_argument('-l2', dest='log_bind_container', type=str, help="directory for output on the rivanna host",
                    required=True)

parser.add_argument('-s1', dest='script_bind_host', type=str, help="directory for output on the rivanna host",
                    required=True)
parser.add_argument('-s2', dest='script_bind_container', type=str, help="directory for output on the rivanna host",
                    required=True)
parser.add_argument('-s3', dest='script', type=str, help="script to execute",
                    required=True)





args = vars(parser.parse_args())

#add list of env variables to pass to Apptainer container
env_vars = [f"ENV={args['env']}",
            f"ROWS={args['rows']}",
            f"PARTITIONS={args['it']}",
            f"UNIQUENESS={args['unique']}",
            f"SCALING={args['scaling']}",
            f"CYLON_OPERATION={args['operation']}",
            f"WORLD_SIZE={args['world_size']}",
            f"REDIS_HOST={args['redis_host']}",
            f"REDIS_PORT={args['redis_port']}",
            f"OUTPUT_SCALING_FILENAME={args['output_scaling_filename']}",
            f"OUTPUT_SUMMARY_FILENAME={args['output_summary_filename']}",
            #f"UCX_TCP_PORT_RANGE={args['ucx_port_range']}",
            #f"EXPOSE_ENV={args['ucx_port_range']}",
            f"SCRIPT={args['script']}"]

env_vars_str = ",".join(env_vars)

print(f"env args to pass to apptainer: {env_vars_str}")


memspec = ""

if args['memory'] != "DefMemPerNode":
    memspec = f"#SBATCH --mem={args['memory']}"




# (nodes, threads, cpus, rows, partition, "exclusive")
combination = [ \
    # (1,4, 5000, "parallel", "exclusive"), # always pending
    (args['nodes'], args['threads'], args['cpus'], args['partition'], ""),
    # ("54.227.18.138", 4,8, 16, args['rows'], "parallel", ""),
    # ("44.213.71.107", 4,8, 16, args['rows'], "parallel", ""),
    # ("52.90.116.44", 4,8, 16, args['rows'], "parallel", ""),
    # (2,37, 1000000, "parallel", ""),
    # (4,37, 35000000, "parallel", ""),
    # (6,37, 35000000, "parallel", ""),
    # (8,37, 35000000, "parallel", ""),
    # (10,37, 35000000, "parallel", ""),
    # (12,37, 35000000, "parallel", ""),
    # (14,37, 35000000, "parallel", ""),
]

'''
combination = []
for nodes in range(0,50):
  for threads in range(0,37):
    combination.append((nodes+1, threads+1, "parallel", "")) 
'''

total = len(combination)
jobid = "-%j"
# jobid=""

f = open("../../../../rivanna/scripts/submit.log", "w")
for nodes, threads, cpus, partition, exclusive in combination:
    counter = counter + 1

    if exclusive == "exclusive":
        exclusive = "#SBATCH --exclusive"
        e = "e1"
    else:
        exclusive = ""
        e = "e0"

    usable_threads = nodes * threads

    '''
    cores_per_node = nodes * threads - 2
  
    print (cores_per_node)
  
    config = readfile("raptor.in.cfg")
  
    config = config.replace("CORES_PER_NODE", str(cores_per_node))
    config = config.replace("NO_OF_ROWS", str(rows))
  
  
    print (config)
  
    cfg_filename = f"raptor-{nodes}-{threads}.cfg"
  
    writefile(cfg_filename, config)
    '''
    banner(f"SLURM {nodes} {threads} {counter}/{total}")
    script = dedent(f"""
  #!/bin/bash
  #!/bin/bash
  #SBATCH --job-name=h-n={nodes:02d}-t={threads:02d}-e={e}
  #SBATCH --nodes={nodes}
  #SBATCH --ntasks={threads}
  #SBATCH --cpus-per-task={cpus}
  {memspec}
  #SBATCH --time=15:00
  #SBATCH --time=15:00
  #SBATCH --output=out-{nodes:02d}-{threads:02d}{jobid}.log
  #SBATCH --error=out-{nodes:02d}-{threads:02d}{jobid}.err
  #SBATCH --partition={partition}
  #SBATCH -A bii_dsc_community
  {exclusive}
  echo "..............................................................."
  source /scratch/qad5gv/cylon/CYLON-ENV/bin/activate
  echo "..............................................................." 
  echo "..............................................................."
  module load apptainer
  echo "..............................................................."  
  lscpu
  echo "..............................................................."
  time srun --exact --nodes {nodes} apptainer run --env {env_vars_str} --bind {args['log_bind_host']}:{args['log_bind_container']},{args['script_bind_host']}:{args['script_bind_container']}  --containall {args['docker_image']}
  echo "..............................................................."
  """).strip()

    print(script)
    filename = f"script-{nodes:02d}-{threads:02d}.slurm"
    writefile(filename, script)

    if not debug:

        r = os.system(f"sbatch {filename}")
        total = nodes * threads
        if r == 0:
            msg = f"{counter} submitted: nodes={nodes:02d} threads={threads:02d} total={total}"
            Console.ok(msg)
        else:
            msg = f"{counter} failed: nodes={nodes:02d} threads={threads:02d} total={total}"
            Console.error(msg)
        f.writelines([msg, "\n"])
f.close()
