from pycylon.frame import CylonEnv, DataFrame
from pycylon.net.fmi_config import FMIConfig

from pycylon.net.reduce_op import ReduceOp



def cylon_communicator(data = None):

    nonblocking = data['fmioptions'] == 'nonblocking'
    resolverendip = bool(data['resolverendip'])
    fmi_config = FMIConfig(data['rank'], data['world_size'], f"{data['rendezvous_host']}", data['rendezvous_port'], 
                           data['maxtimeout'], resolverendip, "fmi_pair", nonblocking, f"{data['redis_host']}",
                           data['redis_port'], f"{data['redis_namespace']}")

    if fmi_config is None:
        print("unable to initialize fmi_config")

    env = CylonEnv(config=fmi_config, distributed=True)

    context = env.context

    if context is None:
        print("unable to retrieve cylon context")

    communicator = context.get_communicator()

    return communicator, env
