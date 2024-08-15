from pycylon.frame import CylonEnv, DataFrame
from pycylon.net.ucc_config import UCCConfig
from pycylon.net.redis_ucc_oob_context import UCCRedisOOBContext
from pycylon.net.reduce_op import ReduceOp



def cylon_communicator(data = None):
    redis_context = UCCRedisOOBContext(data['world_size'], f"tcp://{data['redis_host']}:{data['redis_port']}")

    if redis_context is not None:
        ucc_config = UCCConfig(redis_context)

    if ucc_config is None:
        print("unable to initialize uccconfig")

    env = CylonEnv(config=ucc_config, distributed=True)

    context = env.context

    if context is None:
        print("unable to retrieve cylon context")

    communicator = context.get_communicator()

    return communicator, env