import fmi


def fmi_communicator(data = None):
    world_size = int(data["world_size"])
    rank = int(data["rank"])
    communicator = fmi.Communicator(rank, world_size, "fmi.json", "fmi_pair", 512)

    if communicator is None:
        print("unable to create FMI Communicator")
        return

    communicator.hint(fmi.hints.fast)

    print("retrieved fmi communicator")

    return communicator