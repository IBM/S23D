import os

import torch
import torch.distributed


def rank_zero():
    # check if we are in a distributed setting (if we are, must check that we are also on global rank 0)
    distributed = ("NODE_RANK" in os.environ) and ("LOCAL_RANK" in os.environ)
    if distributed:
        node_rank = int(os.environ["NODE_RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        if (node_rank == 0) and (local_rank == 0):
            return True
    else:
        # if not in a distributed setting, we can just start logger
        return True
    return False


def get_rank():
    distributed = ("NODE_RANK" in os.environ) and ("LOCAL_RANK" in os.environ)
    if distributed:
        rank = torch.distributed.get_rank()
        return rank
    return 0
