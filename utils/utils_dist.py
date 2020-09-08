import os
import datetime
import torch.distributed as dist

def init_process(rank, size, backend='gloo', master_addr='127.0.0.1', master_port='12355', timeout_h=5):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    dist.init_process_group(backend, rank=rank, world_size=size, timeout=datetime.timedelta(hours=timeout_h))

def cleanup():
    dist.destroy_process_group()