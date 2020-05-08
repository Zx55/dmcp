# -*- coding:utf-8  -*-

import functools
import os
import socket
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


gpu_id = None


def init_dist(distributed=True, backend='nccl'):
    global gpu_id

    mp.set_start_method('spawn')
    if distributed:
        if dist.is_initialized():
            return

        # for slurm
        if os.environ.get('SLURM_PROCID', None) is not None:
            rank, world_size, url = _slurm_init_distributed()

        # for environment variable
        else:
            rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            url = None

        num_gpus = torch.cuda.device_count()
        gpu_id = rank % num_gpus
        torch.cuda.set_device(gpu_id)

        dist.init_process_group(backend, init_method=url, rank=rank, world_size=world_size)


def _slurm_init_distributed():
    def find_free_port():
        s = socket.socket()
        s.bind(('', 0))
        return s.getsockname()[1]

    rank = int(os.environ['SLURM_PROCID'])
    world_size = int(os.environ['SLURM_NPROCS'])
    job_id = os.environ['SLURM_JOBID']
    host_file = 'config/dist_url.' + job_id + '.txt'

    # for master
    if rank == 0:
        ip = socket.gethostbyname(socket.gethostname())
        port = find_free_port()
        dist_url = 'tcp://{}:{}'.format(ip, port)
        with open(host_file, 'w') as f:
            f.write(dist_url)

    else:
        while not os.path.exists(host_file):
            time.sleep(1)
        with open(host_file, 'r') as f:
            dist_url = f.read()

    return rank, world_size, dist_url


def get_rank():
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def get_world_size():
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def is_master():
    return get_rank() == 0


def master(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_master():
            return func(*args, **kwargs)
        else:
            return None
    return wrapper


def all_reduce(tensor, div=False):
    world_size = get_world_size()
    if world_size == 1:
        return tensor

    with torch.no_grad():
        dist.all_reduce(tensor)
        if div:
            tensor.div_(world_size)

    return tensor


def average_gradient(params):
    world_size = get_world_size()
    if world_size == 1:
        return

    for param in params:
        if param.requires_grad and param.grad is not None:
            dist.all_reduce(param.grad.data)


def barrier():
    dist.barrier()
