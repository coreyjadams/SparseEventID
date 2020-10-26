import os
import sys
import time
import math
from collections import OrderedDict
import socket


import numpy
import torch

from mpi4py import MPI

# Pytorch data parallel:
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

import horovod.torch as hvd

from larcv.distributed_queue_interface import queue_interface

from .torch_trainer import torch_trainer

import tensorboardX



class distributed_trainer(torch_trainer):
    '''
    This class is the core interface for training.  Each function to
    be overridden for a particular interface is marked and raises
    a NotImplemented error.

    '''
    def __init__(self, args):

        torch_trainer.__init__(self, args)
        # Rely on the base class for most standard parameters, only
        # search for parameters relevant for distributed computing here

        # Put the IO rank as the last rank in the COMM, since rank 0 does tf saves
        #
        # if self.args.compute_mode == "GPU":
        #     print(os.environ['CUDA_VISIBLE_DEVICES'])

        # We use MPI to pick the rank and world size:

        # Rank0

        self.rank = MPI.COMM_WORLD.Get_rank()
        world_size = MPI.COMM_WORLD.Get_size()


        if self.args.distributed_backend == "DDP":
            init_method = f'file:///home/cadams/.torch_ddp_init'
            dist.init_process_group("nccl", init_method=init_method,rank=self.rank, world_size=world_size)

        # if self.rank == 0:
        #     hostname = socket.gethostname()
        #     IPAddr = socket.gethostbyname(hostname)


    def print(self, *argv):
        if self.rank == 0:
            torch_trainer.print(self, *argv)


    def save_model(self):

        if self.rank == 0:
            torch_trainer.save_model(self)


    def init_optimizer(self):

        # This takes the base optimizer (self._opt) and replaces
        # it with a distributed version

        torch_trainer.init_optimizer(self)

        if self.args.distributed_backend == "horovod":
            self._opt = hvd.DistributedOptimizer(self._opt, named_parameters=self._net.named_parameters())

    def init_network(self):

        torch_trainer.init_network(self)
        if self.args.distributed_backend == "DDP":
            self._net = DDP(self._net)

    def init_saver(self):
        if self.rank == 0:
            torch_trainer.init_saver(self)
        else:
            self._saver = None
            self._aux_saver = None


    def restore_model(self):

        if self.rank == 0:
            # On the root rank, load the model:
            state = torch_trainer.restore_model(self)
        else:
            state = {}

        # Here, we need to broad cast the entire state:
        state = hvd.broadcast_object(state, root_rank = 0)

        return state

    def summary(self, metrics, saver=""):
        if self.rank == 0:
            torch_trainer.summary(self, metrics, saver)
        return

    def _compute_metrics(self, logits, minibatch_data, loss):
        # This function calls the parent function which computes local metrics.
        # Then, it performs an all reduce on all metrics:
        metrics = torch_trainer._compute_metrics(self, logits, minibatch_data, loss)


        for key in metrics:
            # self.print("All reducing ", key)
            if self.args.distributed_backend == "horovod":
                metrics[key] = hvd.allreduce(metrics[key], name = key)
            else:
                pass

        return metrics



    def log(self, metrics, saver=""):
        if self.rank == 0:
            torch_trainer.log(self, metrics, saver)
