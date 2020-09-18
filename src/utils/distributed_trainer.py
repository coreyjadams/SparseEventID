import os
import sys
import time
import math
from collections import OrderedDict

import numpy

import horovod.torch as hvd
hvd.init()
os.environ["CUDA_VISIBLE_DEVICES"] = str(hvd.local_rank())
print("CUDA_VISIBLE_DEVICES: ", os.environ["CUDA_VISIBLE_DEVICES"])
import torch


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
        root_rank = hvd.size() - 1

        if self.args.compute_mode == "GPU":
            os.environ['CUDA_VISIBLE_DEVICES'] = str(hvd.local_rank())
            print(os.environ['CUDA_VISIBLE_DEVICES'])

        self._rank            = hvd.rank()

    def print(self, *argv):
        if self._rank == 0:
            torch_trainer.print(self, *argv)


    def save_model(self):

        if hvd.rank() == 0:
            torch_trainer.save_model(self)


    def init_optimizer(self):

        # This takes the base optimizer (self._opt) and replaces
        # it with a distributed version

        torch_trainer.init_optimizer(self)

        self._opt = hvd.DistributedOptimizer(self._opt, named_parameters=self._net.named_parameters())



    def init_saver(self):
        if hvd.rank() == 0:
            torch_trainer.init_saver(self)
        else:
            self._saver = None
            self._aux_saver = None


    def load_state(self):
        # Load the state on rank 0:
        if state is not None and hvd.rank() == 0:
            self.load_state(state)


        # Broadcast the global step:
        self._global_step = hvd.broadcast_object(self._global_step, root_rank = 0)

        # Broadcast the state of the model:
        hvd.broadcast_parameters(self._net.state_dict(), root_rank = 0)

        # Broadcast the optimizer state:
        hvd.broadcast_optimizer_state(self._opt, root_rank = 0)

        # Horovod doesn't actually move the optimizer onto a GPU:
        if self.args.compute_mode == "GPU":
            for state in self._opt.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        # Broadcast the LR Schedule state:
        state_dict = hvd.broadcast_object(self.lr_scheduler.state_dict(), root_rank = 0)
        self.lr_scheduler.load_state_dict(state_dict)

    def restore_model(self):

        if hvd.rank() == 0:
            # On the root rank, load the model:
            state = torch_trainer.restore_model(self)
        else:
            state = None


    def summary(self, metrics, saver=""):
        if hvd.rank() == 0:
            torch_trainer.summary(self, metrics, saver)
        return

    def _compute_metrics(self, logits, minibatch_data, loss):
        # This function calls the parent function which computes local metrics.
        # Then, it performs an all reduce on all metrics:
        metrics = torch_trainer._compute_metrics(self, logits, minibatch_data, loss)


        for key in metrics:
            # self.print("All reducing ", key)
            metrics[key] = hvd.allreduce(metrics[key], name = key)

        return metrics



    def log(self, metrics, saver=""):
        if hvd.rank() == 0:
            torch_trainer.log(self, metrics, saver)
