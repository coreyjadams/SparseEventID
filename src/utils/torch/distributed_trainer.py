import os
import sys
import time
from collections import OrderedDict

import numpy

import torch


# Always want mpi, but horovod is imported below.
from mpi4py import MPI
comm = MPI.COMM_WORLD



from . trainer import trainer

class distributed_trainer(trainer):
    '''
    This class is the core interface for training.  Each function to
    be overridden for a particular interface is marked and raises
    a NotImplemented error.

    '''
    def __init__(self, args):

        trainer.__init__(self, args)
        # Rely on the base class for most standard parameters, only
        # search for parameters relevant for distributed computing here


        # Put the IO rank as the last rank in the COMM, since rank 0 does tf saves
        if self.args.framework.distributed_mode == "horovod":
            global hvd
            import horovod.torch as hvd
            hvd.init()
            self._rank            = hvd.rank()
            self._local_rank      = hvd.local_rank()

            # If we're using the sparse network, we try setting CUDA_VISIBLE_DEVICES
            if self.args.network.data_format == "sparse":
                os.environ['CUDA_VISIBLE_DEVICES'] = str(self._local_rank)

        else:

            import socket
            import torch.distributed as dist
            from torch.nn.parallel import DistributedDataParallel as DDP

            rank = MPI.COMM_WORLD.Get_rank()


            # Pytorch will look for these:
            local_rank = os.environ['OMPI_COMM_WORLD_LOCAL_RANK']
            size = MPI.COMM_WORLD.Get_size()
            rank = MPI.COMM_WORLD.Get_rank()

            # If we're using the sparse network, we try setting CUDA_VISIBLE_DEVICES
            if self.args.network.data_format == "sparse":
                os.environ['CUDA_VISIBLE_DEVICES'] = str(local_rank)

            os.environ["RANK"] = str(rank)
            os.environ["WORLD_SIZE"] = str(size)

            self._rank = rank
            self._size = size
            self._local_rank = local_rank

            # It will want the master address too, which we'll broadcast:
            if rank == 0:
                master_addr = socket.gethostname()
            else:
                master_addr = None

            master_addr = MPI.COMM_WORLD.bcast(master_addr, root=0)
            os.environ["MASTER_ADDR"] = master_addr
            os.environ["MASTER_PORT"] = str(2345)

            # What backend?  nccl on GPU, gloo on CPU
            if self.args.run.compute_mode == "GPU": backend = 'nccl'
            elif self.args.run.compute_mode == "CPU": backend = 'gloo'

            torch.distributed.init_process_group(
                backend=backend, init_method='env://')


    def save_model(self):

        if self._rank == 0:
            trainer.save_model(self)


    def default_device_context(self):
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            return trainer.default_device_context(self)

        # Convert the input data to torch tensors
        if self.args.run.compute_mode == "GPU":
            return torch.cuda.device(int(self._local_rank))
        elif self.args.run.compute_mode == "XPU":
            return contextlib.nullcontext
            # device = torch.device("xpu")
        elif self.args.run.compute_mode == "DPCPP":
            return contextlib.nullcontext
            # device = torch.device("dpcpp")
        else:
            return contextlib.nullcontext
            # device = torch.device('cpu')

    def default_device(self):
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            return trainer.default_device(self)

        if self.args.run.compute_mode == "GPU":
            return torch.device(f"cuda:{int(self._local_rank)}")
        elif self.args.run.compute_mode == "XPU":
            device = torch.device("xpu")
        elif self.args.run.compute_mode == "DPCPP":
            device = torch.device("dpcpp")
        else:
            device = torch.device('cpu')

    def init_optimizer(self):

        # This takes the base optimizer (self._opt) and replaces
        # it with a distributed version

        trainer.init_optimizer(self)

        if self.args.framework.distributed_mode == "horovod":
            self._opt = hvd.DistributedOptimizer(self._opt, named_parameters=self._net.named_parameters())

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self._opt, self.lr_calculator, last_epoch=-1)


    def init_saver(self):
        if self._rank == 0:
            trainer.init_saver(self)
        else:
            self._savers = {}


    def print_network_info(self):
        if self._rank == 0:
            trainer.print_network_info(self)
        return

    def restore_model(self):

        if self._rank == 0:
            state = self.load_state_from_file()
        else:
            state = None

        if state is not None and self._rank == 0:
            self.restore_state(state)



        if self.args.framework.distributed_mode == "horovod":

            # Broadcast the global step:
            self._global_step = hvd.broadcast_object(self._global_step, root_rank = 0)

            # Broadcast the state of the model:
            hvd.broadcast_parameters(self._net.state_dict(), root_rank = 0)

            # Broadcast the optimizer state:
            hvd.broadcast_optimizer_state(self._opt, root_rank = 0)

            device = self.default_device()
            # Horovod doesn't actually move the optimizer onto a GPU:
            if self.args.run.compute_mode == "GPU":
                for state in self._opt.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.to(device)



            # Broadcast the LR Schedule state:
            state_dict = hvd.broadcast_object(self.lr_scheduler.state_dict(), root_rank = 0)

        elif self.args.framework.distributed_mode == "DDP":


            self._net = torch.nn.parallel.DistributedDataParallel(self._net)



            self._global_step = MPI.COMM_WORLD.bcast(self._global_step, root=0)
            state_dict = MPI.COMM_WORLD.bcast(self.lr_scheduler.state_dict(), root=0)

        # Load the state dict:
        self.lr_scheduler.load_state_dict(state_dict)

    def summary(self, metrics, kind):

        if self._rank == 0:
            trainer.summary(self, metrics, kind)
        return


    def _compute_metrics(self, logits, minibatch_data, loss):

        # This function calls the parent function which computes local metrics.
        # Then, it performs an all reduce on all metrics:
        metrics = trainer._compute_metrics(self, logits, minibatch_data, loss)

        if self.args.framework.distributed_mode == "horovod":
            for key in metrics:
                metrics[key] = hvd.allreduce(metrics[key], name = key)
        elif self.args.framework.distributed_mode == "DDP":
            for key in metrics:
                torch.distributed.all_reduce(metrics[key])
                metrics[key] /= self._size

        return metrics



    def log(self, metrics, kind):

        if self._rank == 0:
            trainer.log(self, metrics, kind)
