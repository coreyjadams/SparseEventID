import os
import sys
import time
import math
from collections import OrderedDict

import numpy

import torch
import horovod.torch as hvd
hvd.init()


from larcv.distributed_queue_interface import queue_interface


from .torch_trainer import torch_trainer

import tensorboardX
#
#
# # max_steps = 5000
# # base_lr = 0.003
# peak_lr = 1.5
# cycle_len = 0.8
#
# def constant_lr(step):
#     return 1.0
#
# def decay_after_epoch(step):
#     if step > self.args.iterations*cycle_len:
#         return 0.1
#     else:
#         return 1.0
#
# def lr_increase(step):
#
#     # This function actually ignores the input and uses the global step variable
#     # This allows it to get the learning rate correct after restore.
#
#     # For this problem, the dataset size is 1e5.
#     # So the epoch can be calculated easily:
#     # epoch = (step * self.args.MINIBATCH_SIZE) / (1e5)
#
#     base_lr   = self.args.learning_rate
#     step_size = 5.0
#
#     return 1.0 + step*step_size
#
#     # # return 1.0 + max_lr
#
#     # # Perform 500 warmup steps, gradually ramping the rate:
#     # if epoch <= flat_warmup:
#     #     return 1.0
#     # elif epoch < flat_warmup + linear_warmup:
#     #     return 1.0 + (target - 1) * (epoch - flat_warmup) / linear_warmup
#     # elif epoch <= flat_warmup + linear_warmup + full:
#     #     return target
#     # else:
#     #     return target * numpy.exp(-0.001*(epoch-(full+linear_warmup+flat_warmup)))
#
#
# def one_cycle_clr(step):
#
#     peak = peak_lr / self.args.learning_rate
#
#     cycle_steps  = int(self.args.iterations*cycle_len)
#     end_steps = self.args.iterations - cycle_steps
#     # Which cycle are we in?
#
#     cycle = int(step / cycle_steps)
#     intra_step = 1.0 * (step % cycle_steps)
#
#     base_multiplier = 1.0
#
#     if cycle < 1:
# #         base_multiplier *= 0.5
#
#         if intra_step > cycle_steps*0.5:
#             intra_step = cycle_steps - intra_step
#
#         value = intra_step * (peak) /(0.5*cycle_steps)
#
#     else:
#         value = (intra_step / end_steps)*-1.0
#
#     print ('using', base_multiplier + value)
#     return base_multiplier + value
#
# min_lr = {}
# max_lr = {}
# min_lr['2d'] = 0.0002
# max_lr['2d'] = 0.0018
# min_lr['3d'] = 0.0001
# max_lr['3d'] = 0.0035
#
# def triangle_clr(step):
#     '''
#     Implements the triangular cycle
#     learning rate schedule
#     '''
#     step_size = 100
#     cycle = math.floor(1 + step / (2 * step_size))
#     func = 1 - abs(step / step_size - 2 * cycle + 1)
#     diff = max_lr[self.args.image_type] - min_lr[self.args.image_type]
#
#     return (min_lr[self.args.image_type] + diff * max(0, func)) / self.args.learning_rate
#
# def exp_range_clr(step,
#                   step_size = 100,
#                   min_lr=min_lr[self.args.image_type],
#                   max_lr=max_lr[self.args.image_type],
#                   mode='exp_range',
#                   gamma=0.999):
#     '''
#     Implements the cyclical lr with exp decrease
#     learning rate schedule
#     '''
#     scale_func = 1
#     if mode == 'exp_range':
#         scale_func = gamma**step
#
#     max_lr *= scale_func
#
#     if max_lr <= min_lr:
#         max_lr = min_lr
#
#     step_size = 100
#     cycle = math.floor(1 + step / (2 * step_size))
#     func = 1 - abs(step / step_size - 2 * cycle + 1)
#     diff = max_lr - min_lr
#
#     return (min_lr + diff * max(0, func)) / self.args.learning_rate
#
#
#
# def exp_increase_lr(step):
#   '''
#   This function increases the learning rate exponentialy
#   from start_lr to end_lr. It can be used to study the loss
#   vs. learning rate and fins a proper interaval in which
#   to vary the learning rate.
#   '''
#
#   start_lr = self.args.learning_rate  # 1.e-7
#   end_lr = self.args.learning_rate * 1.e8
#
#   return math.exp(step * math.log(end_lr / start_lr) / self.args.iterations)

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

        # if self.args.lr_schedule == '1cycle':
        #     self._lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        #         self._opt, one_cycle_clr, last_epoch=-1)
        # elif self.args.lr_schedule == 'triangle_clr':
        #     self._lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        #         self._opt, triangle_clr, last_epoch=-1)
        # elif self.args.lr_schedule == 'exp_range_clr':
        #     self._lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        #         self._opt, exp_range_clr, last_epoch=-1)
        # elif self.args.lr_schedule == 'decay':
        #     self._lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        #         self._opt, decay_after_epoch, last_epoch=-1)
        # elif self.args.lr_schedule == 'expincrease':
        #     self._lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        #         self._opt, exp_increase_lr, last_epoch=-1)
        # else:
        #     self._lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        #         self._opt, constant_lr, last_epoch=-1)


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
