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


from .trainercore import trainercore

import tensorboardX


# max_steps = 5000
# base_lr = 0.003
peak_lr = 1.5
cycle_len = 0.8

def constant_lr(step):
    return 1.0

def decay_after_epoch(step):
    if step > self.args.iterations*cycle_len:
        return 0.1
    else:
        return 1.0

def lr_increase(step):

    # This function actually ignores the input and uses the global step variable
    # This allows it to get the learning rate correct after restore.

    # For this problem, the dataset size is 1e5.
    # So the epoch can be calculated easily:
    # epoch = (step * self.args.MINIBATCH_SIZE) / (1e5)

    base_lr   = self.args.learning_rate
    step_size = 5.0

    return 1.0 + step*step_size

    # # return 1.0 + max_lr

    # # Perform 500 warmup steps, gradually ramping the rate:
    # if epoch <= flat_warmup:
    #     return 1.0
    # elif epoch < flat_warmup + linear_warmup:
    #     return 1.0 + (target - 1) * (epoch - flat_warmup) / linear_warmup 
    # elif epoch <= flat_warmup + linear_warmup + full:
    #     return target
    # else:
    #     return target * numpy.exp(-0.001*(epoch-(full+linear_warmup+flat_warmup)))


def one_cycle_clr(step):
    
    peak = peak_lr / self.args.learning_rate
    
    cycle_steps  = int(self.args.iterations*cycle_len)
    end_steps = self.args.iterations - cycle_steps
    # Which cycle are we in?

    cycle = int(step / cycle_steps)
    intra_step = 1.0 * (step % cycle_steps)

    base_multiplier = 1.0

    if cycle < 1:
#         base_multiplier *= 0.5
        
        if intra_step > cycle_steps*0.5:
            intra_step = cycle_steps - intra_step

        value = intra_step * (peak) /(0.5*cycle_steps) 

    else:
        value = (intra_step / end_steps)*-1.0

    print ('using', base_multiplier + value)
    return base_multiplier + value

min_lr = {}
max_lr = {}
min_lr['2d'] = 0.0002
max_lr['2d'] = 0.0018
min_lr['3d'] = 0.0001
max_lr['3d'] = 0.0035

def triangle_clr(step):
    '''
    Implements the triangular cycle 
    learning rate schedule
    '''
    step_size = 100
    cycle = math.floor(1 + step / (2 * step_size))
    func = 1 - abs(step / step_size - 2 * cycle + 1)
    diff = max_lr[self.args.image_type] - min_lr[self.args.image_type]

    return (min_lr[self.args.image_type] + diff * max(0, func)) / self.args.learning_rate

def exp_range_clr(step, 
                  step_size = 100,
                  min_lr=min_lr[self.args.image_type], 
                  max_lr=max_lr[self.args.image_type], 
                  mode='exp_range', 
                  gamma=0.999):
    '''
    Implements the cyclical lr with exp decrease 
    learning rate schedule
    '''
    scale_func = 1
    if mode == 'exp_range':
        scale_func = gamma**step 

    max_lr *= scale_func

    if max_lr <= min_lr:
        max_lr = min_lr

    step_size = 100
    cycle = math.floor(1 + step / (2 * step_size))
    func = 1 - abs(step / step_size - 2 * cycle + 1)
    diff = max_lr - min_lr

    return (min_lr + diff * max(0, func)) / self.args.learning_rate



def exp_increase_lr(step):
  '''
  This function increases the learning rate exponentialy 
  from start_lr to end_lr. It can be used to study the loss 
  vs. learning rate and fins a proper interaval in which 
  to vary the learning rate.
  '''

  start_lr = self.args.learning_rate  # 1.e-7
  end_lr = self.args.learning_rate * 1.e8

  return math.exp(step * math.log(end_lr / start_lr) / self.args.iterations)

class distributed_trainer(trainercore):
    '''
    This class is the core interface for training.  Each function to
    be overridden for a particular interface is marked and raises
    a NotImplemented error.

    '''
    def __init__(self):
        # Rely on the base class for most standard parameters, only
        # search for parameters relevant for distributed computing here

        # Put the IO rank as the last rank in the COMM, since rank 0 does tf saves
        root_rank = hvd.size() - 1 

        if self.args.COMPUTE_MODE == "GPU":
            os.environ['CUDA_VISIBLE_DEVICES'] = str(hvd.local_rank())
            

        self._larcv_interface = queue_interface()#read_option='read_from_single_local_rank')
        self._iteration       = 0
        self._rank            = hvd.rank()
        self._cleanup         = []
        self._global_step     = torch.as_tensor(-1)

        if self._rank == 0:
            self.args.dump_config()
        # Make sure that 'learning_rate' and 'TRAINING'
        # are in net network parameters:


    def __del__(self):
        if hvd.rank() == 0:
            trainercore.__del__(self)

    def save_model(self):

        if hvd.rank() == 0:
            trainercore.save_model(self)
            

    def init_optimizer(self):

        # This takes the base optimizer (self._opt) and replaces
        # it with a distributed version

        trainercore.init_optimizer(self)

        if self.args.lr_schedule == '1cycle':
            self._lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self._opt, one_cycle_clr, last_epoch=-1)
        elif self.args.lr_schedule == 'triangle_clr':
            self._lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self._opt, triangle_clr, last_epoch=-1)
        elif self.args.lr_schedule == 'exp_range_clr':
            self._lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self._opt, exp_range_clr, last_epoch=-1)
        elif self.args.lr_schedule == 'decay':
            self._lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self._opt, decay_after_epoch, last_epoch=-1)
        elif self.args.lr_schedule == 'expincrease':
            self._lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self._opt, exp_increase_lr, last_epoch=-1)
        else:
            self._lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self._opt, constant_lr, last_epoch=-1)


        self._opt = hvd.DistributedOptimizer(self._opt, named_parameters=self._net.named_parameters())





    def init_saver(self):
        if hvd.rank() == 0:
            trainercore.init_saver(self)
        else:
            self._saver = None
            self._aux_saver = None


    def restore_model(self):
        if hvd.rank() == 0:
            state = trainercore.restore_model(self)

            if state is not None:
                self.load_state(state)
            else:
                self._global_step = torch.as_tensor(0)
        return


    def initialize(self, io_only = False):

        print("HVD rank: {}".format(hvd.rank()))

        self._initialize_io(color=self._rank)

        # print("Rank {}".format(hvd.rank()) + " Initialized IO")
        if io_only:
            return

        dims = self._larcv_interface.fetch_minibatch_dims('primary')
        # print("Rank {}".format(hvd.rank()) + " Recieved Dimensions")

        # This sets up the necessary output shape:
        if self.args.label_mode == 'split':
            output_shape = { key : dims[key] for key in self.args.keyword_label}
        else:
            output_shape = dims[self.args.keyword_label]

        self._net = self.args._net(output_shape)
        # print("Rank {}".format(hvd.rank()) + " Built network")


        if self.args.training: 
            self._net.train(True)



        if hvd.rank() == 0:
            n_trainable_parameters = 0
            for var in self._net.parameters():
                n_trainable_parameters += numpy.prod(var.shape)
            # print("Rank {}".format(hvd.rank()) + " Total number of trainable parameters in this network: {}".format(n_trainable_parameters))

        self.init_optimizer()
        # print("Rank {}".format(hvd.rank()) + " Initialized Optimizer")

        self.init_saver()
        # print("Rank {}".format(hvd.rank()) + " Initialized Saver")

        # If restoring, this will restore the model on the root node
        self.restore_model()
        # print("Rank {}".format(hvd.rank()) + " Restored Model if necessary")
        
        self._global_step = hvd.broadcast(self._global_step, root_rank = 0)

        # This is important to ensure LR continuity after restoring:
        # Step the learning rate scheduler up to the right amount
        if self._global_step > 0:
            i = 0
            while i < self._global_step:
                self._lr_scheduler.step()
                i += 1

        # Now broadcast the model to syncronize the optimizer and model:
        hvd.broadcast_parameters(self._net.state_dict(), root_rank = 0)
        hvd.broadcast_optimizer_state(self._opt, root_rank = 0)
        print("Rank {}".format(hvd.rank()) + " Parameters broadcasted")


        if self.args.compute_mode == "CPU":
            pass
        if self.args.compute_mode == "GPU":
            self._net.cuda()
            # This moves the optimizer to the GPU:
            for state in self._opt.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
            

        print("Rank ", hvd.rank(), next(self._net.parameters()).device)

        if self.args.label_mode == 'all':
            self._log_keys = ['loss', 'accuracy']
        elif self.args.label_mode == 'split':
            self._log_keys = ['loss']
            for key in self.args.keyword_label: 
                self._log_keys.append('acc/{}'.format(key))





    def summary(self, metrics, saver=""):
        if hvd.rank() == 0:
            trainercore.summary(self, metrics, saver)
        return
        
    def _compute_metrics(self, logits, minibatch_data, loss):
        # This function calls the parent function which computes local metrics.
        # Then, it performs an all reduce on all metrics:
        metrics = trainercore._compute_metrics(self, logits, minibatch_data, loss)


        for key in metrics:
            # print("All reducing ", key)
            metrics[key] = hvd.allreduce(metrics[key], name = key)

        return metrics

    def on_epoch_end(self):
        pass

    def on_step_end(self):
        self._lr_scheduler.step()
        # pass



    def to_torch(self, minibatch_data):

        # This function wraps the to-torch function but for a gpu forces

        device = self.get_device()

        minibatch_data = trainercore.to_torch(self, minibatch_data, device)

        return minibatch_data

    def log(self, metrics, saver=""):
        if hvd.rank() == 0:
            trainercore.log(self, metrics, saver)
