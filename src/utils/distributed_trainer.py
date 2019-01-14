import os
import sys
import time
from collections import OrderedDict

import numpy

import torch
import horovod.torch as hvd
hvd.init()


from larcv.distributed_larcv_interface import larcv_interface


from . import flags
# from . import data_transforms
FLAGS = flags.FLAGS()

from .trainercore import trainercore

import tensorboardX


def lambda_warmup(epoch):
    # Constant terms:
    flat_warmup = 50
    linear_warmup = 500
    full = 3000
    size=hvd.size()
    target = numpy.sqrt(size)
    # Perform 500 warmup steps, gradually ramping the rate:
    if epoch <= flat_warmup:
        return 1.0
    elif epoch < flat_warmup + linear_warmup:
        return 1.0 + (target - 1) * (epoch - flat_warmup) / linear_warmup 
    elif epoch <= flat_warmup + linear_warmup + full:
        return target
    else:
        return target * numpy.exp(-0.001*(epoch-(full+linear_warmup+flat_warmup)))


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

        if FLAGS.COMPUTE_MODE == "GPU":
            os.environ['CUDA_VISIBLE_DEVICES'] = str(hvd.local_rank())
            

        self._larcv_interface = larcv_interface(root=root_rank)
        self._iteration       = 0
        self._rank            = hvd.rank()

        if self._rank == 0:
            FLAGS.dump_config()
        # Make sure that 'LEARNING_RATE' and 'TRAINING'
        # are in net network parameters:


    def save_model(self):

        if hvd.rank() == 0:
            trainercore.save_model(self)
            

    def init_optimizer(self):

        # This takes the base optimizer (self._opt) and replaces
        # it with a distributed version

        trainercore.init_optimizer(self)


        self._lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self._opt, lambda_warmup, last_epoch=-1)

        self._opt = hvd.DistributedOptimizer(self._opt, named_parameters=self._net.named_parameters())



    def init_saver(self):
        if hvd.rank() == 0:
            trainercore.init_saver(self)
        else:
            self._saver = None
            self._aux_saver = None


    def initialize(self, io_only = False):

        print("HVD rank: {}".format(hvd.rank()))

        self._initialize_io()


        if io_only:
            return

        dims = self._larcv_interface.fetch_minibatch_dims('primary')

        # This sets up the necessary output shape:
        if FLAGS.LABEL_MODE == 'split':
            output_shape = { key : dims[key] for key in FLAGS.KEYWORD_LABEL}
        else:
            output_shape = dims[FLAGS.KEYWORD_LABEL]


        self._net = FLAGS._net(output_shape)


        if FLAGS.TRAINING: 
            self._net.train(True)



        if hvd.rank() == 0:
            n_trainable_parameters = 0
            for var in self._net.parameters():
                n_trainable_parameters += numpy.prod(var.shape)
            print("Total number of trainable parameters in this network: {}".format(n_trainable_parameters))

        self.init_optimizer()

        self.init_saver()
        # hooks = self.get_standard_hooks()


        # Here, either restore the weights of the network or initialize it:
        self._global_step = 0
        # Restore the state from the root rank:
        # if hvd.rank() == 0:
        #     self.restore_model()

        # Broadcast the state of the model:
        # print(self._net.state_dict().keys())
        # print(self._opt.state_dict()['state'])
        hvd.broadcast_parameters(self._net.state_dict(), root_rank = 0)
        # hvd.broadcast_parameters(self._global_step, root_rank = 0)

        # Broadcast the state of the optimizer?

        if FLAGS.COMPUTE_MODE == "CPU":
            pass
        if FLAGS.COMPUTE_MODE == "GPU":
            self._net.cuda()
            # self._opt.cuda()
            
        hvd.broadcast_optimizer_state(self._opt, root_rank = 0)

        print("Rank ", hvd.rank(), next(self._net.parameters()).device)

        if FLAGS.LABEL_MODE == 'all':
            self._log_keys = ['loss', 'accuracy']
        elif FLAGS.LABEL_MODE == 'split':
            self._log_keys = ['loss']
            for key in FLAGS.KEYWORD_LABEL_SPLIT: 
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


    def to_torch(self, minibatch_data):

        # This function wraps the to-torch function but for a gpu forces
        # the right device
        if FLAGS.COMPUTE_MODE == 'GPU':
            device = torch.device('cuda')
            # device = torch.device('cuda:{}'.format(hvd.local_rank()))
        else:
            device = None
        minibatch_data = trainercore.to_torch(self, minibatch_data, device)

        return minibatch_data

    def log(self, metrics, saver=""):
        if hvd.rank() == 0:
            trainercore.log(self, metrics, saver)
