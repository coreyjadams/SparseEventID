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


        # Make sure that 'LEARNING_RATE' and 'TRAINING'
        # are in net network parameters:


    def save_model(self):

        if hvd.rank() == 0:
            trainercore.save_model(self)
            


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


        # Create an optimizer:
        if FLAGS.LEARNING_RATE <= 0:
            self._opt = torch.optim.Adam(self._net.parameters())
        else:
            self._opt = torch.optim.Adam(self._net.parameters(), FLAGS.LEARNING_RATE)

        self._opt = hvd.DistributedOptimizer(self._opt, named_parameters=self._net.named_parameters())


        # self._train_op = opt.minimize(self._loss, self._global_step)
        self._criterion = torch.nn.CrossEntropyLoss()

        # hooks = self.get_standard_hooks()


        # This sets up the summary saver:
        if hvd.rank() == 0:
            self._saver = tensorboardX.SummaryWriter(FLAGS.LOG_DIRECTORY)
        else:
            self._saver = None
        # This code is supposed to add the graph definition.
        # It doesn't currently work
        # temp_dims = list(dims['image'])
        # temp_dims[0] = 1
        # dummy_input = torch.randn(size=tuple(temp_dims), requires_grad=True)
        # self._saver.add_graph(self._net, (dummy_input,))

        # Here, either restore the weights of the network or initialize it:
        self._global_step = 0
        # Restore the state from the root rank:
        if hvd.rank() == 0:
            self.restore_model()

        # Broadcast the state of the model:
        hvd.broadcast_parameters(self._net.state_dict(), root_rank = 0)

        if FLAGS.COMPUTE_MODE == "CPU":
            pass
        if FLAGS.COMPUTE_MODE == "GPU":
            self._net.cuda()

        print("Rank ", hvd.rank(), next(self._net.parameters()).device)

        if FLAGS.LABEL_MODE == 'all':
            self._log_keys = ['loss', 'accuracy']
        elif FLAGS.LABEL_MODE == 'split':
            self._log_keys = ['loss']
            for key in FLAGS.KEYWORD_LABEL_SPLIT: 
                self._log_keys.append('acc_{}'.format(key))





    def summary(self, metrics, level=""):
        if hvd.rank() == 0:
            trainercore.summary(self, metrics, level)
        return
        
    def _compute_metrics(self, logits, minibatch_data, loss):
        # This function calls the parent function which computes local metrics.
        # Then, it performs an all reduce on all metrics:
        metrics = trainercore._compute_metrics(self, logits, minibatch_data, loss)


        for key in metrics:
            # print("All reducing ", key)
            metrics[key] = hvd.allreduce(metrics[key], name = key)

        return metrics

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

    def log(self, metrics, level=""):
        if hvd.rank() == 0:
            trainercore.log(self, metrics, level)
