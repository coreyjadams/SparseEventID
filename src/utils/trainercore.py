import os
import sys
import time
from collections import OrderedDict

import numpy

import torch

from larcv import larcv_interface

from . import flags
from . import data_transforms
FLAGS = flags.FLAGS()

import datetime

# This uses tensorboardX to save summaries and metrics to tensorboard compatible files.

import tensorboardX

class trainercore(object):
    '''
    This class is the core interface for training.  Each function to
    be overridden for a particular interface is marked and raises
    a NotImplemented error.

    '''
    def __init__(self,):
        self._larcv_interface = larcv_interface.larcv_interface()
        self._iteration       = 0
        self._global_step     = -1

    def _initialize_io(self):

        # Prepare data managers:
        io_config = {
            'filler_name' : FLAGS.FILLER,
            'filler_cfg'  : FLAGS.FILE,
            'verbosity'   : FLAGS.VERBOSITY,
            'make_copy'   : True
        }

        if FLAGS.LABEL_MODE == 'all':

            data_keys = OrderedDict({
                'image': FLAGS.KEYWORD_DATA, 
                'label': FLAGS.KEYWORD_LABEL
                })
        else:
            data_keys = OrderedDict({
                'image': FLAGS.KEYWORD_DATA,
                })
            for label in FLAGS.KEYWORD_LABEL:
                data_keys.update({label : label})


        self._larcv_interface.prepare_manager('primary', io_config, FLAGS.MINIBATCH_SIZE, data_keys)

        # All of the additional tools are in case there is a test set up:
        if FLAGS.AUX_FILE is not None:

            io_config = {
                'filler_name' : FLAGS.AUX_FILLER,
                'filler_cfg'  : FLAGS.AUX_FILE,
                'verbosity'   : FLAGS.VERBOSITY,
                'make_copy'   : True
            }

            if FLAGS.LABEL_MODE == 'all':

                data_keys = OrderedDict({
                    'image': FLAGS.AUX_KEYWORD_DATA, 
                    'label': FLAGS.AUX_KEYWORD_LABEL
                    })
            else:
                data_keys = OrderedDict({
                    'image': FLAGS.AUX_KEYWORD_DATA,
                    })
                for label in FLAGS.AUX_KEYWORD_LABEL:
                    data_keys.update({label : label})


            self._larcv_interface.prepare_manager('aux', io_config, FLAGS.AUX_MINIBATCH_SIZE, data_keys)


    def initialize(self, io_only=False):

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



        n_trainable_parameters = 0
        for var in self._net.parameters():
            n_trainable_parameters += numpy.prod(var.shape)
        print("Total number of trainable parameters in this network: {}".format(n_trainable_parameters))

        # self._construct_graph()

        # Create an optimizer:
        if FLAGS.LEARNING_RATE <= 0:
            self._opt = torch.optim.Adam(self._net.parameters())
        else:
            self._opt = torch.optim.Adam(self._net.parameters(), FLAGS.LEARNING_RATE)



        # self._train_op = opt.minimize(self._loss, self._global_step)
        self._criterion = torch.nn.CrossEntropyLoss()

        # hooks = self.get_standard_hooks()


        # This sets up the summary saver:
        self._saver = tensorboardX.SummaryWriter(FLAGS.LOG_DIRECTORY)

        # This code is supposed to add the graph definition.
        # It doesn't currently work
        # temp_dims = list(dims['image'])
        # temp_dims[0] = 1
        # dummy_input = torch.randn(size=tuple(temp_dims), requires_grad=True)
        # self._saver.add_graph(self._net, (dummy_input,))

        # Here, either restore the weights of the network or initialize it:
        self._global_step = 0
        self.restore_model()


        if FLAGS.COMPUTE_MODE == "CPU":
            pass
        if FLAGS.COMPUTE_MODE == "GPU":
            self._net.cuda()


        self._log_keys = ['loss', 'accuracy']



    def restore_model(self):
        ''' This function attempts to restore the model from file
        '''

        _, checkpoint_file_path = self.get_model_filepath()

        if not os.path.isfile(checkpoint_file_path):
            return
        # Parse the checkpoint file and use that to get the latest file path

        with open(checkpoint_file_path, 'r') as _ckp:
            for line in _ckp.readlines():
                if line.startswith("latest: "):
                    chkp_file = line.replace("latest: ", "").rstrip('\n')
                    chkp_file = os.path.dirname(checkpoint_file_path) + "/" + chkp_file
                    print("Restoring weights from ", chkp_file)
                    break

        state = torch.load(chkp_file)

        self._net.load_state_dict(state['state_dict'])
        self._opt.load_state_dict(state['optimizer'])
        self._global_step = state['global_step']

        return

    def save_model(self):
        '''Save the model to file
        
        '''

        current_file_path, checkpoint_file_path = self.get_model_filepath()

        # save the model state into the file path:
        state_dict = {
            'global_step' : self._global_step,
            'state_dict'  : self._net.state_dict(),
            'optimizer'   : self._opt.state_dict()
        }

        # Make sure the path actually exists:
        if not os.path.isdir(os.path.dirname(current_file_path)):
            os.makedirs(os.path.dirname(current_file_path))

        torch.save(state_dict, current_file_path)

        # Parse the checkpoint file to see what the last checkpoints were:

        # Keep only the last 5 checkpoints
        n_keep = 5


        past_checkpoint_files = {}
        try:
            with open(checkpoint_file_path, 'r') as _chkpt:
                for line in _chkpt.readlines():
                    line = line.rstrip('\n')
                    vals = line.split(":")
                    if vals[0] != 'latest':
                        past_checkpoint_files.update({int(vals[0]) : vals[1].replace(' ', '')})
        except:
            pass
        

        # Remove the oldest checkpoints while the number is greater than n_keep
        while len(past_checkpoint_files) >= n_keep:
            min_index = min(past_checkpoint_files.keys())
            file_to_remove = os.path.dirname(checkpoint_file_path) + "/" + past_checkpoint_files[min_index]
            os.remove(file_to_remove)
            past_checkpoint_files.pop(min_index)



        # Update the checkpoint file
        with open(checkpoint_file_path, 'w') as _chkpt:
            _chkpt.write('latest: {}\n'.format(os.path.basename(current_file_path)))
            _chkpt.write('{}: {}\n'.format(self._global_step, os.path.basename(current_file_path)))
            for key in past_checkpoint_files:
                _chkpt.write('{}: {}\n'.format(key, past_checkpoint_files[key]))


    def get_model_filepath(self):
        '''Helper function to build the filepath of a model for saving and restoring:
        
        
        '''

        # Find the base path of the log directory
        file_path= FLAGS.LOG_DIRECTORY  + "/checkpoints/"


        name = file_path + 'model-{}.ckpt'.format(self._global_step)
        checkpoint_file_path = file_path + "checkpoint"

        return name, checkpoint_file_path

    def _create_softmax(self, logits):
        '''Must return a dict type

        [description]

        Arguments:
            logits {[type]} -- [description]

        Raises:
            NotImplementedError -- [description]
        '''

        # For the logits, we compute the softmax and the predicted label


        output = dict()

        # Take the logits (which are one per plane) and create a softmax and prediction (one per plane)

        output['softmax'] = nn.softmax(logits)
        output['prediction'] = nn.argmax(logits, axis=1)


        return output



    def _calculate_loss(self, inputs, logits):
        ''' Calculate the loss.

        returns a single scalar for the optimizer to use.
        '''



        if FLAGS.LABEL_MODE == 'all':
            values, target = torch.max(inputs[FLAGS.KEYWORD_LABEL], dim = 1)
            loss = self._criterion(logits, target=target)
            return loss


    def _calculate_accuracy(self, logits, minibatch_data):
        ''' Calculate the accuracy.

        '''

        # Compare how often the input label and the output prediction agree:

        values, indices = torch.max(minibatch_data[FLAGS.KEYWORD_LABEL], dim = 1)
        values, predict = torch.max(logits, dim=1)
        correct_prediction = torch.eq(predict,indices)
        accuracy = torch.mean(correct_prediction.float())

        return accuracy




    # def image_to_point_cloud(self, image):

    #     # This function maps an image to a point cloud


    #     # Have to do this image by image to get the right shape:
    #     outputs = []
    #     max_points= 0
    #     for b in range(image.shape[0]):

    #         non_zero_locs = list(numpy.where(image[b] != 0))
    #         # Calculate the values and put them last:
    #         values = image[b][tuple(non_zero_locs)]
    #         if values.shape[0] > max_points:
    #             max_points = values.shape[0]
    #         # Merge everything into an N by 2 array:
    #         non_zero_locs.append(values)
    #         point_dim_len = len(non_zero_locs)
    #         # Stack everythin together and take the transpose:
    #         res = numpy.stack((non_zero_locs)).T
    #         outputs.append(res)

    #     # Stack the results into a uniform numpy array padded with zeros:
    #     output = numpy.zeros([len(outputs), max_points, point_dim_len])
    #     for i, o in enumerate(outputs):
    #         npoints = len(o)
    #         output[i,0:npoints,:] = o

    #     return output

    def _compute_metrics(self, logits, minibatch_data, loss):

        # Call all of the functions in the metrics dictionary:
        metrics = {}

        metrics['loss']     = loss.data
        metrics['accuracy'] = self._calculate_accuracy(logits, minibatch_data)

        return metrics

    def log(self, metrics):


        if self._global_step % FLAGS.LOGGING_ITERATION == 0:
            
            self._current_log_time = datetime.datetime.now()

            # Build up a string for logging:
            if self._log_keys != []:
                s = ", ".join(["{0}: {1:.4}".format(key, metrics[key]) for key in self._log_keys])
            else:
                s = ", ".join(["{0}: {1:.4}".format(key, metrics[key]) for key in metrics])
      
            try:
                s += " ({} s)".format((self._current_log_time - self._previous_log_time).total_seconds())
            except:
                pass

            self._previous_log_time = self._current_log_time

            print("Global Step {} metrics: {}".format(self._global_step, s))



    def summary(self, metrics):

        if self._global_step % FLAGS.SUMMARY_ITERATION == 0:
            for metric in metrics:
                self._saver.add_scalar(metric, metrics[metric], self._global_step)

            pass

    def fetch_next_batch(self, mode='primary'):

        minibatch_data = self._larcv_interface.fetch_minibatch_data(mode)
        minibatch_dims = self._larcv_interface.fetch_minibatch_dims(mode)


        for key in minibatch_data:
            minibatch_data[key] = numpy.reshape(minibatch_data[key], minibatch_dims[key])

        # Here, do some massaging to convert the input data to another format, if necessary:
        if FLAGS.IMAGE_MODE == 'dense' and not FLAGS.SPARSE:
            # Don't have to do anything here
            pass
        elif FLAGS.IMAGE_MODE == 'dense' and FLAGS.SPARSE:
            # Have to convert the input image from dense to sparse format:
            if '3d' in FLAGS.FILE:
                minibatch_data['image'] = data_transforms.larcvdense_to_scnsparse_3d(minibatch_data['image'])
            else:
                minibatch_data['image'] = data_transforms.larcvdense_to_scnsparse_2d(minibatch_data['image'])
        elif FLAGS.IMAGE_MODE == 'sparse' and not FLAGS.SPARSE:
            # Need to convert sparse larcv into a dense numpy array:
            minibatch_data['image'] = data_transforms.larcvsparse_to_dense(minibatch_data['image'])
        elif FLAGS.IMAGE_MODE == 'sparse' and FLAGS.SPARSE:
            if '3d' in FLAGS.FILE:
                minibatch_data['image'] = data_transforms.larcvsparse_to_scnsparse_3d(minibatch_data['image'])
            else:
                minibatch_data['image'] = data_transforms.larcvsparse_to_scnsparse_2d(minibatch_data['image'])

        return minibatch_data

    def increment_global_step(self):

        self._global_step += 1

    def train_step(self):


        # For a train step, we fetch data, run a forward and backward pass, and
        # if this is a logging step, we compute some logging metrics.


        global_start_time = datetime.datetime.now()

        # Reset the gradient values for this step:
        self._opt.zero_grad()

        # Fetch the next batch of data with larcv
        io_start_time = datetime.datetime.now()
        minibatch_data = self.fetch_next_batch()
        io_end_time = datetime.datetime.now()

        # Convert the input data to torch tensors
        if FLAGS.COMPUTE_MODE == "GPU":
            device = torch.device('cuda')
            minibatch_data = {key : torch.tensor(minibatch_data[key],device=device) for key in minibatch_data }
        else:
            minibatch_data = {key : torch.tensor(minibatch_data[key]) for key in minibatch_data }



        # Run a forward pass of the model on the input image:
        logits = self._net(minibatch_data['image'])


        # Compute the loss based on the logits
        loss = self._calculate_loss(minibatch_data, logits)

        # Compute the gradients for the network parameters:
        loss.backward()

        # Compute any necessary metrics:
        metrics = self._compute_metrics(logits, minibatch_data, loss)
        
        self.log(metrics) 

        # Add the global step / second to the tensorboard log:
        try:
            metrics['global_step/sec'] = self._global_step_per_second
        except:
            metrics['global_step/sec'] = 0.0

        metrics['io_fetch_time'] = (io_end_time - io_start_time).total_seconds()



        self.summary(metrics)       

        # Apply the parameter update:
        self._opt.step()
        global_end_time = datetime.datetime.now()

        # Compute global step per second:
        self._global_step_per_second = (global_end_time - global_start_time).total_seconds()

        # Increment the global step value:
        self.increment_global_step()

        return metrics

    def val_step(self, n_iterations=1):
        # perform a validation step
        # Validation steps can optionally accumulate over several minibatches, to
        # fit onto a gpu or other accelerator

        total_metrics = {}
        for iteration in range(n_iterations):

            # Fetch the next batch of data with larcv
            # (Make sure to pull from the validation set)
            minibatch_data = self.fetch_next_batch('aux')        

            # Convert the input data to torch tensors
            minibatch_data = {key : torch.Tensor(minibatch_data[key]) for key in minibatch_data }
            
            # if using cuda, copy the input data to GPU:
            if FLAGS.COMPUTE_MODE == "GPU":
                for key in minibatch_data: minibatch_data[key].cuda()

            # Run a forward pass of the model on the input image:
            logits = self._net(minibatch_data['image'])

            # Compute the metrics for this iteration:
            metrics = self._compute_metrics()


            # Add them to the total metrics for this validation step:
            if total_metrics == {}:
                total_metrics = metrics
            else:
                total_metrics = { total_metrics[key] + metrics[key] for key in metrics}


        # Average metrics over the total number of iterations:
        total_metrics = { total_metrics[key] / n_iterations for key in metrics}

        return metrics


    def stop(self):
        # Mostly, this is just turning off the io:
        self._larcv_interface.stop()

    def checkpoint(self):

        if self._global_step % FLAGS.CHECKPOINT_ITERATION == 0 and self._global_step != 0:
            # Save a checkpoint, but don't do it on the first pass
            self.save_model()


    def batch_process(self):

        # At the begining of batch process, figure out the epoch size:
        self._epoch_size = self._larcv_interface.size('primary')

        # This is the 'master' function, so it controls a lot



        # Run iterations
        for i in range(FLAGS.ITERATIONS):
            if FLAGS.TRAINING and self._iteration >= FLAGS.ITERATIONS:
                print('Finished training (iteration %d)' % self._iteration)
                self.checkpoint()
                break

            # Start IO thread for the next batch while we train the network
            self.train_step()

            self.checkpoint()

        self._saver.close()

