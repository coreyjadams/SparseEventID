import os
import tempfile
import sys
import time
from collections import OrderedDict

import numpy

import torch

from larcv import larcv_interface

from . import flags
from . import data_transforms
from . import io_templates
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

        self._cleanup         = []

    def __del__(self):
        for f in self._cleanup:
            os.unlink(f.name)
            

    def _initialize_io(self):

        # Use the templates to generate a configuration string, which we store into a temporary file
        if FLAGS.TRAINING:
            config = io_templates.train_io(input_file=FLAGS.FILE, image_dim=FLAGS.INPUT_DIMENSION, 
                label_mode=FLAGS.LABEL_MODE)
        else:
            config = io_templates.ana_io(input_file=FLAGS.FILE, image_dim=FLAGS.INPUT_DIMENSION,
                label_mode=FLAGS.LABEL_MODE)


        # Generate a named temp file:
        main_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
        main_file.write(config.generate_config_str())

        main_file.close()
        self._cleanup.append(main_file)

        # Prepare data managers:
        io_config = {
            'filler_name' : config._name,
            'filler_cfg'  : main_file.name,
            'verbosity'   : FLAGS.VERBOSITY,
            'make_copy'   : True
        }

        # Build up the data_keys:
        data_keys = OrderedDict()
        data_keys['image'] = 'data'
        for proc in config._process_list._processes:
            if proc._name == 'data': 
                continue
            else:
                data_keys[proc._name] = proc._name

        # Assign the keywords here:
        if FLAGS.LABEL_MODE == 'all':
            FLAGS.KEYWORD_LABEL = 'label'
        else:
            FLAGS.KEYWORD_LABEL = []
            for key in data_keys.keys():
                if key != 'image':
                    FLAGS.KEYWORD_LABEL.append(key)

        self._larcv_interface.prepare_manager('primary', io_config, FLAGS.MINIBATCH_SIZE, data_keys)

        if not FLAGS.TRAINING:
            self._larcv_interface._dataloaders['primary'].set_next_index(0)

        # All of the additional tools are in case there is a test set up:
        if FLAGS.AUX_FILE is not None:


            if FLAGS.TRAINING:
                config = io_templates.test_io(input_file=FLAGS.AUX_FILE, image_dim=FLAGS.INPUT_DIMENSION, 
                    label_mode=FLAGS.LABEL_MODE)

                # Generate a named temp file:
                aux_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
                aux_file.write(config.generate_config_str())
                # print([proc._name for proc in config._process_list._processes])

                aux_file.close()
                self._cleanup.append(aux_file)
                io_config = {
                    'filler_name' : config._name,
                    'filler_cfg'  : aux_file.name,
                    'verbosity'   : FLAGS.VERBOSITY,
                    'make_copy'   : True
                }

                # Build up the data_keys:
                data_keys = OrderedDict()
                data_keys['image'] = 'aux_data'
                for proc in config._process_list._processes:
                    if proc._name == 'aux_data': 
                        continue
                    else:
                        data_keys[proc._name] = proc._name



                self._larcv_interface.prepare_manager('aux', io_config, FLAGS.AUX_MINIBATCH_SIZE, data_keys)

        if FLAGS.OUTPUT_FILE is not None:
            if not FLAGS.TRAINING:
                config = io_templates.output_io(input_file=FLAGS.FILE, output_file=FLAGS.OUTPUT_FILE)
                
                out_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
                out_file.write(config.generate_config_str())
                print(config.generate_config_str())

                out_file.close()
                self._cleanup.append(out_file)

                self._larcv_interface.prepare_writer(out_file.name, FLAGS.OUTPUT_FILE)



    def init_network(self):

        dims = self._larcv_interface.fetch_minibatch_dims('primary')

        # This sets up the necessary output shape:
        if FLAGS.LABEL_MODE == 'split':
            output_shape = { key : dims[key] for key in FLAGS.KEYWORD_LABEL}
        else:
            output_shape = dims[FLAGS.KEYWORD_LABEL]


        self._net = FLAGS._net(output_shape)


        if FLAGS.TRAINING: 
            self._net.train(True)

    def initialize(self, io_only=False):



        self._initialize_io()


        if io_only:
            return

        FLAGS.dump_config()

        self.init_network()

        n_trainable_parameters = 0
        for var in self._net.parameters():
            n_trainable_parameters += numpy.prod(var.shape)
        print("Total number of trainable parameters in this network: {}".format(n_trainable_parameters))

        self.init_optimizer()

        self.init_saver()

        state = self.restore_model()

        if state is not None:
            self.load_state(state)
        else:
            self._global_step = 0


        if FLAGS.COMPUTE_MODE == "CPU":
            pass
        if FLAGS.COMPUTE_MODE == "GPU":
            self._net.cuda()

        if FLAGS.LABEL_MODE == 'all':
            self._log_keys = ['loss', 'accuracy']
        elif FLAGS.LABEL_MODE == 'split':
            self._log_keys = ['loss']
            for key in FLAGS.KEYWORD_LABEL: 
                self._log_keys.append('acc/{}'.format(key))


    def get_device(self):
        # Convert the input data to torch tensors
        if FLAGS.COMPUTE_MODE == "GPU":
            device = torch.device('cuda')
            # print(device)
        else:
            device = torch.device('cpu')


        return device

    def init_optimizer(self):

        # Create an optimizer:
        if FLAGS.OPTIMIZER == "SDG":
            self._opt = torch.optim.SGD(self._net.parameters(), lr=FLAGS.LEARNING_RATE,
                weight_decay=FLAGS.WEIGHT_DECAY)
        else:
            self._opt = torch.optim.Adam(self._net.parameters(), lr=FLAGS.LEARNING_RATE,
                weight_decay=FLAGS.WEIGHT_DECAY)





        device = self.get_device()

        # here we store the loss weights:
        if FLAGS.LABEL_MODE == 'all':
            self._label_weights = torch.tensor([ 
                4930., 247., 2311., 225., 11833., 1592., 3887., 378., 4966., 1169., 1944., 335., 
                5430., 201., 1630., 67., 13426., 1314., 3111., 243., 5070., 788., 1464., 163.,
                5851.,3267.,1685.,183.,7211.,3283.,2744.,302.,5804.,1440.,1302., 204.
                ], device=device)
            weights = torch.sum(self._label_weights) / self._label_weights
            self._label_weights = weights / torch.sum(weights)

            self._criterion = torch.nn.CrossEntropyLoss(weight=self._label_weights)


        elif FLAGS.LABEL_MODE == 'split':
            # These are the raw category occurences
            self._label_weights = {
                'label_cpi'  : torch.tensor([7784., 2216.], device=device),
                'label_prot' : torch.tensor([2658., 4940., 2402.], device=device), 
                'label_npi'  : torch.tensor([8448., 1552.], device=device),
                'label_neut' : torch.tensor([3322., 3317., 3361.], device=device)
            }

            self._criterion = {}

            for key in self._label_weights:
                weights = torch.sum(self._label_weights[key]) / self._label_weights[key]
                self._label_weights[key] = weights / torch.sum(weights)



            for key in self._label_weights:
                self._criterion[key] = torch.nn.CrossEntropyLoss(weight=self._label_weights[key])


    def init_saver(self):

        # This sets up the summary saver:
        if FLAGS.TRAINING:
            self._saver = tensorboardX.SummaryWriter(FLAGS.LOG_DIRECTORY)

        if FLAGS.AUX_FILE is not None and FLAGS.TRAINING:
            self._aux_saver = tensorboardX.SummaryWriter(FLAGS.LOG_DIRECTORY + "/test/")
        elif FLAGS.AUX_FILE is not None and not FLAGS.TRAINING:
            self._aux_saver = tensorboardX.SummaryWriter(FLAGS.LOG_DIRECTORY + "/val/")

        else:
            self._aux_saver = None
        # This code is supposed to add the graph definition.
        # It doesn't currently work
        # temp_dims = list(dims['image'])
        # temp_dims[0] = 1
        # dummy_input = torch.randn(size=tuple(temp_dims), requires_grad=True)
        # self._saver.add_graph(self._net, (dummy_input,))

        # Here, either restore the weights of the network or initialize it:


    def restore_model(self):
        ''' This function attempts to restore the model from file
        '''

        _, checkpoint_file_path = self.get_model_filepath()

        print(checkpoint_file_path)

        if not os.path.isfile(checkpoint_file_path):
            print("Returning none!")
            return None
        # Parse the checkpoint file and use that to get the latest file path

        with open(checkpoint_file_path, 'r') as _ckp:
            for line in _ckp.readlines():
                if line.startswith("latest: "):
                    chkp_file = line.replace("latest: ", "").rstrip('\n')
                    chkp_file = os.path.dirname(checkpoint_file_path) + "/" + chkp_file
                    print("Restoring weights from ", chkp_file)
                    break

        if FLAGS.COMPUTE_MODE == "CPU":
            state = torch.load(chkp_file, map_location='cpu')
        else:
            state = torch.load(chkp_file)

        return state

    def load_state(self, state):


        self._net.load_state_dict(state['state_dict'])
        self._opt.load_state_dict(state['optimizer'])
        self._global_step = state['global_step']

        # If using GPUs, move the model to GPU:
        if FLAGS.COMPUTE_MODE == "GPU":
            for state in self._opt.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        return True


    def save_model(self):
        '''Save the model to file
        
        '''

        current_file_path, checkpoint_file_path = self.get_model_filepath()

        # save the model state into the file path:
        state_dict = {
            'global_step' : self._global_step,
            'state_dict'  : self._net.state_dict(),
            'optimizer'   : self._opt.state_dict(),
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
        if FLAGS.CHECKPOINT_DIRECTORY == None:
            file_path= FLAGS.LOG_DIRECTORY  + "/checkpoints/"
        else:
            file_path= FLAGS.CHECKPOINT_DIRECTORY  + "/checkpoints/"


        name = file_path + 'model-{}.ckpt'.format(self._global_step)
        checkpoint_file_path = file_path + "checkpoint"

        return name, checkpoint_file_path

    def _calculate_loss(self, inputs, logits):
        ''' Calculate the loss.

        returns a single scalar for the optimizer to use.
        '''


        # This dataset is not balanced across labels.  So, we can weight the loss according to the labels
        # 
        # 'label_cpi': array([1523.,  477.]), 
        # 'label_prot': array([528., 964., 508.]), 
        # 'label_npi': array([1699.,  301.]), 
        # 'label_neut': array([655., 656., 689.])

        # You can see that the only category that's truly balanced is the neutrino category.  
        # The proton category has a ratio of 1 : 2 : 1 which isn't terrible, but can be fixed.  
        # Both the proton and neutrino categories learn well.
        #
        #
        # The pion categories learn poorly, and slowly.  They quickly reach ~75% and ~85% accuracy for c/n pi
        # Which is just the ratio of the 0 : 1 label in each category.  So, they are learning to predict always zero, 
        # And it is difficult to bust out of that.


        if FLAGS.LABEL_MODE == 'all':
            values, target = torch.max(inputs[FLAGS.KEYWORD_LABEL], dim = 1)
            loss = self._criterion(logits, target=target)
            return loss
        elif FLAGS.LABEL_MODE == 'split':
            loss = None
            for key in logits:
                values, target = torch.max(inputs[key], dim=1)

                temp_loss = self._criterion[key](logits[key], target= target)
                # print(temp_loss.shape)
                # temp_loss *= self._label_weights[key]
                # print(temp_loss.shape)
                # temp_loss = torch.sum(temp_loss)
                # print(temp_loss.shape)

                if loss is None:
                    loss = temp_loss
                else:
                    loss += temp_loss

            return loss


    def _calculate_accuracy(self, logits, minibatch_data):
        ''' Calculate the accuracy.

        '''

        # Compare how often the input label and the output prediction agree:

        if FLAGS.LABEL_MODE == 'all':

            values, indices = torch.max(minibatch_data[FLAGS.KEYWORD_LABEL], dim = 1)
            values, predict = torch.max(logits, dim=1)
            correct_prediction = torch.eq(predict,indices)
            accuracy = torch.mean(correct_prediction.float())

        elif FLAGS.LABEL_MODE == 'split':
            accuracy = {}
            for key in logits:

                values, indices = torch.max(minibatch_data[key], dim = 1)
                values, predict = torch.max(logits[key], dim=1)
                correct_prediction = torch.eq(predict,indices)
                accuracy[key] = torch.mean(correct_prediction.float())

        return accuracy


    def _compute_metrics(self, logits, minibatch_data, loss):

        # Call all of the functions in the metrics dictionary:
        metrics = {}

        metrics['loss']     = loss.data
        accuracy = self._calculate_accuracy(logits, minibatch_data)
        if FLAGS.LABEL_MODE == 'all':
            metrics['accuracy'] = accuracy
        elif FLAGS.LABEL_MODE == 'split':
            for key in accuracy:
                metrics['acc/{}'.format(key)] = accuracy[key]

        return metrics

    def log(self, metrics, saver=''):


        if self._global_step % FLAGS.LOGGING_ITERATION == 0:
            
            self._current_log_time = datetime.datetime.now()

            s = ""

            if 'it.' in metrics:
                # This prints out the iteration for ana steps
                s += "it.: {}, ".format(metrics['it.'])

            # Build up a string for logging:
            if self._log_keys != []:
                s += ", ".join(["{0}: {1:.3}".format(key, metrics[key]) for key in self._log_keys])
            else:
                s += ", ".join(["{0}: {1:.3}".format(key, metrics[key]) for key in metrics])
      

            try:
                s += " ({:.2}s / {:.2} IOs / {:.2})".format(
                    (self._current_log_time - self._previous_log_time).total_seconds(), 
                    metrics['io_fetch_time'],
                    metrics['step_time'])
            except:
                pass

            self._previous_log_time = self._current_log_time

            print("{} Step {} metrics: {}".format(saver, self._global_step, s))



    def summary(self, metrics,saver=""):

        if self._saver is None:
            return

        if self._global_step % FLAGS.SUMMARY_ITERATION == 0:
            for metric in metrics:
                name = metric
                if saver == "test":
                    self._aux_saver.add_scalar(metric, metrics[metric], self._global_step)
                else:
                    self._saver.add_scalar(metric, metrics[metric], self._global_step)


            # try to get the learning rate
            # print self._lr_scheduler.get_lr()
            self._saver.add_scalar("learning_rate", self._opt.state_dict()['param_groups'][0]['lr'], self._global_step)
            pass

    def fetch_next_batch(self, mode='primary', metadata=False):

        # For the serial mode, call next here:
        if not FLAGS.DISTRIBUTED:
            self._larcv_interface.next(mode)

        minibatch_data = self._larcv_interface.fetch_minibatch_data(mode, fetch_meta_data=metadata)
        minibatch_dims = self._larcv_interface.fetch_minibatch_dims(mode)


        for key in minibatch_data:
            if key == 'entries' or key == 'event_ids':
                continue
            minibatch_data[key] = numpy.reshape(minibatch_data[key], minibatch_dims[key])

        # Strip off the primary/aux label in the keys:
        for key in minibatch_data:
            new_key = key.replace('aux_','')
            minibatch_data[new_key] = minibatch_data.pop(key)            



        # Here, do some massaging to convert the input data to another format, if necessary:
        if FLAGS.IMAGE_MODE == 'dense' and not FLAGS.SPARSE:
            # Don't have to do anything here
            pass
        elif FLAGS.IMAGE_MODE == 'dense' and FLAGS.SPARSE:
            # Have to convert the input image from dense to sparse format:
            if FLAGS.INPUT_DIMENSION == '3D':
                minibatch_data['image'] = data_transforms.larcvdense_to_scnsparse_3d(minibatch_data['image'])
            else:
                minibatch_data['image'] = data_transforms.larcvdense_to_scnsparse_2d(minibatch_data['image'])
        elif FLAGS.IMAGE_MODE == 'sparse' and not FLAGS.SPARSE:
            # Need to convert sparse larcv into a dense numpy array:
            if FLAGS.INPUT_DIMENSION == '3D':
                minibatch_data['image'] = data_transforms.larcvsparse_to_dense_3d(minibatch_data['image'])
            else:
                minibatch_data['image'] = data_transforms.larcvsparse_to_dense_2d(minibatch_data['image'])


        elif FLAGS.IMAGE_MODE == 'sparse' and FLAGS.SPARSE:
            if FLAGS.INPUT_DIMENSION == '3D':
                minibatch_data['image'] = data_transforms.larcvsparse_to_scnsparse_3d(minibatch_data['image'])
            else:
                minibatch_data['image'] = data_transforms.larcvsparse_to_scnsparse_2d(minibatch_data['image'])

        return minibatch_data

    def increment_global_step(self):

        previous_epoch = int((self._global_step * FLAGS.MINIBATCH_SIZE) / self._epoch_size)
        self._global_step += 1
        current_epoch = int((self._global_step * FLAGS.MINIBATCH_SIZE) / self._epoch_size)

        self.on_step_end()

        if previous_epoch != current_epoch:
            self.on_epoch_end()

    def on_step_end(self):
        pass

    def on_epoch_end(self):
        pass

    def to_torch(self, minibatch_data, device=None):

        # Convert the input data to torch tensors
        if FLAGS.COMPUTE_MODE == "GPU":
            if device is None:
                device = torch.device('cuda')
            # print(device)

        else:
            if device is None:
                device = torch.device('cpu')


        for key in minibatch_data:
            if key == 'entries' or key =='event_ids':
                continue
            if key == 'image' and FLAGS.SPARSE:
                if FLAGS.INPUT_DIMENSION == '3D':
                    minibatch_data['image'] = (
                            torch.tensor(minibatch_data['image'][0]).long(),
                            torch.tensor(minibatch_data['image'][1], device=device),
                            minibatch_data['image'][2],
                        )
                else:
                    minibatch_data['image'] = (
                            torch.tensor(minibatch_data['image'][0]).long(),
                            torch.tensor(minibatch_data['image'][1], device=device),
                            minibatch_data['image'][2],
                        )
            else:
                minibatch_data[key] = torch.tensor(minibatch_data[key],device=device)
        
        return minibatch_data

    def train_step(self):


        # For a train step, we fetch data, run a forward and backward pass, and
        # if this is a logging step, we compute some logging metrics.

        self._net.train()

        global_start_time = datetime.datetime.now()

        # Reset the gradient values for this step:
        self._opt.zero_grad()

        # Fetch the next batch of data with larcv
        io_start_time = datetime.datetime.now()
        minibatch_data = self.fetch_next_batch()
        io_end_time = datetime.datetime.now()

        minibatch_data = self.to_torch(minibatch_data)


        # Run a forward pass of the model on the input image:
        logits = self._net(minibatch_data['image'])

        # print("Completed Forward pass")
        # Compute the loss based on the logits
        loss = self._calculate_loss(minibatch_data, logits)
        # print("Completed loss")

        # Compute the gradients for the network parameters:
        loss.backward()
        # print("Completed backward pass")

        # Compute any necessary metrics:
        metrics = self._compute_metrics(logits, minibatch_data, loss)
        


        # Add the global step / second to the tensorboard log:
        try:
            metrics['global_step_per_sec'] = 1./self._seconds_per_global_step
            metrics['images_per_second'] = FLAGS.MINIBATCH_SIZE / self._seconds_per_global_step
        except:
            metrics['global_step_per_sec'] = 0.0
            metrics['images_per_second'] = 0.0

        metrics['io_fetch_time'] = (io_end_time - io_start_time).total_seconds()

        # print("Calculated metrics")


        step_start_time = datetime.datetime.now()
        # Apply the parameter update:
        self._opt.step()
        # print("Updated Weights")
        global_end_time = datetime.datetime.now()

        metrics['step_time'] = (global_end_time - step_start_time).total_seconds()


        self.log(metrics, saver="train") 

        # print("Completed Log")

        self.summary(metrics, saver="train")       

        # print("Summarized")

        # Compute global step per second:
        self._seconds_per_global_step = (global_end_time - global_start_time).total_seconds()

        # Increment the global step value:
        self.increment_global_step()



        return metrics

    def val_step(self, n_iterations=1):

        # First, validation only occurs on training:
        if not FLAGS.TRAINING: return

        # Second, validation can not occur without a validation dataloader.
        if FLAGS.AUX_FILE is None: return

        # perform a validation step
        # Validation steps can optionally accumulate over several minibatches, to
        # fit onto a gpu or other accelerator

        # self._net.eval()

        with torch.no_grad():

            if self._global_step != 0 and self._global_step % FLAGS.AUX_ITERATION == 0:


                # Fetch the next batch of data with larcv
                # (Make sure to pull from the validation set)
                minibatch_data = self.fetch_next_batch('aux')        


                # Convert the input data to torch tensors
                minibatch_data = self.to_torch(minibatch_data)
                
                # Run a forward pass of the model on the input image:
                logits = self._net(minibatch_data['image'])

                # # Here, we have to map the logit keys to aux keys
                # for key in logits.keys():
                #     new_key = 'aux_' + key
                #     logits[new_key] = logits.pop(key)



                # Compute the loss
                loss = self._calculate_loss(minibatch_data, logits)

                # Compute the metrics for this iteration:
                metrics = self._compute_metrics(logits, minibatch_data, loss)


                self.log(metrics, saver="test")
                self.summary(metrics, saver="test")

                return metrics

    def ana_step(self, iteration=None):

        # First, validation only occurs on training:
        if FLAGS.TRAINING: return

        # perform a validation step

        # Set network to eval mode
        self._net.eval()
        # self._net.train()

        # Fetch the next batch of data with larcv
        minibatch_data = self.fetch_next_batch(metadata=True)


        # Convert the input data to torch tensors
        minibatch_data = self.to_torch(minibatch_data)
        
        # Run a forward pass of the model on the input image:
        with torch.no_grad():
            logits = self._net(minibatch_data['image'])

        if FLAGS.LABEL_MODE == 'all':
            softmax = torch.nn.Softmax(dim=-1)(logits)
        else:
            softmax = { key : torch.nn.Softmax(dim=-1)(logits[key]) for key in logits } 

        # print('label_neut', minibatch_data['label_neut'])
        # print('label_npi', minibatch_data['label_npi'])
        # print('label_cpi', minibatch_data['label_cpi'])
        # print('label_prot', minibatch_data['label_prot'])
        # print(softmax)

        # Call the larcv interface to write data:
        if FLAGS.OUTPUT_FILE is not None:
            if FLAGS.LABEL_MODE == 'all':
                writable_logits = numpy.asarray(softmax.cpu())
                self._larcv_interface.write_output(data=writable_logits[0], datatype='meta', producer='all',
                    entries=minibatch_data['entries'], event_ids=minibatch_data['event_ids'])
            else:
                for key in softmax:
                    writable_logits = numpy.asarray(softmax[key].cpu())
                    self._larcv_interface.write_output(data=writable_logits[0], datatype='meta', producer=key,
                        entries=minibatch_data['entries'], event_ids=minibatch_data['event_ids'])

        # If the input data has labels available, compute the metrics:
        if (FLAGS.LABEL_MODE == 'all' and 'label' in minibatch_data) or \
           (FLAGS.LABEL_MODE == 'split' and 'label_neut' in minibatch_data):
            # Compute the loss
            loss = self._calculate_loss(minibatch_data, logits)

            # Compute the metrics for this iteration:
            metrics = self._compute_metrics(logits, minibatch_data, loss)

            if iteration is not None:
                metrics.update({'it.' : iteration})


            self.log(metrics, saver="test")
            # self.summary(metrics, saver="test")

            return metrics


    def stop(self):
        # Mostly, this is just turning off the io:
        self._larcv_interface.stop()

    def checkpoint(self):

        if FLAGS.CHECKPOINT_ITERATION == -1:
            return

        if self._global_step % FLAGS.CHECKPOINT_ITERATION == 0 and self._global_step != 0:
            # Save a checkpoint, but don't do it on the first pass
            self.save_model()


    def batch_process(self):

        # At the begining of batch process, figure out the epoch size:
        self._epoch_size = self._larcv_interface.size('primary')

        # This is the 'master' function, so it controls a lot

        # If we're not training, force the number of iterations to the epoch size or less
        if not FLAGS.TRAINING:
            if FLAGS.ITERATIONS > self._epoch_size:
                FLAGS.ITERATIONS = self._epoch_size


        # Run iterations
        for i in range(FLAGS.ITERATIONS):
            if FLAGS.TRAINING and self._iteration >= FLAGS.ITERATIONS:
                print('Finished training (iteration %d)' % self._iteration)
                self.checkpoint()
                break

            if FLAGS.TRAINING:
                self.val_step()
                self.train_step()
                self.checkpoint()
            else:
                self.ana_step(i)


        if FLAGS.TRAINING:
            if self._saver is not None:
                self._saver.close()
            if self._aux_saver is not None:
                self._aux_saver.close()

