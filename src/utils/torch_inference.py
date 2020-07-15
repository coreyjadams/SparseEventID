import os
import tempfile
import sys
import time
from collections import OrderedDict

import numpy

import torch

from . larcvio   import larcv_fetcher

import datetime

# This uses tensorboardX to save summaries and metrics to tensorboard compatible files.

import tensorboardX

class trainercore(object):
    '''
    This class is the core interface for training.  Each function to
    be overridden for a particular interface is marked and raises
    a NotImplemented error.

    '''
    def __init__(self, args):
        self.args = args
        self.larcv_fetcher = larcv_fetcher.larcv_fetcher(
            mode            = args.mode,
            distributed     = args.distributed,
            image_mode      = args.image_mode,
            label_mode      = self.args.label_mode,
            input_dimension = self.args.input_dimension,
        )

        # self.larcv_fetcher = threadloader.thread_interface()
        self._iteration       = 0
        self._global_step     = -1



    def print(self, *argv):
        ''' Function for logging as needed.  Works correctly in distributed mode'''

        message = " ".join([ str(s) for s in argv] )

        sys.stdout.write(message + "\n")
        sys.stdout.flush()


    def _initialize_io(self, color=0):

        # Prepare the training sample:
        self._train_data_size = self.larcv_fetcher.prepare_sample(
            name            = "primary",
            input_file      = self.args.file,
            batch_size      = self.args.minibatch_size,
            color           = color
        )

        # Check that the training file exists:
        if not self.args.aux_file.exists():
            if self.args.mode == "train":
                self.print("WARNING: Aux file does not exist.  Setting to None for training")
                self.args.aux_file = None
            else:
                # In inference mode, we are creating the aux file.  So we need to check
                # that the directory exists.  Otherwise, no writing.
                if not self.args.aux_file.parent.exists():
                    self.print("WARNING: Aux file's directory does not exist, skipping.")
                    self.args.aux_file = None
                elif self.args.aux_file is None or str(self.args.aux_file).lower() == "none":
                    self.print("WARNING: no aux file set, so not writing inference results.")
                    self.args.aux_file = None


        if self.args.aux_file is not None:
            if self.args.mode == "train":
                # Fetching data for on the fly testing:
                self._aux_data_size = self.larcv_fetcher.prepare_sample(
                    name            = "aux",
                    input_file      = self.args.aux_file,
                    batch_size      = self.args.minibatch_size,
                    color           = color
                )
            elif self.args.mode == "inference":
                raise Exception("Need to check the inference writer works")
                self._aux_data_size = self.larcv_fetcher.prepare_writer(
                    input_file = self.args.file, output_file = str(self.args.aux_file))


        # if 'output_file' in self.args and self.args.output_file is not None:
        #     if not self.args.training:
        #         config = io_templates.output_io(input_file=self.args.file, output_file=self.args.output_file)

        #         out_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
        #         out_file.write(config.generate_config_str())
        #         print(config.generate_config_str())

        #         out_file.close()
        #         self._cleanup.append(out_file)

        #         self.larcv_fetcher.prepare_writer(io_config=out_file.name, input_files=self.args.file, output_file=self.args.output_file)



    def init_network(self):


        dims = self.larcv_fetcher.fetch_minibatch_dims('primary')

        # This sets up the necessary output shape:
        output_shape = self.larcv_fetcher.output_shape('primary')


        # To initialize the network, we see what the name is
        # and act on that:
        if self.args.network == "resnet2d":
            from src.networks import resnet
            self._net = resnet.ResNet(output_shape, self.args)
        elif self.args.network == "sparseresnet2d":
            from src.networks import sparseresnet
            self._net = sparseresnet.ResNet(output_shape, self.args)
        elif self.args.network == "sparseresnet3d":
            from src.networks import sparseresnet3d
            self._net = sparseresnet3d.ResNet(output_shape, self.args)
        elif self.args.network == "pointnet":
            from src.networks import pointnet
            self._net = pointnet.PointNet(output_shape, self.args)
        elif self.args.network == "gcn":
            from src.networks import gcn
            self._net = gcn.GCNNet(output_shape, self.args)
        elif self.args.network == "dgcnn":
            from src.networks import dgcnn
            self._net = dgcnn.DGCNN(output_shape, self.args)
        else:
            raise Exception(f"Couldn't identify network {self.args.network}")




        if self.args.training:
            self._net.train(True)

    def initialize(self, io_only=False):



        self._initialize_io()


        if io_only:
            return


        self.init_network()

        n_trainable_parameters = 0
        for var in self._net.parameters():
            n_trainable_parameters += numpy.prod(var.shape)
        self.print("Total number of trainable parameters in this network: {}".format(n_trainable_parameters))

        self.init_optimizer()

        self.init_saver()

        state = self.restore_model()

        if state is not None:
            self.load_state(state)
        else:
            self._global_step = 0


        if self.args.compute_mode == "CPU":
            pass
        if self.args.compute_mode == "GPU":
            self._net.cuda()

        if self.args.label_mode == 'all':
            self._log_keys = ['loss', 'accuracy']
        elif self.args.label_mode == 'split':
            self._log_keys = ['loss']
            for key in self.larcv_fetcher.keyword_label:
                self._log_keys.append('acc/{}'.format(key))


    def get_device(self):
        # Convert the input data to torch tensors
        if self.args.compute_mode == "GPU":
            device = torch.device('cuda')
            # self.print(device)
        else:
            device = torch.device('cpu')


        return device

    def init_optimizer(self):

        # Create an optimizer:
        if self.args.optimizer == "SDG":
            self._opt = torch.optim.SGD(self._net.parameters(), lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay)
        else:
            self._opt = torch.optim.Adam(self._net.parameters(), lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay)





        device = self.get_device()

        # here we store the loss weights:
        if self.args.label_mode == 'all':
            self._label_weights = torch.tensor([
                4930., 247., 2311., 225., 11833., 1592., 3887., 378., 4966., 1169., 1944., 335.,
                5430., 201., 1630., 67., 13426., 1314., 3111., 243., 5070., 788., 1464., 163.,
                5851.,3267.,1685.,183.,7211.,3283.,2744.,302.,5804.,1440.,1302., 204.
                ], device=device)
            weights = torch.sum(self._label_weights) / self._label_weights
            self._label_weights = weights / torch.sum(weights)

            self._criterion = torch.nn.CrossEntropyLoss(weight=self._label_weights)


        elif self.args.label_mode == 'split':
            # These are the raw category occurences
            self._label_weights = {
                'label_cpi'  : torch.tensor([50932., 61269.], device=device),
                'label_prot' : torch.tensor([36583., 46790., 28828.], device=device),
                'label_npi'  : torch.tensor([70572., 41629.], device=device),
                'label_neut' : torch.tensor([39452., 39094., 33655.], device=device)
            }

            self._criterion = {}

            for key in self._label_weights:
                weights = torch.sum(self._label_weights[key]) / self._label_weights[key]
                self._label_weights[key] = weights / torch.sum(weights)
                self.print ('Weights for', key, '=', self._label_weights[key])

            for key in self._label_weights:
                self._criterion[key] = torch.nn.CrossEntropyLoss(weight=self._label_weights[key])


    def init_saver(self):

        # This sets up the summary saver:
        if self.args.training:
            self._saver = tensorboardX.SummaryWriter(self.args.log_directory)

        if self.args.aux_file is not None and self.args.training:
            self._aux_saver = tensorboardX.SummaryWriter(self.args.log_directory + "/test/")
        elif self.args.aux_file is not None and not self.args.training:
            self._aux_saver = tensorboardX.SummaryWriter(self.args.log_directory + "/val/")

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

        self.print(checkpoint_file_path)

        if not os.path.isfile(checkpoint_file_path):
            self.print("Returning none!")
            return None
        # Parse the checkpoint file and use that to get the latest file path

        with open(checkpoint_file_path, 'r') as _ckp:
            for line in _ckp.readlines():
                if line.startswith("latest: "):
                    chkp_file = line.replace("latest: ", "").rstrip('\n')
                    chkp_file = os.path.dirname(checkpoint_file_path) + "/" + chkp_file
                    self.print("Restoring weights from ", chkp_file)
                    break

        if self.args.compute_mode == "CPU":
            state = torch.load(chkp_file, map_location='cpu')
        else:
            state = torch.load(chkp_file)

        return state

    def load_state(self, state):


        self._net.load_state_dict(state['state_dict'])
        self._opt.load_state_dict(state['optimizer'])
        self._global_step = state['global_step']

        # If using GPUs, move the model to GPU:
        if self.args.compute_mode == "GPU":
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
        n_keep = 100


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
        if self.args.checkpoint_directory == None:
            file_path= self.args.log_directory  + "/checkpoints/"
        else:
            file_path= self.args.checkpoint_directory  + "/checkpoints/"


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


        if self.args.label_mode == 'all':
            values, target = torch.max(inputs[self.args.keyword_label], dim = 1)
            loss = self._criterion(logits, target=target)
            return loss
        elif self.args.label_mode == 'split':
            loss = None
            for key in logits:
                values, target = torch.max(inputs[key], dim=1)

                temp_loss = self._criterion[key](logits[key], target=target)
                # self.print(temp_loss.shape)
                # temp_loss *= self._label_weights[key]
                # self.print(temp_loss.shape)
                # temp_loss = torch.sum(temp_loss)
                # self.print(temp_loss.shape)

                if loss is None:
                    loss = temp_loss
                else:
                    loss += temp_loss

            return loss


    def _calculate_accuracy(self, logits, minibatch_data):
        ''' Calculate the accuracy.

        '''

        # Compare how often the input label and the output prediction agree:

        if self.args.label_mode == 'all':

            values, indices = torch.max(minibatch_data[self.args.keyword_label], dim = 1)
            values, predict = torch.max(logits, dim=1)
            correct_prediction = torch.eq(predict,indices)
            accuracy = torch.mean(correct_prediction.float())

        elif self.args.label_mode == 'split':
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
        if self.args.label_mode == 'all':
            metrics['accuracy'] = accuracy
        elif self.args.label_mode == 'split':
            for key in accuracy:
                metrics['acc/{}'.format(key)] = accuracy[key]

        return metrics

    def log(self, metrics, saver=''):


        if self._global_step % self.args.logging_iteration == 0:

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

            try:
                s += " (LR: {:.4})".format(
                    self._opt.state_dict()['param_groups'][0]['lr'])
            except:
                pass

            self._previous_log_time = self._current_log_time

            self.print("{} Step {} metrics: {}".format(saver, self._global_step, s))



    def summary(self, metrics,saver=""):

        if self._saver is None:
            return

        if self._global_step % self.args.summary_iteration == 0:
            for metric in metrics:
                name = metric
                if saver == "test":
                    self._aux_saver.add_scalar(metric, metrics[metric], self._global_step)
                else:
                    self._saver.add_scalar(metric, metrics[metric], self._global_step)


            # try to get the learning rate
            # self.print self._lr_scheduler.get_lr()
            if saver == "test":
                self._aux_saver.add_scalar("learning_rate", self._opt.state_dict()['param_groups'][0]['lr'], self._global_step)
            else:
                self._saver.add_scalar("learning_rate", self._opt.state_dict()['param_groups'][0]['lr'], self._global_step)
            pass


    def increment_global_step(self):

        previous_epoch = int((self._global_step * self.args.minibatch_size) / self._train_data_size)
        self._global_step += 1
        current_epoch = int((self._global_step * self.args.minibatch_size) / self._train_data_size)

        self.on_step_end()

        if previous_epoch != current_epoch:
            self.on_epoch_end()

    def on_step_end(self):
        pass

    def on_epoch_end(self):
        pass

    def to_torch(self, minibatch_data, device=None):

        # Convert the input data to torch tensors
        if self.args.compute_mode == "GPU":
            if device is None:
                device = torch.device('cuda')
            # print(device)

        else:
            if device is None:
                device = torch.device('cpu')


        for key in minibatch_data:
            if key == 'entries' or key =='event_ids':
                continue
            if key == 'image' and self.args.image_mode == "sparse":
                if self.args.input_dimension == 3:
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
            elif key == 'image' and self.args.image_mode == 'graph':
                minibatch_data[key] = minibatch_data[key].to(device)
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
        minibatch_data = self.larcv_fetcher.fetch_next_batch("primary",force_pop = True)
        io_end_time = datetime.datetime.now()

        minibatch_data = self.to_torch(minibatch_data)


        # Run a forward pass of the model on the input image:
        logits = self._net(minibatch_data['image'])

        # print("Completed Forward pass")
        # Compute the loss based on the logits
        loss = self._calculate_loss(minibatch_data, logits)

        # Compute the gradients for the network parameters:
        loss.backward()
        # print("Completed backward pass")

        # Compute any necessary metrics:
        metrics = self._compute_metrics(logits, minibatch_data, loss)



        # Add the global step / second to the tensorboard log:
        try:
            metrics['global_step_per_sec'] = 1./self._seconds_per_global_step
            metrics['images_per_second'] = self.args.minibatch_size / self._seconds_per_global_step
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
        if not self.args.training: return

        # Second, validation can not occur without a validation dataloader.
        if self.args.aux_file is None: return

        # perform a validation step
        # Validation steps can optionally accumulate over several minibatches, to
        # fit onto a gpu or other accelerator

        # self._net.eval()

        with torch.no_grad():

            if self._global_step != 0 and self._global_step % self.args.aux_iteration == 0:


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

        if self.args.training: return

        # Set network to eval mode
        self._net.eval()

        # Fetch the next batch of data with larcv
        minibatch_data = self.fetch_next_batch(metadata=True)

        # Convert the input data to torch tensors
        minibatch_data = self.to_torch(minibatch_data)

        # Run a forward pass of the model on the input image:
        with torch.no_grad():
            logits = self._net(minibatch_data['image'])


        if self.args.label_mode == 'all':
            softmax = torch.nn.Softmax(dim=-1)(logits)
        else:
            softmax = { key : torch.nn.Softmax(dim=-1)(logits[key]) for key in logits }

        # Call the larcv interface to write data:
        if self.args.output_file is not None:
            if self.args.label_mode == 'all':
                writable_logits = numpy.asarray(softmax.cpu())
                self.larcv_fetcher.write_output(data=writable_logits[0], datatype='meta', producer='all',
                    entries=minibatch_data['entries'], event_ids=minibatch_data['event_ids'])
            else:
                for entry in range(self.args.minibatch_size):
                    if iteration > 1 and minibatch_data['entries'][entry] == 0:
                        self.print ('Reached max number of entries.')
                        break
                    this_entry = [minibatch_data['entries'][entry]]
                    this_event_id = [minibatch_data['event_ids'][entry]]
                    for key in softmax:
                        writable_logits = numpy.asarray(softmax[key].cpu())[entry]
                        self.larcv_fetcher.write_output(data=writable_logits, datatype='tensor1d', producer=key,
                            entries=this_entry, event_ids=this_event_id)

        # If the input data has labels available, compute the metrics:
        if (self.args.label_mode == 'all' and 'label' in minibatch_data) or \
           (self.args.label_mode == 'split' and 'label_neut' in minibatch_data):
            # Compute the loss
            loss = self._calculate_loss(minibatch_data, logits)

            # Compute the metrics for this iteration:
            metrics = self._compute_metrics(logits, minibatch_data, loss)

            if iteration is not None:
                metrics.update({'it.' : iteration})


            self.log(metrics, saver="ana")
            # self.summary(metrics, saver="test")

            return metrics


    def stop(self):
        # Mostly, this is just turning off the io:
        # self.larcv_fetcher.stop()
        pass

    def checkpoint(self):

        if self.args.checkpoint_iteration == -1:
            return

        if self._global_step % self.args.checkpoint_iteration == 0 and self._global_step != 0:
            # Save a checkpoint, but don't do it on the first pass
            self.save_model()


    def batch_process(self):

        # If we're not training, force the number of iterations to the epoch size or less
        if not self.args.training:
            if self.args.iterations > int(self._train_data_size/self.args.minibatch_size) + 1:
                self.args.iterations = int(self._train_data_size/self.args.minibatch_size) + 1
                self.print('Number of iterations set to', self.args.iterations)


        # Run iterations
        for i in range(self.args.iterations):
            if self.args.training and self._iteration >= self.args.iterations:
                self.print('Finished training (iteration %d)' % self._iteration)
                self.checkpoint()
                break

            if self.args.training:
                self.val_step()
                self.train_step()
                self.checkpoint()
            else:
                self.ana_step(i)


        if self.args.training:
            if self._saver is not None:
                self._saver.close()
            if self._aux_saver is not None:
                self._aux_saver.close()
