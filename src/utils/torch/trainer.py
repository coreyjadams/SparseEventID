import os
import sys
import time
from collections import OrderedDict

import numpy

import torch

import datetime

# Import most of the IO functions:
from src.utils.core.trainercore import trainercore
from torch.utils.tensorboard import SummaryWriter

import logging
logger = logging.getLogger()

# torch profiler imports
from torch.profiler import profile, record_function, ProfilerActivity


import contextlib
@contextlib.contextmanager
def dummycontext():
    yield None


class trainer(trainercore):
    '''
    This class is the core interface for training.  Each function to
    be overridden for a particular interface is marked and raises
    a NotImplemented error.

    '''
    def __init__(self, args):
        trainercore.__init__(self, args)

        # self.larcv_fetcher = threadloader.thread_interface()
        self._iteration       = 0.
        self._global_step     = -1.
        self._rank            = None

    def init_saver(self):

        self._savers = {}

        # This sets up the summary saver:
        if self.args.mode.name == "train":
            self._savers['train'] = SummaryWriter(self.args.run.output_dir + '/train')

            if self._val_data_size is not None:
                self._savers['val'] = SummaryWriter(self.args.run.output_dir + '/val')



    def init_network(self):


        # This sets up the necessary output shape:
        output_shape = self.larcv_fetcher.output_shape('primary')


        # To initialize the network, we see what the name is
        # and act on that:
        if self.args.network.name == "resnet":
            if self.args.network.data_format == 'sparse':
                if self.args.dataset.dimension == 2:
                    from src.networks.torch import sparseresnet
                    self._net = sparseresnet.ResNet(output_shape, self.args)
                else:
                    from src.networks.torch import sparseresnet3d
                    self._net = sparseresnet3d.ResNet(output_shape, self.args)
            else:
                if self.args.dataset.dimension == 2:
                    from src.networks.torch import resnet
                    self._net = resnet.ResNet(output_shape, self.args)
                else:
                    raise Exception("No Resnet3d Implemented!")
        elif self.args.network.name == "pointnet":
            if self.args.dataset.dimension == 2:
                from src.networks.torch import pointnet
                self._net = pointnet.PointNet(output_shape, self.args)
            else:
                from src.networks.torch import pointnet3d
                self._net = pointnet3d.PointNet(output_shape, self.args)
        elif self.args.network.name == "dgcnn":
            if self.args.dataset.dimension == 2:
                from src.networks.torch import dgcnn2d
                self._net = dgcnn2d.DGCNN(output_shape, self.args)
            else:
                from src.networks.torch import dgcnn3d
                self._net = dgcnn3d.DGCNN(output_shape, self.args)
        else:
            raise Exception(f"Couldn't identify network {self.args.network.name}")

        logger.debug(self._net)


        if self.args.mode.name == "train":
            self._net.train(True)


    def initialize(self, io_only=False):


        trainercore.initialize(self, io_only)

        if io_only:
            return

        with self.default_device_context():


            self.init_network()

            self._net.to(self.default_device())

            n_trainable_parameters = 0
            for var in self._net.parameters():
                n_trainable_parameters += numpy.prod(var.shape)
            logger.info("Total number of trainable parameters in this network: {}".format(n_trainable_parameters))

            self.init_optimizer()

            self.init_saver()

            self.restore_model()

            from src.networks.torch import LossCalculator
            vertex_shape = self._net.vertex_shape()
            if vertex_shape is not None:
                vertex_shape = vertex_shape.to(self.default_device())
            self.loss_calculator = LossCalculator.LossCalculator(vertex_shape)


        self._log_keys = ['loss']
        for key in self.larcv_fetcher.keyword_label:
            self._log_keys.append('acc/{}'.format(key))

        if self.args.network['detect_vertex']:
            self._log_keys.append("acc/Vertex")
            self.image_dimensions = torch.tensor(self.larcv_fetcher.image_dimensions, device=self.default_device())
            self.vertex_origin = torch.tensor(self.larcv_fetcher.vertex_origin, device=self.default_device())
            self.loss_calculator.set_image_dimensions(self.image_dimensions, self.vertex_origin)

    def init_optimizer(self):

        self.build_lr_schedule()

        # Create an optimizer:
        if self.args.mode.optimizer.name.upper() == "SDG":
            self._opt = torch.optim.SGD(self._net.parameters(), lr = 1.0)
        else:
            self._opt = torch.optim.Adam(self._net.parameters(), lr = 1.0)

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self._opt, self.lr_calculator, last_epoch=-1)




        device = self.default_device()





    def _calculate_accuracy(self, logits, minibatch_data):
        ''' Calculate the accuracy.

        '''
        accuracy = {}

        with torch.no_grad():

            if 'detect_vertex' in self.args.network and self.args.network.detect_vertex:
                # Split things up again
                logits, vertex = logits
                # First, compute the difference between target and real in
                # normalized units:
                acc_temp = (minibatch_data['vertex'] - vertex*self.image_dimensions)

                accuracy['Vx'] = torch.mean(torch.abs(acc_temp[:,0]))
                accuracy['Vy'] = torch.mean(torch.abs(acc_temp[:,1]))
                accuracy['Vz'] = torch.mean(torch.abs(acc_temp[:,2]))
                abs_acc = torch.sqrt(torch.sum(acc_temp**2, axis=-1))
                accuracy['Vertex'] = torch.mean(abs_acc)

            # Compare how often the input label and the output prediction agree:
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
        for key in accuracy:
            metrics['acc/{}'.format(key)] = accuracy[key]


        return metrics


    def increment_global_step(self):

        previous_epoch = int((self._global_step * self.args.run.minibatch_size) / self._train_data_size)
        self._global_step += 1
        current_epoch = int((self._global_step * self.args.run.minibatch_size) / self._train_data_size)

        self.on_step_end()

        if previous_epoch != current_epoch:
            self.on_epoch_end()

    def on_step_end(self):
        pass

    def on_epoch_end(self):
        pass


    def default_device_context(self):

        if self.args.run.compute_mode == "GPU":
            return torch.cuda.device(0)
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

        if self.args.run.compute_mode == "GPU":
            return torch.device("cuda")
        elif self.args.run.compute_mode == "XPU":
            device = torch.device("xpu")
        elif self.args.run.compute_mode == "DPCPP":
            device = torch.device("dpcpp")
        else:
            device = torch.device('cpu')


    def to_torch(self, minibatch_data, device=None):

        with self.default_device_context():

            device = self.default_device()

            with torch.no_grad():
                for key in minibatch_data:
                    if key == 'entries' or key =='event_ids' or key == 'particle':
                        continue
                    if key == 'image' and self.args.network.data_format == 'sparse':
                        if self.args.dataset.dimension == 3:
                            minibatch_data['image'] = [
                                    torch.tensor(minibatch_data['image'][0]).long(),
                                    torch.tensor(minibatch_data['image'][1], device=device),
                                    minibatch_data['image'][2],
                                ]
                        else:
                            minibatch_data['image'] = [
                                    torch.tensor(minibatch_data['image'][0]).long(),
                                    torch.tensor(minibatch_data['image'][1], device=device),
                                    minibatch_data['image'][2],
                                ]
                    elif key == 'image' and self.args.network.data_format == 'graph':
                        if self.args.dataset.dimension == 2:
                            minibatch_data[key] = [ torch.tensor(m, device=device) for m in minibatch_data[key]]
                        else:
                            # This is repetitive from below, but might need to be adjusted eventually.
                            minibatch_data[key] = torch.tensor(minibatch_data[key], device=device)
                    else:
                        minibatch_data[key] = torch.tensor(minibatch_data[key],device=device)

        return minibatch_data

    def forward_pass(self, minibatch_data):
        '''
        Run the forward pass for all target objectives
        '''


        minibatch_data = self.to_torch(minibatch_data)

        predictions={}

        if self.args.run.precision == "mixed" and self.args.run.compute_mode == "GPU":
            with torch.cuda.amp.autocast():
                logits = self._net(minibatch_data['image'])
        else:
            logits = self._net(minibatch_data['image'])

        if 'detect_vertex' in self.args.network and self.args.network.detect_vertex:
            logits, vertex = logits
            predictions['vertex'] = vertex

        predictions['logits'] = logits

        return predictions

    def train_step(self):


        # For a train step, we fetch data, run a forward and backward pass, and
        # if this is a logging step, we compute some logging metrics.

        self._net.train()

        global_start_time = datetime.datetime.now()

        # Reset the gradient values for this step:
        self._opt.zero_grad()

        metrics = {}
        io_fetch_time = 0.0

        grad_accum = self.args.mode.optimizer.gradient_accumulation

        for interior_batch in range(grad_accum):

            # Fetch the next batch of data with larcv
            io_start_time = datetime.datetime.now()
            minibatch_data = self.larcv_fetcher.fetch_next_batch("primary",force_pop = True)
            io_end_time = datetime.datetime.now()
            io_fetch_time = (io_end_time - io_start_time).total_seconds()

            network_predictions = self.forward_pass(minibatch_data)

            use_cuda=torch.cuda.is_available()

            losses = self.loss_calculator(minibatch_data, network_predictions)

            # # Compute the loss based on the logits
            # loss = self._calculate_loss(minibatch_data, logits)
            # if 'detect_vertex' in self.args.network and self.args.network.detect_vertex:
            #     vtx_loss = self._calculate_loss_vertex(minibatch_data, vertex)
            #     loss += vtx_loss

            # Compute the gradients for the network parameters:
            if self.args.run.precision == "mixed" and self.args.run.compute_mode == "GPU":
                self.scaler.scale(loss).backward()
            else:
                loss.backward()



            # Compute any necessary metrics:
            # if 'detect_vertex' in self.args.network and self.args.network.detect_vertex:
                # interior_metrics['loss/vertex'] = vtx_loss
                # interior_metrics['loss/classification'] = loss - vtx_loss
            interior_metrics = self._compute_metrics(logits, minibatch_data, loss)

            for key in interior_metrics:
                if key in metrics:
                    metrics[key] += interior_metrics[key]
                else:
                    metrics[key] = interior_metrics[key]

        # Here, make sure to normalize the interior metrics:
        for key in metrics:
            metrics[key] /= grad_accum

        # Add the global step / second to the tensorboard log:
        try:
            metrics['global_step_per_sec'] = 1./self._seconds_per_global_step
            metrics['images_per_second'] = grad_accum * self.args.run.minibatch_size / self._seconds_per_global_step
        except:
            metrics['global_step_per_sec'] = 0.0
            metrics['images_per_second'] = 0.0

        metrics['io_fetch_time'] = io_fetch_time / grad_accum

        step_start_time = datetime.datetime.now()
        # Apply the parameter update:
        self._opt.step()
        self.lr_scheduler.step()
        global_end_time = datetime.datetime.now()

        metrics['step_time'] = (global_end_time - step_start_time).total_seconds()


        self.log(metrics, kind="train")

        self.summary(metrics, kind="train")

        # Compute global step per second:
        self._seconds_per_global_step = (global_end_time - global_start_time).total_seconds()

        # Increment the global step value:
        self.increment_global_step()



        return metrics

    def summary(self, metrics, kind):
        '''kind should be a string
        metrics should be a dict
        '''

        if kind in self._savers:
            for metric in metrics.keys():
                self._savers[kind].add_scalar(metric, metrics[metric], self._global_step)

            # try to get the learning rate
            if kind == "train":
                self._savers[kind].add_scalar("learning_rate", self._opt.state_dict()['param_groups'][0]['lr'], self._global_step)
            return

    def val_step(self, n_iterations=1):

        # First, validation only occurs on training:
        if self.args.mode.name != "train": return

        # Second, validation can not occur without a validation dataloader.
        if self._val_data_size is None: return

        self._net.eval()

        with torch.no_grad():

            if self._global_step != 0 and self._global_step % self.args.run.aux_iterations == 0:


                # Fetch the next batch of data with larcv
                # (Make sure to pull from the validation set)
                minibatch_data = self.larcv_fetcher.fetch_next_batch("val",force_pop = True)
                # Convert the input data to torch tensors
                minibatch_data = self.to_torch(minibatch_data)

                # Run a forward pass of the model on the input image:
                logits = self._net(minibatch_data['image'])


                if 'detect_vertex' in self.args.network and self.args.network.detect_vertex:
                    logits, vertex = logits



                # Compute the loss
                loss = self._calculate_loss(minibatch_data, logits)
                if 'detect_vertex' in self.args.network and self.args.network.detect_vertex:
                    vtx_loss = self._calculate_loss_vertex(minibatch_data, vertex)

                    loss += vtx_loss
                    # Compute the metrics for this iteration:
                    metrics = self._compute_metrics((logits, vertex), minibatch_data, loss)
                    metrics['loss/vertex'] = vtx_loss
                    metrics['loss/classification'] = loss - vtx_loss
                else:
                    metrics = self._compute_metrics(logits, minibatch_data, loss)



                self.log(metrics, kind="val")
                self.summary(metrics, kind="val")

                return metrics

    def stop(self):
        # Mostly, this is just turning off the io:
        # self.larcv_fetcher.stop()
        pass

    def checkpoint(self):

        if self.args.mode.checkpoint_iteration == -1:
            return

        if self._global_step % self.args.mode.checkpoint_iteration == 0 and self._global_step != 0:
            # Save a checkpoint, but don't do it on the first pass
            self.save_model()

    def restore_model(self):

        state = self.load_state_from_file()

        if state is not None:
            self.restore_state(state)


    def load_state_from_file(self):
        ''' This function attempts to restore the model from file
        '''

        _, checkpoint_file_path = self.get_model_filepath()


        if not os.path.isfile(checkpoint_file_path):
            return None
        # Parse the checkpoint file and use that to get the latest file path

        with open(checkpoint_file_path, 'r') as _ckp:
            for line in _ckp.readlines():
                if line.startswith("latest: "):
                    chkp_file = line.replace("latest: ", "").rstrip('\n')
                    chkp_file = os.path.dirname(checkpoint_file_path) + "/" + chkp_file
                    logger.info(f"Restoring weights from {chkp_file}")
                    break
        try:
            state = torch.load(chkp_file)
            return state
        except:
            logger.info("Could not load from checkpoint file, starting fresh.")
            return None

    def restore_state(self, state):



        new_state_dict = {}
        for key in state['state_dict']:
            if key.startswith("module."):
                new_key = key.lstrip("module.")
            else:
                new_key = key
            new_state_dict[new_key] = state['state_dict'][key]

        state['state_dict'] = new_state_dict

        self._net.load_state_dict(state['state_dict'])

        if self.args.mode.name == "train":
            self._opt.load_state_dict(state['optimizer'])
            self.lr_scheduler.load_state_dict(state['scheduler'])

        self._global_step = state['global_step']

        device = self.default_device()

        # If using GPUs, move the model to GPU:
        if self.args.run.compute_mode == "GPU" and self.args.mode.name == "train":
            for state in self._opt.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)

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
            'scheduler'   : self.lr_scheduler.state_dict(),
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
        file_path= self.args.run.output_dir  + "/checkpoints/"


        name = file_path + 'model-{}.ckpt'.format(self._global_step)
        checkpoint_file_path = file_path + "checkpoint"

        return name, checkpoint_file_path



    def batch_process(self):


        # If we're not training, force the number of iterations to the epoch size or less
        if self.args.mode.name != "train":
            if self.args.run.iterations > int(self._train_data_size/self.args.minibatch_size) + 1:
                self.args.run.iterations = int(self._train_data_size/self.args.minibatch_size) + 1
                logger.info(f'Number of iterations set to {self.args.run.iterations}')


        start = time.time()
        # Run iterations
        for i in range(self.args.run.iterations):
            if self.args.mode.name == "train" and self._iteration >= self.args.run.iterations:
                logger.info('Finished training (iteration %d)' % self._iteration)
                self.checkpoint()
                break

            if self.args.mode.name == "train":
                self.val_step()
                self.train_step()
                self.checkpoint()
            else:
                self.ana_step(i)

        if self.args.mode.name == "train":
            self.save_model()

        logger.info(f"Total time to batch process: { time.time() - start}")

        # Close savers:
        for saver in self._savers.keys():
            self._savers[saver].flush()
            self._savers[saver].close()
