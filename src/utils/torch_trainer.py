import os
import sys
import time
from collections import OrderedDict

import numpy

import torch

import datetime

from .iocore import iocore

# This uses tensorboardX to save summaries and metrics to tensorboard compatible files.

import tensorboardX

class torch_trainer(iocore):
    '''
    This class is the core interface for training.  Each function to
    be overridden for a particular interface is marked and raises
    a NotImplemented error.

    '''
    def __init__(self, args):
        iocore.__init__(self, args)

        # self.larcv_fetcher = threadloader.thread_interface()
        self._iteration       = 0.
        self._global_step     = -1.




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

        self.print(self._net)


        if self.args.training:
            self._net.train(True)

        if self.args.compute_mode == "CPU":
            pass
        if self.args.compute_mode == "GPU":
            self._net.cuda()

    def build_lr_schedule(self, learning_rate_schedule = None):
        # Define the learning rate sequence:

        if learning_rate_schedule is None:
            learning_rate_schedule = {
                'warm_up' : {
                    'function'      : 'linear',
                    'start'         : 0,
                    'n_epochs'      : 1,
                    'initial_rate'  : 0.00001,
                },
                'flat' : {
                    'function'      : 'flat',
                    'start'         : 1,
                    'n_epochs'      : 20,
                },
                'decay' : {
                    'function'      : 'decay',
                    'start'         : 21,
                    'n_epochs'      : 4,
                    'floor'         : 0.00001,
                    'decay_rate'    : 0.999
                },
            }


        # one_cycle_schedule = {
        #     'ramp_up' : {
        #         'function'      : 'linear',
        #         'start'         : 0,
        #         'n_epochs'      : 10,
        #         'initial_rate'  : 0.00001,
        #         'final_rate'    : 0.001,
        #     },
        #     'ramp_down' : {
        #         'function'      : 'linear',
        #         'start'         : 10,
        #         'n_epochs'      : 10,
        #         'initial_rate'  : 0.001,
        #         'final_rate'    : 0.00001,
        #     },
        #     'decay' : {
        #         'function'      : 'decay',
        #         'start'         : 20,
        #         'n_epochs'      : 5,
        #         'rate'          : 0.00001
        #         'floor'         : 0.00001,
        #         'decay_rate'    : 0.99
        #     },
        # }
        # learning_rate_schedule = one_cycle_schedule

        # We build up the functions we need piecewise:
        func_list = []
        cond_list = []

        for i, key in enumerate(learning_rate_schedule):

            # First, create the condition for this stage
            start    = learning_rate_schedule[key]['start']
            length   = learning_rate_schedule[key]['n_epochs']

            if i +1 == len(learning_rate_schedule):
                # Make sure the condition is open ended if this is the last stage
                condition = lambda x, s=start, l=length: x >= s
            else:
                # otherwise bounded
                condition = lambda x, s=start, l=length: x >= s and x < s + l


            if learning_rate_schedule[key]['function'] == 'linear':

                initial_rate = learning_rate_schedule[key]['initial_rate']
                if 'final_rate' in learning_rate_schedule[key]: final_rate = learning_rate_schedule[key]['final_rate']
                else: final_rate = self.args.learning_rate

                function = lambda x, s=start, l=length, i=initial_rate, f=final_rate : numpy.interp(x, [s, s + l] ,[i, f] )

            elif learning_rate_schedule[key]['function'] == 'flat':
                if 'rate' in learning_rate_schedule[key]: rate = learning_rate_schedule[key]['rate']
                else: rate = self.args.learning_rate

                function = lambda x : rate

            elif learning_rate_schedule[key]['function'] == 'decay':
                decay    = learning_rate_schedule[key]['decay_rate']
                floor    = learning_rate_schedule[key]['floor']
                if 'rate' in learning_rate_schedule[key]: rate = learning_rate_schedule[key]['rate']
                else: rate = self.args.learning_rate

                function = lambda x, s=start, d=decay, f=floor: (rate-f) * numpy.exp( -(d * (x - s))) + f

            cond_list.append(condition)
            func_list.append(function)

        self.lr_calculator = lambda x: numpy.piecewise(
            x * (self.args.minibatch_size / self._train_data_size),
            [c(x * (self.args.minibatch_size / self._train_data_size)) for c in cond_list], func_list)


    def initialize(self, io_only=False):


        iocore.initialize(self, io_only)

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

        self.build_lr_schedule()

        # Create an optimizer:
        if self.args.optimizer == "SDG":
            self._opt = torch.optim.SGD(self._net.parameters(), lr = 1.0,
                weight_decay=self.args.weight_decay)
        else:
            self._opt = torch.optim.Adam(self._net.parameters(), lr = 1.0,
                weight_decay=self.args.weight_decay)

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self._opt, self.lr_calculator, last_epoch=-1)




        device = self.get_device()

        if self.args.loss_mode == "focal":
            reduction = "none"
        else:
            reduction = "mean"

        # here we store the loss weights:
        if self.args.label_mode == 'all':
            self._criterion = torch.nn.CrossEntropyLoss(reduction=reduction)


        elif self.args.label_mode == 'split':
            # # These are the raw category occurences
            self._label_weights = {
                'label_cpi'  : torch.tensor([50932., 61269.], device=device),
                'label_prot' : torch.tensor([36583., 46790., 28828.], device=device),
                'label_npi'  : torch.tensor([70572., 41629.], device=device),
                'label_neut' : torch.tensor([39452., 39094., 33655.], device=device)
            }
            #
            self._criterion = torch.nn.CrossEntropyLoss(reduction = reduction)

    def focal_loss(self, loss, logits, target, num_classes):

        softmax = torch.nn.functional.softmax(logits.float(), dim=1)
        onehot  = torch.nn.functional.one_hot(target, num_classes=num_classes)
        weights = onehot * (1 - softmax)**2
        weights = torch.mean(weights, dim=1)

        loss = weights * loss
        loss = torch.mean(loss)

        return loss


    def _calculate_loss(self, inputs, logits):
        ''' Calculate the loss.

        returns a single scalar for the optimizer to use.
        '''

        self.num_classes = {
            'label_cpi' : 2,
            'label_prot' : 3,
            'label_npi' : 2,
            'label_neut' : 3,
        }



        if self.args.label_mode == 'all':
            values, target = torch.max(inputs[self.args.keyword_label], dim = 1)
            loss = self._criterion(logits, target=target)
            if self.args.loss_mode == "focal":
                loss = self.focal_loss(loss, logits, target, num_classes = 36)
        elif self.args.label_mode == 'split':
            loss = None
            for key in logits:
                values, target = torch.max(inputs[key], dim=1)
                temp_loss = self._criterion(logits[key], target = target)
                if self.args.loss_mode == "focal":
                    temp_loss = self.focal_loss(temp_loss, logits[key], target, num_classes=self.num_classes[key])

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

        first_param = next(self._net.parameters())

        # print(first_param)
        # print(first_param.grad)

        # print("First layer mean of grad ", torch.mean(torch.abs(first_param)))
        # print("First layer mean of parameter", torch.mean(torch.abs(first_param.grad)))
        # print("First layer mean ratio of grad to parameter", torch.mean(torch.abs(first_param.grad) / torch.abs(first_param)))

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
        self.lr_scheduler.step()

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
                minibatch_data = self.larcv_fetcher.fetch_next_batch("aux",force_pop = True)

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

        start = time.time()
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

        self.print("Total time to batch process: ", time.time() - start)

        if self.args.training:
            if self._saver is not None:
                self._saver.close()
            if self._aux_saver is not None:
                self._aux_saver.close()
