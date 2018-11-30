import os
import sys
import time
from collections import OrderedDict

import numpy

import torch
import code

from larcv import larcv_interface

from . import flags
FLAGS = flags.FLAGS()

class trainercore(object):
    '''
    This class is the core interface for training.  Each function to
    be overridden for a particular interface is marked and raises
    a NotImplemented error.

    '''
    def __init__(self,):
        self._larcv_interface = larcv_interface.larcv_interface()
        self._iteration       = 0

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


    def _construct_graph(self):

        # Net construction:
        start = time.time()
        sys.stdout.write("Begin constructing network\n")

        # Make sure all required dimensions are present:

        dims = self._larcv_interface.fetch_minibatch_dims('primary')


        # Call the function to define the inputs
        self._input   = self._initialize_input(dims)

        # Apply a softmax and argmax:
        self._outputs = self._create_softmax(self._logits)

        self._accuracy = self._calculate_accuracy(self._input, self._outputs)

        # Create the loss function
        self._loss    = self._calculate_loss(self._input, self._logits, self._transformations)



        end = time.time()
        sys.stdout.write("Done constructing network. ({0:.2}s)\n".format(end-start))


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
        data = self.fetch_next_batch()['image']

        input_tensor = torch.Tensor(data)

        logits = self._net.forward(input_tensor)

        print(logits.shape)
        code.interact(local=locals())

        print (self._net.parameters())

        n_trainable_parameters = 0
        for var in self._net.parameters():
            n_trainable_parameters += numpy.prod(var.shape)
        print("Total number of trainable parameters in this network: {}".format(n_trainable_parameters))

        # self._construct_graph()

        # Create an optimizer:
        if FLAGS.LEARNING_RATE <= 0:
            opt = torch.optim.Adam(self._net.parameters())
        else:
            opt = tf.train.AdamOptimizer(FLAGS.LEARNING_RATE)

        self._global_step = tf.train.get_or_create_global_step()
        self._train_op = opt.minimize(self._loss, self._global_step)


        hooks = self.get_standard_hooks()

        config = tf.ConfigProto()

        if FLAGS.COMPUTE_MODE == "CPU":
            pass
        if FLAGS.COMPUTE_MODE == "GPU":
            self.model.cuda()

        # self._sess = tf.train.MonitoredTrainingSession(config=config)
        self._sess = tf.train.MonitoredTrainingSession(config=config, 
            hooks = hooks,
            checkpoint_dir= "{}/checkpoints/".format(FLAGS.LOG_DIRECTORY),
            save_checkpoint_steps=FLAGS.CHECKPOINT_ITERATION)


    def get_standard_hooks(self):

        print("LOG_DIRECTORY: ", FLAGS.LOG_DIRECTORY)
        loss_is_nan_hook = tf.train.NanTensorHook(
            self._loss,
            fail_on_nan_loss=True,
        )

        # Create a hook to manage the summary saving:
        summary_saver_hook = tf.train.SummarySaverHook(
            save_steps = FLAGS.SUMMARY_ITERATION,
            output_dir = FLAGS.LOG_DIRECTORY,
            summary_op = tf.summary.merge_all()
            )


        # Create a profiling hook for tracing:
        profile_hook = tf.train.ProfilerHook(
            save_steps    = FLAGS.PROFILE_ITERATION,
            output_dir    = FLAGS.LOG_DIRECTORY,
            show_dataflow = True,
            show_memory   = True
        )

        logging_hook = tf.train.LoggingTensorHook(
            tensors       = { 'global_step' : self._global_step,
                              'accuracy'    : self._accuracy,
                              'loss'        : self._loss},
            every_n_iter  = FLAGS.LOGGING_ITERATION,
            )

        hooks = [
            loss_is_nan_hook,
            summary_saver_hook,
            profile_hook,
            logging_hook,
        ]

        return hooks

    # def _initialize_input(self, dims):
    #     '''Initialize input parameters of the network.  Must return a dict type

    #     For exampe, paremeters of the dict can be 'image', 'label', 'weights', etc

    #     Arguments:
    #         dims {[type]} -- [description]

    #     Keyword Arguments:
    #         label_dims {[type]} -- [description] (default: {None})

    #     Raises:
    #         NotImplementedError -- [description]
    #     '''

    #     inputs = dict()

    #     batch_size = self._larcv_interface.fetch_minibatch_dims('primary')['label'][0]

    #     inputs.update({
    #         'image' : torch.tensor(tf.float32, [batch_size, None, 3]),
    #         'label' :  tf.placeholder(tf.int64,   dims['label'], name="input_label"),
    #         }
    #     )

    #     # inputs.update({
    #     #     'image' :  tf.placeholder(tf.float32, dims['image'], name="input_image"),
    #     #     'label' :  tf.placeholder(tf.int64,   dims['label'], name="input_label"),
    #     # })




    #     return inputs


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

        output['softmax'] = tf.nn.softmax(logits)
        output['prediction'] = tf.argmax(logits, axis=-1)


        return output



    # def compute_weights(self, labels, boost_labels = None):
    #     '''
    #     This is NOT a tensorflow implementation, but a numpy implementation.
    #     Running on CPUs this might not make a difference.  Running on GPUs
    #     it might be good to move this to a GPU, but I suspect it's not needed.
    #     '''
    #     # Take the labels, and compute the per-label weight


    #     # Prepare output weights:
    #     weights = numpy.zeros(labels.shape)

    #     i = 0
    #     for batch in labels:
    #         # First, figure out what the labels are and how many of each:
    #         values, counts = numpy.unique(batch, return_counts=True)

    #         n_pixels = numpy.sum(counts)
    #         for value, count in zip(values, counts):
    #             weight = 1.0*(n_pixels - count) / n_pixels
    #             if boost_labels is not None and value in boost_labels.keys():
    #                 weight *= boost_labels[value]
    #             mask = labels[i] == value
    #             weights[i, mask] += weight
    #         weights[i] *= 1. / numpy.sum(weights[i])
    #         i += 1



    #     # Normalize the weights to sum to 1 for each event:
    #     return weights



    def _calculate_loss(self, inputs, logits, transformations):
        ''' Calculate the loss.

        returns a single scalar for the optimizer to use.
        '''


        with tf.name_scope('cross_entropy'):

            # Traditional Classification loss:            
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(labels=inputs['label'],
                                                           logits=logits)
                )
            tf.summary.scalar("Classification_Loss",loss)

            # If desired, add weight regularization loss:
            if FLAGS.REGULARIZE_WEIGHTS != 0.0:
                reg_loss = tf.reduce_mean(tf.losses.get_regularization_losses())
                tf.summary.scalar("Weight_Regularization", reg_loss)
                loss += reg_loss

            # Add a regularization loss against the matrix transformations:
            t_loss = None
            for transformation in transformations:
                mat_dim = transformation.get_shape().as_list()[-1]
                batch_size = self._larcv_interface.fetch_minibatch_dims(FLAGS.MODE)['label'][0]
                difference_to_identity = tf.eye(mat_dim, batch_shape = [batch_size]) 
                difference_to_identity -= tf.matmul(transformation, tf.matrix_transpose(transformation))
                this_t_loss = tf.reduce_mean(tf.nn.l2_normalize(difference_to_identity))
                this_t_loss = FLAGS.REGULARIZE_TRANSFORMS*this_t_loss
                if t_loss is None:
                    t_loss = this_t_loss
                else:
                    t_loss += this_t_loss

            if t_loss is not None:
                tf.summary.scalar("Transformation_Loss", t_loss)
                loss += t_loss

            # Total summary:
            tf.summary.scalar("Total_Loss",loss)

            return loss




    def _calculate_accuracy(self, inputs, outputs):
        ''' Calculate the accuracy.  Computes total average accuracy,
        accuracy on just non zero pixels, and accuracy on just neutrino pixels.

        '''

        # Compare how often the input label and the output prediction agree:


        with tf.name_scope('accuracy'):

            correct_prediction = tf.equal(tf.argmax(inputs['label'], -1),
                                          outputs['prediction'])
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar("Accuracy", accuracy)

        return accuracy




    def feed_dict(self, inputs):
        '''Build the feed dict

        Take input images, labels and match
        to the correct feed dict tensorrs

        This is probably overridden in the subclass, but here you see the idea

        Arguments:
            images {dict} -- Dictionary containing the input tensors

        Returns:
            [dict] -- Feed dictionary for a tf session run call

        '''
        fd = dict()

        for key in inputs:
            if inputs[key] is not None:
                fd.update({self._input[key] : inputs[key]})

        return fd


    def image_to_point_cloud(self, image):

        # This function maps an image to a point cloud


        # Have to do this image by image to get the right shape:
        outputs = []
        max_points= 0
        for b in range(image.shape[0]):

            non_zero_locs = list(numpy.where(image[b] != 0))
            # Calculate the values and put them last:
            values = image[b][tuple(non_zero_locs)]
            if values.shape[0] > max_points:
                max_points = values.shape[0]
            # Merge everything into an N by 2 array:
            non_zero_locs.append(values)
            point_dim_len = len(non_zero_locs)
            # Stack everythin together and take the transpose:
            res = numpy.stack((non_zero_locs)).T
            outputs.append(res)

        # Stack the results into a uniform numpy array padded with zeros:
        output = numpy.zeros([len(outputs), max_points, point_dim_len])
        for i, o in enumerate(outputs):
            npoints = len(o)
            output[i,0:npoints,:] = o

        return output


    def fetch_next_batch(self, mode='primary'):

        minibatch_data = self._larcv_interface.fetch_minibatch_data(mode)
        minibatch_dims = self._larcv_interface.fetch_minibatch_dims(mode)


        for key in minibatch_data:
            minibatch_data[key] = numpy.reshape(minibatch_data[key], minibatch_dims[key])


        return minibatch_data

    def train_step(self):


        minibatch_data = self.fetch_next_batch()

        self._sess.run(self._train_op, 
                       feed_dict = self.feed_dict(inputs = minibatch_data))


    def stop(self):
        # Mostly, this is just turning off the io:
        self._larcv_interface.stop()

    def batch_process(self):

        # Run iterations
        for i in range(FLAGS.ITERATIONS):
            if FLAGS.TRAINING and self._iteration >= FLAGS.ITERATIONS:
                print('Finished training (iteration %d)' % self._iteration)
                break

            # Start IO thread for the next batch while we train the network
            self.train_step()
