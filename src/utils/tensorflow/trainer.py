import os
import sys
import time
import tempfile
from collections import OrderedDict

import numpy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'

from src.utils.core.trainercore import trainercore

import datetime



import tensorflow as tf

floating_point_format = tf.float32
integer_format = tf.int64

import logging
logger = logging.getLogger()


class trainer(trainercore):
    '''
    This is the tensorflow version of the trainer

    '''

    def __init__(self, args):
        trainercore.__init__(self, args)
        self._rank = None

    def local_batch_size(self):
        return self.args.run.minibatch_size

    def init_network(self):

        # This function builds the compute graph.
        # Optionally, it can build a 'subset' graph if this mode is

        # Net construction:
        start = time.time()

        # Here, if using mixed precision, set a global policy:
        if self.args.run.precision == "mixed":
            from tensorflow.keras.mixed_precision import experimental as mixed_precision
            self.policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_policy(self.policy)

        if self.args.run.precision == "bfloat16":
            from tensorflow.keras.mixed_precision import experimental as mixed_precision
            self.policy = mixed_precision.Policy('mixed_bfloat16')
            mixed_precision.set_policy(self.policy)


        #
        self._global_step = tf.Variable(0, dtype=tf.int64)


        # Add the dataformat for the network construction:

        # This sets up the necessary output shape:
        output_shape = self.larcv_fetcher.output_shape('primary')

        # Build the network object, forward pass only:
        # To initialize the network, we see what the name is
        # and act on that:
        if self.args.network.name == "resnet":
            if self.args.network.data_format == 'sparse':
                raise Exception("No sparse networks available in tensorflow")
            else:
                if self.args.dataset.dimension == 2:
                    from src.networks.tensorflow import resnet
                    self._net = resnet.ResNet(output_shape, self.args)
                else:
                    raise Exception("No Resnet3d Implemented!")
        elif self.args.network.name == "pointnet":
            if self.args.dataset.dimension == 2:
                from src.networks.tensorflow import pointnet
                self._net = pointnet.PointNet(output_shape, self.args)
            else:
                from src.networks.tensorflow import pointnet3d
                self._net = pointnet3d.PointNet(output_shape, self.args)
        elif self.args.network.name == "dgcnn":
            from src.networks.tensorflow import dgcnn
            self._net = dgcnn.DGCNN(output_shape, self.args)
        else:
            raise Exception(f"Couldn't identify network {self.args.network.name}")

        self._net.trainable = True

        # self._logits = self._net(self._input['image'], training=self.args.mode.name == "train")

        # # If channels first, need to permute the logits:
        # if self._channels_dim == 1:
        #     permutation = tf.keras.layers.Permute((2, 3, 1))
        #     self._loss_logits = [ permutation(l) for l in self._logits ]
        # else:
        #     self._loss_logits = self._logits


        # TO PROPERLY INITIALIZE THE NETWORK, NEED TO DO A FORWARD PASS
        minibatch_data = self.larcv_fetcher.fetch_next_batch("primary",force_pop=False)
        minibatch_data = self.cast_input(minibatch_data)

        self.forward_pass(minibatch_data['image'], training=False)


        self.acc_calculator  = AccuracyCalculator.AccuracyCalculator()
        self.loss_calculator = LossCalculator.LossCalculator(
            self.args.mode.optimizer.loss_balance_scheme, self._channels_dim)

        self._log_keys = ["loss", "Average/Non_Bkg_Accuracy", "Average/mIoU"]

        end = time.time()
        return end - start

    def print_network_info(self, verbose=False):
        n_trainable_parameters = 0
        for var in self._net.variables:
            n_trainable_parameters += numpy.prod(var.get_shape())
            if verbose:
                logger.debug(var.name, var.get_shape())
        logger.info(f"Total number of trainable parameters in this network: {n_trainable_parameters}")


    def n_parameters(self):
        n_trainable_parameters = 0
        for var in tf.compat.v1.trainable_variables():
            n_trainable_parameters += numpy.prod(var.get_shape())

        return n_trainable_parameters

    def current_step(self):
        return self._global_step


    def set_compute_parameters(self):

        self._config = tf.compat.v1.ConfigProto()

        if self.args.run.compute_mode == "CPU":
            self._config.inter_op_parallelism_threads = self.args.framework.inter_op_parallelism_threads
            self._config.intra_op_parallelism_threads = self.args.framework.intra_op_parallelism_threads
        elif self.args.run.compute_mode == "GPU":
            gpus = tf.config.experimental.list_physical_devices('GPU')


            # The code below is for MPS mode.  It is a bit of a hard-coded
            # hack.  Use with caution since the memory limit is set by hand.
            ####################################################################
            # print(gpus)
            # if gpus:
            #   # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
            #   try:
            #     tf.config.experimental.set_virtual_device_configuration(
            #         gpus[0],
            #         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=15000)])
            #     # tf.config.experimental.set_memory_growth(gpus[0], True)
            #     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            #     # print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            #   except RuntimeError as e:
            #     # Virtual devices must be set before GPUs have been initialized
            #     print(e)
            ####################################################################



    def initialize(self, io_only=False):


        self._initialize_io(color=0)


        if io_only:
            return

        if self.args.mode.name == "train":
            self.build_lr_schedule()

        start = time.time()

        net_time = self.init_network()

        logger.info("Done constructing network. ({0:.2}s)\n".format(time.time()-start))


        self.print_network_info()

        if self.args.mode != "inference":
            self.init_optimizer()

        self.init_saver()

        self.set_compute_parameters()


        # Try to restore a model?
        restored = self.restore_model()


    def init_learning_rate(self):
        # Use a place holder for the learning rate :
        self._learning_rate = tf.Variable(initial_value=0.0, trainable=False, dtype=floating_point_format, name="lr")



    def restore_model(self):
        ''' This function attempts to restore the model from file
        '''

        file_path = self.get_checkpoint_dir()

        path = tf.train.latest_checkpoint(file_path)


        if path is None:
            logger.info("No checkpoint found, starting from scratch")
            return False
        # Parse the checkpoint file and use that to get the latest file path
        logger.info(f"Restoring checkpoint from {path}")
        self._net.load_weights(path)

        # self.scheduler.set_current_step(self.current_step())

        return True

    def checkpoint(self):

        gs = int(self.current_step().numpy())

        if gs % self.args.mode.checkpoint_iteration == 0 and gs != 0:
            # Save a checkpoint, but don't do it on the first pass
            self.save_model(gs)

    def get_checkpoint_dir(self):

        # Find the base path of the log directory
        file_path= self.args.run.output_dir + "/checkpoints/"

        return file_path

    def save_model(self, global_step):
        '''Save the model to file

        '''

        file_path = self.get_checkpoint_dir()

        # # Make sure the path actually exists:
        # if not os.path.isdir(os.path.dirname(file_path)):
        #     os.makedirs(os.path.dirname(file_path))

        saved_path = self._net.save_weights(file_path + "model_{}.ckpt".format(global_step))


    def get_model_filepath(self, global_step):
        '''Helper function to build the filepath of a model for saving and restoring:

        '''

        file_path = self.get_checkpoint_dir()

        name = file_path + 'model-{}.ckpt'.format(global_step)
        checkpoint_file_path = file_path + "checkpoint"

        return name, checkpoint_file_path


    def init_saver(self):

        file_path = self.get_checkpoint_dir()

        try:
            os.makedirs(file_path)
        except:
            logger.warning("Could not make file path")

        # # Create a saver for snapshots of the network:
        # self._saver = tf.compat.v1.train.Saver()

        # Create a file writer for training metrics:
        self._main_writer = tf.summary.create_file_writer(self.args.run.output_dir +  "/train/")

        # Additionally, in training mode if there is aux data use it for validation:
        if hasattr(self, "_aux_data_size"):
            self._val_writer = tf.summary.create_file_writer(self.args.run.output_dir + "/test/")


    def init_optimizer(self):

        self.init_learning_rate()


        if self.args.mode.optimizer.name == "rmsprop":
            # Use RMS prop:
            self._opt = tf.keras.optimizers.RMSprop(self._learning_rate)
        else:
            # default is Adam:
            self._opt = tf.keras.optimizers.Adam(self._learning_rate)

        if self.args.run.precision == "mixed":
            from tensorflow.keras.mixed_precision import experimental as mixed_precision
            self._opt = mixed_precision.LossScaleOptimizer(self._opt, loss_scale='dynamic')


        self.tape = tf.GradientTape()

    def _compute_metrics(self, logits, prediction, labels, loss):

        # self._output['softmax'] = [ tf.nn.softmax(x) for x in self._logits]
        # self._output['prediction'] = [ tf.argmax(input=x, axis=self._channels_dim) for x in self._logits]
        accuracy = self.acc_calculator(prediction=prediction, labels=labels)

        metrics = {}
        for p in [0,1,2]:
            metrics[f"plane{p}/Total_Accuracy"]          = accuracy["total_accuracy"][p]
            metrics[f"plane{p}/Non_Bkg_Accuracy"]        = accuracy["non_bkg_accuracy"][p]
            metrics[f"plane{p}/Neutrino_IoU"]            = accuracy["neut_iou"][p]
            metrics[f"plane{p}/Cosmic_IoU"]              = accuracy["cosmic_iou"][p]

        metrics["Average/Total_Accuracy"]          = float(tf.reduce_mean(accuracy["total_accuracy"]).numpy())
        metrics["Average/Non_Bkg_Accuracy"]        = float(tf.reduce_mean(accuracy["non_bkg_accuracy"]).numpy())
        metrics["Average/Neutrino_IoU"]            = float(tf.reduce_mean(accuracy["neut_iou"]).numpy())
        metrics["Average/Cosmic_IoU"]              = float(tf.reduce_mean(accuracy["cosmic_iou"]).numpy())
        metrics["Average/mIoU"]                    = float(tf.reduce_mean(accuracy["miou"]).numpy())

        metrics['loss'] = loss

        return metrics

    def log(self, metrics, kind, step):

        log_string = ""

        log_string += "{} Global Step {}: ".format(kind, step)


        for key in metrics:
            if key in self._log_keys and key != "global_step":
                log_string += "{}: {:.3}, ".format(key, metrics[key])

        if kind == "Train":
            log_string += "Img/s: {:.2} ".format(metrics["images_per_second"])
            log_string += "IO: {:.2} ".format(metrics["io_fetch_time"])
        else:
            log_string.rstrip(", ")

        logger.info(log_string)

        return


    # @tf.function
    def cast_input(self, minibatch_data):
        if self.args.run.precision == "float32" or self.args.run.precision == "mixed":
            input_dtype = tf.float32
        elif self.args.run.precision == "bfloat16":
            input_dtype = tf.bfloat16


        for key in minibatch_data.keys():
            if key == 'entries' or key == 'event_ids': continue

            if key == 'image' and self.args.network.data_format == 'graph':
                if self.args.dataset.dimension == 2:
                    minibatch_data[key] = [ tf.convert_to_tensor(m, dtype=input_dtype) for m in minibatch_data[key]]
                else:
                    # This is repetitive from below, but might need to be adjusted eventually.
                    minibatch_data[key] = tf.convert_to_tensor(minibatch_data[key], dtype=input_dtype)
            else:
                if key == 'image' or 'label' in key:
                    minibatch_data[key]= tf.convert_to_tensor(minibatch_data[key], dtype=input_dtype)

        return minibatch_data

    @tf.function
    def forward_pass(self, image, training):

        # Run a forward pass of the model on the input image:
        logits = self._net(image, training=training)

        prediction = [tf.argmax(l, axis=self._channels_dim, output_type = tf.dtypes.int32) for l in logits]

        return logits, prediction

    @tf.function
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



        loss = None
        for key in logits:
            values, target = tf.argmax(inputs[key], dim=1)
            temp_loss = self._criterion(logits[key], target = target)

            if loss is None:
                loss = temp_loss
            else:
                loss += temp_loss

        return loss


    # @tf.function(experimental_relax_shapes=True)
    def summary(self, metrics, saver=""):

        if self.current_step() % self.args.mode.summary_iteration == 0:

            if saver == "":
                saver = self._main_writer

            with saver.as_default():
                for metric in metrics:
                    name = metric
                    tf.summary.scalar(metric, metrics[metric], self.current_step())
        return

    # @tf.function
    def get_gradients(self, loss, tape, trainable_weights):

        return tape.gradient(loss, self._net.trainable_weights)

    @tf.function
    def apply_gradients(self, gradients):
        self._opt.apply_gradients(zip(gradients, self._net.trainable_variables))




    def metrics(self, metrics):
        # This function looks useless, but it is not.
        # It allows a handle to the distributed network to allreduce metrics.
        return metrics

    def val_step(self):

        if not hasattr(self, "_aux_data_size"):
            return

        if self.args.data.synthetic:
            return

        if self._val_writer is None:
            return

        gs = self.current_step()

        if gs % self.args.run.aux_iterations == 0:

            # Fetch the next batch of data with larcv
            minibatch_data = self.larcv_fetcher.fetch_next_batch('aux', force_pop = True)
            image, label = self.cast_input(minibatch_data['image'], minibatch_data['label'])

            labels, logits, prediction = self.forward_pass(image, label, training=False)

            loss = self.loss_calculator(labels, logits)


            metrics = self._compute_metrics(logits, prediction, labels, loss)


            # Report metrics on the terminal:
            self.log(metrics, kind="Test", step=int(self.current_step().numpy()))


            self.summary(metrics=metrics, saver=self._val_writer)
            self.summary_images(labels, prediction, saver=self._val_writer)

        return


    @tf.function
    def gradient_step(self, image, label):

        with self.tape:
            labels, logits, prediction = self.forward_pass(image, label, training=True)

            # The loss function has to be in full precision or automatic mixed.
            # bfloat16 is not supported
            if self.args.run.precision == "bfloat16":
                logits = [ tf.cast(l, dtype=tf.float32) for  l in logits ]

            loss = self.loss_calculator(labels, logits)
        #
            if self.args.run.precision == "mixed":
                scaled_loss = self._opt.get_scaled_loss(loss)

        # Do the backwards pass for gradients:
        if self.args.run.precision == "mixed":
            scaled_gradients = self.get_gradients(scaled_loss, self.tape, self._net.trainable_weights)
            gradients = self._opt.get_unscaled_gradients(scaled_gradients)
        else:
            gradients = self.get_gradients(loss, self.tape, self._net.trainable_weights)

        return logits, labels, prediction, loss, gradients

    def train_step(self):

        global_start_time = datetime.datetime.now()

        io_fetch_time = 0.0

        gradients = None
        metrics = {}

        gradient_accumulation = self.args.mode.optimizer.gradient_accumulation
        for i in range(gradient_accumulation):

            # Fetch the next batch of data with larcv
            io_start_time = datetime.datetime.now()
            minibatch_data = self.larcv_fetcher.fetch_next_batch("primary",force_pop=True)
            image, label = self.cast_input(minibatch_data['image'], minibatch_data['label'])

            io_end_time = datetime.datetime.now()
            io_fetch_time += (io_end_time - io_start_time).total_seconds()

            if self.args.run.profile:
                if not self.args.distributed or self._rank == 0:
                    tf.profiler.experimental.start(self.args.run.output_dir + "/train/")
            logits, labels, prediction, loss, internal_gradients = self.gradient_step(image, label)

            if self.args.run.profile:
                if not self.args.distributed or self._rank == 0:
                    tf.profiler.experimental.stop()

            # Accumulate gradients if necessary
            if gradients is None:
                gradients = internal_gradients
            else:
                gradients += internal_gradients


            # Compute any necessary metrics:
            interior_metrics = self._compute_metrics(logits, prediction, labels, loss)

            for key in interior_metrics:
                if key in metrics:
                    metrics[key] += interior_metrics[key]
                else:
                    metrics[key] = interior_metrics[key]

        # Normalize the metrics:
        for key in metrics:
            metrics[key] /= gradient_accumulation

        # Add the global step / second to the tensorboard log:
        try:
            metrics['global_step_per_sec'] = 1./self._seconds_per_global_step
            metrics['images_per_second'] = self.args.run.minibatch_size / self._seconds_per_global_step
        except:
            metrics['global_step_per_sec'] = 0.0
            metrics['images_per_second'] = 0.0

        metrics['io_fetch_time'] = io_fetch_time
        metrics['learning_rate'] = self._learning_rate

        # After the accumulation, weight the gradients as needed and apply them:
        if gradient_accumulation != 1:
            gradients = [ g / gradient_accumulation for g in gradients ]

        self.apply_gradients(gradients)


        # Add the global step / second to the tensorboard log:
        try:
            metrics['global_step_per_sec'] = 1./self._seconds_per_global_step
            metrics['images_per_second'] = (self.args.run.minibatch_size*gradient_accumulation) / self._seconds_per_global_step
        except AttributeError:
            metrics['global_step_per_sec'] = 0.0
            metrics['images_per_second'] = 0.0



        self.summary(metrics)
        self.summary_images(labels, prediction)

        # Report metrics on the terminal:
        self.log(metrics, kind="Train", step=int(self.current_step().numpy()))


        global_end_time = datetime.datetime.now()

        # Compute global step per second:
        self._seconds_per_global_step = (global_end_time - global_start_time).total_seconds()

        # Update the global step:
        self._global_step.assign_add(1)
        # Update the learning rate:
        self._learning_rate.assign(self.lr_calculator(int(self._global_step.numpy())))
        return self.current_step()


    def current_step(self):
        return self._global_step


    def stop(self):
        # Mostly, this is just turning off the io:
        # self._larcv_interface.stop()
        pass


    def ana_step(self):

        raise NotImplementedError("You must implement this function")



    def close_savers(self):
        pass
        # if self.args.mode == 'inference':
        #     if self.larcv_fetcher._writer is not None:
        #         self.larcv_fetcher._writer.finalize()
