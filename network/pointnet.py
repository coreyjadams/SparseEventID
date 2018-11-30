import sys

import tensorflow as tf

from .utils import transform_net, mlp

from .flags import FLAGS

# Main class
class pointnet(object):
    '''Define a network model and run training

    resnet implementation
    '''
    def __init__(self):
        '''initialization

        Requires a list of parameters as python dictionary

        Arguments:
            params {dict} -- Network parameters

        Raises:
            ConfigurationException -- Missing a required parameter
        '''

        return


    def _build_network(self, inputs):

        ''' verbosity 0 = no printouts
            verbosity 1 = sparse information
            verbosity 2 = debug
        '''

        
        # The input is a point cloud
        x = inputs['image']

        # We have to determine if this is a network operating on 3D data, or mulitple
        # planes of 2D data


        # First, apply a transformation net to the input:

        with tf.variable_scope("input_transformation"):
            input_transformation = transform_net(x, 
                                                 FLAGS.TRAINING,
                                                 name       = "input_transformation",
                                                 # mlp_layers = [64, 256, 1024],
                                                 use_bias   = FLAGS.USE_BIAS,
                                                 batch_norm = FLAGS.BATCH_NORM,
                                                 regularize = FLAGS.REGULARIZE_WEIGHTS,)

            # Do a matrix multiplication to transform the point:

            # Squeeze out the empty dimension in the point cloud:

            x = tf.matmul(x, input_transformation)



        # Next is a multilayered perceptron to map each point's k-vector to
        # a list of 64 points:

        with tf.variable_scope("first_mlp"):

            # Use 1D bottleneck convolutions to map k to 64, through a 64 dimension hidden layer

            x = mlp(x,
                    mlp_layers = [64,64],
                    is_training= FLAGS.TRAINING, 
                    name       = "mlp",
                    use_bias   = FLAGS.USE_BIAS,
                    batch_norm = FLAGS.BATCH_NORM,
                    regularize = FLAGS.REGULARIZE_WEIGHTS,)

        with tf.variable_scope("feature_transformation"):
            feature_transformation = transform_net(x, FLAGS.TRAINING,
                                                   name       = "input_transformation",
                                                   # mlp_layers = [64, 256, 1024],
                                                   use_bias   = FLAGS.USE_BIAS,
                                                   batch_norm = FLAGS.BATCH_NORM,
                                                   regularize = FLAGS.REGULARIZE_WEIGHTS,)
            # Apply the feature transformation:
            x = tf.matmul(x, feature_transformation)


        with tf.variable_scope("second_mlp"):

            x = mlp(x,
                    mlp_layers = [64,128, 1024],
                    is_training= FLAGS.TRAINING, 
                    name       = "mlp",
                    use_bias   = FLAGS.USE_BIAS,
                    batch_norm = FLAGS.BATCH_NORM,
                    regularize = FLAGS.REGULARIZE_WEIGHTS,)

        # Now, apply the symmetric function (reduce_max) to map to (B, 1024) global features:
        x = tf.reduce_max(
            input_tensor=x,
            axis=1,
            name=None,
        )


        with tf.variable_scope("classification_mlp"):
            for n_hidden in [512, 256, FLAGS.NUM_CLASSES]:

                if FLAGS.BATCH_NORM:
                    with tf.variable_scope("batch_norm_{}".format(n_hidden)) as scope:
                        # updates_collections=None is important here
                        # it forces batchnorm parameters to be saved immediately,
                        # rather than have to be saved into snapshots manually.
                        x = tf.contrib.layers.batch_norm(x,
                                                         updates_collections=None,
                                                         decay=0.9,
                                                         is_training=FLAGS.TRAINING,
                                                         trainable=FLAGS.TRAINING,
                                                         scope=scope)

                x = tf.layers.dense(x, n_hidden, 
                                    activation=None,
                                    use_bias=FLAGS.USE_BIAS,
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=FLAGS.REGULARIZE_WEIGHTS),
                                    trainable=FLAGS.TRAINING,
                                    name="dense_{}".format(n_hidden),
                                    reuse=None)

                # Skip the relu on the final layer:
                if n_hidden != FLAGS.NUM_CLASSES:
                    x = tf.nn.relu(x)

        # To compute the loss, we return not just the logits but also the transormation matrices:
        return x, [input_transformation, feature_transformation]








