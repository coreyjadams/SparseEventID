# import sys
# import numpy

# import tensorflow as tf


# def transform_net(input_points, 
#                   is_training,
#                   name       = "transform_net",
#                   mlp_layers = [64, 256, 1024],
#                   use_bias   = True,
#                   batch_norm = True,
#                   regularize = 0.0,):
#     '''Create a T-Net to transform the points through a learned kxk transformation
    
#     Build a mini point net and regress a kxk transformation matrix
    
#     Arguments:
#         input_points {[type]} -- [description]
#     '''

#     with tf.variable_scope(name) as scope:
#         # Number of values for each point:
#         point_dimension_length = input_points.get_shape().as_list()[-1]

#         # First, pass these points through an mlp:
#         x = mlp(input_points,
#                 mlp_layers = mlp_layers,
#                 is_training=is_training,
#                 name       = "mlp",
#                 use_bias   = use_bias,
#                 batch_norm = batch_norm,
#                 regularize = regularize)

#         # Now, perform a max pooling across ALL points to map to (B, 1024)
#         # max pool can't actually handle the dynamic reduction, but reduce_max can
#         x = tf.reduce_max(
#             input_tensor=x,
#             axis=1,
#             name=None,
#         )
#         #


#         # Apply mlp to regress this to point_dimension_length*point_dimension_length points, then reshape into a matrix

#         for n_hidden in [512, 256, point_dimension_length*point_dimension_length]:

#             if batch_norm:
#                 with tf.variable_scope("batch_norm_{}".format(n_hidden)) as scope:
#                     # updates_collections=None is important here
#                     # it forces batchnorm parameters to be saved immediately,
#                     # rather than have to be saved into snapshots manually.
#                     x = tf.contrib.layers.batch_norm(x,
#                                                      updates_collections=None,
#                                                      decay=0.9,
#                                                      is_training=is_training,
#                                                      trainable=is_training,
#                                                      scope=scope)

#             x = tf.layers.dense(x, n_hidden, 
#                                 activation=None,
#                                 use_bias=use_bias,
#                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=regularize),
#                                 trainable=is_training,
#                                 name="dense_{}".format(n_hidden),
#                                 reuse=None)
#             if n_hidden != point_dimension_length**2:
#                 x = tf.nn.relu(x)
#             else:
#                 x = tf.nn.sigmoid(x)

#         # Reshape the matrix into the proper dimensions:

#         x = tf.reshape(x, shape=[x.get_shape()[0], point_dimension_length, point_dimension_length])

#         # Add a tensor of identity matrixes to keep this transformation close to 1:
#         x += tf.eye(point_dimension_length, batch_shape=[x.get_shape()[0]])



#         return x


# def mlp(input_points,
#         mlp_layers,
#         is_training, 
#         name       = "mlp",
#         use_bias   = True,
#         batch_norm = True,
#         regularize = 0.0,):

#     '''Multi layer perceptron to map the input points to a different space of real numbers
    
#     [description]
#     '''


#     with tf.variable_scope(name) as scope:

#         # Each layer in the mlp applies a convolution with a kernel size of [1, k]
#         # The convolutional kernels are 


#         x = input_points
#         for i, intermediate_layer in enumerate(mlp_layers):
#             name = "mlp_{}_{}".format(i, intermediate_layer)

#             # Batch normalization:
#             if batch_norm:
#                 with tf.variable_scope("batch_norm_{}_{}".format(i, intermediate_layer)) as scope:
#                     # updates_collections=None is important here
#                     # it forces batchnorm parameters to be saved immediately,
#                     # rather than have to be saved into snapshots manually.
#                     x = tf.contrib.layers.batch_norm(x,
#                                                      updates_collections=None,
#                                                      decay=0.9,
#                                                      is_training=is_training,
#                                                      trainable=is_training,
#                                                      scope=scope)

#             x = tf.layers.conv1d(x, intermediate_layer,
#                                 kernel_size = 1,
#                                 strides=1,
#                                 use_bias=use_bias,
#                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=regularize),
#                                 trainable=is_training,
#                                 name=name,
#                                 reuse=None
#                             )

#             # Activation:
#             x = tf.nn.relu(x)



#         return x