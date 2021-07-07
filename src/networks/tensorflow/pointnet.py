
import tensorflow as tf


class MLP(tf.keras.layers.Layer):
    def __init__(self, input_size, output_size, data_format):
        tf.keras.layers.Layer.__init__(self)

        self.mlp = tf.keras.layers.Conv1D(
            filters     = output_size,
            kernel_size = (1),
            data_format = data_format
        )

        self.bn = tf.keras.layers.BatchNormalization()

        self.relu = tf.keras.layers.ReLU()

    def call(self, data):
        return self.relu(self.bn(self.mlp(data)))

class TNet(tf.keras.layers.Layer):
    def __init__(self, input_shape, output_shape, data_format):
        tf.keras.layers.Layer.__init__(self)

        self.data_format = data_format

        # Initialize these to small but not zero
        self.trainable_matrix = tf.Variable(
            (0.01/256)*tf.random.uniform(shape=[256,output_shape**2]),
            trainable=True
        )

        self.trainable_biases  = tf.Variable(
            tf.eye(output_shape),
            trainable=True
        )

        self.identity = tf.eye(output_shape)

        self.output_dimension = output_shape

        self.mlps = tf.keras.Sequential([
            MLP(input_shape, 64, data_format),
            MLP(64, 128, data_format),
            MLP(128, 1024, data_format),
            ]
        )

        self.fully_connected = tf.keras.Sequential([
            tf.keras.layers.Dense(512), tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(256), tf.keras.layers.ReLU()]
            )

    def call(self, data):


        x = self.mlps(data)
        shape = x.shape[2:]

        # How many points are we pooling?
        if self.data_format == "channels_last":
            pool_size = data.shape[1]
        elif self.data_format == "channels_first":
            pool_size = data.shape[2]
        
        #pooling layer:
        x = tf.keras.layers.MaxPool1D(pool_size=pool_size, data_format=self.data_format)(x)

        x = tf.squeeze(x)

        x = self.fully_connected(x)

        matrix = tf.linalg.matmul(x, self.trainable_matrix)
        matrix = tf.reshape(matrix, (-1, self.output_dimension, self.output_dimension))
        matrix = matrix + self.trainable_biases

        transpose = tf.transpose(matrix, perm=(0, 2,1))

        ortho_loss = tf.reduce_sum( (self.identity - tf.linalg.matmul(matrix, transpose) )**2 )

        return matrix, ortho_loss

class PointNet(tf.keras.layers.Layer):
    def __init__(self, output_shape, args):
        tf.keras.layers.Layer.__init__(self)

        # What's the dataformat for tensorflow?
        self.data_format = args.framework.data_format

        # TNets and MLPs are _shared_ across planes - they should be learning the same features
        self.tnet0 = TNet(3, 3, data_format = self.data_format)

        self.mlp0 =  tf.keras.Sequential(
                [
                    MLP(3, 64, data_format = self.data_format), 
                    MLP(64, 64, data_format = self.data_format)
                ]
            )

        self.tnet1 = TNet(64, 64, data_format = self.data_format)

        self.mlp1 =  tf.keras.Sequential(
                [
                    MLP(64, 128, data_format = self.data_format), 
                    MLP(128, 1024, data_format = self.data_format)
                ]
            )


        self.final_mlp = {
                    key : tf.keras.Sequential([
                        MLP(3*1024, 512, data_format = self.data_format),
                        MLP(512, 256, data_format = self.data_format),
                        MLP(256, output_shape[key][-1], data_format = self.data_format)
                        ])
                    for key in output_shape.keys()
                }


    def call(self, data, training=None):


        # in 2D, shape is [batch_size, max_points, x/y/val] x 3 planes

        # in 3D, shape is [batch_size, max_points, x/y/z/val]


        tnets = [ self.tnet0(d) for d in data]

        rotations, losses1 = list(zip(*tnets))


        # Now, we have a rotation matrix for each plane.
        # Apply it to all points.
        transpose = self.data_format == "channels_last"
        data = [ tf.linalg.matmul(r, p, transpose_b=transpose) for r, p in zip(rotations, data)]

        if transpose:
            data = [tf.transpose(d, (0,2,1)) for d in data]


        # Now apply the MLP to 64 features:
        data = [ self.mlp0(d) for d in data]

        # Now, another TNet call:

        tnets = [self.tnet1(d) for d in data]

        rotations, losses2 = list(zip(*tnets))
        # Apply the new rotations:
        transpose = self.data_format == "channels_last"
        data = [ tf.linalg.matmul(r, p, transpose_b=transpose) for r, p in zip(rotations, data)]

        if transpose:
            data = [tf.transpose(d, (0,2,1)) for d in data]


        # The next MLPs:
        data = [ self.mlp1(d) for d in data]

        # Next, we maxpool over each plane:
        shape = data[0].shape[2:]


        # How many points are we pooling?
        if self.data_format == "channels_last":
            pool_size = data[0].shape[1]
        elif self.data_format == "channels_first":
            pool_size = data[0].shape[2]
        
        #pooling layer:
        pooling = tf.keras.layers.MaxPool1D(pool_size=pool_size, data_format=self.data_format)


        data = [ tf.squeeze(pooling(d)) for d in data ]


        # For the outputs, we concatenate across all 3 planes and have a unique series of MLPs for each
        # Category of output


        data = tf.concat(data, axis=-1)

        data = tf.reshape(data, data.shape + (1,))
        if transpose:
            data = tf.transpose(data, (0,2,1))


        outputs = { key : tf.squeeze(self.final_mlp[key](data)) for key in self.final_mlp.keys() }


        return outputs
