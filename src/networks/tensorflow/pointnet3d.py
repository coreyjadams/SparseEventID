
import tensorflow as tf

from . pointnet import MLP, TNet


class PointNet(tf.keras.layers.Layer):
    def __init__(self, output_shape, args):
        tf.keras.layers.Layer.__init__(self)

        # What's the dataformat for tensorflow?
        self.data_format = args.framework.data_format

        # TNets and MLPs are _shared_ across planes - they should be learning the same features
        self.tnet0 = TNet(4, 4, data_format = self.data_format)

        self.mlp0 =  tf.keras.Sequential(
                [
                    MLP(4, 64, data_format = self.data_format), 
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
                        MLP(1024, 512, data_format = self.data_format),
                        MLP(512, 256, data_format = self.data_format),
                        MLP(256, output_shape[key][-1], data_format = self.data_format)
                        ])
                    for key in output_shape.keys()
                }

    def call(self, data, training=None):
        #print("entered")

        # in 2D, shape is [batch_size, max_points, x/y/val] x 3 planes

        # in 3D, shape is [batch_size, max_points, x/y/z/val]


        rotation, losses1 = self.tnet0(data)


        # Now, we have a rotation matrix for each plane.
        # Apply it to all points.
        transpose = self.data_format == "channels_last"
        data = tf.linalg.matmul(rotation, data, transpose_b=transpose)

        if transpose:
            data = tf.transpose(data, (0,2,1))


        # Now apply the MLP to 64 features:
        data = self.mlp0(data)

        # Now, another TNet call:

        rotation, losses2 = self.tnet1(data)

        # Apply the new rotation:
        transpose = self.data_format == "channels_last"
        data = tf.linalg.matmul(rotation, data, transpose_b=transpose)

        if transpose:
            data = tf.transpose(data, (0,2,1))


        # The next MLPs:
        data = self.mlp1(data)


        # How many points are we pooling?
        if self.data_format == "channels_last":
            pool_size = data.shape[1]
        elif self.data_format == "channels_first":
            pool_size = data.shape[2]
        
        
        #pooling layer:
        pooling = tf.keras.layers.MaxPool1D(pool_size=pool_size, data_format=self.data_format)

        data = tf.squeeze(pooling(data))


        # For the outputs, we concatenate across all 3 planes and have a unique series of MLPs for each
        # Category of output


        data = tf.reshape(data, data.shape + (1,))
        if transpose:
            data = tf.transpose(data, (0,2,1))


        outputs = { key : tf.squeeze(self.final_mlp[key](data)) for key in self.final_mlp.keys() }


        return outputs

