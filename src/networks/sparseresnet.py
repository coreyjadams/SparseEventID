import torch
import torch.nn as nn
import sparseconvnet as scn

from src import utils


#####################################################################
# This code is copied from torch/vision/resnet.py:

DIMENSION = 2 + 1


def submanifold_sparse_conv5x5(in_planes, out_planes, stride=1):
    """5x5 convolution with padding"""
    return scn.SubmanifoldConvolution(DIMENSION, in_planes, out_planes, 
        filter_size=[1,1,5], bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv5x5(in_planes, out_planes, stride=1):
    """5x5 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=2, bias=False)



class SparseBasicBlock(nn.Module):

    def __init__(self, inplanes, outplanes, stride=1, nplanes=1):
        nn.Module.__init__(self)
        
        if stride == 1:
            self.conv1 = scn.SubmanifoldConvolution(dimension=3, 
                nIn=inplanes, 
                nOut=outplanes, 
                filter_size=[nplanes,3,3], 
                bias=False)
        else:
            self.conv1 = scn.Convolution(dimension=3, 
                nIn             = inplanes,
                nOut            = outplanes,
                filter_size     = [1,2,2],
                filter_stride   = [1,stride,stride],
                bias            = False)

        self.bn1 = scn.BatchNormReLU(outplanes)

        self.conv2 = scn.SubmanifoldConvolution(dimension=3, 
                nIn             = outplanes,
                nOut            = outplanes,
                filter_size     = [nplanes,3,3],
                bias            = False)
        self.bn2 = scn.BatchNormalization(outplanes)

        if stride == 1:
            self.residual = scn.Identity()
        else:
            self.residual = scn.Convolution(dimension=3, 
                nIn             = inplanes,
                nOut            = outplanes,
                filter_size     = [1,2,2],
                filter_stride   = [1,stride,stride],
                bias            = False)

        self.relu = scn.ReLU()

        self.add = scn.AddTable()

    def forward(self, x):

        residual = self.residual(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # The addition of sparse tensors is not straightforward, since
        # the may not share active sites.
        # Instead, concatenate the tables, then add the tables:

        out = self.add([out, residual])

        # # out += residual
        out = self.relu(out)

        return out


class SparseBlockSeries(torch.nn.Module):


    def __init__(self, inplanes, outplanes, n_blocks, nplanes=1):
        torch.nn.Module.__init__(self)


        self.blocks = [ 
            SparseBasicBlock(inplanes, 
                inplanes, 
                nplanes=nplanes) 
            for i in range(n_blocks-1) 
        ]
        self.blocks.append(
            SparseBasicBlock(inplanes, outplanes, nplanes=nplanes, stride=2))

        for i, block in enumerate(self.blocks):
            self.add_module('block_{}'.format(i), block)


    def forward(self, x):
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

        return x



class ResNet(torch.nn.Module):

    def __init__(self, output_shape):
        torch.nn.Module.__init__(self)
        # All of the parameters are controlled via the flags module
        FLAGS = utils.flags.FLAGS()


        # Create the sparse input tensor:
        # (first spatial dim is plane)
        self.input_tensor = scn.InputLayer(dimension=3, spatial_size=[FLAGS.NPLANES,512,512])

        spatial_size = 512


        # Here, define the layers we will need in the forward path:


        
        # The convolutional layers, which can be shared or not across planes,
        # are defined below

        # We apply an initial convolution, to each plane, to get n_inital_filters


        self.initial_convolution = scn.SubmanifoldConvolution(dimension=3, 
            nIn=1, 
            nOut=FLAGS.N_INITIAL_FILTERS, 
            filter_size=[1,5,5], 
            bias=False)
        n_filters = FLAGS.N_INITIAL_FILTERS
        # Next, build out the convolution steps


        self.pre_convolutional_layers = []
        for layer in range(FLAGS.NETWORK_DEPTH_PRE_MERGE):

            self.pre_convolutional_layers.append(
                SparseBlockSeries(n_filters, 
                    FLAGS.N_INITIAL_FILTERS + n_filters, 
                    FLAGS.RES_BLOCKS_PER_LAYER)
                )
            n_filters += FLAGS.N_INITIAL_FILTERS
            spatial_size /= 2
            self.add_module("pre_merge_conv_{}".format(layer), 
                self.pre_convolutional_layers[-1])


        # This operation takes the list of features and locations and merges them
        self.concat = scn.append_tensors

        # n_filters *= FLAGS.NPLANES
        self.post_convolutional_layers = []
        for layer in range(FLAGS.NETWORK_DEPTH_POST_MERGE):

            self.post_convolutional_layers.append(
                SparseBlockSeries(n_filters, 
                    FLAGS.N_INITIAL_FILTERS + n_filters, 
                    FLAGS.RES_BLOCKS_PER_LAYER,
                    nplanes=FLAGS.NPLANES)
                )
            # A downsample happens after the convolution, so it doubles the number of filters
            n_filters += FLAGS.N_INITIAL_FILTERS
            spatial_size /= 2

        for i, layer in enumerate(self.post_convolutional_layers):
            self.add_module("post_merge_layer_{}".format(i), layer)
        # Now prepare the output operations.  In general, it's a Block Series,
        # then a 1x1 to get the right number of filters, then a global average pooling,
        # then a reshape

        # This is either once to get one set of labels, or several times to split the network
        # output to multiple labels

        if FLAGS.LABEL_MODE == 'all':
            self.final_layer = SparseBlockSeries(n_filters, 
                n_filters, 
                FLAGS.RES_BLOCKS_PER_LAYER,
                nplanes=FLAGS.NPLANES)
            spatial_size /= 2

            self.bottleneck = scn.Convolution(dimension=3, 
                        nIn             = n_filters,
                        nOut            = output_shape[-1],
                        filter_size     = [FLAGS.NPLANES,spatial_size,spatial_size],
                        filter_stride   = [1,1,1],
                        bias            = False)

            self.sparse_to_dense = scn.SparseToDense(dimension=3, nPlanes=output_shape[-1])
        else:
            self.final_layer = { 
                    key : SparseBlockSeries(n_filters, 
                            n_filters, 
                            FLAGS.RES_BLOCKS_PER_LAYER,
                            nplanes=FLAGS.NPLANES)
                    for key in output_shape
                }
            spatial_size /= 2
            self.bottleneck  = { 
                    key : scn.Convolution(dimension=3, 
                        nIn             = n_filters,
                        nOut            = output_shape[key][-1],
                        filter_size     = [FLAGS.NPLANES,spatial_size,spatial_size],
                        filter_stride   = [1,1,1],
                        bias            = False)
                    for key in output_shape
                }
            self.sparse_to_dense = {
                    key : scn.SparseToDense(dimension=3, nPlanes=output_shape[key][-1])
                    for key in output_shape
                }

            for key in self.final_layer:
                self.add_module("final_layer_{}".format(key), self.final_layer[key])
                self.add_module("bottleneck_{}".format(key), self.bottleneck[key])
                self.add_module("sparse_to_dense_{}".format(key), self.sparse_to_dense[key])


        # Sparse to Dense conversion to apply before global average pooling:

        # The rest of the final operations (reshape, softmax) are computed in the forward pass


        # Configure initialization:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, scn.SubmanifoldConvolution):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, scn.BatchNormReLU) or isinstance(m, scn.BatchNormalization):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        
        FLAGS = utils.flags.FLAGS()

        # # Split the input into NPLANES streams
        # x = [ _ for _ in torch.split(x, 1, dim=1)]
        # for the sparse input data, it's ALREADY split



        # Convert to the right format:
        x = self.input_tensor(x )

        # Apply all of the forward layers:
        x = self.initial_convolution(x)
        for i in range(len(self.pre_convolutional_layers)):
            x = self.pre_convolutional_layers[i](x)

        # # Merge the 3 streams into one with a concat:
        # x = self.concat(x)
        # Apply the after-concat convolutions:
        for i in range(len(self.post_convolutional_layers)):
            x = self.post_convolutional_layers[i](x)


        # Apply the final steps to get the right output shape

        if FLAGS.LABEL_MODE == 'all':
            # Apply the final residual block:
            output = self.final_layer(x)
            # Apply the bottle neck to make the right number of output filters:
            output = self.bottleneck(output)

            # Apply global average pooling 
            kernel_size = output.shape[2:]
            output = torch.squeeze(nn.AvgPool2d(kernel_size, ceil_mode=False)(output))

            output = nn.Softmax(dim=1)(output)

        else:
            output = {}
            for key in self.final_layer:
                # Apply the final residual block:
                output[key] = self.final_layer[key](x)

                # Apply the bottle neck to make the right number of output filters:
                output[key] = self.bottleneck[key](output[key])

                # Apply global average pooling 
                kernel_size = output[key].spatial_size

                # Convert to dense tensor:
                output[key] = self.sparse_to_dense[key](output[key])

                # Squeeze off the last few dimensions:
                output[key] = output[key].view(output[key].shape[0:2])

                # output[key] = scn.AveragePooling(dimension=3,
                #     pool_size=kernel_size, pool_stride=kernel_size)(output[key])

                # print (output[key].spatial_size)
                # print (output[key])
    
                # print (output[key].size())

                output[key] = nn.Softmax(dim=1)(output[key])

        return output



