import torch
import torch.nn as nn
import sparseconvnet as scn

from src import utils


#####################################################################
# This code is copied from torch/vision/resnet.py:

def sparse_conv3x3(in_planes, out_planes, stride=1):
    return scn.Convolution(2, in_planes, out_planes, filter_size=3, filter_stride=stride, bias=False)

def sparse_conv1x1(in_planes, out_planes, stride=1):
    return scn.Convolution(2, in_planes, out_planes, filter_size=1, filter_stride=stride, bias=False)



def submanifold_sparse_conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return scn.SubmanifoldConvolution(2, in_planes, out_planes, filter_size=3, bias=False)

def submanifold_sparse_conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return scn.SubmanifoldConvolution(2, in_planes, out_planes, filter_size=1, bias=False)

def submanifold_sparse_conv5x5(in_planes, out_planes, stride=1):
    """5x5 convolution with padding"""
    return scn.SubmanifoldConvolution(2, in_planes, out_planes, filter_size=5, bias=False)


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

    def __init__(self, inplanes, outplanes, stride=1):
        nn.Module.__init__(self)
        
        if stride == 1:
            self.conv1 = scn.SubmanifoldConvolution(dimension=2, 
                nIn=inplanes, nOut=outplanes, filter_size=2, bias=False)
        else:
            self.conv1 = scn.Convolution(dimension=2, 
                nIn=inplanes, nOut=outplanes, filter_size=2, filter_stride=stride, bias=False)
        self.bn1 = scn.BatchNormReLU(outplanes)
        self.conv2 = scn.SubmanifoldConvolution(dimension=2, 
                nIn=outplanes, nOut=outplanes, filter_size=2, bias=False)
        self.bn2 = scn.BatchNormalization(outplanes)
        if stride == 1:
            self.residual = scn.Identity()
        else:
            self.residual = scn.Convolution(dimension=2, 
                nIn=inplanes, nOut=outplanes, filter_size=2, filter_stride=stride, bias=False)

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
        # out = self.relu(out)

        return out

class DenseBasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        nn.Module.__init__(self)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SparseBlockSeries(torch.nn.Module):


    def __init__(self, inplanes, outplanes, n_blocks):
        torch.nn.Module.__init__(self)


        self.blocks = [ SparseBasicBlock(inplanes, inplanes) for i in range(n_blocks-1) ]
        self.blocks.append(SparseBasicBlock(inplanes, outplanes, stride=2))

        for i, block in enumerate(self.blocks):
            self.add_module('block_{}'.format(i), block)


    def forward(self, x):
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

        return x


class DenseBlockSeries(torch.nn.Module):


    def __init__(self, inplanes, outplanes, n_blocks):
        torch.nn.Module.__init__(self)

        downsample = nn.Sequential(
                conv1x1(inplanes, outplanes, 2),
                nn.BatchNorm2d(outplanes),
            )
        self.blocks = [ DenseBasicBlock(inplanes, inplanes) for i in range(n_blocks-1) ]
        self.blocks.append(DenseBasicBlock(inplanes, outplanes, stride=2, downsample=downsample))

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


        # Create the sparse input tensor:
        self.input_tensor = scn.InputLayer(dimension=2, spatial_size=512)


        # Here, define the layers we will need in the forward path:


        
        # The convolutional layers, which can be shared or not across planes,
        # are defined below

        # We apply an initial convolution, to each plane, to get n_inital_filters

        FLAGS = utils.flags.FLAGS()

        if FLAGS.SHARE_WEIGHTS:
            self.initial_convolution = submanifold_sparse_conv5x5(1, FLAGS.N_INITIAL_FILTERS)
        else:
            self.initial_convolution = []
            for i in FLAGS.NPLANES:
                self.initial_convolution.append(
                        submanifold_sparse_conv5x5(1, FLAGS.N_INITIAL_FILTERS)
                    )
                self.add_module("initial_convolution_plane_{}".format(i), self.initial_convolution[-1])

        n_filters = FLAGS.N_INITIAL_FILTERS
        # Next, build out the convolution steps


        self.pre_convolutional_layers = []
        for layer in range(FLAGS.NETWORK_DEPTH_PRE_MERGE):

            if FLAGS.SHARE_WEIGHTS:
                self.pre_convolutional_layers.append(SparseBlockSeries(n_filters, FLAGS.N_INITIAL_FILTERS + n_filters, FLAGS.RES_BLOCKS_PER_LAYER))
                n_filters += FLAGS.N_INITIAL_FILTERS
                self.add_module("pre_merge_conv_{}".format(layer), self.pre_convolutional_layers[-1])

            else:
                self.pre_convolutional_layers.append(
                    [ SparseBlockSeries(n_filters, FLAGS.N_INITIAL_FILTERS + n_filters, FLAGS.RES_BLOCKS_PER_LAYER) for x in range(FLAGS.NPLANES)]
                    )
                n_filters += FLAGS.N_INITIAL_FILTERS
                for plane in FLAGS.NPLANES:
                    self.add_module("pre_merge_conv_{}_plane_{}".format(layer, plane),
                                     self.pre_convolutional_layers[-1][plane])


        # The next operations apply after the merge across planes has happened.
        self.sparse_to_dense = scn.SparseToDense(2, n_filters)

        n_filters *= FLAGS.NPLANES
        self.post_convolutional_layers = []
        for layer in range(FLAGS.NETWORK_DEPTH_POST_MERGE):

            self.post_convolutional_layers.append(DenseBlockSeries(n_filters, FLAGS.NPLANES*FLAGS.N_INITIAL_FILTERS + n_filters, FLAGS.RES_BLOCKS_PER_LAYER))
            # A downsample happens after the convolution, so it doubles the number of filters
            n_filters += FLAGS.NPLANES*FLAGS.N_INITIAL_FILTERS

        for i, layer in enumerate(self.post_convolutional_layers):
            self.add_module("post_merge_layer_{}".format(i), layer)
        # Now prepare the output operations.  In general, it's a Block Series,
        # then a 1x1 to get the right number of filters, then a global average pooling,
        # then a reshape

        # This is either once to get one set of labels, or several times to split the network
        # output to multiple labels

        if FLAGS.LABEL_MODE == 'all':
            self.final_layer = DenseBlockSeries(n_filters, n_filters, FLAGS.RES_BLOCKS_PER_LAYER)
            self.bottleneck  = conv1x1(n_filters, output_shape[-1])
        else:
            self.final_layer = { key : DenseBlockSeries(n_filters, n_filters, FLAGS.RES_BLOCKS_PER_LAYER) for key in output_shape}
            self.bottleneck  = { key : conv1x1(n_filters, output_shape[key][-1]) for key in output_shape}
            for key in self.final_layer:
                self.add_module("final_layer_{}".format(key), self.bottleneck[key])
                self.add_module("bottleneck_{}".format(key), self.final_layer[key])


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

        for p in range(len(x)):

            # Convert to the right format:
            x[p] = self.input_tensor(x[p] )

            # Apply all of the forward layers:
            if FLAGS.SHARE_WEIGHTS:
                x[p] = self.initial_convolution(x[p])
                for i in range(len(self.pre_convolutional_layers)):
                    x[p] = self.pre_convolutional_layers[i](x[p])
            else:
                x[p] = self.initial_convolution[p](x[p])
                for i in range(len(self.pre_convolutional_layers)):
                    x[p] = self.pre_convolutional_layers[i][p](x[p])

        # Merge the 3 streams into one with a concat:
        x = [self.sparse_to_dense(plane) for plane in x ]
        
        x = torch.cat(x, dim=1)

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
                kernel_size = output[key].shape[2:]
                output[key] = torch.squeeze(nn.AvgPool2d(kernel_size, ceil_mode=False)(output[key]))

                output[key] = nn.Softmax(dim=1)(output[key])

        return output



