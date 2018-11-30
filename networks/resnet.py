import torch
import torch.nn as nn

from utils import flags


#####################################################################
# This code is copied from torch/vision/resnet.py:

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


#####################################################################


def conv5x5(in_planes, out_planes, stride=1):
    """5x5 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=2, bias=False)


class BlockSeries(torch.nn.Module):


    def __init__(self, inplanes, outplanes, n_blocks):
        torch.nn.Module.__init__(self)

        downsample = nn.Sequential(
                conv1x1(inplanes, outplanes, 2),
                nn.BatchNorm2d(outplanes),
            )
        self.blocks = [ BasicBlock(inplanes, inplanes) for i in range(n_blocks-1) ]
        self.blocks.append(BasicBlock(inplanes, outplanes, stride=2, downsample=downsample))

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

        # Here, define the layers we will need in the forward path:
        
        # The convolutional layers, which can be shared or not across planes,
        # are defined below

        # We apply an initial convolution, to each plane, to get n_inital_filters

        FLAGS = flags.FLAGS()

        if FLAGS.SHARE_WEIGHTS:
            self.initial_convolution = conv5x5(1, FLAGS.N_INITIAL_FILTERS)
        else:
            self.initial_convolution = []
            for i in FLAGS.NPLANES:
                self.initial_convolution.append(
                        conv5x5(1, FLAGS.N_INITIAL_FILTERS)
                    )
                self.add_module("initial_convolution_plane_{}".format(i), self.initial_convolution[-1])

        n_filters = FLAGS.N_INITIAL_FILTERS
        # Next, build out the convolution steps



        self.RES_BLOCKS_PER_LAYER       = 2
        self.NETWORK_DEPTH_PRE_MERGE    = 3
        self.NETWORK_DEPTH_POST_MERGE   = 3
        self.NPLANES                    = 3
        self.SHARE_WEIGHTS              = True


        self.pre_convolutional_layers = []
        for layer in range(FLAGS.NETWORK_DEPTH_PRE_MERGE):

            if FLAGS.SHARE_WEIGHTS:
                self.pre_convolutional_layers.append(BlockSeries(n_filters, 2*n_filters, FLAGS.RES_BLOCKS_PER_LAYER))
                n_filters *= 2
                self.add_module("pre_merge_conv_{}".format(layer), self.pre_convolutional_layers[-1])

            else:
                self.pre_convolutional_layers.append(
                    [ BlockSeries(n_filters, n_filters, FLAGS.RES_BLOCKS_PER_LAYER) for x in range(FLAGS.NPLANES)]
                    )
                n_filters *= 2
                for plane in FLAGS.NPLANES:
                    self.add_module("pre_merge_conv_{}_plane_{}".format(layer, plane),
                                     self.pre_convolutional_layers[-1][plane])


        # The next operations apply after the merge across planes has happened.
        n_filters *= FLAGS.NPLANES
        self.post_convolutional_layers = []
        for layer in range(FLAGS.NETWORK_DEPTH_POST_MERGE):

            self.post_convolutional_layers.append(BlockSeries(n_filters, 2*n_filters, FLAGS.RES_BLOCKS_PER_LAYER))
            # A downsample happens after the convolution, so it doubles the number of filters
            n_filters *= 2

        for i, layer in enumerate(self.post_convolutional_layers):
            self.add_module("post_merge_layer_{}".format(i), layer)
        # Now prepare the output operations.  In general, it's a Block Series,
        # then a 1x1 to get the right number of filters, then a global average pooling,
        # then a reshape

        # This is either once to get one set of labels, or several times to split the network
        # output to multiple labels

        if FLAGS.LABEL_MODE == 'all':
            self.final_layer = BlockSeries(n_filters, n_filters, FLAGS.RES_BLOCKS_PER_LAYER)
            self.bottleneck  = conv1x1(n_filters, output_shape[-1])
        else:
            self.final_layer = { key : BlockSeries(n_filters, n_filters, FLAGS.RES_BLOCKS_PER_LAYER) for key in output_shape}
            self.bottleneck  = { key : conv1x1(n_filters, output_shape[key][-1]) for key in output_shape}

            ### TODO:
            # Add the split paths with add_module

        # The rest of the final operations (reshape, softmax) are computed in the forward pass


    def forward(self, x):
        
        FLAGS = flags.FLAGS()

        # Split the input into NPLANES streams
        x = [ _ for _ in torch.split(x, 1, dim=1)]

        for p in range(len(x)):

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
        x = torch.cat(x, dim=1)
        print(x.shape)

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
            output = torch.squeeze(nn.AvgPool2d(kernel_size)(output))

            output = nn.Softmax(dim=-1)(output)

        else:
            output = {}
            for key in self.final_layer:
                # Apply the final residual block:
                output[key] = self.final_layer[key](x)

                # Apply the bottle neck to make the right number of output filters:
                output[key] = self.bottleneck[key](output[key])

                # Apply global average pooling 
                kernel_size = output[key].shape[1:-1]
                output[key] = nn.AvgPool2d(kernel_size)(output[key])

                output[key] = nn.Softmax(dim=-1)(output[key])

        return output



