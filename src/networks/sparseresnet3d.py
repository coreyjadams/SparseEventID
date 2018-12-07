import torch
import torch.nn as nn
import sparseconvnet as scn

from src import utils


#####################################################################
# This code is copied from torch/vision/resnet.py:

def sparse_conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return scn.SubmanifoldConvolution(3, in_planes, out_planes, filter_size=3, bias=False)

def sparse_conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return scn.SubmanifoldConvolution(3, in_planes, out_planes, filter_size=1, bias=False)

def sparse_conv5x5(in_planes, out_planes, stride=1):
    """5x5 convolution with padding"""
    return scn.SubmanifoldConvolution(3, in_planes, out_planes, filter_size=5, bias=False)


# def conv3x3(in_planes, out_planes, stride=1):
#     """3x3 convolution with padding"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=[1,1,1], stride=stride,
                     padding=1, bias=False)

# def conv5x5(in_planes, out_planes, stride=1):
#     """5x5 convolution with padding"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride,
#                      padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = sparse_conv3x3(inplanes, planes)
        self.bn1 = scn.BatchNormalization(planes)
        self.relu = scn.ReLU()
        # self.conv2 = sparse_conv3x3(planes, planes)
        # self.bn2 = scn.BatchNormalization(planes)
        # self.downsample = downsample
        # self.stride = stride

    def forward(self, x):
        # residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # out = self.conv2(out)
        # out = self.bn2(out)

        # # if self.downsample is not None:
        # #     residual = self.downsample(x)

        # # out += residual
        # out = self.relu(out)

        return out


# class Bottleneck(nn.Module):
#     expansion = 4

#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(Bottleneck, self).__init__()
#         self.conv1 = conv1x1(inplanes, planes)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = conv3x3(planes, planes, stride)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = conv1x1(planes, planes * self.expansion)
#         self.bn3 = nn.BatchNorm2d(planes * self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)

#         out = self.conv3(out)
#         out = self.bn3(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         out += residual
#         out = self.relu(out)

#         return out


#####################################################################




class BlockSeries(torch.nn.Module):


    # def __init__(self, inplanes, outplanes):
    def __init__(self, inplanes, outplanes, n_blocks):
        torch.nn.Module.__init__(self)

        downsample = scn.Sequential(
                sparse_conv1x1(inplanes, outplanes, 2),
                scn.BatchNormalization(outplanes),
            )
        self.block1 = BasicBlock(inplanes, inplanes)
        self.block2 = BasicBlock(inplanes, inplanes)
        self.block3 = BasicBlock(inplanes, inplanes)
        self.block4 = BasicBlock(inplanes, outplanes)

        # Apply a max pooling:
        self.pool   = scn.MaxPooling(dimension=3,pool_size=2, pool_stride=2)

        # self.blocks = [ BasicBlock(inplanes, inplanes) for i in range(n_blocks-1) ]
        # self.blocks.append(BasicBlock(inplanes, outplanes, stride=2, downsample=downsample))

        # for i, block in enumerate(self.blocks):
        #     self.add_module('block_{}'.format(i), block)


    def forward(self, x):
        # for i in range(len(self.blocks)):
        #     x = self.blocks[i](x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x)

        return x



class ResNet(torch.nn.Module):

    def __init__(self, output_shape):
        torch.nn.Module.__init__(self)
        # All of the parameters are controlled via the flags module


        # Create the sparse input tensor:
        self.input_tensor = scn.InputLayer(dimension=3, spatial_size=(512))

        # Here, define the layers we will need in the forward path:


        
        # The convolutional layers, which can be shared or not across planes,
        # are defined below

        # We apply an initial convolution, to each plane, to get n_inital_filters

        FLAGS = utils.flags.FLAGS()

        self.initial_convolution = sparse_conv5x5(1, FLAGS.N_INITIAL_FILTERS)

        n_filters = FLAGS.N_INITIAL_FILTERS
        # Next, build out the convolution steps




        self.convolutional_layers = []
        for layer in range(FLAGS.NETWORK_DEPTH):

            self.convolutional_layers.append(BlockSeries(n_filters, 2*n_filters, FLAGS.RES_BLOCKS_PER_LAYER))
            n_filters *= 2
            self.add_module("conv_{}".format(layer), self.convolutional_layers[-1])

        # Here, take the final output and convert to a dense tensor:
        self.sparse_to_dense = scn.SparseToDense(512, n_filters)


        if FLAGS.LABEL_MODE == 'all':
            self.bottleneck  = conv1x1(n_filters, output_shape[-1])
        else:
            self.final_layer = { key : BlockSeries(n_filters, n_filters, FLAGS.RES_BLOCKS_PER_LAYER) for key in output_shape}
            self.bottleneck  = { key : conv1x1(n_filters, output_shape[key][-1]) for key in output_shape}
            for key in self.final_layer:
                self.add_module("final_layer_{}".format(key), self.final_layer[key])
                self.add_module("bottleneck_{}".format(key), self.bottleneck[key])


        # # The rest of the final operations (reshape, softmax) are computed in the forward pass


        # Configure initialization:
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        
        batch_size = x[2]

        FLAGS = utils.flags.FLAGS()


        x = self.input_tensor(x) 

        x = self.initial_convolution(x)




        for i in range(len(self.convolutional_layers)):
            x = self.convolutional_layers[i](x)

        # Apply the final steps to get the right output shape

        if FLAGS.LABEL_MODE == 'all':
            # Apply the final residual block:
            # Apply the bottle neck to make the right number of output filters:
            
            x = self.sparse_to_dense(x)
            output = self.bottleneck(x)


            # Apply global average pooling 
            kernel_size = output.shape[2:]
            output = torch.squeeze(nn.AvgPool3d(kernel_size, ceil_mode=False)(output))

            # Make sure the shape is right, in case batch size is 1:
            output = output.view([batch_size, output.shape[-1]])


            output = nn.Softmax(dim=1)(output)

        else:
            output = {}
            for key in self.final_layer:
                # Apply the final residual block:
                output[key] = self.final_layer[key](x)
                # print(key, " 1 shape: ", output[key].shape)
                output[key] = self.sparse_to_dense(output[key])

                # Apply the bottle neck to make the right number of output filters:
                output[key] = self.bottleneck[key](output[key])

                # Apply global average pooling 
                kernel_size = output[key].shape[2:]
                output[key] = torch.squeeze(nn.AvgPool3d(kernel_size, ceil_mode=False)(output[key]))
                output[key] = output[key].view([batch_size, output[key].shape[-1]])

                output[key] = nn.Softmax(dim=1)(output[key])

        return output



