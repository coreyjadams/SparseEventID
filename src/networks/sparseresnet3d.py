import torch
import torch.nn as nn
import sparseconvnet as scn

from src import utils

FLAGS = utils.flags.FLAGS()

class SparseBlock(nn.Module):

    def __init__(self, inplanes, outplanes):

        nn.Module.__init__(self)
        
        self.conv1 = scn.SubmanifoldConvolution(dimension=3, 
            nIn=inplanes, 
            nOut=outplanes, 
            filter_size=3, 
            bias=False)
        
        # if FLAGS.BATCH_NORM:
        self.bn1 = scn.BatchNormReLU(outplanes)
        # self.relu = scn.ReLU()

    def forward(self, x):

        out = self.conv1(x)
        # if FLAGS.BATCH_NORM:
        out = self.bn1(out)
        # else:
            # out = self.relu(out)

        return out



class SparseResidualBlock(nn.Module):

    def __init__(self, inplanes, outplanes):
        nn.Module.__init__(self)
        
        
        self.conv1 = scn.SubmanifoldConvolution(dimension=3, 
            nIn         = inplanes, 
            nOut        = outplanes, 
            filter_size = 3, 
            bias=False)
        

        # if FLAGS.BATCH_NORM:
        self.bn1 = scn.BatchNormReLU(outplanes)

        self.conv2 = scn.SubmanifoldConvolution(dimension=3, 
            nIn         = outplanes,
            nOut        = outplanes,
            filter_size = 3,
            bias        = False)

        # if FLAGS.BATCH_NORM:
        self.bn2 = scn.BatchNormalization(outplanes)

        self.residual = scn.Identity()
        self.relu = scn.ReLU()

        self.add = scn.AddTable()

    def forward(self, x):

        residual = self.residual(x)

        out = self.conv1(x)
        # if FLAGS.BATCH_NORM:
        out = self.bn1(out)
        # else:
            # out = self.relu(out)
        out = self.conv2(out)

        # if FLAGS.BATCH_NORM:
        out = self.bn2(out)

        # The addition of sparse tensors is not straightforward, since

        out = self.add([out, residual])

        out = self.relu(out)

        return out




class SparseConvolutionDownsample(nn.Module):

    def __init__(self, inplanes, outplanes):
        nn.Module.__init__(self)

        self.conv = scn.Convolution(dimension=3,
            nIn             = inplanes,
            nOut            = outplanes,
            filter_size     = 2,
            filter_stride   = 2,
            bias            = False
        )
        # if FLAGS.BATCH_NORM:
        self.bn   = scn.BatchNormalization(outplanes)
        self.relu = scn.ReLU()

    def forward(self, x):
        out = self.conv(x)

        # if FLAGS.BATCH_NORM:
        out = self.bn(out)

        out = self.relu(out)
        return out


class SparseBlockSeries(torch.nn.Module):


    def __init__(self, inplanes, n_blocks, residual=False):
        torch.nn.Module.__init__(self)

        if residual:
            self.blocks = [ SparseResidualBlock(inplanes, inplanes) for i in range(n_blocks) ]
        else:
            self.blocks = [ SparseBlock(inplanes, inplanes) for i in range(n_blocks)]

        for i, block in enumerate(self.blocks):
            self.add_module('block_{}'.format(i), block)


    def forward(self, x):
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
        return x


def filter_increase(input_filters):
    # return input_filters * 2
    return input_filters + FLAGS.N_INITIAL_FILTERS

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

        self.initial_convolution = scn.SubmanifoldConvolution(3, 1, FLAGS.N_INITIAL_FILTERS, filter_size=5, bias=False)

        n_filters = FLAGS.N_INITIAL_FILTERS
        # Next, build out the convolution steps




        self.convolutional_layers = []
        for layer in range(FLAGS.NETWORK_DEPTH):

            self.convolutional_layers.append(SparseBlockSeries(
                n_filters, 
                FLAGS.RES_BLOCKS_PER_LAYER,
                residual = True))
            out_filters = filter_increase(n_filters)
            self.convolutional_layers.append(SparseConvolutionDownsample(
                inplanes = n_filters,
                outplanes = out_filters))
                # outplanes = n_filters + FLAGS.N_INITIAL_FILTERS))
            n_filters = out_filters

            self.add_module("conv_{}".format(layer), self.convolutional_layers[-2])
            self.add_module("down_{}".format(layer), self.convolutional_layers[-1])

        # Here, take the final output and convert to a dense tensor:


        if FLAGS.LABEL_MODE == 'all':
            self.final_layer = SparseBlockSeries(
                        inplanes = n_filters, 
                        n_blocks = FLAGS.RES_BLOCKS_PER_LAYER,
                        residual = True)
                    
            self.bottleneck  = scn.SubmanifoldConvolution(dimension=3, 
                        nIn=n_filters, 
                        nOut=output_shape[key][-1], 
                        filter_size=3, 
                        bias=False)

            self.sparse_to_dense = scn.SparseToDense(dimension=3, nPlanes=output_shape[-1])

        else:

            self.final_layer = { 
                    key : SparseBlockSeries(
                        inplanes = n_filters, 
                        n_blocks = FLAGS.RES_BLOCKS_PER_LAYER,
                        residual = True)
                    for key in output_shape
                }
            self.bottleneck  = { 
                    key : scn.SubmanifoldConvolution(dimension=3, 
                        nIn=n_filters, 
                        nOut=output_shape[key][-1], 
                        filter_size=1, 
                        bias=False)
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


        # # The rest of the final operations (reshape, softmax) are computed in the forward pass


        # # Configure initialization:
        # for m in self.modules():
        #     if isinstance(m, nn.Conv3d) or isinstance(m, scn.SubmanifoldConvolution):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, scn.BatchNormReLU):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)



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
            output = self.final_layer(x)
            # print " 1 shape: ", output.shape)

            # Apply the bottle neck to make the right number of output filters:
            output = self.bottleneck(output)

            output = self.sparse_to_dense(output)

            # Apply global average pooling 
            kernel_size = output.shape[2:]
            output = torch.squeeze(nn.AvgPool3d(kernel_size, ceil_mode=False)(output))
            output = output.view([batch_size, output.shape[-1]])



        else:
            output = {}
            for key in self.final_layer:
                # Apply the final residual block:
                output[key] = self.final_layer[key](x)
                # print(key, " 1 shape: ", output[key].shape)

                # Apply the bottle neck to make the right number of output filters:
                output[key] = self.bottleneck[key](output[key])

                output[key] = self.sparse_to_dense[key](output[key])

                # Apply global average pooling 
                kernel_size = output[key].shape[2:]
                output[key] = torch.squeeze(nn.AvgPool3d(kernel_size, ceil_mode=False)(output[key]))
                output[key] = output[key].view([batch_size, output[key].shape[-1]])


        return output



