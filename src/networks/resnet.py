import torch
import torch.nn as nn


from src import utils

FLAGS = utils.flags.FLAGS()


class Block(nn.Module):

    def __init__(self, inplanes, outplanes, nplanes=1):

        nn.Module.__init__(self)
        
        padding = [0,1,1] if nplanes == 1 else [1,1,1]

        self.conv1 = torch.nn.Conv3d(
            in_channels  = inplanes, 
            out_channels = outplanes, 
            kernel_size  = [nplanes,3,3], 
            stride       = [1, 1, 1],
            padding      = padding,
            bias=False)
        
        # if FLAGS.BATCH_NORM:
        self.bn1  = torch.nn.BatchNorm3d(outplanes)
        self.relu = torch.nn.ReLU()

    def forward(self, x):

        out = self.conv1(x)
        # if FLAGS.BATCH_NORM:
        out = self.bn1(out)
        out = self.relu(out)
        # else:
            # out = self.relu(out)

        return out



class ResidualBlock(nn.Module):

    def __init__(self, inplanes, outplanes, nplanes=1):
        nn.Module.__init__(self)
        
        padding = [0,1,1] if nplanes == 1 else [1,1,1]
        
        self.conv1 = torch.nn.Conv3d(
            in_channels  = inplanes, 
            out_channels = outplanes, 
            kernel_size  = [nplanes,3,3], 
            stride       = [1, 1, 1],
            padding      = padding,
            bias         = False)
        

        # if FLAGS.BATCH_NORM:
        self.bn1 = torch.nn.BatchNorm3d(outplanes)

        self.conv2 = torch.nn.Conv3d(
            in_channels  = outplanes,
            out_channels = outplanes,
            kernel_size  = [nplanes,3,3],
            stride       = [1, 1, 1],
            padding      = padding,
            bias         = False)

        # if FLAGS.BATCH_NORM:
        self.bn2 = torch.nn.BatchNorm3d(outplanes)


        self.relu = torch.nn.ReLU()


    def forward(self, x):

        residual = x

        out = self.conv1(x)        
        # if FLAGS.BATCH_NORM:
        out = self.bn1(out)
        # else:
            # out = self.relu(out)
        out = self.conv2(out)
        # if FLAGS.BATCH_NORM:
        out = self.bn2(out)

        out = self.relu(out + residual)

        return out




class ConvolutionDownsample(nn.Module):

    def __init__(self, inplanes, outplanes,nplanes=1):
        nn.Module.__init__(self)

        self.conv = torch.nn.Conv3d(
            in_channels  = inplanes,
            out_channels = outplanes,
            kernel_size  = [nplanes,2,2],
            stride       = [1,2,2],
            padding      = [0,0,0],
            bias         = False
        )
        # if FLAGS.BATCH_NORM:
        self.bn   = torch.nn.BatchNorm3d(outplanes)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        out = self.conv(x)

        # if FLAGS.BATCH_NORM:
        out = self.bn(out)

        out = self.relu(out)
        return out

class BlockSeries(torch.nn.Module):


    def __init__(self, inplanes, n_blocks, nplanes, residual=False):
        torch.nn.Module.__init__(self)

        if residual:
            self.blocks = [ ResidualBlock(inplanes, inplanes, nplanes=nplanes) for i in range(n_blocks) ]
        else:
            self.blocks = [ Block(inplanes, inplanes, nplanes=nplanes) for i in range(n_blocks)]

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

        
        # We apply an initial convolution, to each plane, to get n_inital_filters


        self.initial_convolution = torch.nn.Conv3d(
            in_channels  = 1, 
            out_channels = FLAGS.N_INITIAL_FILTERS, 
            kernel_size  = [1, 5, 5], 
            stride       = [1, 1, 1],
            padding      = [0, 2, 2],
            bias         = False)


        n_filters = FLAGS.N_INITIAL_FILTERS
        # Next, build out the convolution steps


        self.pre_convolutional_layers = []
        for layer in range(FLAGS.NETWORK_DEPTH_PRE_MERGE):

            self.pre_convolutional_layers.append(
                BlockSeries(inplanes = n_filters, 
                    n_blocks = FLAGS.RES_BLOCKS_PER_LAYER,
                    nplanes =1,
                    residual = True)
                )
            self.pre_convolutional_layers.append(
                ConvolutionDownsample(inplanes=n_filters,
                    outplanes = n_filters + FLAGS.N_INITIAL_FILTERS,
                    nplanes = 1)
                )

            n_filters += FLAGS.N_INITIAL_FILTERS
            self.add_module("pre_merge_conv_{}".format(layer), 
                self.pre_convolutional_layers[-2])
            self.add_module("pre_merge_down_{}".format(layer), 
                self.pre_convolutional_layers[-1])


        # This operation takes the list of features and locations and merges them

        # n_filters *= FLAGS.NPLANES
        self.post_convolutional_layers = []
        for layer in range(FLAGS.NETWORK_DEPTH_POST_MERGE):

            self.post_convolutional_layers.append(
                BlockSeries(
                    inplanes = n_filters, 
                    n_blocks = FLAGS.RES_BLOCKS_PER_LAYER,
                    nplanes  = FLAGS.NPLANES,
                    residual = True)
                )
            self.post_convolutional_layers.append(
                ConvolutionDownsample(
                    inplanes  = n_filters,
                    outplanes = n_filters + FLAGS.N_INITIAL_FILTERS,
                    nplanes   = 1)
                )

            n_filters += FLAGS.N_INITIAL_FILTERS

            self.add_module("post_merge_conv_{}".format(layer), 
                self.post_convolutional_layers[-2])
            self.add_module("post_merge_down_{}".format(layer), 
                self.post_convolutional_layers[-1])

        # Now prepare the output operations.  In general, it's a Block Series,
        # then a 1x1 to get the right number of filters, then a global average pooling,
        # then a reshape

        # This is either once to get one set of labels, or several times to split the network
        # output to multiple labels

        if FLAGS.LABEL_MODE == 'all':
            # self.final_layer = SparseBlockSeries(n_filters, 
            #     n_filters, 
            #     FLAGS.RES_BLOCKS_PER_LAYER,
            #     nplanes=FLAGS.NPLANES)

            # self.bottleneck = scn.Convolution(dimension=3, 
            #             nIn             = n_filters,
            #             nOut            = output_shape[-1],
            #             filter_size     = [FLAGS.NPLANES,spatial_size,spatial_size],
            #             filter_stride   = [1,1,1],
            #             bias            = False)

            # self.sparse_to_dense = scn.SparseToDense(dimension=3, nPlanes=output_shape[-1])
            pass
        else:
            self.final_layer = { 
                    key : BlockSeries(
                        inplanes = n_filters, 
                        n_blocks = FLAGS.RES_BLOCKS_PER_LAYER,
                        nplanes  = FLAGS.NPLANES,
                        residual = True)
                    for key in output_shape
                }
            self.bottleneck  = { 
                    key : torch.nn.Conv3d(
                        in_channels  = n_filters, 
                        out_channels = output_shape[key][-1], 
                        kernel_size  = [1,1,1], 
                        stride       = [1,1,1],
                        padding      = [0,0,0],
                        bias=False)
                    for key in output_shape
                }
        

            for key in self.final_layer:
                self.add_module("final_layer_{}".format(key), self.final_layer[key])
                self.add_module("bottleneck_{}".format(key), self.bottleneck[key])




    def forward(self, x):
        
        batch_size = x.shape[0]

        x = x.view([x.shape[0], 1, x.shape[1], x.shape[2], x.shape[3]])

        # Apply all of the forward layers:
        x = self.initial_convolution(x)
        for i in range(len(self.pre_convolutional_layers)):
            x = self.pre_convolutional_layers[i](x)



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

            # output = nn.Softmax(dim=1)(output)

        else:
            output = {}
            for key in self.final_layer:
                # Apply the final residual block:
                output[key] = self.final_layer[key](x)

                # Apply the bottle neck to make the right number of output filters:
                output[key] = self.bottleneck[key](output[key])
                
                kernel_size = output[key].shape[2:]

                output[key] = torch.squeeze(nn.AvgPool3d(kernel_size, ceil_mode=False)(output[key]))

                # Squeeze off the last few dimensions:
                output[key] = output[key].view([batch_size, output[key].shape[-1]])

                # output[key] = scn.AveragePooling(dimension=3,
                #     pool_size=kernel_size, pool_stride=kernel_size)(output[key])

                # print (output[key].spatial_size)
                # print (output[key])
    
                # print (output[key].size())

                # output[key] = nn.Softmax(dim=1)(output[key])

        return output




