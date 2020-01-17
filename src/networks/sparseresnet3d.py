import torch
import torch.nn as nn
import sparseconvnet as scn

from src import utils

FLAGS = utils.flags.FLAGS()

# class resnet3D(FLAGS):
#     ''' Sparse Resnet specific flags
#     '''

#     def __init__(self):
#         FLAGS.__init__(self)


#     def set_net(self, net):
#         # For the resnet object, we set the network as resnet:
#         self._net = net


#     def _set_defaults(self):

#         self.VERBOSITY             = 0
#         self.N_INITIAL_FILTERS     = 32
#         self.RES_BLOCKS_PER_LAYER  = 2
#         self.NETWORK_DEPTH         = 8
#         self.WEIGHT_DECAY          = 1e-4
#         self.SPARSE                = True
#         self.INPUT_DIMENSION       = '3D'

#         self.BATCH_NORM                 = True
#         self.LEAKY_RELU                 = False   

#         # self.BOTTLENECK_FC         = False

#         FLAGS._set_defaults(self)

#     def _add_default_network_configuration(self, parser):

#         parser.add_argument('-v', '--verbosity', type=int,default=self.VERBOSITY,
#             help="Network verbosity at construction [default: {}]".format(self.VERBOSITY))


#         parser.add_argument('--n-initial-filters', type=int, default=self.N_INITIAL_FILTERS,
#             help="Number of filters applied, per plane, for the initial convolution [default: {}]".format(self.N_INITIAL_FILTERS))
#         parser.add_argument('--res-blocks-per-layer', type=int, default=self.RES_BLOCKS_PER_LAYER,
#             help="Number of residual blocks per layer [default: {}]".format(self.RES_BLOCKS_PER_LAYER))
#         parser.add_argument('--network-depth', type=int, default=self.NETWORK_DEPTH,
#             help="Total number of downsamples to apply [default: {}]".format(self.NETWORK_DEPTH))

#         parser.add_argument('--weight-decay', type=float, default=self.WEIGHT_DECAY,
#             help="Weight decay strength [default: {}]".format(self.WEIGHT_DECAY))

#         # parser.add_argument('--bottleneck-fully-connected', type=str2bool, default=self.BOTTLENECK_FC,
#         #     help="Whether or not to apply a fully connected layer with dropout as bottleneck [default: {}]".format(self.BOTTLENECK_FC))

#         parser.add_argument('--sparse', type=str2bool, default=self.SPARSE,
#             help="Run using submanifold sparse convolutions [default: {}]".format(self.SPARSE))

#         parser.add_argument('--batch-norm', type=str2bool, default=self.BATCH_NORM,
#             help="Run using batch normalization [default: {}]".format(self.BATCH_NORM))
#         parser.add_argument('--leaky-relu', type=str2bool, default=self.LEAKY_RELU,
#             help="Run using leaky relu [default: {}]".format(self.LEAKY_RELU))

#         return parser



class SparseBlock(nn.Module):

    def __init__(self, inplanes, outplanes):

        nn.Module.__init__(self)
        
        self.conv1 = scn.SubmanifoldConvolution(dimension=3, 
            nIn=inplanes, 
            nOut=outplanes, 
            filter_size=3, 
            bias=False)

        if FLAGS.BATCH_NORM:
            if FLAGS.LEAKY_RELU: self.bn1 = scn.BatchNormLeakyReLU(outplanes)
            else:                self.bn1 = scn.BatchNormReLU(outplanes)
        else:
            if FLAGS.LEAKY_RELU: self.relu = scn.LeakyReLU()
            else:                self.relu = scn.ReLU()

    def forward(self, x):

        out = self.conv1(x)
        if FLAGS.BATCH_NORM:
            out = self.bn1(out)
        else:
            out = self.relu(out)

        return out



class SparseResidualBlock(nn.Module):

    def __init__(self, inplanes, outplanes):
        nn.Module.__init__(self)
        
        
        self.conv1 = scn.SubmanifoldConvolution(dimension=3, 
            nIn         = inplanes, 
            nOut        = outplanes, 
            filter_size = 3, 
            bias=False)
        
        if FLAGS.BATCH_NORM:
            if FLAGS.LEAKY_RELU: self.bn1 = scn.BatchNormLeakyReLU(outplanes)
            else:                self.bn1 = scn.BatchNormReLU(outplanes)

        self.conv2 = scn.SubmanifoldConvolution(dimension=3, 
            nIn         = outplanes,
            nOut        = outplanes,
            filter_size = 3,
            bias        = False)

        if FLAGS.BATCH_NORM:
            self.bn2 = scn.BatchNormalization(outplanes)

        self.residual = scn.Identity()
        
        if FLAGS.LEAKY_RELU: self.relu = scn.LeakyReLU()
        else:                self.relu = scn.ReLU()

        self.add = scn.AddTable()

    def forward(self, x):

        residual = self.residual(x)

        out = self.conv1(x)

        if FLAGS.BATCH_NORM:
            out = self.bn1(out)
        else:
            out = self.relu(out)

        out = self.conv2(out)

        if FLAGS.BATCH_NORM:
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

        if FLAGS.BATCH_NORM:
            self.bn   = scn.BatchNormalization(outplanes)
            
        if FLAGS.LEAKY_RELU: self.relu = scn.LeakyReLU()
        else:                self.relu = scn.ReLU()

    def forward(self, x):
        out = self.conv(x)

        if FLAGS.BATCH_NORM:
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

class FullyConnectedSeries(torch.nn.Module):

    def __init__(self, inplanes, outplanes1, outplanes2):
        torch.nn.Module.__init__(self)

        self.linear1 = nn.Linear(inplanes, outplanes1)
        self.dropout1 = nn.Dropout()

        self.linear2 = nn.Linear(outplanes1, outplanes2)
        self.dropout2 = nn.Dropout()

    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout1(x)

        x = self.linear2(x)
        x = self.dropout2(x)

        return x


def filter_increase(input_filters):
    # return input_filters * 2
    return input_filters + FLAGS.N_INITIAL_FILTERS

class ResNet(torch.nn.Module):

    def __init__(self, output_shape):
        torch.nn.Module.__init__(self)
        # All of the parameters are controlled via the flags module

        # Create the sparse input tensor:
        # The real spatial size of the inputs is (1333, 1333, 1666)
        # But, this size is stupid.
        self.input_tensor = scn.InputLayer(dimension=3, spatial_size=(1536,1536,1536))

        # Here, define the layers we will need in the forward path:


        
        # The convolutional layers, which can be shared or not across planes,
        # are defined below

        # We apply an initial convolution, to each plane, to get n_inital_filters

        self.initial_convolution = scn.SubmanifoldConvolution(3, 1, 
            FLAGS.N_INITIAL_FILTERS, filter_size=5, bias=False)

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
                    
            # self.bottleneck  = scn.SubmanifoldConvolution(dimension=3, 
            #             nIn=n_filters, 
            #             nOut=output_shape[key][-1], 
            #             filter_size=3, 
            #             bias=False)

            self.sparse_to_dense = scn.SparseToDense(dimension=3, nPlanes=output_shape[-1])

        else:

            self.final_layer = { 
                    key : SparseBlockSeries(
                        inplanes = n_filters, 
                        n_blocks = FLAGS.RES_BLOCKS_PER_LAYER,
                        residual = True)
                    for key in output_shape
                }

            # if not FLAGS.BOTTLENECK_FC:

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

            # else:

            #     self.bottleneck  = { 
            #             key : scn.SubmanifoldConvolution(dimension=3, 
            #                 nIn=n_filters, 
            #                 nOut=8, 
            #                 filter_size=1, 
            #                 bias=False)
            #             for key in output_shape
            #         }

            #     self.sparse_to_dense = {
            #             key : scn.SparseToDense(dimension=3, nPlanes=8)
            #             for key in output_shape
            #         }

            #     self.fully_connected = {
            #             key : FullyConnectedSeries(inplanes=1728,
            #                 outplanes1=64,
            #                 outplanes2=output_shape[key][-1]
            #                 )
            #             for key in output_shape
            #     }

            for key in self.final_layer:
                self.add_module("final_layer_{}".format(key), self.final_layer[key])
                self.add_module("bottleneck_{}".format(key), self.bottleneck[key])
                self.add_module("sparse_to_dense_{}".format(key), self.sparse_to_dense[key])
                # if FLAGS.BOTTLENECK_FC:
                #     self.add_module("fully_connected_{}".format(key), self.fully_connected[key])



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

                # if FLAGS.BOTTLENECK_FC:
                #     # Flatten
                #     output[key] = output[key].reshape(output[key].size(0), -1)
                #     # Fully connected layers
                #     output[key] = self.fully_connected[key](output[key])                
                # else:
                # Apply global average pooling 
                kernel_size = output[key].shape[2:]
                output[key] = torch.squeeze(nn.AvgPool3d(kernel_size, ceil_mode=False)(output[key]))
                output[key] = output[key].view([batch_size, output[key].shape[-1]])


        return output



