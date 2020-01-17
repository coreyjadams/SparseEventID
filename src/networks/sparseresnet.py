import torch
import torch.nn as nn
import sparseconvnet as scn

from src import utils

FLAGS = utils.flags.FLAGS()
    #     parser.add_argument('--leaky-relu', type=str2bool, default=self.LEAKY_RELU,
    #         help="Run using leaky relu [default: {}]".format(self.LEAKY_RELU))


# class resnet(FLAGS):
#     ''' Sparse Resnet specific flags
#     '''

#     def __init__(self):
#         FLAGS.__init__(self)


#     def set_net(self, net):
#         # For the resnet object, we set the network as resnet:
#         self._net = net


#     def _set_defaults(self):

#         self.VERBOSITY                  = 0
#         self.N_INITIAL_FILTERS          = 2
#         self.RES_BLOCKS_PER_LAYER       = 2
#         self.NETWORK_DEPTH_PRE_MERGE    = 4
#         self.NETWORK_DEPTH_POST_MERGE   = 2
#         self.NPLANES                    = 3
#         self.SHARE_WEIGHTS              = True
#         self.WEIGHT_DECAY               = 1e-4
#         self.BATCH_NORM                 = True
#         self.LEAKY_RELU                 = False   

#         # self.BOTTLENECK_FC              = False

#         self.SPARSE                     = True

#         self.INPUT_DIMENSION            = '2D'

#         FLAGS._set_defaults(self)

#     def _add_default_network_configuration(self, parser):



#         parser.add_argument('-v', '--verbosity', type=int,default=self.VERBOSITY,
#             help="Network verbosity at construction [default: {}]".format(self.VERBOSITY))

#         parser.add_argument('--weight-decay', type=float, default=self.WEIGHT_DECAY,
#             help="Weight decay strength [default: {}]".format(self.WEIGHT_DECAY))

#         parser.add_argument('--n-initial-filters', type=int, default=self.N_INITIAL_FILTERS,
#             help="Number of filters applied, per plane, for the initial convolution [default: {}]".format(self.N_INITIAL_FILTERS))
#         parser.add_argument('--res-blocks-per-layer', type=int, default=self.RES_BLOCKS_PER_LAYER,
#             help="Number of residual blocks per layer [default: {}]".format(self.RES_BLOCKS_PER_LAYER))
#         parser.add_argument('--network-depth-pre-merge', type=int, default=self.NETWORK_DEPTH_PRE_MERGE,
#             help="Total number of downsamples to apply before merging planes [default: {}]".format(self.NETWORK_DEPTH_PRE_MERGE))
#         parser.add_argument('--network-depth-post-merge', type=int, default=self.NETWORK_DEPTH_POST_MERGE,
#             help="Total number of downsamples to apply after merging planes [default: {}]".format(self.NETWORK_DEPTH_POST_MERGE))
#         parser.add_argument('--nplanes', type=int, default=self.NPLANES,
#             help="Number of planes to split the initial image into [default: {}]".format(self.NPLANES))
#         parser.add_argument('--share-weights', type=str2bool, default=self.SHARE_WEIGHTS,
#             help="Whether or not to share weights across planes [default: {}]".format(self.SHARE_WEIGHTS))

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

    def __init__(self, inplanes, outplanes, nplanes=1):

        nn.Module.__init__(self)
        
        self.conv1 = scn.SubmanifoldConvolution(dimension=3, 
            nIn=inplanes, 
            nOut=outplanes, 
            filter_size=[nplanes,3,3], 
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

    def __init__(self, inplanes, outplanes, nplanes=1):
        nn.Module.__init__(self)
        
        
        self.conv1 = scn.SubmanifoldConvolution(dimension=3, 
            nIn         = inplanes, 
            nOut        = outplanes, 
            filter_size = [nplanes,3,3], 
            bias=False)
        

        if FLAGS.BATCH_NORM:
            if FLAGS.LEAKY_RELU: self.bn1 = scn.BatchNormLeakyReLU(outplanes)
            else:                self.bn1 = scn.BatchNormReLU(outplanes)

        self.conv2 = scn.SubmanifoldConvolution(dimension=3, 
            nIn         = outplanes,
            nOut        = outplanes,
            filter_size = [nplanes,3,3],
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

    def __init__(self, inplanes, outplanes,nplanes=1):
        nn.Module.__init__(self)

        self.conv = scn.Convolution(dimension=3,
            nIn             = inplanes,
            nOut            = outplanes,
            filter_size     = [nplanes,2,2],
            filter_stride   = [1,2,2],
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


    def __init__(self, inplanes, n_blocks, nplanes, residual=False):
        torch.nn.Module.__init__(self)

        if residual:
            self.blocks = [ SparseResidualBlock(inplanes, inplanes, nplanes=nplanes) for i in range(n_blocks) ]
        else:
            self.blocks = [ SparseBlock(inplanes, inplanes, nplanes=nplanes) for i in range(n_blocks)]

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


        # self.add_module('fc1', self.linear1)
        # self.add_module('dropout1', self.dropout1)
        # self.add_module('fc2', self.linear2)
        # self.add_module('dropout2', self.dropout2)

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
        # (first spatial dim is plane)
        self.input_tensor = scn.InputLayer(dimension=3, spatial_size=[FLAGS.NPLANES,2048, 1280])

        spatial_size = [2048, 1280]

        
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
            out_filters = filter_increase(n_filters)

            self.pre_convolutional_layers.append(
                SparseBlockSeries(inplanes = n_filters, 
                    n_blocks = FLAGS.RES_BLOCKS_PER_LAYER,
                    nplanes  = 1,
                    residual = True)
                )
            self.pre_convolutional_layers.append(
                SparseConvolutionDownsample(inplanes=n_filters,
                    outplanes = out_filters,
                    nplanes = 1)
                )
            n_filters = out_filters

            spatial_size =  [ ss / 2 for ss in spatial_size ]
            self.add_module("pre_merge_conv_{}".format(layer), 
                self.pre_convolutional_layers[-2])
            self.add_module("pre_merge_down_{}".format(layer), 
                self.pre_convolutional_layers[-1])



        # n_filters *= FLAGS.NPLANES
        self.post_convolutional_layers = []
        for layer in range(FLAGS.NETWORK_DEPTH_POST_MERGE):
            out_filters = filter_increase(n_filters)

            self.post_convolutional_layers.append(
                SparseBlockSeries(
                    inplanes = n_filters, 
                    n_blocks = FLAGS.RES_BLOCKS_PER_LAYER,
                    nplanes  = FLAGS.NPLANES,
                    residual = True)
                )
            self.post_convolutional_layers.append(
                SparseConvolutionDownsample(
                    inplanes  = n_filters,
                    outplanes = out_filters,
                    nplanes   = 1)
                )
            n_filters = out_filters

            spatial_size =  [ ss / 2 for ss in spatial_size ]

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
            self.final_layer = SparseBlockSeries(n_filters, 
                n_filters, 
                FLAGS.RES_BLOCKS_PER_LAYER,
                nplanes=FLAGS.NPLANES)
            spatial_size =  [ ss / 2 for ss in spatial_size ]

            self.bottleneck = scn.SubmanifoldConvolution(dimension=3, 
                        nIn=n_filters, 
                        nOut=output_shape[-1], 
                        filter_size=1, 
                        bias=False)

            self.sparse_to_dense = scn.SparseToDense(dimension=3, nPlanes=output_shape[-1])
        else:
            self.final_layer = { 
                    key : SparseBlockSeries(
                        inplanes = n_filters, 
                        n_blocks = FLAGS.RES_BLOCKS_PER_LAYER,
                        nplanes  = FLAGS.NPLANES,
                        residual = True)
                    for key in output_shape
                }
            spatial_size =  [ ss / 2 for ss in spatial_size ]

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

                # self.fully_connected = {
                #         key : FullyConnectedSeries(inplanes=15360,
                #             outplanes1=64,
                #             outplanes2=output_shape[key][-1]
                #             )
                #         for key in output_shape
                # }

            for key in self.final_layer:
                self.add_module("final_layer_{}".format(key), self.final_layer[key])
                self.add_module("bottleneck_{}".format(key), self.bottleneck[key])
                self.add_module("sparse_to_dense_{}".format(key), self.sparse_to_dense[key])
                # if FLAGS.BOTTLENECK_FC:
                #     self.add_module("fully_connected_{}".format(key), self.fully_connected[key])


        # Sparse to Dense conversion to apply before global average pooling:

        # The rest of the final operations (reshape, softmax) are computed in the forward pass


        # # Configure initialization:
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, scn.SubmanifoldConvolution):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, scn.BatchNormReLU) or isinstance(m, scn.BatchNormalization):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)


    def forward(self, x):
        
        FLAGS = utils.flags.FLAGS()

        # # Split the input into NPLANES streams
        # x = [ _ for _ in torch.split(x, 1, dim=1)]
        # for the sparse input data, it's ALREADY split

        batch_size = x[-1]

        # Convert to the right format:
        x = self.input_tensor(x )
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
                

                # Convert to dense tensor:
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

                # Squeeze off the last few dimensions:
                output[key] = output[key].view([batch_size, output[key].shape[-1]])

                # output[key] = scn.AveragePooling(dimension=3,
                #     pool_size=kernel_size, pool_stride=kernel_size)(output[key])

                # print (output[key].spatial_size)
                # print (output[key])
    
                # print (output[key].size())

                # output[key] = nn.Softmax(dim=1)(output[key])

        return output



