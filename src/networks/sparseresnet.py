import torch
import torch.nn as nn
try:
    import sparseconvnet as scn
except:
    scn = None

from . network_config import network_config, str2bool


class ResNetFlags(network_config):

    def __init__(self):
        network_config.__init__(self)
        self._name = "sparseresnet2d"
        self._help = "Sparse Resnet with siamese tower structure"

    def build_parser(self, network_parser):
        # this_parser = network_parser
        this_parser = network_parser.add_parser(self._name, help=self._help)

        this_parser.add_argument("--n-initial-filters",
            type    = int,
            default = 2,
            help    = "Number of filters applied, per plane, for the initial convolution")

        this_parser.add_argument("--res-blocks-per-layer",
            help    = "Number of residual blocks per layer",
            type    = int,
            default = 2)

        this_parser.add_argument("--network-depth-pre-merge",
            help    = "Total number of downsamples to apply before merging planes",
            type    = int,
            default = 4)

        this_parser.add_argument("--network-depth-post-merge",
            help    = "Total number of downsamples to apply after merging planes",
            type    = int,
            default = 2)

        this_parser.add_argument("--nplanes",              
            help    = "Number of planes to split the initial image into",
            type    = int,
            default = 3)

        this_parser.add_argument("--batch-norm",
            help    = "Run using batch normalization",
            type    = str2bool,
            default = True)

        this_parser.add_argument("--leaky-relu",
            help    = "Run using leaky relu",
            type    = str2bool,
            default = False)        






class SparseBlock(nn.Module):

    def __init__(self, inplanes, outplanes, batch_norm, leaky_relu, nplanes=1):

        nn.Module.__init__(self)
        
        self.batch_norm = batch_norm
        self.leaky_relu = leaky_relu
        
        self.conv1 = scn.SubmanifoldConvolution(dimension=3, 
            nIn=inplanes, 
            nOut=outplanes, 
            filter_size=[nplanes,3,3], 
            bias=False)

        if self.batch_norm:
            if self.leaky_relu: self.bn1 = scn.BatchNormLeakyReLU(outplanes)
            else:               self.bn1 = scn.BatchNormReLU(outplanes)
        else:
            if self.leaky_relu: self.relu = scn.LeakyReLU()
            else:               self.relu = scn.ReLU()

    def forward(self, x):

        out = self.conv1(x)
        if self.batch_norm:
            out = self.bn1(out)
        else:
            out = self.relu(out)

        return out



class SparseResidualBlock(nn.Module):

    def __init__(self, inplanes, outplanes, batch_norm, leaky_relu, nplanes=1):
        nn.Module.__init__(self)
        
        self.batch_norm = batch_norm
        self.leaky_relu = leaky_relu

        self.conv1 = scn.SubmanifoldConvolution(dimension=3, 
            nIn         = inplanes, 
            nOut        = outplanes, 
            filter_size = [nplanes,3,3], 
            bias=False)
        

        if self.batch_norm:
            if self.leaky_relu: self.bn1 = scn.BatchNormLeakyReLU(outplanes)
            else:                self.bn1 = scn.BatchNormReLU(outplanes)

        self.conv2 = scn.SubmanifoldConvolution(dimension=3, 
            nIn         = outplanes,
            nOut        = outplanes,
            filter_size = [nplanes,3,3],
            bias        = False)

        if self.batch_norm:
            self.bn2 = scn.BatchNormalization(outplanes)

        self.residual = scn.Identity()

        if self.leaky_relu: self.relu = scn.LeakyReLU()
        else:               self.relu = scn.ReLU()

        self.add = scn.AddTable()

    def forward(self, x):

        residual = self.residual(x)

        out = self.conv1(x)

        if self.batch_norm:
            out = self.bn1(out)
        else:
            out = self.relu(out)

        out = self.conv2(out)

        if self.batch_norm:
            out = self.bn2(out)

        # The addition of sparse tensors is not straightforward, since

        out = self.add([out, residual])

        out = self.relu(out)

        return out




class SparseConvolutionDownsample(nn.Module):

    def __init__(self, inplanes, outplanes, batch_norm, leaky_relu, nplanes=1):
        nn.Module.__init__(self)

        self.batch_norm = batch_norm
        self.leaky_relu = leaky_relu

        self.conv = scn.Convolution(dimension=3,
            nIn             = inplanes,
            nOut            = outplanes,
            filter_size     = [nplanes,2,2],
            filter_stride   = [1,2,2],
            bias            = False
        )

        if self.batch_norm:
            self.bn   = scn.BatchNormalization(outplanes)
            
        if self.leaky_relu: self.relu = scn.LeakyReLU()
        else:               self.relu = scn.ReLU()

    def forward(self, x):
        out = self.conv(x)

        if self.batch_norm:
            out = self.bn(out)

        out = self.relu(out)
        return out

class SparseBlockSeries(torch.nn.Module):


    def __init__(self, inplanes, n_blocks, nplanes, batch_norm, leaky_relu, residual=False):
        torch.nn.Module.__init__(self)



        if residual:
            self.blocks = [ SparseResidualBlock(inplanes, inplanes, batch_norm, leaky_relu, nplanes=nplanes) for i in range(n_blocks) ]
        else:
            self.blocks = [ SparseBlock(inplanes, inplanes, batch_norm, leaky_relu, nplanes=nplanes) for i in range(n_blocks)]

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
    return input_filters + self.n_initial_filters


class ResNet(torch.nn.Module):

    def __init__(self, output_shape, args):
        torch.nn.Module.__init__(self)
        # All of the parameters are controlled via the self module

        if scn is None:
            raise Exception("Couldn't import sparse conv net!")

        # Create the sparse input tensor:
        # (first spatial dim is plane)
        self.input_tensor = scn.InputLayer(dimension=3, spatial_size=[self.NPLANES,2048, 1280])

        spatial_size = [2048, 1280]

        
        # The convolutional layers, which can be shared or not across planes,
        # are defined below

        # We apply an initial convolution, to each plane, to get n_inital_filters


        self.initial_convolution = scn.SubmanifoldConvolution(dimension=3, 
            nIn=1, 
            nOut=args.n_initial_filters, 
            filter_size=[1,5,5], 
            bias=False)
        n_filters = args.n_initial_filters
        # Next, build out the convolution steps


        self.pre_convolutional_layers = []
        for layer in range(args.network_depth_pre_merge):
            out_filters = filter_increase(n_filters)

            self.pre_convolutional_layers.append(
                SparseBlockSeries(inplanes = n_filters, 
                    n_blocks = args.res_blocks_per_layer,
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



        # n_filters *= args.NPLANES
        self.post_convolutional_layers = []
        for layer in range(args.network_depth_post_merge):
            out_filters = filter_increase(n_filters)

            self.post_convolutional_layers.append(
                SparseBlockSeries(
                    inplanes = n_filters, 
                    n_blocks = args.res_blocks_per_layer,
                    nplanes  = args.NPLANES,
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
        self.label_mode = args.label_mode

        if args.label_mode == 'all':
            self.final_layer = SparseBlockSeries(n_filters, 
                n_filters, 
                args.res_blocks_per_layer,
                nplanes=args.NPLANES)
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
                        n_blocks = args.res_blocks_per_layer,
                        nplanes  = args.NPLANES,
                        residual = True)
                    for key in output_shape
                }
            spatial_size =  [ ss / 2 for ss in spatial_size ]

            # if not args.BOTTLENECK_FC:
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
                # if args.BOTTLENECK_FC:
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

        if self.label_mode == 'all':
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



