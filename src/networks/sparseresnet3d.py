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
        self._name = "sparseresnet3d"
        self._help = "Sparse Resnet"

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

        this_parser.add_argument("--network-depth",
            help    = "Total number of downsamples to apply",
            type    = int,
            default = 8)

        this_parser.add_argument("--batch-norm",
            help    = "Run using batch normalization",
            type    = str2bool,
            default = True)

        this_parser.add_argument("--leaky-relu",
            help    = "Run using leaky relu",
            type    = str2bool,
            default = False)


class SparseBlock(nn.Module):

    def __init__(self, inplanes, outplanes, batch_norm, leaky_relu):

        nn.Module.__init__(self)

        self.batch_norm = batch_norm
        self.leaky_relu = leaky_relu

        self.conv1 = scn.SubmanifoldConvolution(dimension=3,
            nIn=inplanes,
            nOut=outplanes,
            filter_size=3,
            bias=False)

        if self.batch_norm:
            if self.leaky_relu: self.bn1 = scn.BatchNormLeakyReLU(outplanes)
            else:                self.bn1 = scn.BatchNormReLU(outplanes)
        else:
            if self.leaky_relu: self.relu = scn.LeakyReLU()
            else:                self.relu = scn.ReLU()

    def forward(self, x):

        out = self.conv1(x)
        if self.batch_norm:
            out = self.bn1(out)
        else:
            out = self.relu(out)

        return out



class SparseResidualBlock(nn.Module):

    def __init__(self, inplanes, outplanes, batch_norm, leaky_relu):
        nn.Module.__init__(self)

        self.batch_norm = batch_norm
        self.leaky_relu = leaky_relu

        self.conv1 = scn.SubmanifoldConvolution(dimension=3,
            nIn         = inplanes,
            nOut        = outplanes,
            filter_size = 3,
            bias=False)

        if self.batch_norm:
            if self.leaky_relu: self.bn1 = scn.BatchNormLeakyReLU(outplanes)
            else:                self.bn1 = scn.BatchNormReLU(outplanes)

        self.conv2 = scn.SubmanifoldConvolution(dimension=3,
            nIn         = outplanes,
            nOut        = outplanes,
            filter_size = 3,
            bias        = False)

        if self.batch_norm:
            self.bn2 = scn.BatchNormalization(outplanes)

        self.residual = scn.Identity()

        if self.leaky_relu: self.relu = scn.LeakyReLU()
        else:                self.relu = scn.ReLU()

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

    def __init__(self, inplanes, outplanes, batch_norm, leaky_relu):
        nn.Module.__init__(self)

        self.batch_norm = batch_norm
        self.leaky_relu = leaky_relu

        self.conv = scn.Convolution(dimension=3,
            nIn             = inplanes,
            nOut            = outplanes,
            filter_size     = 2,
            filter_stride   = 2,
            bias            = False
        )

        if self.batch_norm:
            self.bn   = scn.BatchNormalization(outplanes)

        if self.leaky_relu: self.relu = scn.LeakyReLU()
        else:                self.relu = scn.ReLU()

    def forward(self, x):
        out = self.conv(x)

        if self.batch_norm:
            out = self.bn(out)

        out = self.relu(out)
        return out



class SparseBlockSeries(torch.nn.Module):


    def __init__(self, inplanes, n_blocks, batch_norm, leaky_relu, residual=False):
        torch.nn.Module.__init__(self)

        self.batch_norm = batch_norm
        self.leaky_relu = leaky_relu

        if residual:
            self.blocks = [ SparseResidualBlock(inplanes, inplanes, batch_norm, leaky_relu,) for i in range(n_blocks) ]
        else:
            self.blocks = [ SparseBlock(inplanes, inplanes, batch_norm, leaky_relu,) for i in range(n_blocks)]

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


def filter_increase(input_filters, initial_filters):
    # return input_filters * 2
    return input_filters + initial_filters

class ResNet(torch.nn.Module):

    def __init__(self, output_shape, args):
        torch.nn.Module.__init__(self)
        # All of the parameters are controlled via the flags module


        if scn is None:
            raise Exception("Couldn't import sparse conv net!")


        # Create the sparse input tensor:
        # The real spatial size of the inputs is (1333, 1333, 1666)
        # But, this size is stupid.
        self.input_tensor = scn.InputLayer(dimension=3, spatial_size=(1536,1536,1536))

        # Here, define the layers we will need in the forward path:



        # The convolutional layers, which can be shared or not across planes,
        # are defined below

        # We apply an initial convolution, to each plane, to get n_initial_filters

        self.initial_convolution = scn.SubmanifoldConvolution(3, 1,
            args.n_initial_filters, filter_size=5, bias=False)

        n_filters = args.n_initial_filters
        # Next, build out the convolution steps




        self.convolutional_layers = []
        for layer in range(args.network_depth):

            self.convolutional_layers.append(SparseBlockSeries(
                n_filters,
                args.res_blocks_per_layer,
                batch_norm = args.batch_norm,
                leaky_relu = args.leaky_relu,
                residual = True))
            out_filters = filter_increase(n_filters, args.n_initial_filters)
            self.convolutional_layers.append(SparseConvolutionDownsample(
                inplanes = n_filters,
                batch_norm = args.batch_norm,
                leaky_relu = args.leaky_relu,
                outplanes = out_filters))
                # outplanes = n_filters + args.n_initial_filters))
            n_filters = out_filters

            self.add_module("conv_{}".format(layer), self.convolutional_layers[-2])
            self.add_module("down_{}".format(layer), self.convolutional_layers[-1])

        # Here, take the final output and convert to a dense tensor:

        self.label_mode = args.label_mode
        if args.label_mode == 'all':
            self.final_layer = SparseBlockSeries(
                        inplanes = n_filters,
                        n_blocks = args.res_blocks_per_layer,
                        batch_norm = args.batch_norm,
                        leaky_relu = args.leaky_relu,
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
                        n_blocks = args.res_blocks_per_layer,
                        batch_norm = args.batch_norm,
                        leaky_relu = args.leaky_relu,
                        residual = True)
                    for key in output_shape
                }

            # if not args.bottleneck_fc:

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
                # if args.bottleneck_fc:
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

        x = self.input_tensor(x)

        x = self.initial_convolution(x)



        for i in range(len(self.convolutional_layers)):
            x = self.convolutional_layers[i](x)

        # Apply the final steps to get the right output shape

        if self.label_mode == 'all':

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
