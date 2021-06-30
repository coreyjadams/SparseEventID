import torch
import torch.nn as nn
try:
    import sparseconvnet as scn
except:
    scn = None





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


def filter_increase(input_filters, initial_filters):
    # return input_filters * 2
    return input_filters + initial_filters


class ResNet(torch.nn.Module):

    def __init__(self, output_shape, args):
        torch.nn.Module.__init__(self)
        # All of the parameters are controlled via the self module

        if scn is None:
            raise Exception("Couldn't import sparse conv net!")

        # Create the sparse input tensor:
        # (first spatial dim is plane)
        self.input_tensor = scn.InputLayer(dimension=3, spatial_size=[3,2048, 1280])

        spatial_size = [2048, 1280]


        # The convolutional layers, which can be shared or not across planes,
        # are defined below

        # We apply an initial convolution, to each plane, to get n_inital_filters


        self.initial_convolution = scn.SubmanifoldConvolution(dimension=3,
            nIn=1,
            nOut=args.network.n_initial_filters,
            filter_size=[1,5,5],
            bias=False)
        n_filters = args.network.n_initial_filters
        # Next, build out the convolution steps

        depth_pre_merge = args.network.depth_pre_merge
        depth_post_merge = args.network.network_depth - args.network.depth_pre_merge

        self.pre_convolutional_layers = torch.nn.ModuleList()

        for layer in range(depth_pre_merge):
            out_filters = filter_increase(n_filters, args.network.n_initial_filters)

            self.pre_convolutional_layers.append(
                SparseBlockSeries(inplanes = n_filters,
                    n_blocks = args.network.res_blocks_per_layer,
                    nplanes  = 1,
                    batch_norm = args.network.batch_norm,
                    leaky_relu = args.network.leaky_relu,
                    residual = True)
                )
            self.pre_convolutional_layers.append(
                SparseConvolutionDownsample(inplanes=n_filters,
                    outplanes = out_filters,
                    batch_norm = args.network.batch_norm,
                    leaky_relu = args.network.leaky_relu,
                    nplanes = 1)
                )
            n_filters = out_filters

            spatial_size =  [ ss / 2 for ss in spatial_size ]



        # n_filters *= args.NPLANES
        self.post_convolutional_layers = torch.nn.ModuleList()
        for layer in range(depth_post_merge):
            out_filters = filter_increase(n_filters, args.network.n_initial_filters)

            self.post_convolutional_layers.append(
                SparseBlockSeries(
                    inplanes = n_filters,
                    n_blocks = args.network.res_blocks_per_layer,
                    nplanes  = 3,
                    batch_norm = args.network.batch_norm,
                    leaky_relu = args.network.leaky_relu,
                    residual = True)
                )
            self.post_convolutional_layers.append(
                SparseConvolutionDownsample(
                    inplanes  = n_filters,
                    outplanes = out_filters,
                    batch_norm = args.network.batch_norm,
                    leaky_relu = args.network.leaky_relu,
                    nplanes   = 1)
                )
            n_filters = out_filters

            spatial_size =  [ ss / 2 for ss in spatial_size ]


        # Now prepare the output operations.  In general, it's a Block Series,
        # then a 1x1 to get the right number of filters, then a global average pooling,
        # then a reshape


        self.final_layer = torch.nn.ModuleDict({
                key : torch.nn.Sequential(
                    SparseBlockSeries(
                        inplanes = n_filters,
                        n_blocks = args.network.res_blocks_per_layer,
                        nplanes  = 3,
                        batch_norm = args.network.batch_norm,
                        leaky_relu = args.network.leaky_relu,
                        residual = True),
                    scn.SubmanifoldConvolution(dimension=3,
                        nIn=n_filters,
                        nOut=output_shape[key][-1],
                        filter_size=1,
                        bias=False),
                    scn.SparseToDense(dimension=3, nPlanes=output_shape[key][-1])
                    )
                for key in output_shape
            })

    def forward(self, x):

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

        output = {}
        for key in self.final_layer:
            # Apply the final residual block:
            output[key] = self.final_layer[key](x)

            # Apply global average pooling
            kernel_size = output[key].shape[2:]
            output[key] = torch.squeeze(nn.AvgPool3d(kernel_size, ceil_mode=False)(output[key]))

            # Squeeze off the last few dimensions:
            output[key] = output[key].view([batch_size, output[key].shape[-1]])

        return output
