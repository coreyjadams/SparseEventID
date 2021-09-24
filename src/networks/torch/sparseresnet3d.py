import torch
import torch.nn as nn
try:
    import sparseconvnet as scn
except:
    scn = None

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
        # The real spatial size of the inputs is (900, 500, 1250)
        # But, this size is stupid.
        self.input_tensor = scn.InputLayer(dimension=3, spatial_size=(1024,512,1280))

        # Here, define the layers we will need in the forward path:



        # The convolutional layers, which can be shared or not across planes,
        # are defined below

        # We apply an initial convolution, to each plane, to get n_initial_filters

        self.initial_convolution = scn.SubmanifoldConvolution(3, 1,
            args.network.n_initial_filters, filter_size=5, bias=False)

        n_filters = args.network.n_initial_filters
        # Next, build out the convolution steps




        self.convolutional_layers = []
        for layer in range(args.network.network_depth):

            self.convolutional_layers.append(SparseBlockSeries(
                n_filters,
                args.network.res_blocks_per_layer,
                batch_norm = args.network.batch_norm,
                leaky_relu = args.network.leaky_relu,
                residual = True))
            out_filters = filter_increase(n_filters, args.network.n_initial_filters)
            self.convolutional_layers.append(SparseConvolutionDownsample(
                inplanes = n_filters,
                batch_norm = args.network.batch_norm,
                leaky_relu = args.network.leaky_relu,
                outplanes = out_filters))
                # outplanes = n_filters + args.network.n_initial_filters))
            n_filters = out_filters

            self.add_module("conv_{}".format(layer), self.convolutional_layers[-2])
            self.add_module("down_{}".format(layer), self.convolutional_layers[-1])

        # Here, take the final output and convert to a dense tensor:


        self.final_layer = torch.nn.ModuleDict({
                key :
                    torch.nn.Sequential(
                        SparseBlockSeries(
                            inplanes    = n_filters,
                            n_blocks    = args.network.res_blocks_per_layer,
                            batch_norm  = args.network.batch_norm,
                            leaky_relu  = args.network.leaky_relu,
                            residual    = True),
                        scn.SubmanifoldConvolution(dimension=3,
                            nIn         = n_filters,
                            nOut        = output_shape[key][-1],
                            filter_size = 1,
                            bias        = False),
                        scn.SparseToDense(dimension=3, nPlanes=output_shape[key][-1])
                    )
                for key in output_shape
            })

        if 'detect_vertex' in args.network and args.network.detect_vertex:
            self.detect_vertex=True
            # How many dense planes?  in 3D yolo we need a classification
            # score (vertex or not) + regression layer x 3D = 4
            self.vertex_layer = torch.nn.Sequential(
                scn.SubmanifoldConvolution(dimension=3,
                    nIn         = n_filters,
                    nOut        = 16,
                    filter_size = 1,
                    bias        = False),
                scn.SparseToDense(dimension=3, nPlanes=16),
                torch.nn.Flatten(),
                torch.nn.Linear(640, 3),
                torch.nn.Sigmoid()
            )
        else:
            self.detect_vertex = False


    def forward(self, x):

        torch.cuda.synchronize()
        batch_size = x[2]

        x = self.input_tensor(x)

        x = self.initial_convolution(x)



        for i in range(len(self.convolutional_layers)):
            x = self.convolutional_layers[i](x)

        # Apply the final steps to get the right output shape



        output = {}
        for key in self.final_layer:
            # Apply the final residual block:
            output[key] = self.final_layer[key](x)

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


        if self.detect_vertex:
            vertex = self.vertex_layer(x)

            return output, vertex
        else:
            return output
