import torch
from torch import nn


from src.config.network import  Norm


class InputNorm(nn.Module):

    def __init__(self, *, nIn, nOut):
        nn.Module.__init__(self)

        self.layer = nn.BatchNorm3d(nOut)

    def forward(self, x):

        return self.layer(x)


class Block(nn.Module):

    def __init__(self, *,
            nIn,
            nOut,
            kernel     = [3,3,3],
            padding    = "same",
            strides    = [1,1,1],
            activation = nn.functional.leaky_relu,
            params):
        nn.Module.__init__(self)

        dimension = len(kernel)
        assert len(kernel) == len(strides)

        self.nOut = nOut
        if dimension == 2:
            self.conv = nn.Conv2d(
                in_channels  = nIn,
                out_channels = nOut,
                kernel_size  = kernel,
                stride       = strides,
                padding      = padding,
                bias         = params.bias)         
        elif dimension == 3:
            self.conv = nn.Conv3d(
                in_channels  = nIn,
                out_channels = nOut,
                kernel_size  = kernel,
                stride       = strides,
                padding      = padding,
                bias         = params.bias)




        if params.normalization == Norm.batch:
            self._do_normalization = True
            if dimension == 2:
                self.norm = nn.BatchNorm2d(nOut)
            elif dimension == 3:
                self.norm = nn.BatchNorm3d(nOut)
        elif params.normalization == Norm.layer:
            self._do_normalization = True
            self.norm = nn.LayerNorm(nOut)
        elif: params.normalization == Norm.instance:
            self._do_normalization = True
            if dimension == 2:
                self.norm = nn.InstanceNorm2d(nOut)
            elif dimension == 3:
                self.norm = nn.InstanceNorm3d(nOut)
        else:
            self._do_normalization = False


        self.activation = activation

    def forward(self, x):
        out = self.conv(x)
        if self._do_normalization:
            out = self.norm(out)
        out = self.activation(out)
        return out



class ResidualBlock(nn.Module):

    def __init__(self, *,
            nIn,
            nOut,
            kernel  = [3,3,3],
            padding = "same",
            params):
        nn.Module.__init__(self)



        self.convolution_1 = Block(
            nIn         = nIn,
            nOut        = nOut,
            kernel      = kernel,
            padding     = padding,
            params      = params)

        self.convolution_2 = Block(
            nIn         = nIn,
            nOut        = nOut,
            kernel      = kernel,
            padding     = padding,
            activation  = torch.nn.Identity(),
            params      = params)




    def forward(self, x):
        residual = x

        out = self.convolution_1(x)

        out = self.convolution_2(out)


        out += residual
        out = nn.functional.leaky_relu(out)

        return out


class ConvolutionDownsample(nn.Module):

    def __init__(self, *, nIn, nOut, params):
        nn.Module.__init__(self)

        self.conv = nn.Conv3d(
            in_channels  = nIn,
            out_channels = nOut,
            kernel_size  = [2, 2, 2],
            stride       = [2, 2, 2],
            padding      = "valid",
            bias         = params.bias)


        if params.normalization == Norm.batch:
            self._do_normalization = True
            self.norm = nn.BatchNorm3d(nOut)
        elif params.normalization == Norm.layer:
            self._do_normalization = True
            self.norm = nn.LayerNorm(nOut)
        else:
            self._do_normalization = False
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        if self._do_normalization:
            out = self.norm(out)

        out = self.relu(out)
        return out


class ConvolutionUpsample(nn.Module):

    def __init__(self, *, nIn, nOut, params):
        nn.Module.__init__(self)

        self.conv = nn.ConvTranspose3d(
            in_channels  = nIn,
            out_channels = nOut,
            kernel_size  = [2, 2, 2],
            stride       = [2, 2, 2],
            padding      = [0, 0, 0],
            bias         = params.bias)


        if params.normalization == Norm.batch:
            self._do_normalization = True
            self.norm = nn.BatchNorm3d(nOut)
        elif params.normalization == Norm.layer:
            self._do_normalization = True
            self.norm = nn.LayerNorm(nOut)
        else:
            self._do_normalization = False
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        out = self.conv(x)
        if self._do_normalization:
            out = self.norm(out)
        out = self.relu(out)
        return out


class BlockSeries(torch.nn.Module):


    def __init__(self, *, nIn, n_blocks, params, kernel=[3,3,3], padding="same"):
        torch.nn.Module.__init__(self)

        if not params.residual:
            self.blocks = [ Block(nIn = nIn, nOut = nIn,
                kernel=kernel, padding=padding, params = params) for i in range(n_blocks) ]
        else:
            self.blocks = [ ResidualBlock(nIn = nIn, nOut = nIn,
                kernel=kernel, padding=padding, params = params) for i in range(n_blocks)]

        for i, block in enumerate(self.blocks):
            self.add_module('block_{}'.format(i), block)


    def forward(self, x):
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

        return x


class MaxPooling(nn.Module):

    def __init__(self,*, nIn, nOut, params):
        nn.Module.__init__(self)


        self.pool = torch.nn.MaxPool3d(stride=[2,2,2], kernel_size=[2,2,2])

        self.bottleneck = Block(
            nIn     = nIn,
            nOut    = nOut,
            kernel  = [1,1,1],
            padding = "same",
            params  = params)

    def forward(self, x):
        x = self.pool(x)

        return self.bottleneck(x)

class InterpolationUpsample(nn.Module):

    def __init__(self, *, nIn, nOut, params):
        nn.Module.__init__(self)


        self.up = torch.nn.Upsample(scale_factor=(2,2,2), mode="nearest")

        self.bottleneck = Block(
            nIn     = nIn,
            nOut    = nOut,
            kernel  = [1,1,1],
            padding = "same",
            params  = params)

    def forward(self, x):
        x = self.up(x)
        return self.bottleneck(x)
