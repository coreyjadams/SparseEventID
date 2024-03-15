import torch
import sparseconvnet as scn

import numpy

from src.config.network import GrowthRate, DownSampling
from src.config.framework import DataMode


class Encoder(torch.nn.Module):

    def __init__(self, params, image_size):

        super().__init__()

        dimension = params.data.dimension

        Block, BlockSeries, ConvDonsample, Pool, InputNorm = \
            self.import_building_blocks(params.framework.mode)

        # How many filters did we start with?
        current_number_of_filters = params.encoder.n_initial_filters
        if params.framework.mode == DataMode.sparse:

            if dimension == 2:
                self.input_layer = scn.InputLayer(
                    dimension    = 3,
                    spatial_size = torch.tensor(image_size)
                )
                self.initial_convolution = scn.SubmanifoldConvolution(
                    dimension   = 3,
                    nIn         = 1,
                    nOut        = params.encoder.n_initial_filters,
                    filter_size = [1,5,5],
                    bias        = params.encoder.bias
                )
            else:
                # Padding the 3D data to make things a little easier
                # image_size3d = (1024, 512, 1280)
                self.input_layer = scn.InputLayer(
                    dimension    = 3,
                    spatial_size = torch.tensor(image_size[1:])
                )
                self.initial_convolution = scn.SubmanifoldConvolution(
                    dimension   = 3,
                    nIn         = 1,
                    nOut        = params.encoder.n_initial_filters,
                    filter_size = [5,5,5],
                    bias        = params.encoder.bias
                )   

        else:
            self.input_layer = torch.nn.Identity()
            self.initial_convolution = nn.Conv2d(
                dimension   = 3,
                nIn         = 1,
                nOut        = params.encoder.n_initial_filters,
                filter_size = [1,5,5],
                bias        = params.encoder.bias
            )
        # self.first_block = Block(
        #     nIn    = 1,
        #     nOut   = params.encoder.n_initial_filters,
        #     params = params.encoder
        # )




        if params.encoder.downsampling == DownSampling.convolutional:
            downsampler = ConvDonsample
        else:
            downsampler = Pool

        self.network_layers = torch.nn.ModuleList()



        for i_layer in range(params.encoder.depth):
            self.network_layers.append(
                BlockSeries(
                    nIn      = current_number_of_filters,
                    n_blocks = params.encoder.blocks_per_layer,
                    dim      = dimension,
                    params   = params.encoder
                )
            )
            next_filters = self.increase_filters(current_number_of_filters, params.encoder)
            self.network_layers.append(downsampler(
                    nIn    = current_number_of_filters,
                    nOut   = next_filters,
                    dim    = dimension,
                    params = params.encoder
                )
            )
            current_number_of_filters = next_filters

        self.final_layer = BlockSeries(
                    nIn      = current_number_of_filters,
                    n_blocks = params.encoder.blocks_per_layer,
                    dim      = dimension,
                    params   = params.encoder
                )

        self.bottleneck  = scn.SubmanifoldConvolution(
                    dimension=3,
                    nIn  = current_number_of_filters,
                    nOut = params.encoder.n_output_filters,
                    filter_size=1,
                    bias = params.encoder.bias)



        final_shape = [i // 2**params.encoder.depth for i in image_size]
        if dimension == 2:
            final_shape[0] = 3
        else:
            final_shape = final_shape[1:]
        self.output_shape = [params.encoder.n_output_filters,] +  final_shape

        # We apply a global pooling layer to the image, to produce the encoding:
        if params.framework.mode == DataMode.sparse:
            self.pool = torch.nn.Sequential(
                scn.SparseToDense(
                    dimension=3, nPlanes=self.output_shape[0]),
                # torch.nn.AvgPool3d(self.output_shape[1:]),
                # torch.nn.AvgPool3d(self.output_shape[1:], divisor_override=1),
            )
        else:
            self.pool = torch.nn.AvgPool3d(self.output_shape[1:])


        # if params.framework.mode == DataMode.sparse:
        #     self.final_activation = scn.Tanh()
        # else:
        #     self.final_activation = torch.nn.Tanh()

        # self.flatten = torch.nn.Flatten(start_dim=1, end_dim=-1)


    def forward(self, x):

        x = self.input_layer(x)
        x = self.initial_convolution(x)
        
        for i, l in enumerate(self.network_layers):
            x = l(x)

        # Last convolutional Layers:
        x = self.final_layer(x)
        
        # Bottleneck for the right number of outputs:
        x = self.bottleneck(x)
        
        # # Pool correctly: 
        x = self.pool(x)

        # Normalize the output features?
        x = torch.tanh(x)

        return x


    def increase_filters(self, current_filters, params):
        if params.growth_rate == GrowthRate.multiplicative:
            return current_filters * 2
        else: # GrowthRate.additive
            return current_filters + params.n_initial_filters

    def import_building_blocks(self, mode):
        if mode == DataMode.sparse:
            from . sparse_building_blocks import Block
            from . sparse_building_blocks import ConvolutionDownsample
            from . sparse_building_blocks import BlockSeries
            from . sparse_building_blocks import InputNorm
            from . sparse_building_blocks import Pooling
        else:
            from . building_blocks import Block, BlockSeries
            from . building_blocks import ConvolutionDownsample
            from . building_blocks import MaxPooling
            from . building_blocks import InputNorm
        return Block, BlockSeries, ConvolutionDownsample, Pooling, InputNorm
