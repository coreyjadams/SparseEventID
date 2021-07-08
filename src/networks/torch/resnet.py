import torch
import torch.nn as nn



class Block(nn.Module):

    def __init__(self, inplanes, outplanes, batch_norm):

        nn.Module.__init__(self)
        
        self.batch_norm = batch_norm

        self.conv1 = torch.nn.Conv2d(
            in_channels  = inplanes, 
            out_channels = outplanes, 
            kernel_size  = [3,3], 
            stride       = [1, 1],
            padding      = [1, 1],
            bias         = False)
        
        if batch_norm:
            self.bn1  = torch.nn.BatchNorm2d(outplanes)
        self.relu = torch.nn.ReLU()

    def forward(self, x):

        out = self.conv1(x)
        if self.batch_norm:
            out = self.bn1(out)
        out = self.relu(out)

        return out



class ResidualBlock(nn.Module):

    def __init__(self, inplanes, outplanes, batch_norm):
        nn.Module.__init__(self)
        
        self.batch_norm = batch_norm

        self.conv1 = torch.nn.Conv2d(
            in_channels  = inplanes, 
            out_channels = outplanes, 
            kernel_size  = [3,3], 
            stride       = [1, 1],
            padding      = [1, 1],
            bias         = False)
        
        if batch_norm:
            self.bn1  = torch.nn.BatchNorm2d(outplanes)


        self.conv2 = torch.nn.Conv2d(
            in_channels  = inplanes, 
            out_channels = outplanes, 
            kernel_size  = [3,3], 
            stride       = [1, 1],
            padding      = [1, 1],
            bias         = False)

        if batch_norm:
            self.bn2  = torch.nn.BatchNorm2d(outplanes)

        self.relu = torch.nn.ReLU()


    def forward(self, x):

        residual = x

        out = self.conv1(x)        
        if self.batch_norm:
            out = self.bn1(out)

        out = self.conv2(out)

        if self.batch_norm:
            out = self.bn2(out)

        out = self.relu(out + residual)

        return out




class ConvolutionDownsample(nn.Module):

    def __init__(self, inplanes, outplanes, batch_norm):
        nn.Module.__init__(self)
        self.batch_norm = batch_norm

        self.conv = torch.nn.Conv2d(
            in_channels  = inplanes,
            out_channels = outplanes,
            kernel_size  = [2,2],
            stride       = [2,2],
            padding      = [0,0],
            bias         = False
        )

        if batch_norm:
            self.bn1  = torch.nn.BatchNorm2d(outplanes)


        self.relu = torch.nn.ReLU()

    def forward(self, x):
        out = self.conv(x)

        if self.batch_norm:
            out = self.bn1(out)

        out = self.relu(out)
        return out

class BlockSeries(torch.nn.Module):


    def __init__(self, inplanes, n_blocks, batch_norm, residual=False):
        torch.nn.Module.__init__(self)

        if residual:
            self.blocks = [ ResidualBlock(inplanes, inplanes, batch_norm) for i in range(n_blocks) ]
        else:
            self.blocks = [ Block(inplanes, inplanes, batch_norm) for i in range(n_blocks)]

        for i, block in enumerate(self.blocks):
            self.add_module('block_{}'.format(i), block)


    def forward(self, x):
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
        return x





def filter_increase(input_filters, initial_filters):
    # return input_filters * 2
    return input_filters + initial_filters


class ResNet(torch.nn.Module):

    def __init__(self, output_shape, args):
        torch.nn.Module.__init__(self)
        # All of the parameters are controlled via the args module

        if args.image_mode != "dense":
            raise Exception("You have selected resnet2d but have not set image-mode to dense, error!")

        if args.input_dimension != 2:
            raise Exception("You have selected resnet2d but have not set input-dimension to 2, error!")


        self.nplanes    = args.nplanes
        self.label_mode = args.label_mode
        
        # We apply an initial convolution, to each plane, to get n_inital_filters


        self.initial_convolution = torch.nn.Conv2d(
            in_channels  = 1, 
            out_channels = args.n_initial_filters, 
            kernel_size  = [5, 5], 
            stride       = [1, 1],
            padding      = [2, 2],
            bias         = False)


        n_filters = args.n_initial_filters
        # Next, build out the convolution steps


        self.pre_convolutional_layers = []
        for layer in range(args.network_depth_pre_merge):
            out_filters = filter_increase(n_filters, args.n_initial_filters)

            self.pre_convolutional_layers.append(
                BlockSeries(inplanes = n_filters, 
                    n_blocks   = args.res_blocks_per_layer,
                    residual   = True,
                    batch_norm = args.batch_norm)
                )
            self.pre_convolutional_layers.append(
                ConvolutionDownsample(
                    inplanes   = n_filters,
                    outplanes  = out_filters,
                    batch_norm = args.batch_norm)
                )

            n_filters = out_filters
            self.add_module("pre_merge_conv_{}".format(layer), 
                self.pre_convolutional_layers[-2])
            self.add_module("pre_merge_down_{}".format(layer), 
                self.pre_convolutional_layers[-1])

        n_filters *= args.nplanes
        
        self.post_convolutional_layers = []

        for layer in range(args.network_depth_post_merge):
            out_filters = filter_increase(n_filters, args.n_initial_filters)

            self.post_convolutional_layers.append(
                BlockSeries(
                    inplanes   = n_filters, 
                    n_blocks   = args.res_blocks_per_layer,
                    residual   = True,
                    batch_norm = args.batch_norm)
                )
            self.post_convolutional_layers.append(
                ConvolutionDownsample(
                    inplanes   = n_filters,
                    outplanes  = out_filters,
                    batch_norm = args.batch_norm)
                )
            n_filters = out_filters


            self.add_module("post_merge_conv_{}".format(layer), 
                self.post_convolutional_layers[-2])
            self.add_module("post_merge_down_{}".format(layer), 
                self.post_convolutional_layers[-1])

        # Now prepare the output operations.  In general, it's a Block Series,
        # then a 1x1 to get the right number of filters, then a global average pooling,
        # then a reshape

        # This is either once to get one set of labels, or several times to split the network
        # output to multiple labels


        self.final_layer = { 
                key : BlockSeries(
                    inplanes    = n_filters, 
                    n_blocks    = args.res_blocks_per_layer,
                    residual    = True,
                    batch_norm  = args.batch_norm)
                for key in output_shape
            }
        self.bottleneck  = { 
                key : torch.nn.Conv2d(
                    in_channels  = n_filters, 
                    out_channels = output_shape[key][-1], 
                    kernel_size  = [1,1], 
                    stride       = [1,1],
                    padding      = [0,0],
                    bias         = False)
                for key in output_shape
            }
    

        for key in self.final_layer:
            self.add_module("final_layer_{}".format(key), self.final_layer[key])
            self.add_module("bottleneck_{}".format(key), self.bottleneck[key])



        # Configure initialization:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        
        batch_size = x.shape[0]



        # Reshape this tensor into the right shape to apply this multiplane network.

        x = torch.chunk(x, chunks=self.nplanes, dim=1)


        # Apply the initial convolutions:
        x = [ self.initial_convolution(_x) for _x in x ]
        for i in range(len(self.pre_convolutional_layers)):
            x = [ self.pre_convolutional_layers[i](_x) for _x in x ]

        # Merge the paths together:
        x = torch.cat(x, dim=1)


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
                
                kernel_size = output[key].shape[2:]

                output[key] = torch.squeeze(nn.AvgPool2d(kernel_size, ceil_mode=False)(output[key]))

                # Squeeze off the last few dimensions:
                output[key] = output[key].view([batch_size, output[key].shape[-1]])

                # output[key] = scn.AveragePooling(dimension=3,
                #     pool_size=kernel_size, pool_stride=kernel_size)(output[key])

                # print (output[key].spatial_size)
                # print (output[key])
    
                # print (output[key].size())

                # output[key] = nn.Softmax(dim=1)(output[key])

        return output




