import os.path as osp

import torch

from . graph_net_utils import MLP
from . network_config  import network_config, str2bool


class DGCNNFlags(network_config):

    def __init__(self):
        network_config.__init__(self)
        self._name = "dgcnn"
        self._help = "Dynamic Graph CNN with extra linear layers for multi-key classification"

    def build_parser(self, network_parser):
        # this_parser = network_parser
        this_parser = network_parser.add_parser(self._name, help=self._help)

        this_parser.add_argument("-k",
            type    = int,
            default = 20,
            help    = "Number of points to use in neighborhood")


        this_parser.add_argument("--aggregation",
            type    = str,
            default = 'max',
            choices = ['add', 'mean', 'max'],
            help    = "Function for aggregation")




class DGCNN(torch.nn.Module):
    def __init__(self, output_shape, args):
        torch.nn.Module.__init__(self)
        # We can modify the first three for a 4 if we want coordinates + pixel intensity
        self.conv1 = torch_geometric.nn.DynamicEdgeConv(MLP([2 * 4, 64, 64, 64]), args.k, args.aggregation)
        self.conv2 = torch_geometric.nn.DynamicEdgeConv(MLP([2 * 64, 128]), args.k, args.aggregation)
        self.lin1 = MLP([128 + 64, 1024])


        self.mlp = torch.nn.Sequential(
            MLP([1024, 512]), torch.nn.Dropout(0.5), MLP([512, 256]), torch.nn.Dropout(0.5))

        self.lin  = { key : torch.nn.Linear(256, output_shape[key][1]) for key in output_shape }


        for key in self.lin:
            self.add_module("lin_{}".format(key), self.lin[key])


    def forward(self, data):
        pos, batch = data.pos, data.batch
        x1 = self.conv1(pos, batch)
        x2 = self.conv2(x1, batch)
        out = self.lin1(torch.cat([x1, x2], dim=1))
        out = torch_geometric.nn.global_max_pool(out, batch)
        out = self.mlp(out)

        output = { key : self.lin[key](out) for key in self.lin }

        return output
