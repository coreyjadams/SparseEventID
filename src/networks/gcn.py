# import os.path as osp
# import argparse

# import torch
# import torch_geometric.transforms as T
# from torch_geometric.nn import GCNConv, ChebConv  # noqa


# from . network_config import network_config, str2bool

# #
# # The 3D convolution method for 2D images is very, very slow.
# # Much faster to use 2D convolutions 3 times

# class GCNFlags(network_config):

#     def __init__(self):
#         network_config.__init__(self)
#         self._name = "gcn"
#         self._help = "GCN with extra linear layers for multi-key classification"

#     def build_parser(self, network_parser):
#         # this_parser = network_parser
#         this_parser = network_parser.add_parser(self._name, help=self._help)

#         # this_parser.add_argument("--n-initial-filters",
#         #     type    = int,
#         #     default = 2,
#         #     help    = "Number of filters applied, per plane, for the initial convolution")






# class GCNNet(torch.nn.Module):
#     def __init__(self, output_shape, args):
#         torch.nn.Module.__init__(self)
#         self.conv1 = GCNConv(dataset.num_features, 16, cached=True,
#                              normalize=not args.use_gdc)
#         self.conv2 = GCNConv(16, dataset.num_classes, cached=True,
#                              normalize=not args.use_gdc)
#         # self.conv1 = ChebConv(data.num_features, 16, K=2)
#         # self.conv2 = ChebConv(16, data.num_features, K=2)

#         self.reg_params = self.conv1.parameters()
#         self.non_reg_params = self.conv2.parameters()

#     def forward(self):
#         x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
#         x = torch.nn.functional.relu(self.conv1(x, edge_index, edge_weight))
#         x = torch.nn.functional.dropout(x, training=self.training)
#         x = self.conv2(x, edge_index, edge_weight)
#         return torch.nn.functional.log_softmax(x, dim=1)

