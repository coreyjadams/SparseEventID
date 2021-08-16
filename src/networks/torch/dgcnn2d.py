
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Modified from original version by Yue Wang (yuewangx@mit.edu)
"""


import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(torch.nn.Module):
    def __init__(self, input_size, output_size):
        torch.nn.Module.__init__(self)

        self.mlp = torch.nn.Conv1d(
            in_channels  = input_size,
            out_channels = output_size,
            kernel_size  = (1))

        self.bn = torch.nn.BatchNorm1d(num_features=output_size)

        self.relu = torch.nn.ReLU()

    def forward(self, data):
        return self.relu(self.bn(self.mlp(data)))

def knn(x, k):

    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def knn_cdist(x, k):

    pairwise_cdist = torch.cdist(x.transpose(2, 1), x.transpose(2, 1))

    idx = pairwise_cdist.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        # idx1 = knn(x, k=k)
        idx = knn_cdist(x, k=k)   # (batch_size, num_points, k)
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)

    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)

    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature



class DGCNN(nn.Module):
    def __init__(self, output_shape, args):
        super(DGCNN, self).__init__()
        self.args = args
        self.k = args.network.k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.network.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.network.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.network.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.network.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.network.dropout)

        # Modification:
        # self.linear3 = nn.Linear(256, output_channels)

        self.linear3 = torch.nn.ModuleDict(
                { key : MLP(256*3, output_shape[key][-1])
                   for key in output_shape.keys()
                }
            )

        # self.final_mlp = torch.nn.ModuleDict(
        #        { key : torch.nn.Sequential(
        #                MLP(3*1024, 512),
        #                MLP(512, 256),
        #                MLP(256, output_shape[key][-1])
        #                )
        #            for key in output_shape.keys()
        #        }
        #    )


    def forward(self, data):
        # TODO: adapt this to the data dimmensions
        # a thought: wrap in for loop and add each x to a data list, make data a tensor, and then at the end do
        # outputs = { key : torch.squeeze(self.linear3[key](data)) for key in self.final_mlp.keys() }
        # 2D shape is [batch_size, max_points, x/y/val] x 3 planes

        # go through each plane in the data (there are three of them)

        for i in range(len(data)):
                # get current data plane
                x = data[i]
                batch_size = x.size(0)
                x = get_graph_feature(x, k=self.k)
                x = self.conv1(x)
                x1 = x.max(dim=-1, keepdim=False)[0]

                x = get_graph_feature(x1, k=self.k)
                x = self.conv2(x)
                x2 = x.max(dim=-1, keepdim=False)[0]

                x = get_graph_feature(x2, k=self.k)
                x = self.conv3(x)
                x3 = x.max(dim=-1, keepdim=False)[0]

                x = get_graph_feature(x3, k=self.k)
                x = self.conv4(x)
                x4 = x.max(dim=-1, keepdim=False)[0]

                x = torch.cat((x1, x2, x3, x4), dim=1)

                x = self.conv5(x)
                x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
                x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
                x = torch.cat((x1, x2), 1)

                x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
                x = self.dp1(x)
                x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
                x = self.dp2(x)
                # set current data plane to be x after going through the network
                data[i] = x

        data = torch.cat(data, axis=-1)
        data = torch.reshape(data, data.shape + (1,))
        outputs = { key: torch.squeeze(self.linear3[key](data)) for key in self.linear3.keys() }
        return outputs
