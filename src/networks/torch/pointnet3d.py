
import torch

from . pointnet import MLP, TNet


class PointNet(torch.nn.Module):
    def __init__(self, output_shape, args):
        torch.nn.Module.__init__(self)

        self.tnet0 = TNet(4, 4)

        self.mlp0 = torch.nn.Sequential(MLP(4, 64), MLP(64, 64))

        self.tnet1 = TNet(64, 64)

        self.mlp1 = torch.nn.Sequential(MLP(64, 128), MLP(128, 1024))


        self.final_mlp = torch.nn.ModuleDict(
                { key : torch.nn.Sequential(
                        MLP(1024, 512),
                        MLP(512, 256),
                        MLP(256, output_shape[key][-1])
                        )
                    for key in output_shape.keys()
                }
            )


    def cuda(self, device=None):
        torch.nn.Module.cuda(self, device)
        self.tnet0.cuda()
        self.tnet1.cuda()

    def forward(self, data):
        #print("entered")

        # in 2D, shape is [batch_size, max_points, x/y/val] x 3 planes

        # in 3D, shape is [batch_size, max_points, x/y/z/val]

        rotation, loss1 = self.tnet0(data)


        # Apply it to all points.
        data = torch.matmul(rotation, data)


        # Now apply the MLP to 64 features:
        data = self.mlp0(data)

        # Now, another TNet call:

        rotation, loss2 = self.tnet1(data)

        # Apply the new rotations:
        data = torch.matmul(rotation, data)

        # The next MLPs:
        data = self.mlp1(data)


        # Next, we maxpool over each plane:
        shape = data.shape[2:]
        data = torch.squeeze(torch.nn.functional.max_pool1d(data, kernel_size=shape))


        # For the outputs, we concatenate across all 3 planes and have a unique series of MLPs for each
        # Category of output

        data = torch.reshape(data, data.shape + (1,))

        outputs = { key : torch.squeeze(self.final_mlp[key](data)) for key in self.final_mlp.keys() }

        return outputs
