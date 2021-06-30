
import torch


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

class TNet(torch.nn.Module):
    def __init__(self, input_shape, output_shape):
        torch.nn.Module.__init__(self)

        # Initialize these to small but not zero
        self.trainable_weights = torch.nn.Parameter(
            (0.01/256)*torch.rand([256,output_shape**2]),
            requires_grad=True
        )

        self.trainable_biases  = torch.nn.Parameter(
            torch.eye(output_shape),
            requires_grad=True
        )

        self.identity = torch.eye(output_shape)

        self.output_shape = output_shape

        self.mlps = torch.nn.Sequential(
            MLP(input_shape, 64),
            MLP(64, 128),
            MLP(128, 1024),
        )

        self.fully_connected = torch.nn.Sequential(
            torch.nn.Linear(1024, 512), torch.nn.ReLU(),
            torch.nn.Linear(512, 256), torch.nn.ReLU()
            )

    def cuda(self, device=None):
        torch.nn.Module.cuda(self, device)
        self.identity = self.identity.cuda()

    def forward(self, data):


        x = self.mlps(data)

        shape = x.shape[2:]
        x = torch.nn.functional.max_pool1d(x, kernel_size=shape)
        x = torch.squeeze(x)

        x = self.fully_connected(x)

        matrix = torch.matmul(x, self.trainable_weights)
        matrix = torch.reshape(matrix, (-1, self.output_shape, self.output_shape))
        matrix = matrix + self.trainable_biases


        transpose = torch.transpose(matrix, 1,2)

        ortho_loss = torch.sum( (self.identity - torch.matmul(matrix, transpose) )**2 )

        return matrix, ortho_loss

class PointNet(torch.nn.Module):
    def __init__(self, output_shape, args):
        torch.nn.Module.__init__(self)

        # TNets and MLPs are _shared_ across planes - they should be learning the same features
        self.tnet0 = TNet(3, 3)

        self.mlp0 =  torch.nn.Sequential(MLP(3, 64), MLP(64, 64))

        self.tnet1 = TNet(64, 64)

        self.mlp1 =  torch.nn.Sequential(MLP(64, 128), MLP(128, 1024))


        self.final_mlp = torch.nn.ModuleDict(
                { key : torch.nn.Sequential(
                        MLP(3*1024, 512),
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


        tnets = [ self.tnet0(d) for d in data]

        rotations, losses1 = list(zip(*tnets))


        # Now, we have a rotation matrix for each plane.
        # Apply it to all points.
        data = [ torch.matmul(r, p) for r, p in zip(rotations, data)]


        # Now apply the MLP to 64 features:
        data = [ self.mlp0(d) for d in data]

        # Now, another TNet call:

        tnets = [self.tnet1(d) for d in data]

        rotations, losses2 = list(zip(*tnets))
        # Apply the new rotations:
        data = [ torch.matmul(r, p) for r, p in zip(rotations, data)]




        # The next MLPs:
        data = [ self.mlp1(d) for d in data]

        # Next, we maxpool over each plane:
        shape = data[0].shape[2:]
        data = [ torch.squeeze(torch.nn.functional.max_pool1d(d, kernel_size=shape)) for d in data ]


        # For the outputs, we concatenate across all 3 planes and have a unique series of MLPs for each
        # Category of output

        data = torch.cat(data, axis=-1)
        data = torch.reshape(data, data.shape + (1,))

        outputs = { key : torch.squeeze(self.final_mlp[key](data)) for key in self.final_mlp.keys() }

        return outputs
