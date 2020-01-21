import torch


class LinearBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, batch_norm = True):
        torch.nn.Module.__init__(self)



        self.linear = torch.nn.Linear(in_channels, out_channels)
        self.relu   = torch.nn.ReLU()

        if batch_norm:
            self.batch_norm = torch.nn.BatchNorm1d(out_channels)

    def forward(self, x):

        x = self.linear(x)
        x = self.relu(x)
        if hasattr(self, 'batch_norm'):
            return self.batch_norm(x)
        else:
            return x        

class MLP(torch.nn.Module):

    def __init__(self, channels_list, batch_norm=True):
        torch.nn.Module.__init__(self)

        self.linear_layers = torch.nn.ModuleList()
        for i in range(1, len(channels_list)):
            self.linear_layers.append(LinearBlock(channels_list[i-1], channels_list[i], batch_norm))

    def forward(self,x):

        for layer in self.linear_layers:
            x = layer(x)

        return x