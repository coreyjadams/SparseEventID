import numpy
import torch


from src.config.framework import DataMode

class multi_head_output(torch.nn.Module):

    def __init__(self, spatial_shape, n_in, output_shape):
        super().__init__()
        self.classification_head = torch.nn.ModuleDict({
                key : create_final_dense_chain(spatial_shape, n_in, output_shape[key])
                for key in output_shape.keys()
            })
    def forward(self, x):
        
        return {key : self.classification_head[key](x) for key in self.classification_head.keys() }

def create_final_dense_chain(spatial_shape, n_in, n_out):

    return torch.nn.Sequential(
        torch.nn.AvgPool3d(spatial_shape),
        torch.nn.Flatten(start_dim=1, end_dim=-1),
        torch.nn.Linear(in_features=n_in, out_features=256),
        torch.nn.Dropout(),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(in_features=256,out_features=n_out)
    )

def build_networks(params, input_shape, output_shape):

    if params.framework.mode != DataMode.graph:
        from . resnet import Encoder
        encoder = Encoder(params, input_shape)
    else:
        from . mpnn import Encoder
        encoder = Encoder(params, input_shape)


    # encoder, output_shape = create_resnet(params, input_shape)

    current_number_of_filters = encoder.output_shape[0]
    spatial_shape = encoder.output_shape[1:]
    print(spatial_shape)
    print(current_number_of_filters)
    # First step of the classification head is to pool the spatial size:


    if isinstance(output_shape, dict):

        classification_head = multi_head_output(spatial_shape, current_number_of_filters, output_shape)
    else:
        classification_head = create_final_dense_chain(spatial_shape, current_number_of_filters, output_shape)

    return encoder, classification_head
