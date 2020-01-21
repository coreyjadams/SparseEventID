import numpy

'''
This is a torch-free file that exists to massage data
From sparse to dense or dense to sparse, etc.

This can also convert from sparse to sparse to rearrange formats
For example, larcv BatchFillerSparseTensor2D (and 3D) output data
with the format of
    [B, N_planes, Max_voxels, N_features]

where N_features is 2 or 3 depending on whether or not values are included
(or 3 or 4 in the 3D case)

# The input of a pointnet type format can work with this, but SparseConvNet
# requires a tuple of (coords, features, [batch_size, optional])


''' 
import torch

from torch_geometric import data


def larcvsparse_to_scnsparse_3d(input_array):
    # This format converts the larcv sparse format to
    # the tuple format required for sparseconvnet

    # First, we can split off the features (which is the pixel value)
    # and the indexes (which is everythin else)

    n_dims = input_array.shape[-1]

    print(n_dims)

    split_tensors = numpy.split(input_array, n_dims, axis=-1)

    # The values row is always last:
    values = split_tensors[-1]

    # To map out the non_zero locations now is easy:
    non_zero_inds = numpy.where(values != -999)


    # The batch dimension is just the first piece of the non-zero indexes:
    batch_size  = input_array.shape[0]
    batch_index = non_zero_inds[0]

    print(len(batch_index))
    print(batch_index)

    # Getting the voxel values (features) is also straightforward:
    features = numpy.expand_dims(values[non_zero_inds],axis=-1)

    print(features.shape)

    # Lastly, we need to stack up the coordinates, which we do here:
    dimension_list = []
    for i in range(len(split_tensors) - 1):
        dimension_list.append(split_tensors[i][non_zero_inds])

        print(dimension_list[-1])


    # Tack on the batch index to this list for stacking:
    dimension_list.append(batch_index)


    # And stack this into one numpy array:
    dimension = numpy.stack(dimension_list, axis=-1)

    output_array = (dimension, features, batch_size,)
    return output_array


def larcvsparse_to_torchgeometric(input_array):

    # First, infer the dimension from the input array:

    dim = None

    if len(input_array.shape) == 3:
        # This is 3D
        dim = 3
    elif len(input_array.shape) == 4:
        # This is 2D
        dim = 2

    # The number of batches will be the length of the input's first dim:
    n_batches = input_array.shape[0]

    # How many dimensions?
    n_dims = input_array.shape[-1]




    graph_list = []

    # Loop over the batches to create a data object:
    for batch in range(n_batches):


        # Split the tensor along the final axis of x/z/(z)/(val):
        split_tensors = numpy.split(input_array[batch], n_dims, axis=-1)

        # In 2D, each tensor is now of shape [nplanes, n_pixels, 1] 
        # for the x, y, value across each plane.  3 tensors total.
        # We let the plane index function as a positional variable
        # for the graph networks.

        # In 3D, each tensor is now of the shape [n_voxels, 1]
        # for x, y, z, value.


        # Get all the values 
        # In both dims, the batch is now the first dimension, 
        # while the values are now the last.

        values       = split_tensors[-1]

        # Find the mask for non-zero values:
        non_zero_inds = numpy.where(values != -999)

        # Take the relevant values and reshape to [n_points, n_features=1]
        values = numpy.expand_dims(values[non_zero_inds], axis=-1)

        positional_vars = []

        if dim == 3:

            # X:
            positional_vars.append(split_tensors[0][non_zero_inds])
            # Y:
            positional_vars.append(split_tensors[1][non_zero_inds])
            # Z:
            positional_vars.append(split_tensors[2][non_zero_inds])
            
            positions = numpy.stack(positional_vars, axis=-1)
            # print(positions.shape)

            graph_list.append(data.Data(
                x   = torch.tensor(values), 
                pos = torch.tensor(positions)
                ))

        else:

            # The dimension is 2, so we have n_planes with x/y/val

            # We will use the plane as "just another location" in 2D

            plane = non_zero_inds[0]
            x    = split_tensors[0][non_zero_inds]
            y    = split_tensors[1][non_zero_inds]
            vals = split_tensors[2][non_zero_inds]

            positional_vars.append(x)
            positional_vars.append(y)
            positional_vars.append(plane)


            positions = numpy.stack(positional_vars, axis=-1)
            # print(positions.shape)

            graph_list.append(data.Data(
                x   = torch.tensor(values), 
                pos = torch.tensor(positions)
                ))

    # Finally, turn all the graphs into a batch:
    return data.Batch.from_data_list(graph_list)

        # X is the index 

def larcvsparse_to_scnsparse_2d(input_array):
    # This format converts the larcv sparse format to
    # the tuple format required for sparseconvnet

    # First, we can split off the features (which is the pixel value)
    # and the indexes (which is everythin else)

    # To handle the multiplane networks, we have to split this into
    # n_planes and pass it out as a list


    n_planes = input_array.shape[1]
    batch_size = input_array.shape[0]


    raw_planes = numpy.split(input_array,n_planes, axis=1)

    output_list = []
    output_features = []
    output_dimension = []

    for i, plane in enumerate(raw_planes):
        # First, squeeze off the plane dimension from this image:
        plane = numpy.squeeze(plane, axis=1)

        # Next, figure out the x, y, value coordinates:
        x,y,features = numpy.split(plane, 3, axis=-1)


        non_zero_locs = numpy.where(features != -999)
        # Pull together the different dimensions:
        x = x[non_zero_locs]
        y = y[non_zero_locs]
        p = numpy.full(x.shape, fill_value=i)
        features = features[non_zero_locs]
        features = numpy.expand_dims(features,axis=-1)

        batch = non_zero_locs[0]

        # dimension = numpy.concatenate([x,y,batch], axis=0)
        # dimension = numpy.stack([x,y,batch], axis=-1)
        dimension = numpy.stack([p,x,y,batch], axis=-1)

        output_features.append(features)
        output_dimension.append(dimension)

    output_features = numpy.concatenate(output_features)
    output_dimension = numpy.concatenate(output_dimension)

    output_list = [output_dimension, output_features, batch_size]

    return output_list

def larcvsparse_to_dense_2d(input_array, dense_shape=[1536, 1024]):


    batch_size = input_array.shape[0]
    n_planes   = input_array.shape[1]
    output_array = numpy.zeros((batch_size, n_planes, dense_shape[0], dense_shape[1]), dtype=numpy.float32)

    x_coords = input_array[:,:,:,0]
    y_coords = input_array[:,:,:,1]
    val_coords = input_array[:,:,:,2]


    # Find the non_zero indexes of the input:
    batch_index, plane_index, voxel_index = numpy.where(val_coords != -999)

    values  = val_coords[batch_index, plane_index, voxel_index]
    x_index = numpy.int32(x_coords[batch_index, plane_index, voxel_index])
    y_index = numpy.int32(y_coords[batch_index, plane_index, voxel_index])


    # Fill in the output tensor

    output_array[batch_index, plane_index, x_index, y_index] = values

    return output_array

