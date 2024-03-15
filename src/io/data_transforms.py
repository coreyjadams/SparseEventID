import numpy
import torch_geometric
'''
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


def larcvsparse_to_scnsparse_3d(input_array):
    
    n_dims = input_array.shape[-1]

    split_tensors = numpy.split(input_array, n_dims, axis=-1)


    # To map out the non_zero locations now is easy:
    non_zero_inds = numpy.where(split_tensors[-1] != -999)
    # The batch dimension is just the first piece of the non-zero indexes:
    batch_size  = input_array.shape[0]
    batch_index = non_zero_inds[0]

    # Getting the voxel values (features) is also straightforward:
    features = numpy.expand_dims(split_tensors[-1][non_zero_inds],axis=-1)

    # Lastly, we need to stack up the coordinates, which we do here:
    dimension_list = []
    for i in range(len(split_tensors) - 1):
        dimension_list.append(split_tensors[i][non_zero_inds])

    # Tack on the batch index to this list for stacking:
    dimension_list.append(batch_index)

    # And stack this into one numpy array:
    dimension = numpy.stack(dimension_list, axis=-1)

    output_array = (dimension, features, batch_size,)
    return output_array



def larcvsparse_to_dense_3d(input_array, dense_shape):

    batch_size = input_array.shape[0]
    output_array = numpy.zeros((batch_size, 1, *(dense_shape)) , dtype=numpy.float32)
    # This is the "real" size:
    # output_array = numpy.zeros((batch_size, 1, 45, 45, 275), dtype=numpy.float32)
    x_coords   = input_array[:,0,:,0]
    y_coords   = input_array[:,0,:,1]
    z_coords   = input_array[:,0,:,2]
    val_coords = input_array[:,0,:,3]


    # print(x_coords[0:100])
    # print(y_coords[0:100])
    # print(z_coords[0:100])

    # Find the non_zero indexes of the input:
    batch_index, voxel_index = numpy.where(val_coords != -999)

    values  = val_coords[batch_index, voxel_index]
    x_index = numpy.int32(x_coords[batch_index, voxel_index])
    y_index = numpy.int32(y_coords[batch_index, voxel_index])
    z_index = numpy.int32(z_coords[batch_index, voxel_index])


    # Fill in the output tensor

    output_array[batch_index, 0, x_index, y_index, z_index] = values
    return output_array


import torch

def larcvsparse_to_pytorch_geometric(input_array, image_meta):

    # Need to create node features and an adjacency matrix.
    # Define points as connected if they fall within some radius R
    # For each node, it's node features can be (x, y, z, E) as well as
    # The number of nearby nodes (within R), the average energy of those
    # nodes, and average distance away from its neighbors.
    # For each edge, dunno

    # Ultimately, need to build this into a graph for pytorch_geometric
    
    batch_size = input_array.shape[0]

    # # output_array = numpy.zeros((batch_size, 1, 45, 45, 275), dtype=numpy.float32)
    # x_coords   = input_array[:,0,:,0]
    # y_coords   = input_array[:,0,:,1]
    # z_coords   = input_array[:,0,:,2]
    # val_coords = input_array[:,0,:,3]

    # print("val_coords.shape", val_coords.shape)
    # # Find the non_zero indexes of the input:
    # batch_index, voxel_index = numpy.where(val_coords != -999)

    # values  = val_coords[batch_index, voxel_index]
    # x_index = numpy.int32(x_coords[batch_index, voxel_index])
    # y_index = numpy.int32(y_coords[batch_index, voxel_index])
    # z_index = numpy.int32(z_coords[batch_index, voxel_index])

    # print(batch_index)
    # print(voxel_index)

    # print(x_index)

    voxel_size = (image_meta['size'] / image_meta['n_voxels']).astype(numpy.float32)
    origin     = image_meta['origin']
    graph_data = []

    # Doing this an inefficent way but at least it is working!
    for i in range(batch_size):
        # What index of the input data is this batch?
        this_batch_data = input_array[i,0]
        batch_values = this_batch_data[:,3]

        # Which of this data are valid?
        active_index = numpy.where(batch_values != -999)[0]
        
        # Select the data:
        active_data = this_batch_data[active_index, :]


        xyz = active_data[:,0:3] * voxel_size + image_meta['origin']
        # r = numpy.sqrt(numpy.sum(xyz**2, axis=-1))
        # What is the displacement between each active site?
        edge_displacements = xyz.reshape(-1,1,3) - xyz.reshape(1,-1,3)

        # Compute that as a magnitude:
        r = numpy.sqrt(numpy.sum(edge_displacements**2, axis=-1))
        row, col = numpy.where(r < 50)
        edge_index = numpy.stack([row,col])

        edge_attr = numpy.concatenate([
            r[row,col].reshape(-1,1), 
            edge_displacements[row, col,:]
        ], axis=-1)

        graph_data.append(
            torch_geometric.data.Data(
                x          = torch.tensor(active_data).reshape((-1,4)),
                # x          = torch.tensor(active_data[:,3]).reshape((-1,1)),
                edge_index = torch.tensor(edge_index),
                edge_attr  = torch.tensor(edge_attr),
                pos        = torch.tensor(xyz),
            )
        )

    batch = torch_geometric.data.Batch.from_data_list(graph_data)

    return batch

    


def larcvsparse_to_dense_2d(input_array, dense_shape):

    batch_size = input_array.shape[0]
    n_planes   = input_array.shape[1]

    output_array = numpy.zeros((batch_size, n_planes, dense_shape[0], dense_shape[1]), dtype=numpy.float32)


    x_coords = input_array[:,:,:,0]
    y_coords = input_array[:,:,:,1]
    val_coords = input_array[:,:,:,2]


    filled_locs = val_coords != -999
    non_zero_locs = val_coords != 0.0
    mask = numpy.logical_and(filled_locs,non_zero_locs)
    # Find the non_zero indexes of the input:
    batch_index, plane_index, voxel_index = numpy.where(filled_locs)


    values  = val_coords[batch_index, plane_index, voxel_index]
    x_index = numpy.int32(x_coords[batch_index, plane_index, voxel_index])
    y_index = numpy.int32(y_coords[batch_index, plane_index, voxel_index])

    # Fill in the output tensor
    output_array[batch_index, plane_index, y_index, x_index] = values

    return output_array


def larcvsparse_to_scnsparse_2d(input_array):
    # This format converts the larcv sparse format to
    # the tuple format required for sparseconvnet

    # First, we can split off the features (which is the pixel value)
    # and the indexes (which is everything else)

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

        # print("X: ",numpy.max(x))
        # print("Y: ", numpy.max(y))

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
        dimension = numpy.stack([p,y,x,batch], axis=-1)

        output_features.append(features)
        output_dimension.append(dimension)

    output_features = numpy.concatenate(output_features)
    output_dimension = numpy.concatenate(output_dimension)


    output_list = [output_dimension, output_features, batch_size]

    return output_list