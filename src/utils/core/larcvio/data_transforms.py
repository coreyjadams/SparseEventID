import numpy
import torch

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


def larcvsparse_to_scnsparse_3d(input_array):
    # This format converts the larcv sparse format to
    # the tuple format required for sparseconvnet

    # First, we can split off the features (which is the pixel value)
    # and the indexes (which is everythin else)

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
    output_array = numpy.zeros((batch_size,1) + dense_shape, dtype=numpy.float32)

    # By default, this returns channels_first format with just one channel.
    # You can just reshape since it's an empty dimension, effectively

    x_coords   = input_array[:,:,0]
    y_coords   = input_array[:,:,1]
    z_coords   = input_array[:,:,2]
    val_coords = input_array[:,:,3]


    # Find the non_zero indexes of the input:
    batch_index, voxel_index = numpy.where(val_coords != -999)

    values  = val_coords[batch_index, voxel_index]
    x_index = numpy.int32(x_coords[batch_index, voxel_index])
    y_index = numpy.int32(y_coords[batch_index, voxel_index])
    z_index = numpy.int32(z_coords[batch_index, voxel_index])


    # Fill in the output tensor

    output_array[batch_index, 0, x_index, y_index, z_index] = values

    return output_array

# class GraphData(object):

#     def __init__(self, locations, values):

#         self.locs = point_locations
#         self.vals = values

#         assert len(self.locs) == len(self.vals)

#     def __len__(self):
#         return self.values.shape[0]

#     def dim(self):
#         return len(self.locs[0].shape) - 1

# class GraphBatch(object):

#     def __init__(self, graphs)
#         self.graphs = []

def larcvsparse_to_pointcloud_2d(input_array):

    # This function iterates over each batch to create a pointcloud from each batch sample
    # Each point cloud will generate a torch-geometric Data object using the Data() class
    # The point clouds are appended to a data list which can be input to a Batch.from_data_list object

    # At the end, we have a list of arrays of shape 
    # [batch_size, val, npoints]
    # This is treating x/y/val as features and npoints as the "image"


    nplanes    = 3
    batch_size = input_array.shape[0]
    npoints    = input_array.shape[2]


    # output = [ numpy.zeros(shape=[batch_size, 1, npoints, 3]) for plane in nplanes]

    # Loop over minibatch index:
    
    output = numpy.split(input_array,3, axis=1)


    for o in output:
        # Turn al the -999s to 0.0 for graphs:
        coords = numpy.where(o[:,0,:,-1] == -999)
        o[coords[0],0, coords[1],0] = 0
        o[coords[0],0, coords[1],1] = 0
        o[coords[0],0, coords[1],2] = 0

        # Next We need 

    # for i in range(input_array.shape[0]):
    #     x_coords   = input_array[i,0,:,0]
    #     y_coords   = input_array[i,0,:,1]
    #     # z_coords   = input_array[i,0,:,2]
    #     val_coords = input_array[i,0,:,3]

    #     # Turn all empty pixels to 0 in graph mode:
    #     voxel_index = numpy.where(val_coords != -999)
    #     values = val_coords
    #     values[voxel_index] = 0.0


    output = [ numpy.transpose(numpy.squeeze(o), axes=(0, 2, 1)) for o in output]


    #     zeroes_help = torch.zeros(values.shape[0],1)
    #     values_temp = torch.Tensor(numpy.transpose(numpy.array(values)))
    #     # print(values_temp.shape)
    #     # print(zeroes_help.shape)
    #     zeroes_help[:,0] = values_temp
    #     point_temp = torch.Tensor(numpy.transpose(numpy.array([x_values,y_values,z_values])))


    #     # PointNet takes the features from the X array and the nodes positions from the pos argument
    #     #  Make sure that both x and pos are torch tensors
    #     data_temp_b = Data(x=zeroes_help,pos=point_temp)

    #     # When we have isolated nodes it is necessary to set the number of nodes manually
    #     data_temp_b.num_nodes = point_temp.shape[0]
    #     data_pre_batch.append(data_temp_b)

    # # print(len(data_pre_batch))
    return output
