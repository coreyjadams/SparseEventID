import os
import time

import torch
import numpy
import sparseconvnet as scn

import pandas


def build_data(shape, sparsity, n_filters = 32, sparse=False, cuda=True):

    # First, find out how big the tensor is and how sparse:
    total_size = numpy.prod(shape)

    n_non_zero = int(sparsity * total_size)

    indexes = [ numpy.random.choice(range(s), size=n_non_zero) for s in shape ]

    values  = numpy.random.random(size=n_non_zero*n_filters)
    values  = numpy.reshape(values, [-1, n_filters])


    if sparse:
        indexes.append(numpy.zeros(shape=(len(values))))
        indexes = numpy.stack(indexes, axis=-1)
        indexes = torch.LongTensor(indexes)
        values = torch.FloatTensor(values).view([-1,n_filters])
        if cuda:
            values = values.cuda()
        t = (indexes, values)
        d = len(shape)
        spatial_size = torch.LongTensor(shape)
        data = scn.InputLayer(dimension=d, spatial_size=spatial_size)(t)
    else:
        data = numpy.random.random(size=[n_filters,] + shape)
        # print(data.shape)
        # print(values.shape)
        # if len(shape)  == 2:
        #     data[:,indexes[0], indexes[1]] = values
        data = torch.FloatTensor(data).view([-1, n_filters,] + shape)
        if cuda:
            data = data.cuda()
    return data


def conv_op2d(kernel, n_filters, sparse=False):

    if sparse:
        m = torch.nn.ModuleList()
        m.append(scn.SubmanifoldConvolution(dimension=2,
            nIn=n_filters,
            nOut=n_filters,
            filter_size=kernel,
            bias=True))

        m.append(scn.BatchNormReLU(n_filters))
        return m
    else:

        m = torch.nn.ModuleList()
        m.append(torch.nn.Conv2d(
            in_channels  = n_filters,
            out_channels = n_filters,
            kernel_size  = kernel,
            bias         = True))

        m.append(torch.nn.BatchNorm2d(n_filters))

        m.append(torch.nn.ReLU())

        return m

def conv_op3d(kernel, n_filters, sparse=False):

    if sparse:
        m = torch.nn.ModuleList()
        m.append(scn.SubmanifoldConvolution(dimension=3,
            nIn=n_filters,
            nOut=n_filters,
            filter_size=kernel,
            bias=True))

        m.append(scn.BatchNormReLU(n_filters))
        return m
    else:

        m = torch.nn.ModuleList()
        m.append(torch.nn.Conv3d(
            in_channels  = n_filters,
            out_channels = n_filters,
            kernel_size  = kernel,
            bias         = True))

        m.append(torch.nn.BatchNorm3d(n_filters))

        m.append(torch.nn.ReLU())

        return m


def time_kernel(shape, kernel, n_filters, sparse, cuda, sparsity):

    with torch.no_grad():
        data = build_data(shape, sparsity, n_filters, sparse, cuda)

        if len(kernel) == 2:
            op = conv_op2d(kernel, n_filters, sparse)
        else:
            op = conv_op3d(kernel, n_filters, sparse)


        if cuda:
            op = op.cuda()

        for o in op:
            data = o(data)

        start = time.time()
        for o in op:
            data = o(data)

        end = time.time()

        return end - start

def main():

    sparse = True
    cuda   = True
    kernel = [3,3]
    shape  = [512,512]
    sparsity = 0.1

    df = pandas.DataFrame(columns=['dimension', 'cuda', 'kernel', 'shape', 'sparsity', 'sparse', 'time'])

    i = 0

    n_filters = 3

    for dimension in [2, 3]:
        for sparse in [True, False]:
            for kernel in [1,3,5]:
                for shape in [512]:
                    for sparsity in numpy.logspace(-4, -.5, num=10):
                        full_shape = [shape,] * dimension
                        full_kern  = [kernel,] * dimension
                        print(full_shape)
                        print(full_kern)
                        time = time_kernel(full_shape, full_kern, n_filters, sparse, cuda, sparsity)
                        df.loc[i] = [dimension, cuda, kernel, shape, sparsity, sparse, time]
                        i+= 1

    print(df.info())

    df.to_csv(f"scn_results_{os.uname()[1]}.h5")


if __name__ == '__main__':
    main()
