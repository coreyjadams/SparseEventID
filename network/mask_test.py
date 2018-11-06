import numpy


# Numpy mode: take a matrix of zeros in 2d or 3d and create the list of points

shape = [100, 100, 100, 100,]
n_dims = len(shape)

raw_image = numpy.zeros(shape)

n_filled = numpy.prod(shape) * (0.1 ** n_dims)

for i in range(int(n_filled)):
    loc = []
    for d in range(n_dims):
        loc.append([numpy.random.randint(0,shape[d])])
    raw_image[loc] = numpy.random.random()

non_zero_locs = list(numpy.where(raw_image != 0))


# Merge everything into an N by 2 array:
values = raw_image[non_zero_locs]
non_zero_locs.append(values)
res = numpy.stack((non_zero_locs)).T
# A list of points must be an Nxk tensor, where k = (x, y, z, q  .... )

print(res.shape)
print(res[0])
print(res[-1])