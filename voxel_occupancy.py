from larcv import larcv
import numpy




def count_2d(file_name, product, producer):
    io = larcv.IOManager()
    io.add_in_file(file_name)
    io.initialize()
    voxel_counts = numpy.zeros((io.get_n_entries(), 3))

    for i in range(io.get_n_entries()):
        io.read_entry(i)
        image = larcv.EventSparseTensor2D.to_sparse_tensor(io.get_data("sparse2d", "sbndvoxels"))
        for plane in [0,1,2]:
            voxel_counts[i][plane] = image.as_vector()[plane].size()
            meta = image.as_vector()[plane].meta()
        # image3d = io.get_data("sparse3d", "sbndvoxels")
        # voxel_counts3d[i] = image3d.as_vector().size()

        if i % 100 == 0:
            print("On entry ", i, " of ", io.get_n_entries())

        # if i > 100:
        #     break

    print ("Average Voxel Occupation: ")
    for p in [0,1,2]:
        print("  {p}: {av:.2f} +/- {rms:.2f} ({max} max)".format(
            p   = p, 
            av  = numpy.mean(voxel_counts[:,p]), 
            rms = numpy.std(voxel_counts[:,p]), 
            max = numpy.max(voxel_counts[:,p])
            )
        )


def count_3d(file_name, product, producer):
    io = larcv.IOManager()
    io.add_in_file(file_name)
    io.initialize()
    voxel_counts3d = numpy.zeros((io.get_n_entries(), 1))
    for i in range(io.get_n_entries()):
        io.read_entry(i)
        image3d = larcv.EventSparseTensor3D.to_sparse_tensor(io.get_data("sparse3d", "sbndvoxels"))
        voxel_counts3d[i] = image3d.as_vector()[0].size()

        if i % 100 == 0:
            print("On entry ", i, " of ", io.get_n_entries())

        # if i > 100:
        #     break
    print(" 3D: {av:.2f} +/- {rms:.2f} ({max} max)".format(
        av  = numpy.mean(voxel_counts3d[:]), 
        rms = numpy.std(voxel_counts3d[:]), 
        max = numpy.max(voxel_counts3d[:])
        )
    )



if __name__ == '__main__':
    # count_2d("data_files/out_2d_train_rand.h5", "sparse2d", "sbndvoxels")
    count_3d("data_files/out_3d_train_rand.h5", "sparse3d", "sbndvoxels")


