from ROOT import larcv
import numpy

io = larcv.IOManager()
io.add_in_file("/lus/theta-fs0/projects/datascience/cadams/wire_pixel_fullres_files/sbnd_dl_eventID_val.root")
io.initialize()

voxel_counts = numpy.zeros((io.get_n_entries(), 3))
voxel_counts3d = numpy.zeros((io.get_n_entries(), 1))

for i in range(io.get_n_entries()):
    io.read_entry(i)
    image = io.get_data("sparse2d", "sbndwire")
    for plane in [0,1,2]:
        voxel_counts[i][plane] = image.as_vector()[plane].size()
        meta = image.as_vector()[plane].meta()
    image3d = io.get_data("sparse3d", "sbndvoxels")
    voxel_counts3d[i] = image3d.as_vector().size()

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
print("  {p}: {av:.2f} +/- {rms:.2f} ({max} max)".format(
    p   = p, 
    av  = numpy.mean(voxel_counts3d[:]), 
    rms = numpy.std(voxel_counts3d[:]), 
    max = numpy.max(voxel_counts3d[:])
    )
)
numpy.save("2d_counts.npz",voxel_counts)
numpy.save("3d_counts.npz",voxel_counts3d)