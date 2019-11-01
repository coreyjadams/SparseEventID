import h5py, numpy

f = h5py.File('/gpfs/alpine/scratch/deltutto/nph133/dune_wire_pixel/merged_DUNE_DL_pixsim_150k_samples_20191029/merged_sample_all_train.h5')

labels = ['particle_neutID_group', 'particle_protID_group', 'particle_cpiID_group', 'particle_npiID_group']

dset = f['Data']

for l in labels: 
    g = dset[l]
    p = g['particles']
    p_pdg = p['pdg']

    lb, counts = numpy.unique(p_pdg, return_counts=True)

    print ('Group', l)
    for a, b in zip(lb, counts):
        print('    label =', a, ', events =', b)
