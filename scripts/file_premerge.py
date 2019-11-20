import glob
import os

outname = '/gpfs/alpine/scratch/deltutto/nph133/dune_wire_pixel/new_merged_DUNE_DL_pixsim_highstat_samples_20191108/premerged_sample'
files = '/gpfs/alpine/scratch/deltutto/nph133/dune_wire_pixel/original_DUNE_DL_pixsim_highstat_samples_20191108/merged_sample_*.h5'


files = glob.glob(files)

n_files = len(files)

files_chunks = 8

for i in range(0, 128):

    these_files = files[files_chunks*i:files_chunks*i+files_chunks]
    print ('Merging ', len(these_files), ' files from ', files_chunks*i, ' to ', files_chunks*i+files_chunks)

    command = 'python /ccs/home/deltutto/software/larcv3/bin/merge_larcv3_files.py -il '
    
    for f in these_files:
        command += f
        command += ' '

    command += ' -ol '
    command += outname
    command += '_chunk'
    command += str(files_chunks)
    command += '_'
    command += str(i)
    command += '.h5'

    # print (command)
    os.system(command)
