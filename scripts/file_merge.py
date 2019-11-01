import glob
import os

outname = '/gpfs/alpine/scratch/deltutto/nph133/dune_wire_pixel/merged_DUNE_DL_pixsim_150k_samples_20191029/merged_sample_all2'
files = '/gpfs/alpine/scratch/deltutto/nph133/dune_wire_pixel/original_DUNE_DL_pixsim_150k_samples_20191029/merged_sample_*.h5'


files = glob.glob(files)

n_files = len(files)

training_file_index = int(0.80 * n_files)
# training_file_index = int(0.75 * n_files)
# testing_file_index  = int(0.825 * n_files)

file_dictionary = {
    "train" : files[:training_file_index],
    "test"  : files[training_file_index:-1],
    # "test"  : files[training_file_index:testing_file_index],
    # "val"   : files[testing_file_index:-1]
 }

for mode in ['train', 'test']:

	print ('Merging ', len(file_dictionary[mode]), ' files for mode:', mode)

	command = 'python /ccs/home/deltutto/software/larcv3/bin/merge_larcv3_files.py -il '

	for f in file_dictionary[mode]:
		command += f
		command += ' '

	command += ' -ol '
	command += outname
	command += '_'
	command += mode
	command += '.h5'

	# print (command)
	os.system(command)
