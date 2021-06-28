from . import larcv_io



# Here, we set up a bunch of template IO formats in the form of callable functions:

# These are all doing sparse IO, so there is no dense IO template here.  But you could add it.

def dataset_io(name, input_file, image_dim, prepend_names="", RandomAccess=None):
    if image_dim == 2:
        max_voxels = 20000
        data_proc = gen_sparse2d_data_filler(name=prepend_names + "data", producer="\"dunevoxels\"", max_voxels=max_voxels)
    else:
        max_voxels = 16000
        data_proc = gen_sparse3d_data_filler(name=prepend_names + "data", producer="\"dunevoxels\"", max_voxels=max_voxels)

    label_proc = gen_label_filler(prepend_names)


    config = larcv_io.ThreadIOConfig(name=name)

    config.add_process(data_proc)
    for l in label_proc:
        config.add_process(l)

    config.set_param("InputFiles", input_file)
    if RandomAccess is not None:
        config.set_param("RandomAccess", RandomAccess)

    return config

def ana_io(input_file, image_dim, prepend_names=""):
    if image_dim == 2:
        max_voxels = 20000
        data_proc = gen_sparse2d_data_filler(name=prepend_names + "data", producer="\"dunevoxels\"", max_voxels=max_voxels)
    else:
        max_voxels = 16000
        data_proc = gen_sparse3d_data_filler(name=prepend_names + "data", producer="\"dunevoxels\"", max_voxels=max_voxels)


    label_proc = gen_label_filler(prepend_names)


    config = larcv_io.ThreadIOConfig(name="AnaIO")
    # Force ana files to go in order:

    config._params['RandomAccess'] = "0"
    config.add_process(data_proc)
    for l in label_proc:
        config.add_process(l)

    config.set_param("InputFiles", input_file)

    return config

def output_io(input_file, output_file):




    config = larcv_io.IOManagerConfig(name="IOManager")
    # Force ana files to go in order:

    config._params['RandomAccess'] = "0"

    config.set_param("InputFiles", input_file)
    config.set_param("OutputFile", output_file)

    # These lines slim down the output file.
    # Without them, 25 output events is 2.8M and takes 38s
    # With the, 25 output events is 119K and takes 36s
    # config.set_param("ReadOnlyType", "[\"particle\",\"particle\",\"particle\",\"particle\",\"particle\",\"particle\",\"particle\",\"cluster2d\",\"cluster2d\",\"cluster3d\",\"cluster3d\"]")
    # config.set_param("ReadOnlyName", "[\"neutrino\",\"cpiID\",\"neutID\",\"npiID\",\"protID\",\"all\",\"duneseg\",\"duneseg\",\"segment\",\"duneseg\",\"segment\"]")

    config.set_param("ReadOnlyType", "[\"particle\",\"particle\",\"particle\",\"particle\",\"particle\",\"particle\",\"particle\",\"cluster2d\"]")
    config.set_param("ReadOnlyName", "[\"neutrino\",\"cpiID\",\"neutID\",\"npiID\",\"protID\",\"all\",\"segment\",\"duneseg\"]")

    # config.set_param("ReadOnlyType", "[\"particle\",\"particle\",\"particle\",\"particle\",\"particle\",\"particle\",\"particle\"]")
    # config.set_param("ReadOnlyName", "[\"neutrino\",\"cpiID\",\"neutID\",\"npiID\",\"protID\",\"all\",\"segment\"]")

    return config


def gen_sparse2d_data_filler(name, producer, max_voxels):

    proc = larcv_io.ProcessConfig(proc_name=name, proc_type="BatchFillerSparseTensor2D")

    proc.set_param("Verbosity",         "3")
    proc.set_param("TensorProducer",    producer)
    proc.set_param("IncludeValues",     "true")
    proc.set_param("MaxVoxels",         max_voxels)
    proc.set_param("Channels",          "[0,1,2]")
    proc.set_param("UnfilledVoxelValue","-999")
    proc.set_param("Augment",           "false")

    return proc


def gen_sparse3d_data_filler(name, producer, max_voxels):

    proc = larcv_io.ProcessConfig(proc_name=name, proc_type="BatchFillerSparseTensor3D")

    proc.set_param("Verbosity",         "3")
    proc.set_param("TensorProducer",    producer)
    proc.set_param("IncludeValues",     "true")
    proc.set_param("MaxVoxels",         max_voxels)
    proc.set_param("UnfilledVoxelValue","-999")
    proc.set_param("Augment",           "true")

    return proc


def gen_label_filler(prepend_names):


    procs = []
    for name, l in zip(['neut', 'prot', 'cpi', 'npi'], [3, 3, 2, 2]):
        proc  = larcv_io.ProcessConfig(proc_name=prepend_names + "label_" + name, proc_type="BatchFillerPIDLabel")

        proc.set_param("Verbosity",         "3")
        proc.set_param("ParticleProducer",  name+"ID")
        proc.set_param("PdgClassList",      "[{}]".format(",".join([str(i) for i in range(l)])))

        procs.append(proc)
    return procs
