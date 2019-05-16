from . import larcv_io



# Here, we set up a bunch of template IO formats in the form of callable functions:

def train_io(input_file, image_dim, label_mode, prepend_names=""):
    if image_dim == "2D":
        max_voxels = 10000
        data_proc = gen_sparse2d_data_filler(name=prepend_names + "data", producer="\"sbndwire\"", max_voxels=max_voxels)
    else:
        max_voxels = 18000
        data_proc = gen_sparse3d_data_filler(name=prepend_names + "data", producer="\"sbndvoxels\"", max_voxels=max_voxels)

    label_proc = gen_label_filler(label_mode, prepend_names)


    config = larcv_io.ThreadIOConfig(name="TrainIO")

    config.add_process(data_proc)
    for l in label_proc:
        config.add_process(l)

    config.set_param("InputFiles", input_file)

    return config


def test_io(input_file, image_dim, label_mode, prepend_names="aux_"):
    if image_dim == "2D":
        max_voxels = 10000
        data_proc = gen_sparse2d_data_filler(name=prepend_names + "data", producer="\"sbndwire\"", max_voxels=max_voxels)
    else:
        max_voxels = 18000
        data_proc = gen_sparse3d_data_filler(name=prepend_names + "data", producer="\"sbndvoxels\"", max_voxels=max_voxels)

    label_proc = gen_label_filler(label_mode, prepend_names)


    config = larcv_io.ThreadIOConfig(name="TestIO")

    config.add_process(data_proc)
    for l in label_proc:
        config.add_process(l)

    config.set_param("InputFiles", input_file)

    return config


def ana_io(input_file, image_dim, label_mode, prepend_names=""):
    if image_dim == "2D":
        max_voxels = 10000
        data_proc = gen_sparse2d_data_filler(name=prepend_names + "data", producer="\"sbndwire\"", max_voxels=max_voxels)
    else:
        max_voxels = 18000
        data_proc = gen_sparse3d_data_filler(name=prepend_names + "data", producer="\"sbndvoxels\"", max_voxels=max_voxels)


    label_proc = gen_label_filler(label_mode, prepend_names)


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
    config.set_param("ReadOnlyType", "[\"particle\",\"particle\",\"particle\",\"particle\",\"particle\",\"particle\",\"particle\"]")  
    config.set_param("ReadOnlyName", "[\"sbndneutrino\",\"sbndsegmerged\",\"cpiID\",\"neutID\",\"npiID\",\"protID\",\"all\"]")  

    return config


def gen_sparse2d_data_filler(name, producer, max_voxels):

    proc = larcv_io.ProcessConfig(proc_name=name, proc_type="BatchFillerSparseTensor2D")

    proc.set_param("Verbosity",         "3")
    proc.set_param("Tensor2DProducer",  producer)
    proc.set_param("IncludeValues",     "true")
    proc.set_param("MaxVoxels",         max_voxels)
    proc.set_param("Channels",          "[0,1,2]")
    proc.set_param("UnfilledVoxelValue","-999")
    proc.set_param("Augment",           "true")

    return proc


def gen_sparse3d_data_filler(name, producer, max_voxels):

    proc = larcv_io.ProcessConfig(proc_name=name, proc_type="BatchFillerSparseTensor3D")

    proc.set_param("Verbosity",         "3")
    proc.set_param("Tensor3DProducer",  producer)
    proc.set_param("IncludeValues",     "true")
    proc.set_param("MaxVoxels",         max_voxels)
    proc.set_param("UnfilledVoxelValue","-999")
    proc.set_param("Augment",           "true")

    return proc


def gen_label_filler(label_mode, prepend_names):

    if label_mode == 'all':

        proc = larcv_io.ProcessConfig(proc_name=prepend_names + "label", proc_type="BatchFillerPIDLabel")

        proc.set_param("Verbosity",         "3")
        proc.set_param("ParticleProducer",  "all")
        proc.set_param("PdgClassList",      "[{}]".format(",".join([str(i) for i in range(36)])))

        return [proc]

    else:
        procs = []
        for name, l in zip(['neut', 'prot', 'cpi', 'npi'], [3, 3, 2, 2]): 
            proc  = larcv_io.ProcessConfig(proc_name=prepend_names + "label_" + name, proc_type="BatchFillerPIDLabel")

            proc.set_param("Verbosity",         "3")
            proc.set_param("ParticleProducer",  name+"ID")
            proc.set_param("PdgClassList",      "[{}]".format(",".join([str(i) for i in range(l)])))

            procs.append(proc)
        return procs






