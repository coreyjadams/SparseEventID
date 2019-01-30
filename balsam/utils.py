from balsam.launcher import dag


def build_arg_list(**kwargs):

    s = ""
    for key in kwargs:
        s += "--{key} value ".format(key, kwargs[key])
    return s

def spawn_training_job(num_nodes, walltime, name, workflow, dimension, args=None, **kwargs):

    # There are two optional inputs here.
    # First, args can be passed in a completed form, which is useful for re-spawning a training job 
    # that continues a previous job.

    # Second, args can be built from kwargs.  Any arg not supplied will rely on the default values
    # in the FLAGS class, so it's a YMMV kind of situation

    # TODO: verify kwargs work

    if args is None:
        args = build_arg_list(kwargs)

    if dimension == '2D':
        app = 'event-ID-2D-train'
    else:
        app = 'event-ID-2D-train'

    job = dag.add_job(
            name                = name,
            workflow            = workflow,
            description         = 'Training job for resnet {}'.format(dimension),
            num_nodes           = num_nodes,
            ranks_per_node      = 2,
            threads_per_rank    = 1,
            environ_vars        = "PYTHONPATH:\"\"",
            wall_time_minutes   = walltime,
            args                = args,
            application         = app
        )

    return job

def spawn_inference_job(num_nodes, walltime, name, workflow, dimension, args=None, **kwargs):

    # There are two optional inputs here.
    # First, args can be passed in a completed form, which is useful for re-spawning a training job 
    # that continues a previous job.

    # Second, args can be built from kwargs.  Any arg not supplied will rely on the default values
    # in the FLAGS class, so it's a YMMV kind of situation

    # TODO: verify kwargs work

    if args is None:
        args = build_arg_list(kwargs)

    if dimension == '2D':
        app = 'event-ID-2D-inference'
    else:
        app = 'event-ID-2D-inference'

    job = dag.add_job(
            name                = name,
            workflow            = workflow,
            description         = 'Inference job for resnet {}'.format(dimension),
            num_nodes           = num_nodes,
            ranks_per_node      = 2,
            threads_per_rank    = 1,
            environ_vars        = "PYTHONPATH:\"\"",
            wall_time_minutes   = walltime,
            args                = args,
            application         = app
        )

    return job
    