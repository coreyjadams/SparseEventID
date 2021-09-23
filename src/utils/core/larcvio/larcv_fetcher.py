import os
import time

from . import data_transforms
import tempfile

import numpy
import h5py

import logging
logger = logging.getLogger()

from larcv.config_builder import ConfigBuilder

class larcv_fetcher(object):

    def __init__(self, mode, distributed, dataset, data_format, seed=None):

        if mode not in ['train', 'inference', 'iotest']:
            raise Exception("Larcv Fetcher can't handle mode ", mode)


        random_access_mode = dataset.access_mode

        if random_access_mode != "serial_access" and mode == "inference":
            logger.warn("Using random blocks in inference - possible bug!")

        if distributed:
            from larcv import distributed_queue_interface
            self._larcv_interface = distributed_queue_interface.queue_interface(
                random_access_mode=random_access_mode, seed=seed)
        else:
            from larcv import queueloader
            self._larcv_interface = queueloader.queue_interface(
                random_access_mode=random_access_mode, seed=seed)

        self.mode            = mode
        self.image_mode      = data_format
        self.input_dimension = dataset.dimension
        self.distributed     = distributed


        self.writer     = None


    def __del__(self):
        if hasattr(self, 'writer') and self.writer is not None:
            self.writer.finalize()



    def prepare_sample(self, name, input_file, batch_size, color=None, start_index = 0):



        # First, verify the files exist:
        if not os.path.exists(input_file):
            raise Exception(f"File {input_file} not found")


        cb = ConfigBuilder()
        cb.set_parameter([input_file], "InputFiles")
        cb.set_parameter(5, "ProcessDriver", "IOManager", "Verbosity")
        cb.set_parameter(5, "ProcessDriver", "Verbosity")
        cb.set_parameter(5, "Verbosity")
        
        # Build up the data_keys:
        data_keys = {}
        data_keys['image'] = name+'data'


        # Need to load up on data fillers.
        if self.input_dimension == 2:
            cb.add_batch_filler(
                datatype  = "sparse2d", 
                producer  = "dunevoxels", 
                name      = name+"data",
                MaxVoxels = 20000,
                Augment   = False
                )

        else:
            cb.add_batch_filler(
                datatype  = "sparse3d", 
                producer  = "dunevoxels", 
                name      = name+"data",
                MaxVoxels = 30000,
                Augment   = False
                )

        # Add something to convert the neutrino particles into bboxes:
        cb.add_preprocess(
            datatype = "particle",
            producer = "neutrino",
            process  = "BBoxFromParticle",
            OutputProducer = "neutrino"
        )
        cb.add_batch_filler(
            datatype  = "bbox3d", 
            producer  = "neutrino", 
            name      = name+"bbox",
            MaxBoxes  = 2,
            )

        # Add the label configs:
        for label_name, l in zip(['neut', 'prot', 'cpi', 'npi'], [3, 3, 2, 2]):
            cb.add_batch_filler(
                datatype     = "particle",
                producer     = f"{label_name}ID",
                name         = f'label_{label_name}',
                PdgClassList = [i for i in range(l)]
            )
            data_keys[f'label_{label_name}'] = f'label_{label_name}'

        logger.debug(cb.print_config())

        # Prepare data managers:
        io_config = {
            'filler_name' : name,
            'filler_cfg'  : cb.get_config(),
            'verbosity'   : 5,
            'make_copy'   : False
        }

        # Assign the keywords here:
        self.keyword_label = []
        for key in data_keys.keys():
            if key != 'image':
                self.keyword_label.append(key)

        data_keys.update({"vertex" : name+"bbox"})



        self._larcv_interface.prepare_manager(name, io_config, batch_size, data_keys, color=color)


        if self.mode == "inference":
            self._larcv_interface.set_next_index(name, start_index)

        while self._larcv_interface.is_reading(name):
            time.sleep(0.1)

        # # Here, we pause in distributed mode to make sure all loaders are ready:
        # if self.distributed:
        #     from mpi4py import MPI
        #     MPI.COMM_WORLD.Barrier()

        return self._larcv_interface.size(name)



    def fetch_minibatch_dims(self, name):
        return self._larcv_interface.fetch_minibatch_dims(name)

    def output_shape(self, name):

        dims = self.fetch_minibatch_dims(name)

        # This sets up the necessary output shape:
        output_shape = { key : dims[key] for key in self.keyword_label}

        return output_shape

    def fetch_next_batch(self, name, force_pop=False):

        metadata=True

        pop = True
        if not force_pop:
            pop = False


        while self._larcv_interface.is_reading(name):
            # print("Sleeping in larcv_fetcher")
            time.sleep(0.01)

        minibatch_data = self._larcv_interface.fetch_minibatch_data(name,
            pop=pop,fetch_meta_data=metadata)
        minibatch_dims = self._larcv_interface.fetch_minibatch_dims(name)


        # This brings up the next data to current data
        if pop:
            # print(f"Preparing next {name}")
            self._larcv_interface.prepare_next(name)
            # time.sleep(0.1)


        # If the returned data is None, return none and don't load more:
        if minibatch_data is None:
            return minibatch_data



        # Reshape as needed from larcv:
        for key in minibatch_data:
            if key == 'entries' or key == 'event_ids':
                continue
            minibatch_data[key] = numpy.reshape(minibatch_data[key], minibatch_dims[key])

        # Here, do some massaging to convert the input data to another format, if necessary:
        if self.image_mode == 'dense':
            # Need to convert sparse larcv into a dense numpy array:
            if self.input_dimension == 3:
                minibatch_data['image'] = data_transforms.larcvsparse_to_dense_3d(minibatch_data['image'])
            else:
                minibatch_data['image'] = data_transforms.larcvsparse_to_dense_2d(minibatch_data['image'])
        elif self.image_mode == 'sparse':
            # Have to convert the input image from dense to sparse format:
            if self.input_dimension == 3:
                minibatch_data['image'] = data_transforms.larcvsparse_to_scnsparse_3d(minibatch_data['image'])
            else:
                minibatch_data['image'] = data_transforms.larcvsparse_to_scnsparse_2d(minibatch_data['image'])
        elif self.image_mode == 'graph':

            if self.input_dimension == 3:
                minibatch_data['image'] = data_transforms.larcvsparse_to_pointcloud_3d(minibatch_data['image'])
            else:
                minibatch_data['image'] = data_transforms.larcvsparse_to_pointcloud_2d(minibatch_data['image'])

        else:
            raise Exception("Image Mode not recognized")

        return minibatch_data


    def prepare_writer(self, input_file, output_file):

        from larcv import larcv_writer
        config = io_templates.output_io(input_file  = input_file)

        # Generate a named temp file:
        main_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
        main_file.write(config.generate_config_str())

        main_file.close()

        self.writer = larcv_writer.larcv_writer(main_file.name, output_file)

    def write(self, data, producer, entry, event_id):
        self.writer.write(data, datatype='sparse2d', producer=producer, entry=entry, event_id=event_id)
