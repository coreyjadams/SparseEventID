import os
import time

from . import data_transforms
from . import io_templates
import tempfile

import numpy
import h5py

import logging
logger = logging.getLogger()

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

        config = io_templates.dataset_io(
                name        = name,
                input_file  = input_file,
                image_dim   = self.input_dimension)


        # Generate a named temp file:
        main_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
        main_file.write(config.generate_config_str())


        main_file.close()


        # Prepare data managers:
        io_config = {
            'filler_name' : config._name,
            'filler_cfg'  : main_file.name,
            'verbosity'   : 5,
            'make_copy'   : True
        }

        # Build up the data_keys:
        data_keys = {}
        data_keys['image'] = 'data'
        for proc in config._process_list._processes:
            if proc._name == 'data':
                continue
            else:
                data_keys[proc._name] = proc._name

        # Assign the keywords here:
        self.keyword_label = []
        for key in data_keys.keys():
            if key != 'image':
                self.keyword_label.append(key)



        self._larcv_interface.prepare_manager(name, io_config, batch_size, data_keys, color=color)
        os.unlink(main_file.name)


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
            time.sleep(0.1)

        minibatch_data = self._larcv_interface.fetch_minibatch_data(name,
            pop=pop,fetch_meta_data=metadata)
        minibatch_dims = self._larcv_interface.fetch_minibatch_dims(name)

        # This brings up the next data to current data
        if pop:
            self._larcv_interface.prepare_next(name)

        # If the returned data is None, return none and don't load more:
        if minibatch_data is None:
            return minibatch_data



        # Reshape as needed from larcv:
        for key in minibatch_data:
            if key == 'entries' or key == 'event_ids':
                continue
            minibatch_data[key] = numpy.reshape(minibatch_data[key], minibatch_dims[key])

        # Strip off the primary/aux label in the keys:
        # if self.mode != 'train':
        #     # Can't do this in a loop due to limitations of python's dictionaries.
        #     minibatch_data["label_cpi"]  = minibatch_data.pop("aux_label_cpi")
        #     minibatch_data["label_npi"]  = minibatch_data.pop("aux_label_npi")
        #     minibatch_data["label_prot"] = minibatch_data.pop("aux_label_prot")
        #     minibatch_data["label_neut"] = minibatch_data.pop("aux_label_neut")


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
