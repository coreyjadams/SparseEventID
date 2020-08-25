import os
import time

from . import data_transforms
from . import io_templates
import tempfile

import numpy
import h5py

from torch_geometric.data import Batch

class larcv_fetcher(object):

    def __init__(self, mode, distributed, image_mode, label_mode, input_dimension, seed=None):

        if mode not in ['train', 'inference', 'iotest']:
            raise Exception("Larcv Fetcher can't handle mode ", mode)


        if mode == "inference":
            random_access_mode = "serial_access"
        else:
            random_access_mode = "random_blocks"

        if distributed:
            from larcv import distributed_queue_interface
            self._larcv_interface = distributed_queue_interface.queue_interface(
                random_access_mode=random_access_mode, seed=seed)
        else:
            from larcv import queueloader
            self._larcv_interface = queueloader.queue_interface(
                random_access_mode=random_access_mode, seed=seed)

        self.mode            = mode
        self.image_mode      = image_mode
        self.label_mode      = label_mode
        self.input_dimension = input_dimension

        self.writer     = None


    def __del__(self):
        if self.writer is not None:
            self.writer.finalize()



    def prepare_sample(self, name, input_file, batch_size, color=None, start_index = 0):



        # First, verify the files exist:
        if not os.path.exists(input_file):
            raise Exception(f"File {input_file} not found")

        config = io_templates.dataset_io(
                name        = name,
                input_file  = input_file,
                image_dim   = self.input_dimension,
                label_mode  = self.label_mode)


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
        if self.label_mode == 'all':
            self.keyword_label = 'label'
        else:
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

        return self._larcv_interface.size(name)


#############################################################################



        # # In inference mode, we use h5py to pull out the energy and interaction flavor.
        # if self.mode == "inference":
        #     try:
        #         # Open the file:
        #         f = h5py.File(input_file, 'r')
        #         # Read the right tables from the file:
        #         particle_data = f['Data/particle_sbndneutrino_group/particles']
        #         extents = f['Data/particle_sbndneutrino_group/extents']

        #         # We index into the particle table with extents:
        #         indexes = extents['first']
        #         self.truth_variables[name] = {}
        #         self.truth_variables[name]['energy'] = particle_data['energy_init'][indexes]
        #         self.truth_variables[name]['ccnc']   = particle_data['current_type'][indexes]
        #         self.truth_variables[name]['pdg']    = particle_data['pdg'][indexes]

        #         f.close()
        #     except:
        #         pass

    def fetch_minibatch_dims(self, name):
        return self._larcv_interface.fetch_minibatch_dims(name)

    def output_shape(self, name):

        dims = self.fetch_minibatch_dims(name)

        # This sets up the necessary output shape:
        if self.label_mode == 'split':
            output_shape = { key : dims[key] for key in self.keyword_label}
        else:
            output_shape = dims[self.keyword_label]

        return output_shape

    def fetch_next_batch(self, name, force_pop=False):

        metadata=True

        pop = True
        if not force_pop:
            pop = False

        minibatch_data = self._larcv_interface.fetch_minibatch_data(name,
            pop=pop,fetch_meta_data=metadata)
        minibatch_dims = self._larcv_interface.fetch_minibatch_dims(name)

        # If the returned data is None, return none and don't load more:
        if minibatch_data is None:
            return minibatch_data

        # This brings up the next data to current data
        if pop:
            self._larcv_interface.prepare_next(name)

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
                # Here we use Batch.from_data_list to create a bacth object from a lit of torch geometric Data objects
                minibatch_data['image'] = data_transforms.larcvsparse_to_pointcloud_3d(minibatch_data['image'])
                minibatch_data['image'] = Batch.from_data_list(minibatch_data['image'])
            
        else:
            raise Exception("Image Mode not recognized")

        return minibatch_data


########################################################################################




        for key in minibatch_data:
            if key == 'entries' or key == 'event_ids':
                continue
            minibatch_data[key] = numpy.reshape(minibatch_data[key], minibatch_dims[key])


        if self.image_mode == "dense":
            if self.input_dimension == 3:
                minibatch_data['image']  = data_transforms.larcvsparse_to_dense_3d(
                    minibatch_data['image'],
                    dense_shape =self.image_shape,
                    dataformat  =self.dataformat)
            elif self.input_dimension == 2:
                minibatch_data['image']  = data_transforms.larcvsparse_to_dense_2d(
                    minibatch_data['image'],
                    dense_shape =self.image_shape,
                    dataformat  =self.dataformat)

        elif self.image_mode == "sparse":
            minibatch_data['image']  = data_transforms.larcvsparse_to_scnsparse_2d(
                minibatch_data['image'])
        elif self.image_mode == "graph":
            minibatch_data['image'] = data_transforms.larcvsparse_to_pointcloud_3d(minibatch_data['image'])
            minibatch_data['image'] = Batch.from_data_list(minibatch_data['image'])

        # Label is always dense:
        minibatch_data['label']  = data_transforms.larcvsparse_to_dense_2d(
            minibatch_data['label'],
            dense_shape =self.image_shape,
            dataformat  =self.dataformat)


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
