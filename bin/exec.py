
#!/usr/bin/env python
import os,sys,signal
import time
import pathlib
import logging
from logging import handlers

import numpy

# For configuration:
from omegaconf import DictConfig, OmegaConf
import hydra

#############################

# Add the local folder to the import path:
network_dir = os.path.dirname(os.path.abspath(__file__))
network_dir = os.path.dirname(network_dir)
sys.path.insert(0,network_dir)


import argparse

class SparseEventID(object):

    def __init__(self, config):

        self.args = config

        rank = self.init_mpi()

        # Create the output directory if needed:
        if rank == 0:
            outpath = pathlib.Path(self.args.run.output_dir)
            outpath.mkdir(exist_ok=True, parents=True)

        self.configure_logger(rank)

        self.validate_arguments()

        if config.mode.name == "train":
            self.train()
        if config.mode.name == "iotest":
            self.iotest()



    def init_mpi(self):
        if not self.args.run.distributed:
            return 0
        else:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            return comm.Get_rank()


    def configure_logger(self, rank):

        logger = logging.getLogger()

        # Create a handler for STDOUT, but only on the root rank.
        # If not distributed, we still get 0 passed in here.
        if rank == 0:
            stream_handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            stream_handler.setFormatter(formatter)
            handler = handlers.MemoryHandler(capacity = 0, target=stream_handler)
            logger.addHandler(handler)

            # Add a file handler too:
            log_file = self.args.run.output_dir + "/process.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            file_handler = handlers.MemoryHandler(capacity=10, target=file_handler)
            logger.addHandler(file_handler)

            logger.setLevel(logging.INFO)
        else:
            # in this case, MPI is available but it's not rank 0
            # create a null handler
            handler = logging.NullHandler()
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)


    def train(self):

        logger = logging.getLogger()

        logger.info("Running Training")
        logger.info(self.__str__())

        self.make_trainer()

        self.trainer.initialize()
        self.trainer.batch_process()


    def iotest(self):

        logger = logging.getLogger()

        logger.info("Running IO Test")
        logger.info(self.__str__())

        self.make_trainer()
        print("Initializing")

        configured_keys = self.trainer.initialize(io_only=True)

        print("Initialized")

        if self.args.run.distributed:
            from mpi4py import MPI
            rank = MPI.COMM_WORLD.Get_rank()
        else:
            rank = 0

        # label_stats = numpy.zeros((36,))
        global_start = time.time()
        time.sleep(0.5)
        print("\n\n\n\nBegin Loop \n\n\n\n")
        for i in range(self.args.run.iterations):
            start = time.time()
            for key in configured_keys:
                time.sleep(0.0)
                print(f"Fetching {key}")
                mb = self.trainer.larcv_fetcher.fetch_next_batch(key, force_pop=True)
                print(f"Successfully got {key}")
            end = time.time()

            logger.info(f"{i}: Time to fetch a minibatch of data: {end - start:.2f}s")

        logger.info(f"Total IO Time: {time.time() - global_start:.2f}s")




    def make_trainer(self):

        if self.args.mode.name == "iotest":
            from src.utils.core import trainercore

            self.trainer = trainercore.trainercore(self.args)
            return

        if self.args.framework.name == "tensorflow":

            # Import tensorflow and see what the version is.
            import tensorflow as tf

            if self.args.run.distributed:
                from src.utils.tensorflow import distributed_trainer
                self.trainer = distributed_trainer.distributed_trainer(self.args)
            else:
                from src.utils.tensorflow import trainer
                self.trainer = trainer.trainer(self.args)


        elif self.args.framework.name == "torch":
            if self.args.run.distributed:
                from src.utils.torch import distributed_trainer
                self.trainer = distributed_trainer.distributed_trainer(self.args)
            else:
                from src.utils.torch import trainer
                self.trainer = trainer.trainer(self.args)




    def dictionary_to_str(self, in_dict, indentation = 0):
        substr = ""
        for key in sorted(in_dict.keys()):
            if type(in_dict[key]) == DictConfig or type(in_dict[key]) == dict:
                s = "{none:{fill1}{align1}{width1}}{key}: \n".format(
                        none="", fill1=" ", align1="<", width1=indentation, key=key
                    )
                substr += s + self.dictionary_to_str(in_dict[key], indentation=indentation+2)
            else:
                s = '{none:{fill1}{align1}{width1}}{message:{fill2}{align2}{width2}}: {attr}\n'.format(
                   none= "",
                   fill1=" ",
                   align1="<",
                   width1=indentation,
                   message=key,
                   fill2='.',
                   align2='<',
                   width2=30-indentation,
                   attr = in_dict[key],
                )
                substr += s
        return substr

    def __str__(self):

        s = "\n\n-- CONFIG --\n"
        substring = s +  self.dictionary_to_str(self.args)

        return substring




    def validate_arguments(self):


        # if self.args.framework.name == "torch":
        #     # In torch, only option is channels first:
        #     if self.args.data.data_format == "channels_last":
        #         print("Torch requires channels_first, switching automatically")
        #         self.args.data.data_format = "channels_first"

        pass





@hydra.main(config_path="../src/config", config_name="config")
def main(cfg : OmegaConf) -> None:

    s = SparseEventID(cfg)


if __name__ == '__main__':
    #  Is this good practice?  No.  But hydra doesn't give a great alternative
    import sys
    if "--help" not in sys.argv and "--hydra-help" not in sys.argv:
        sys.argv += ['hydra.run.dir=.', 'hydra/job_logging=disabled']
    main()
