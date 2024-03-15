
#!/usr/bin/env python
import os,sys,signal
import time
import pathlib


# For configuration:
from omegaconf import DictConfig, OmegaConf
import hydra

from hydra.experimental import compose, initialize
from hydra.core.hydra_config import HydraConfig
from hydra.core.utils import configure_log

hydra.output_subdir = None

#############################

# Add the local folder to the import path:
network_dir = os.path.dirname(os.path.abspath(__file__))
network_dir = os.path.dirname(network_dir)
sys.path.insert(0,network_dir)

# from src.config import Config
from src.config.mode import ModeKind

from src.io import create_larcv_dataset

from src import logging

import atexit


class SparseEventID(object):

    def __init__(self, config):

        self.args = config


        rank = self.init_mpi()

        # Add to the output dir:
        # self.args.output_dir += f"/{self.args.network.name}/"
        self.args.output_dir += f"/{self.args.run.id}/"

        # Create the output directory if needed:
        if rank == 0:
            outpath = pathlib.Path(self.args.output_dir)
            outpath.mkdir(exist_ok=True, parents=True)

        self.configure_logger(rank)

        self.validate_arguments()

        # Print the command line args to the log file:
        logger = logging.getLogger("SpEvID")
        logger.info("Dumping launch arguments.")
        logger.info(sys.argv)
        logger.info(self.__str__())

        logger.info("Configuring Datasets.")
        self.datasets  = self.configure_datasets()
        # self.datasets, self.transforms = self.configure_datasets()
        logger.info("Data pipeline ready.")

    def exit(self):
        if hasattr(self, "trainer"):
            self.trainer.exit()

    def run(self):
        if self.args.mode.name == ModeKind.train:
            self.train()
        if self.args.mode.name == ModeKind.iotest:
            self.iotest()
        if self.args.mode.name == ModeKind.inference:
            self.inference()
        if self.args.mode.name == ModeKind.visualize:
            self.visualize()


    def init_mpi(self):
        if not self.args.run.distributed:
            return 0
        else:
            from mpi4py import MPI
            self.args.run.world_size = MPI.COMM_WORLD.Get_size()
            return MPI.COMM_WORLD.Get_rank()


    def configure_lr_schedule(self, epoch_length, max_epochs):



        if self.args.mode.optimizer.lr_schedule.name == "one_cycle":
            from src.utils import OneCycle
            schedule_args = self.args.mode.optimizer.lr_schedule
            lr_schedule = OneCycle(
                min_learning_rate  = 1e-5,
                peak_learning_rate = schedule_args.peak_learning_rate,
                decay_floor        = schedule_args.decay_floor, 
                epoch_length       = epoch_length,
                decay_epochs       = schedule_args.decay_epochs,
                total_epochs       = max_epochs
            )
        elif self.args.mode.optimizer.lr_schedule.name == "standard":
            from src.utils import WarmupFlatDecay
            schedule_args = self.args.mode.optimizer.lr_schedule
            lr_schedule = WarmupFlatDecay(
                peak_learning_rate = schedule_args.peak_learning_rate,
                decay_floor  = schedule_args.decay_floor,
                epoch_length = epoch_length,
                decay_epochs = schedule_args.decay_epochs,
                total_epochs = max_epochs
            )

        return lr_schedule


    def configure_datasets(self):
        """
        This function creates the non-framework iterable datasets used in this app.

        They get converted to framework specific tools, if needed, in the
        framework specific code.
        """

        from src.io import create_torch_larcv_dataloader

        batch_keys = [self.args.data.image_key]
        if self.args.data.transform1:
            batch_keys.append(self.args.data.image_key + "_1")
        if self.args.data.transform2:
            batch_keys.append(self.args.data.image_key + "_2")
        if self.args.name == "yolo":
            batch_keys.append("vertex")
            batch_keys.append("label")
            batch_keys.append("energy")

        elif self.args.name == "supervised_eventID":
            batch_keys.append("labelneutID")
            batch_keys.append("labelprotID")
            batch_keys.append("labelcpiID")
            batch_keys.append("labelnpiID")
        elif self.args.name == "unsupervised_eventID":
            batch_keys.append("energy")
            batch_keys.append("label")

        ds = {}
        print(self.args.data.active)
        for active in self.args.data.active:

            f_name = getattr(self.args.data, active)
            larcv_ds = create_larcv_dataset(self.args.data,
                batch_size   = self.args.run.minibatch_size,
                input_file   = f_name,
                name         = active,
                distributed  = self.args.run.distributed,
                batch_keys   = batch_keys,
                data_mode    = self.args.framework.mode,
            )


            ds.update({
                active :  create_torch_larcv_dataloader(
                    larcv_ds,
                    self.args.run.minibatch_size,
                    self.args.framework.mode,
                )
            })
            # Get the image size:
            # spatial_size = larcv_ds.image_size(self.args.data.image_key)

        return ds
        #

    def configure_logger(self, rank):

        logger = logging.getLogger("SpEvID")
        if rank == 0:
            logger.setFile(self.args.output_dir + "/process.log")
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(999)


    def train(self):

        logger = logging.getLogger("SpEvID")

        logger.info("Running Training")

        self.make_trainer()

        from src.utils import create_trainer

        trainer, model, checkpoint_path = create_trainer(self.args, self.trainer, self.datasets)

        trainer.fit(
            model,
            train_dataloaders= self.datasets["train"],
            val_dataloaders  = self.datasets["val"],
            ckpt_path        = checkpoint_path
        )

    def inference(self):
        logger = logging.getLogger("NEXT")


        logger.info("Running Inference")

        self.make_trainer()

        from src.utils import create_trainer


        trainer, model, checkpoint_path = create_trainer(self.args, self.trainer, self.datasets)

        trainer.validate(
            model,
            dataloaders  = self.datasets["val"],
            ckpt_path    = checkpoint_path
        )

    def iotest(self):

        logger = logging.getLogger("NEXT")

        logger.info("Running IO Test")


        # self.trainer.initialize(io_only=True)

        if self.args.run.distributed:
            from mpi4py import MPI
            rank = MPI.COMM_WORLD.Get_rank()
        else:
            rank = 0



        for key, dataset in self.datasets.items():
            logger.info(f"Reading dataset {key}")
            global_start = time.time()
            total_reads = 0


            # Determine the stopping point:
            break_i = self.args.run.length * len(dataset)
            break_i = 25

            start = time.time()
            for i, minibatch in enumerate(dataset):
                image = minibatch[self.args.data.image_key]
                end = time.time()
                if i >= break_i: break
                logger.info(f"{i}: Time to fetch a minibatch of data: {end - start:.2f}s")
                start = time.time()
                total_reads += 1

            total_time = time.time() - global_start
            images_read = total_reads * self.args.run.minibatch_size
            logger.info(f"{key} - Total IO Time: {total_time:.2f}s")
            logger.info(f"{key} - Total images read per batch: {self.args.run.minibatch_size}")
            logger.info(f"{key} - Average Image IO Throughput: { images_read / total_time:.3f}")



    def make_trainer(self):

        dataset_length = max([len(ds) for ds in self.datasets.values()])


        if self.args.mode.name == ModeKind.train:
            lr_schedule = self.configure_lr_schedule(dataset_length, self.args.run.length)
        else:
            lr_schedule = None

        if self.args.name == "simclr":
            from src.utils.representation_learning import create_lightning_module

        elif self.args.name == "yolo":
            from src.utils.vertex_finding import create_lightning_module
        elif self.args.name == "supervised_eventID":
            from src.utils.supervised_eventID import create_lightning_module
        elif self.args.name == "unsupervised_eventID":
            from src.utils.unsupervised_eventID import create_lightning_module


        transform_keys = []
        if self.args.data.transform1:
            transform_keys.append(self.args.data.image_key+"_1")
        if self.args.data.transform2:
             transform_keys.append(self.args.data.image_key+"_2")
        self.trainer = create_lightning_module(
            self.args,
            self.datasets,
            transform_keys,
            lr_schedule,
        )


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

        pass



from src.config import config

@hydra.main(version_base=None, config_path="../recipes")
def main(cfg : OmegaConf) -> None:

    s = SparseEventID(cfg)
    atexit.register(s.exit)

    s.run()

if __name__ == '__main__':
    #  Is this good practice?  No.  But hydra doesn't give a great alternative
    import sys
    if "--help" not in sys.argv and "--hydra-help" not in sys.argv:
        sys.argv += [
            'hydra/job_logging=disabled',
            'hydra.output_subdir=null',
            'hydra.job.chdir=False',
            'hydra.run.dir=.',
            'hydra/hydra_logging=disabled',
        ]
    main()