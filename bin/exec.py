#!/usr/bin/env python
import os,sys,signal
import time

import numpy

# Add the local folder to the import path:
network_dir = os.path.dirname(os.path.abspath(__file__))
network_dir = os.path.dirname(network_dir)
sys.path.insert(0,network_dir)

# import the necessary
# from src.utils import flags
from src.networks import resnet
from src.networks import sparseresnet
from src.networks import sparseresnet3d
from src.networks import pointnet
from src.networks import gcn
from src.networks import dgcnn


import argparse

class SparseEventID(object):

    def __init__(self):

        # This technique is taken from: https://chase-seibert.github.io/blog/2014/03/21/python-multilevel-argparse.html
        parser = argparse.ArgumentParser(
            description='Run neural networks on Sparse Event ID dataset',
            usage='''exec.py <command> [<args>]

The most commonly used commands are:
   train      Train a network, either from scratch or restart
   inference  Run inference with a trained network
   iotest     Run IO testing without training a network
''')
        parser.add_argument('command', help='Subcommand to run')
        # parse_args defaults to [1:] for args, but you need to
        # exclude the rest of the args too, or validation will fail
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print(f'Unrecognized command {args.command}')
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()


    def train(self):
        self.parser = argparse.ArgumentParser(
            description     = 'Run Network Training',
            formatter_class = argparse.ArgumentDefaultsHelpFormatter)

        self.add_io_arguments(self.parser)
        self.add_core_configuration(self.parser)

        # Define parameters exclusive to training:

        self.parser.add_argument('-lr','--learning-rate', 
            type    = float, 
            default = 0.003,
            help    = 'Initial learning rate')
        self.parser.add_argument('-si','--summary-iteration', 
            type    = int, 
            default = 1,
            help    = 'Period (in steps) to store summary in tensorboard log')
        self.parser.add_argument('-li','--logging-iteration', 
            type    = int, 
            default = 1,
            help    = 'Period (in steps) to print values to log')
        self.parser.add_argument('-ci','--checkpoint-iteration', 
            type    = int, 
            default = 100,
            help    = 'Period (in steps) to store snapshot of weights')
        self.parser.add_argument('--lr-schedule', 
            type    = str, 
            choices = ['flat', '1cycle', 'triangle_clr', 'exp_range_clr', 'decay', 'expincrease'], 
            default = 'flat',
            help    = 'Apply a learning rate schedule')
        self.parser.add_argument('--optimizer', 
            type    = str, 
            choices = ['Adam', 'SGD'], 
            default = 'Adam',
            help    = 'Optimizer to use')
        self.parser.add_argument('-cd','--checkpoint-directory', 
            default = None,
            help    = 'Prefix (directory + file prefix) for snapshots of weights')
        self.parser.add_argument('--weight-decay', 
            type    = float, 
            default = 0.0,
            help    = "Weight decay strength")




        self.add_network_parsers(self.parser)

        self.args = self.parser.parse_args(sys.argv[2:])
        self.args.training = True


        self.make_trainer()

        print("Running Training")
        print(self.__str__())

        self.trainer.initialize()
        self.trainer.batch_process()


    def add_network_parsers(self, parser):
        # Here, we define the networks available.  In io test mode, used to determine what the IO is.
        network_parser = parser.add_subparsers( 
            title          = "Networks", 
            dest           = "network",
            description    = 'Which network architecture to use.')

        # Here, we do a switch on the networks allowed:        
        resnet.ResNetFlags().build_parser(network_parser)
        sparseresnet.ResNetFlags().build_parser(network_parser)
        sparseresnet3d.ResNetFlags().build_parser(network_parser)
        pointnet.PointNetFlags().build_parser(network_parser)
        gcn.GCNFlags().build_parser(network_parser)
        dgcnn.DGCNNFlags().build_parser(network_parser)


    def iotest(self):
        self.parser = argparse.ArgumentParser(
            description     = 'Run IO Testing',
            formatter_class = argparse.ArgumentDefaultsHelpFormatter)
        self.add_io_arguments(self.parser)
        self.add_core_configuration(self.parser)

        # now that we're inside a subcommand, ignore the first
        # TWO argvs, ie the command (exec.py) and the subcommand (iotest)
        self.args = self.parser.parse_args(sys.argv[2:])
        self.args.training = False
        print("Running IO Test")
        print(self.__str__())

        self.make_trainer()

        self.trainer.initialize(io_only=True)

        # label_stats = numpy.zeros((36,))

        time.sleep(0.1)
        for i in range(self.args.iterations):
            start = time.time()
            mb = self.trainer.fetch_next_batch()
            # print(mb.keys())
            # label_stats += numpy.sum(mb['label'], axis=0)

            end = time.time()
            if not self.args.distributed:
                print(i, ": Time to fetch a minibatch of data: {}".format(end - start))
            else:
                if self.trainer._rank == 0:
                    print(i, ": Time to fetch a minibatch of data: {}".format(end - start))
            # time.sleep(0.5)
        # print(label_stats)

    def make_trainer(self):

        if self.args.distributed:
            from src.utils import distributed_trainer

            self.trainer = distributed_trainer.distributed_trainer(self.args)
        else:
            from src.utils import trainercore
            self.trainer = trainercore.trainercore(self.args)

    def inference(self):
        pass


    def __str__(self):
        s = "\n\n-- CONFIG --\n"
        for name in iter(sorted(vars(self.args))):
            # if name != name.upper(): continue
            attribute = getattr(self.args,name)
            # if type(attribute) == type(self.parser): continue
            # s += " %s = %r\n" % (name, getattr(self, name))
            substring = ' {message:{fill}{align}{width}}: {attr}\n'.format(
                   message=name,
                   attr = getattr(self.args, name),
                   fill='.',
                   align='<',
                   width=30,
                )
            s += substring
        return s

    def stop(self):
        if not self.args.distributed:
            self.trainer.stop()



    def add_core_configuration(self, parser):
        # These are core parameters that are important for all modes:
        parser.add_argument('-i', '--iterations', 
            type    = int, 
            default = 5000,
            help    = "Number of iterations to process")

        parser.add_argument('-d','--distributed', 
            action  = 'store_true', 
            default = False,
            help    = "Run with the MPI compatible mode")
        parser.add_argument('-m','--compute-mode', 
            type    = str, 
            choices = ['CPU','GPU'], 
            default = 'CPU',
            help    = "Selection of compute device, CPU or GPU ")
        parser.add_argument('-im','--image-mode',
            type    = str,
            choices = ['dense', 'sparse', 'graph'],
            default = 'sparse',
            help    = "Input image format to the network, dense or sparse")
        parser.add_argument('-ld','--log-directory', 
            default ="log/",
            help    ="Prefix (directory) for logging information")


        return parser

    def add_io_arguments(self, parser):

        # IO PARAMETERS FOR INPUT:
        parser.add_argument('-f','--file', 
            type    = str, 
            default = "/lus/theta-fs0/projects/datascience/cadams/wire_pixel_preprocessed_files_split/train_event_id.root",
            help    = "IO Input File")
        parser.add_argument('--input-dimension', 
            type    = int, 
            default = 3,
            help    = "Dimensionality of data to use",
            choices = [2, 3] )
        parser.add_argument('--start-index', 
            type    = int, 
            default = 0,
            help    = "Start index, only used in inference mode")

        parser.add_argument('--label-mode', 
            type    = str, 
            choices = ['split', 'all'], 
            default = 'split',
            help    = "Run with split labels (multiple classifiers) or all in one" )

        parser.add_argument('-mb','--minibatch-size',
            type    = int, 
            default = 2,
            help    = "Number of images in the minibatch size")
        
        # IO PARAMETERS FOR AUX INPUT:
        parser.add_argument('--aux-file', 
            type    = str, 
            default = None,
            help    = "IO Aux Input File, or output file in inference mode")


        parser.add_argument('--aux-iteration',
            type    = int, 
            default = 10,
            help    = "Iteration to run the aux operations")

        parser.add_argument('--aux-minibatch-size',
            type    = int, 
            default = 2,
            help    = "Number of images in the minibatch size")

        return



def main():

    FLAGS = flags.FLAGS()
    FLAGS.parse_args()
    # FLAGS.dump_config()

    

    if FLAGS.MODE is None:
        raise Exception()


        
    if FLAGS.MODE == 'train' or FLAGS.MODE == 'inference':

        trainer.initialize()
        trainer.batch_process()

       



if __name__ == '__main__':
    s = SparseEventID()  
    s.stop()
