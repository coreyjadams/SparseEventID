import argparse

import os,sys
top_dir = os.path.dirname(os.path.abspath(__file__))
top_dir = os.path.dirname(top_dir)
'''
This script is heavily inspired by the following code from drinkingkazu:
https://github.com/DeepLearnPhysics/dynamic-gcnn/blob/develop/dgcnn/flags.py
'''

class FLAGS:
    '''This class implements global flags through static variables

    It also has the ability to parse the arguments pass on the command line for overrides,
     and can print out help for all of the flag variables
    '''

    # Parameters to control the network implementation
    BATCH_NORM          = True
    USE_BIAS            = True
    NUM_CLASSES         = 3

    # Parameters controlling regularization
    REGULARIZE_WEIGHTS  = 0.001
    TRANSFORMATION_LOSS = 0.0001

    # Parameters controlling training situations
    COMPUTE_MODE                = "CPU"
    TRAINING            = True
    MINIBATCH_SIZE      = 4
    SAVE_ITERATION      = 100
    SUMMARY_ITERATION   = 10
    PROFILE_ITERATION   = 100
    LOGGING_ITERATION   = 1
    LEARNING_RATE       = 0.001
    ITERATIONS          = 5000
    VERBOSITY           = 0
    LOGDIR              = './log'
    # LOGDIR=   '/home/cadams/DeepLearnPhysics/CosmicTagger/log/dev/'

    DISTRIBUTED         = False

    # IO paramters=  
    FILE                = '{}/io/dev_io.cfg'.format(top_dir)
    FILLER              = 'DevIO'
    IO_VERBOSITY        = 3
    KEYWORD_DATA        = 'data'
    KEYWORD_LABEL       = 'label'


    @classmethod
    def _add_default_network_configuration(cls, parser):

        parser.add_argument('-nc','--num-classes', type=int, default=cls.NUM_CLASSES,
            help="Number of classes in the output classification network [default: {}]".format(cls.NUM_CLASSES))
        parser.add_argument('-ub','--bias', type=bool, default=cls.USE_BIAS,
            help="Whether or not to include bias terms in all mlp layers [default: {}]".format(cls.USE_BIAS))
        parser.add_argument('-bn','--batch-norm', type=bool, default=cls.BATCH_NORM,
            help="Whether or not to use batch normalization in all mlp layers [default: {}]".format(cls.BATCH_NORM))

        parser.add_argument('-v', '--verbosity', type=int,default=cls.VERBOSITY,
            help="Network verbosity at construction [default: {}]".format(cls.VERBOSITY))

        parser.add_argument('-m','--compute_mode', type=str, choices=['CPU','GPU'], default=cls.COMPUTE_MODE,
            help="Selection of compute device, CPU or GPU  [default: {}]".format(cls.COMPUTE_MODE))

        parser.add_argument('-mb','--minibatch-size',type=int, default=cls.NUM_CLASSES,
            help="Number of images in the minibatch size [default: {}]".format(cls.NUM_CLASSES))




        return parser

    @classmethod
    def _add_default_io_configuration(cls, parser):

        # IO PARAMETERS FOR INPUT:
        parser.add_argument('-f','--file', type=str, default=cls.FILE,
            help="IO Configuration File [default: {}]".format(cls.FILE))
        parser.add_argument('--filler', type=str, default=cls.FILLER,
            help="IO Larcv Filler [default: {}]".format(cls.FILLER))
        parser.add_argument('--io-verbosity', type=int, default=cls.IO_VERBOSITY,
            help="IO verbosity [default: {}]".format(cls.IO_VERBOSITY))
        parser.add_argument('--keyword-data', type=str, default=cls.KEYWORD_DATA,
            help="Keyword for io data access [default: {}]".format(cls.KEYWORD_DATA))
        parser.add_argument('--keyword-label', type=str, default=cls.KEYWORD_LABEL,
            help="Keyword for io label access [default: {}]".format(cls.KEYWORD_LABEL))
        parser.add_argument('-i','--iterations', type=int, default=cls.ITERATIONS,
            help="Number of iterations to process [default: {}]".format(cls.ITERATIONS))

        parser.add_argument('-d','--distributed', type=bool, default=cls.DISTRIBUTED,
            help="Run with the MPI compatible mode [default: {}]".format(cls.DISTRIBUTED))
        return parser

    @classmethod
    def _create_parsers(cls):

        cls._parser = argparse.ArgumentParser(description="PointNet Configuration Flags")
        subparsers = cls._parser.add_subparsers(title="Modules", 
                                                 description="Valid subcommands", 
                                                 dest='mode', 
                                                 help="Available subcommands: train, iotest")



        # train parser
        train_parser = subparsers.add_parser("train", help="Train PointNet")
        train_parser.add_argument('-ld','--log-directory', default=cls.LOGDIR,
                                  help='Prefix (directory + file prefix) for snapshots of weights [default: {}]'.format(cls.LOGDIR))
        train_parser.add_argument('-lr','--learning-rate', type=float, default=cls.LEARNING_RATE,
                                  help='Initial learning rate [default: {}]'.format(cls.LEARNING_RATE))
        train_parser.add_argument('-si','--summary-iteration', type=int, default=cls.SUMMARY_ITERATION,
                                  help='Period (in steps) to store summary in tensorboard log [default: {}]'.format(cls.SUMMARY_ITERATION))
        train_parser.add_argument('-pi','--profile-iteration', type=int, default=cls.PROFILE_ITERATION,
                                  help='Period (in steps) to store profiling in tensorboard log [default: {}]'.format(cls.PROFILE_ITERATION))
        train_parser.add_argument('-ci','--checkpoint-iteration', type=int, default=cls.SAVE_ITERATION,
                                  help='Period (in steps) to store snapshot of weights [default: {}]'.format(cls.SAVE_ITERATION))


        train_parser.add_argument('-wr','--weight-regularization', type=float, default=cls.REGULARIZE_WEIGHTS,
            help="Regularization strength for all learned weights [default: {}]".format(cls.REGULARIZE_WEIGHTS))
        train_parser.add_argument('-tr','--transform-regularization', type=bool, default=cls.TRANSFORMATION_LOSS,
            help="Regularization strength for transformations [default: {}]".format(cls.TRANSFORMATION_LOSS))


        # # inference parser
        # inference_parser = subparsers.add_parser("inference",help="Run inference of Edge-GCNN")
        # # IO test parser
        cls.iotest_parser = subparsers.add_parser("iotest", help="Test io only (no network)")
        cls.iotest_parser = cls._add_default_io_configuration(cls.iotest_parser)

        # attach common parsers
        cls.train_parser     = cls._add_default_network_configuration(train_parser)
        cls.train_parser     = cls._add_default_io_configuration(cls.train_parser)
        # cls.inference_parser = cls._add_default_parser_configuration(inference_parser)
        # cls.iotest_parser    = cls._add_default_parser_configuration(iotest_parser)


    @classmethod
    def parse_args(cls):
        cls._create_parsers()
        args = cls._parser.parse_args()
        cls.update(vars(args))


    @classmethod
    def dump_config(cls):
        print("\n\n-- CONFIG --")
        for name in vars(cls):
            if name != name.upper(): continue
            attribute = getattr(cls,name)
            if type(attribute) == type(cls._parser): continue
            print("%s = %r" % (name, getattr(cls, name)))
            


        # self.inference_parser.set_defaults(func=inference)
        # self.iotest_parser.set_defaults(func=iotest)


        # Set random seed for reproducibility
        # args.func(self)
                    
    @classmethod
    def update(cls, args):
        for name,value in args.items():
            if name in ['func']: continue
            setattr(cls, name.upper(), args[name])
        # os.environ['CUDA_VISIBLE_DEVICES']=self.GPUS
        # self.GPUS=[int(gpu) for gpu in self.GPUS.split(',')]
        # self.INPUT_FILE=[str(f) for f in self.INPUT_FILE.split(',')]
        # if self.EDGE_CONV_FILTERS.find(',')>0:
        #     self.EDGE_CONV_FILTERS = [int(v) for v in self.EDGE_CONV_FILTERS.split(',')]
        # else:
        #     self.EDGE_CONV_FILTERS = int(self.EDGE_CONV_FILTERS)
        # if self.FC_FILTERS.find(',')>0:
        #     self.FC_FILTERS = [int(v) for v in self.FC_FILTERS.split(',')]
        # else:
        #     self.FC_FILTERS = int(self.FC_FILTERS)
        # if self.SEED < 0:
        #     import time
        #     self.SEED = int(time.time())
    

    
