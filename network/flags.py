import argparse

import os,sys
top_dir = os.path.dirname(os.path.abspath(__file__))
top_dir = os.path.dirname(top_dir)
'''
This script is heavily inspired by the following code from drinkingkazu:
https://github.com/DeepLearnPhysics/dynamic-gcnn/blob/develop/dgcnn/flags.py
'''


def str2bool(v):
    '''Convert string to boolean value
    
    This function is from stackoverflow: 
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    
    Arguments:
        v {str} -- [description]
    
    Returns:
        bool -- [description]
    
    Raises:
        argparse -- [description]
    '''
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class FLAGS:
    '''This class implements global flags through static variables

    It also has the ability to parse the arguments pass on the command line for overrides,
     and can print out help for all of the flag variables
    '''

    # Parameters to control the network implementation
    BATCH_NORM            = True
    USE_BIAS              = True
    MODEL                 = 'pointnet'
    DIMENSIONS            = 2


    # Parameters controlling regularization
    REGULARIZE_WEIGHTS    = 0.001
    REGULARIZE_TRANSFORMS = 0.0001

    # Parameters controlling training situations
    COMPUTE_MODE          = "CPU"
    TRAINING              = True
    MINIBATCH_SIZE        = 2
    CHECKPOINT_ITERATION  = 100
    SUMMARY_ITERATION     = 10
    PROFILE_ITERATION     = 100
    LOGGING_ITERATION     = 1
    LEARNING_RATE         = 0.001
    ITERATIONS            = 5000
    VERBOSITY             = 0
    LOG_DIRECTORY         = './log'

    DISTRIBUTED           = False

    # IO parameters  
    FILE                  = '{}/io/dev/classification_3d_io_all.cfg'.format(top_dir)
    FILLER                = 'DevIO'
    IO_VERBOSITY          = 3
    KEYWORD_DATA          = 'data'
    LABEL_MODE            = 'split' # could also be 'all'

    # These are "background" parameters
    # And are meant to be copied to the 'KEYWORD_LABEL' area
    KEYWORD_LABEL_ALL     = 'label'
    KEYWORD_LABEL_SPLIT   = ['label_neut','label_cpi','label_npi','label_prot']

    KEYWORD_LABEL         = None

    # Relevant parameters for running on KNL:
    INTER_OP_PARALLELISM_THREADS     = 4
    INTRA_OP_PARALLELISM_THREADS     = 64

    @classmethod
    def _add_default_network_configuration(cls, parser):

        parser.add_argument('--model', type=str, default=cls.MODEL, choices=['pointnet'],
            help="Model variant to use [default: {}]".format(cls.MODEL))
        parser.add_argument('--dimensions', type=int, default=cls.DIMENSIONS,
            help="Run in 2 or 3 dimensions [default: {}]".format(cls.DIMENSIONS))

        parser.add_argument('-ub','--use-bias', type=str2bool, default=cls.USE_BIAS,
            help="Whether or not to include bias terms in all mlp layers [default: {}]".format(cls.USE_BIAS))
        parser.add_argument('-bn','--batch-norm', type=str2bool, default=cls.BATCH_NORM,
            help="Whether or not to use batch normalization in all mlp layers [default: {}]".format(cls.BATCH_NORM))

        parser.add_argument('-v', '--verbosity', type=int,default=cls.VERBOSITY,
            help="Network verbosity at construction [default: {}]".format(cls.VERBOSITY))

        parser.add_argument('-m','--compute_mode', type=str, choices=['CPU','GPU'], default=cls.COMPUTE_MODE,
            help="Selection of compute device, CPU or GPU  [default: {}]".format(cls.COMPUTE_MODE))



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

        parser.add_argument('-d','--distributed', action='store_true', default=cls.DISTRIBUTED,
            help="Run with the MPI compatible mode [default: {}]".format(cls.DISTRIBUTED))
        
        parser.add_argument('--label-mode', type=str, choices=['split', 'all'], default=cls.LABEL_MODE,
            help="Run with split labels (multiple classifiers) or all in one [default: {}]".format(cls.LABEL_MODE))



        parser.add_argument('-mb','--minibatch-size',type=int, default=cls.MINIBATCH_SIZE,
            help="Number of images in the minibatch size [default: {}]".format(cls.MINIBATCH_SIZE))

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
        train_parser.add_argument('-ld','--log-directory', default=cls.LOG_DIRECTORY,
                                  help='Prefix (directory + file prefix) for snapshots of weights [default: {}]'.format(cls.LOG_DIRECTORY))
        train_parser.add_argument('-lr','--learning-rate', type=float, default=cls.LEARNING_RATE,
                                  help='Initial learning rate [default: {}]'.format(cls.LEARNING_RATE))
        train_parser.add_argument('-si','--summary-iteration', type=int, default=cls.SUMMARY_ITERATION,
                                  help='Period (in steps) to store summary in tensorboard log [default: {}]'.format(cls.SUMMARY_ITERATION))
        train_parser.add_argument('-pi','--profile-iteration', type=int, default=cls.PROFILE_ITERATION,
                                  help='Period (in steps) to store profiling in tensorboard log [default: {}]'.format(cls.PROFILE_ITERATION))
        train_parser.add_argument('-ci','--checkpoint-iteration', type=int, default=cls.CHECKPOINT_ITERATION,
                                  help='Period (in steps) to store snapshot of weights [default: {}]'.format(cls.CHECKPOINT_ITERATION))


        train_parser.add_argument('-rw','--regularize-weights', type=float, default=cls.REGULARIZE_WEIGHTS,
            help="Regularization strength for all learned weights [default: {}]".format(cls.REGULARIZE_WEIGHTS))
        train_parser.add_argument('-rt','--regularize-transforms', type=str2bool, default=cls.REGULARIZE_TRANSFORMS,
            help="Regularization strength for transformations [default: {}]".format(cls.REGULARIZE_TRANSFORMS))


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
        print(cls.__str__())
            

    @classmethod
    def get_config(cls):
        return cls.__str__()

    @classmethod
    def __str__(cls):
        try:
            _ = getattr(cls, '_parser')
            s = "\n\n-- CONFIG --\n"
            for name in vars(cls):
                if name != name.upper(): continue
                attribute = getattr(cls,name)
                if type(attribute) == type(cls._parser): continue
                s += " %s = %r\n" % (name, getattr(cls, name))
            return s

        except AttributeError:
            return "ERROR: call parse_args()"

        # self.inference_parser.set_defaults(func=inference)
        # self.iotest_parser.set_defaults(func=iotest)


        # Set random seed for reproducibility
        # args.func(self)
                    
    @classmethod
    def update(cls, args):
        for name,value in args.items():
            if name in ['func']: continue
            setattr(cls, name.upper(), args[name])
        # Take special care to reset the keyword label attribute 
        # to match the label mode:
        if cls.LABEL_MODE == "split":
            cls.KEYWORD_LABEL = cls.KEYWORD_LABEL_SPLIT
        elif cls.LABEL_MODE == "all":
            cls.KEYWORD_LABEL = cls.KEYWORD_LABEL_ALL



