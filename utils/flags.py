import argparse

import os,sys
top_dir = os.path.dirname(os.path.abspath(__file__))
top_dir = os.path.dirname(top_dir)
'''
This script is heavily inspired by the following code from drinkingkazu:
https://github.com/DeepLearnPhysics/dynamic-gcnn/blob/develop/dgcnn/flags.py
'''

# This class is from here:
# http://www.aleax.it/Python/5ep.html
# Which is an incredibly simply and elegenant way 
# To enforce singleton behavior
class Borg:
    _shared_state = {}
    def __init__(self):
        self.__dict__ = self._shared_state

# This function is to parse strings from argparse into bool
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


class FLAGS(Borg):
    '''This class implements global flags through static variables
    The static-ness is enforced by inheriting from Borg, which it calls first and 
    foremost in the constructor.

    All classes derived from FLAGS should call this constructor first

    It also has the ability to parse the arguments pass on the command line for overrides,
     and can print out help for all of the flag variables

    In particular, this base class defines a lot of 'shared' parameter configurations:
    IO, snapshotting directories, etc.

    Network parameters are not implemented here but in derived classes.
    '''

    def __init__(self):
        Borg.__init__(self)


        # Parameters controlling training situations
        self.COMPUTE_MODE          = "CPU"
        self.TRAINING              = True
        self.MINIBATCH_SIZE        = 2
        self.CHECKPOINT_ITERATION  = 100
        self.SUMMARY_ITERATION     = 10
        self.LOGGING_ITERATION     = 1
        self.LEARNING_RATE         = 0.001
        self.ITERATIONS            = 5000
        self.VERBOSITY             = 0
        self.LOG_DIRECTORY         = './log'

        self.DISTRIBUTED           = False

        # IO parameters  
        # IO has a 'default' file configuration and an optional
        # 'auxilliary' configuration.  In Train mode, the default
        # is the training data, aux is testing data.
        # In inference mode, default is the validation data, 
        # aux is the outputdata
        self.FILE                  = '{}/io/dev/classification_3d_io_all.cfg'.format(top_dir)
        self.FILLER                = 'DevIO'
        self.IO_VERBOSITY          = 3
        self.KEYWORD_DATA          = 'data'
        # For this classification task, the label can be split or all-in-one
        self.LABEL_MODE            = 'split' # could also be 'all'

        # These are "background" parameters
        # And are meant to be copied to the 'KEYWORD_LABEL' area
        self.KEYWORD_LABEL_ALL     = 'label'
        self.KEYWORD_LABEL_SPLIT   = ['label_neut','label_cpi','label_npi','label_prot']

        self.KEYWORD_LABEL         = None

        # Relevant parameters for running on KNL:
        self.INTER_OP_PARALLELISM_THREADS     = 4
        self.INTRA_OP_PARALLELISM_THREADS     = 64


        # Optional Test IO parameters:
        # To activate the auxilliary IO, the AUX file must be not None
        self.AUX_FILE                  = None
        self.AUX_FILLER                = 'TestIO'
        self.AUX_IO_VERBOSITY          = 3
        self.AUX_KEYWORD_DATA          = 'data'
        self.AUX_KEYWORD_LABEL         = 'label'
        self.AUX_MINIBATCH_SIZE        = 10*self.MINIBATCH_SIZE
    
        # These are "background" parameters
        # And are meant to be copied to the 'KEYWORD_LABEL' area
        self.AUX_KEYWORD_LABEL_ALL     = 'label'
        self.AUX_KEYWORD_LABEL_SPLIT   = ['label_neut','label_cpi','label_npi','label_prot']

        self.AUX_KEYWORD_LABEL         = None



    def _add_default_io_configuration(self, parser):

        # IO PARAMETERS FOR INPUT:
        parser.add_argument('-f','--file', type=str, default=self.FILE,
            help="IO Configuration File [default: {}]".format(self.FILE))
        parser.add_argument('--filler', type=str, default=self.FILLER,
            help="IO Larcv Filler [default: {}]".format(self.FILLER))
        parser.add_argument('--io-verbosity', type=int, default=self.IO_VERBOSITY,
            help="IO verbosity [default: {}]".format(self.IO_VERBOSITY))
        parser.add_argument('--keyword-data', type=str, default=self.KEYWORD_DATA,
            help="Keyword for io data access [default: {}]".format(self.KEYWORD_DATA))
        parser.add_argument('--keyword-label', type=str, default=self.KEYWORD_LABEL,
            help="Keyword for io label access [default: {}]".format(self.KEYWORD_LABEL))

        parser.add_argument('--label-mode', type=str, choices=['split', 'all'], default=self.LABEL_MODE,
            help="Run with split labels (multiple classifiers) or all in one [default: {}]".format(self.LABEL_MODE))

        parser.add_argument('-mb','--minibatch-size',type=int, default=self.MINIBATCH_SIZE,
            help="Number of images in the minibatch size [default: {}]".format(self.MINIBATCH_SIZE))
        return parser


    def _add_aux_io_configuration(self, parser):

        # IO PARAMETERS FOR INPUT:
        parser.add_argument('--aux-file', type=str, default=self.AUX_FILE,
            help="IO Configuration File [default: {}]".format(self.AUX_FILE))
        parser.add_argument('--aux-filler', type=str, default=self.AUX_FILLER,
            help="IO Larcv Filler [default: {}]".format(self.AUX_FILLER))
        parser.add_argument('--aux-io-verbosity', type=int, default=self.AUX_IO_VERBOSITY,
            help="IO verbosity [default: {}]".format(self.AUX_IO_VERBOSITY))
        parser.add_argument('--aux-keyword-data', type=str, default=self.AUX_KEYWORD_DATA,
            help="Keyword for io data access [default: {}]".format(self.AUX_KEYWORD_DATA))
        parser.add_argument('--aux-keyword-label', type=str, default=self.AUX_KEYWORD_LABEL,
            help="Keyword for io label access [default: {}]".format(self.AUX_KEYWORD_LABEL))


        parser.add_argument('--aux-minibatch-size',type=int, default=self.AUX_MINIBATCH_SIZE,
            help="Number of images in the minibatch size [default: {}]".format(self.AUX_MINIBATCH_SIZE))

        return parser

    def _create_parsers(self):

        self._parser = argparse.ArgumentParser(description="Configuration Flags")

        subparsers = self._parser.add_subparsers(title="Modules", 
                                                 description="Valid subcommands", 
                                                 dest='mode', 
                                                 help="Available subcommands: train, iotest, inference")
      
      

        # train parser
        self.train_parser = subparsers.add_parser("train", help="Train")
        self.train_parser.add_argument('-ld','--log-directory', default=self.LOG_DIRECTORY,
                                  help='Prefix (directory + file prefix) for snapshots of weights [default: {}]'.format(self.LOG_DIRECTORY))
        self.train_parser.add_argument('-lr','--learning-rate', type=float, default=self.LEARNING_RATE,
                                  help='Initial learning rate [default: {}]'.format(self.LEARNING_RATE))
        self.train_parser.add_argument('-si','--summary-iteration', type=int, default=self.SUMMARY_ITERATION,
                                  help='Period (in steps) to store summary in tensorboard log [default: {}]'.format(self.SUMMARY_ITERATION))
        # self.train_parser.add_argument('-pi','--profile-iteration', type=int, default=self.PROFILE_ITERATION,
        #                           help='Period (in steps) to store profiling in tensorboard log [default: {}]'.format(self.PROFILE_ITERATION))
        self.train_parser.add_argument('-ci','--checkpoint-iteration', type=int, default=self.CHECKPOINT_ITERATION,
                                  help='Period (in steps) to store snapshot of weights [default: {}]'.format(self.CHECKPOINT_ITERATION))

        # attach common parsers
        self.train_parser  = self._add_default_network_configuration(self.train_parser)
        self.train_parser  = self._add_default_io_configuration(self.train_parser)
        self.train_parser  = self._add_aux_io_configuration(self.train_parser)
        self.train_parser  = self._add_core_configuration(self.train_parser)




        # IO test parser
        self.iotest_parser = subparsers.add_parser("iotest", help="Test io only (no network)")
        self.iotest_parser = self._add_default_io_configuration(self.iotest_parser)
        self.iotest_parser = self._add_aux_io_configuration(self.iotest_parser)
        self.iotest_parser  = self._add_core_configuration(self.iotest_parser)


        # # inference parser
        # inference_parser = subparsers.add_parser("inference",help="Run inference of Edge-GCNN")
       
        # cls.inference_parser = cls._add_default_parser_configuration(inference_parser)
        # cls.iotest_parser    = cls._add_default_parser_configuration(iotest_parser)
      

    def _add_core_configuration(self, parser):
        # These are core parameters that are important for all modes:
        parser.add_argument('-i', '--iterations', type=int, default=self.ITERATIONS,
            help="Number of iterations to process [default: {}]".format(self.ITERATIONS))

        parser.add_argument('-d','--distributed', action='store_true', default=self.DISTRIBUTED,
            help="Run with the MPI compatible mode [default: {}]".format(self.DISTRIBUTED))
        parser.add_argument('-m','--compute_mode', type=str, choices=['CPU','GPU'], default=self.COMPUTE_MODE,
            help="Selection of compute device, CPU or GPU  [default: {}]".format(self.COMPUTE_MODE))

        return parser



    def parse_args(self):
        self._create_parsers()
        args = self._parser.parse_args()
        self.update(vars(args))


    def dump_config(self):
        print(self.__str__())
            

    def get_config(str):
        return str.__str__()

    def __str__(self):
        try:
            _ = getattr(self, '_parser')
            s = "\n\n-- CONFIG --\n"
            for name in vars(self):
                if name != name.upper(): continue
                attribute = getattr(self,name)
                if type(attribute) == type(self._parser): continue
                s += " %s = %r\n" % (name, getattr(self, name))
            return s

        except AttributeError:
            return "ERROR: call parse_args()"

                    
    def update(self, args):
        for name,value in args.items():
            if name in ['func']: continue
            setattr(self, name.upper(), args[name])
        # Take special care to reset the keyword label attribute 
        # to match the label mode:
        if self.LABEL_MODE == "split":
            self.KEYWORD_LABEL = self.KEYWORD_LABEL_SPLIT
            self.AUX_KEYWORD_LABEL = self.AUX_KEYWORD_LABEL_SPLIT
        elif self.LABEL_MODE == "all":
            self.KEYWORD_LABEL = self.KEYWORD_LABEL_ALL
            self.AUX_KEYWORD_LABEL = self.AUX_KEYWORD_LABEL_ALL

    def _add_default_network_configuration(self, parser):
        raise NotImplementedError("Must use a derived class which overrides this function")

class resnet(FLAGS):
    ''' Resnet specific flags
    '''

    def __init__(self):
        FLAGS.__init__(self)

        self.USE_BIAS                   = True
        self.BATCH_NORM                 = True
        self.VERBOSITY                  = 0


        self.N_INITIAL_FILTERS          = 5
        self.RES_BLOCKS_PER_LAYER       = 2
        self.NETWORK_DEPTH_PRE_MERGE    = 3
        self.NETWORK_DEPTH_POST_MERGE   = 3
        self.NPLANES                    = 3
        self.SHARE_WEIGHTS              = True

    def _add_default_network_configuration(self, parser):



        parser.add_argument('-ub','--use-bias', type=str2bool, default=self.USE_BIAS,
            help="Whether or not to include bias terms in all mlp layers [default: {}]".format(self.USE_BIAS))
        parser.add_argument('-bn','--batch-norm', type=str2bool, default=self.BATCH_NORM,
            help="Whether or not to use batch normalization in all mlp layers [default: {}]".format(self.BATCH_NORM))

        parser.add_argument('-v', '--verbosity', type=int,default=self.VERBOSITY,
            help="Network verbosity at construction [default: {}]".format(self.VERBOSITY))


        parser.add_argument('--n-initial-filters', type=int, default=self.N_INITIAL_FILTERS,
            help="Number of filters applied, per plane, for the initial convolution [default: {}]".format(self.N_INITIAL_FILTERS))
        parser.add_argument('--res-blocks-per-layer', type=int, default=self.RES_BLOCKS_PER_LAYER,
            help="Number of residual blocks per layer [default: {}]".format(self.RES_BLOCKS_PER_LAYER))
        parser.add_argument('--network-depth-pre-merge', type=int, default=self.NETWORK_DEPTH_PRE_MERGE,
            help="Total number of downsamples to apply before merging planes [default: {}]".format(self.NETWORK_DEPTH_PRE_MERGE))
        parser.add_argument('--network-depth-post-merge', type=int, default=self.NETWORK_DEPTH_POST_MERGE,
            help="Total number of downsamples to apply after merging planes [default: {}]".format(self.NETWORK_DEPTH_POST_MERGE))
        parser.add_argument('--nplanes', type=int, default=self.NPLANES,
            help="Number of planes to split the initial image into [default: {}]".format(self.NPLANES))
        parser.add_argument('--share-weights', type=str2bool, default=self.SHARE_WEIGHTS,
            help="Whether or not to share weights across planes [default: {}]".format(self.SHARE_WEIGHTS))



        return parser

# class sparseresnet(FLAGS):
#     '''FLAGS for a sparse residual network
    
    
#     Extends:
#         FLAGS
#     '''

#     def __init__(self):
#         FLAGS.__init__(self)

#     # # # Parameters to control the network implementation
#     # # BATCH_NORM            = True
#     # # USE_BIAS              = True
#     # # # Options are "resnet", "pointnet", "sparseresnet"
#     # # MODEL                 = 'resnet'
#     # # DIMENSIONS            = 2


#     #     train_parser.add_argument('-rw','--regularize-weights', type=float, default=cls.REGULARIZE_WEIGHTS,
#     #         help="Regularization strength for all learned weights [default: {}]".format(cls.REGULARIZE_WEIGHTS))
#     #     train_parser.add_argument('-rt','--regularize-transforms', type=str2bool, default=cls.REGULARIZE_TRANSFORMS,
#     #         help="Regularization strength for transformations [default: {}]".format(cls.REGULARIZE_TRANSFORMS))




#     # # # Parameters controlling regularization
#     # # REGULARIZE_WEIGHTS    = 0.001
#     # # REGULARIZE_TRANSFORMS = 0.0001