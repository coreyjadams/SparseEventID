import sys
import time


import tensorflow as tf


# Declaring exception names:
class ConfigurationException(Exception): pass



# Main class
class networkcore(object):
    '''Define a network model and run training

    U resnet implementation
    '''
    def __init__(self):
        '''initialization

        Requires a list of parameters as python dictionary

        Arguments:
            params {dict} -- Network parameters

        Raises:
            ConfigurationException -- Missing a required parameter
        '''
        super(networkcore, self).__init__()
        # This defines a core set of parameters needed to define the network model
        # It gets extended for each network that may inherit this class.
        self._core_network_params = [
            'TRAINING',
        ]


    def set_params(self, params):
        ''' Check the parameters, if passes set the paramers'''
        if self._check_params(params):
            self._params = params
            self._apply_default_params()

    def _check_params(self, params):
        for param in self._core_network_params:
            if param not in params:
                raise ConfigurationException("Missing paragmeter "+ str(param))
        return True
        # self._params = params

    def _apply_default_params(self):
        raise NotImplementedError("Must implement _apply_default_params in subclass.")
