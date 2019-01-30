#!/usr/bin/env python

from balsam.launcher import dag

# This script runs after train has finished for a job.
# It determines some basic information:
# - Did the training reach the requested number of iterations?
# - If not, is the model overfitting?
# 
# If the job has reached the required number of iterations, this script spawns inference jobs
# that process the validation dataset.

# If the model has not reached the requested number of iterations, AND 
# it does not appear to be overfitting, then the job is extended by 
# more iterations.  It passes the same arguments, and therefore doubles the 
# number of iterations

# To measure the overfitting of a network, we require the testing loss 
# divided by the training loss to be less than 2.
# This is a totally arbitrary metric.

# To parse the loss, we use tensorboardX to skim the data off of the log files.
# We get the log location by parsing the arguments to the training with argparse

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
def tabulate_events(dpath):
    ''' Go into a tensorboard even log and scrape up all of the scalars, return the output in a usable way
    '''
    summary_iterators = [EventAccumulator(os.path.join(dpath, dname)).Reload() for dname in os.listdir(dpath)]

    for summary_iterator in summary_iterators:
        if summary_iterator.Tags()['scalars'] == []:
            continue
        else:
            break

    tags = summary_iterator.Tags()['scalars']

    out = {}
    steps = []

    for tag in tags:
        events = summary_iterator.Scalars(tag)
        out[tag] = [e.value for e in events]
        if steps == []:
            steps = [e.step for e in events]
        else:
            assert steps == [e.step for e in events]

    return out, steps

import argparse
def generic_parser():
    # This parser only knows enough about the arguments to find the log path, number of iterations, etc
    
    ITERATIONS = None
    LOG_DIRECTORY = None

    parser = argparse.ArgumentParser(description="Configuration Flags")
    parser.add_argument('-i', '--iterations', type=int, default=ITERATIONS,
            help="Number of iterations to process [default: {}]".format(ITERATIONS))
        parser.add_argument('-ld','--log-directory', default=LOG_DIRECTORY,
            help='Prefix (directory + file prefix) for snapshots of weights [default: {}]'.format(LOG_DIRECTORY))


    args, unknown = parser.parse_known_args(dag.current_job.args)

    return LOG_DIRECTORY


def postprocess_training():

    # First, get this current job:
    current_job = dag.current_job

    # Let's parse the arguments to get the logdir:
    LOG_DIRECTORY = generic_parser()

    # We should be able to see the log dir.
    print("Attempting to scrape tensorboard information from {}".format(LOG_DIRECTORY))

    output, steps = tabulate_events(LOG_DIRECTORY)

    print(steps)


if __name__ == '__main__':
    postprocess_training()











