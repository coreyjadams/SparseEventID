#!/usr/bin/env python
import os
import argparse
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy

from balsam.launcher import dag

from utils import spawn_training_job, spawn_inference_job

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


def tabulate_events(dpath):
    ''' Go into a tensorboard even log and scrape up all of the scalars, return the output in a usable way
    '''

    # We want to get the test and real event logs seperately

    train_logs = []
    test_logs  = []

    for d in os.listdir(dpath):
        if 'events' in d:
            train_logs.append(os.path.join(dpath, d))
    for d in os.listdir(dpath + "/test/"):
        if 'events' in d:
            test_logs.append(os.path.join(dpath + "/test/", d))


    train_summary_iterators = [EventAccumulator(d).Reload() for d in train_logs]
    test_summary_iterators = [EventAccumulator(d).Reload() for d in test_logs]

    train_steps = None
    train_loss = None

    test_steps = None
    test_loss = None

    print("Found {} training summaries".format(len(train_summary_iterators)))

    for train_summary_iterator in train_summary_iterators:
        if train_summary_iterator.Tags()['scalars'] == []:
            continue

        train_events = train_summary_iterator.Scalars('loss')
        temp_steps =  numpy.asarray([e.step for e in train_events])
        temp_loss = numpy.asarray([e.value for e in train_events])
        if train_steps is None:
            train_steps = temp_steps
        else:
            train_steps = numpy.concatenate([train_steps, temp_steps])

        if train_loss is None:
            train_loss = temp_loss
        else:
            train_loss = numpy.concatenate([train_loss, temp_loss])

    for test_summary_iterator in test_summary_iterators:
        if test_summary_iterator.Tags()['scalars'] == []:
            continue

        test_events = test_summary_iterator.Scalars('loss')
        temp_steps =  numpy.asarray([e.step for e in test_events])
        temp_loss = numpy.asarray([e.value for e in test_events])
        if test_steps is None:
            test_steps = temp_steps
        else:
            test_steps = numpy.concatenate([test_steps, temp_steps])

        if test_loss is None:
            test_loss = temp_loss
        else:
            test_loss = numpy.concatenate([test_loss, temp_loss])

    return train_steps, train_loss, test_steps, test_loss



def quantify_overtraining(minibatch_size, train_loss, test_loss, train_steps, test_steps):
    # Use the latest N steps from training, and the test steps that match those values
    # Return values:
    # -1 == stop training, overfitting happened
    # 0 == keep training, no changes
    # 1 == keep training, lower the learning rate
    # 2 == stop training, start inference

    max_step = numpy.max(train_steps)

    print(max_step)

    # This forces an inference job:
    return 2

    # If this job has exceeded 20 epochs (20 * 1e5 = 3M events)
    # then stop

    if minibatch_size * max_step >= 20*1e5:
        return 2

    # Take the last 10%, 50 steps, or 1 point (whichever is largest)

    N_10 = numpy.sum(train_steps > 0.9*max_step)
    N_50 = numpy.sum(train_steps > max_step - 50)


    if N_10 > N_50:
        min_step = 0.9*max_step
    elif N_50 > 1:
        min_step =  max_step - 50
    else:
        min_step = max_step

    average_train_loss = numpy.mean(train_loss[train_steps >= min_step])

    average_test_loss = numpy.mean(test_loss[test_steps >= min_step])

    if average_test_loss == 0:
        return 0
    elif average_test_loss / average_train_loss > 2.0:
        return -1

    # If it hasn't returned, that means the average test loss is still close to the average
    # train loss.
    # We can see if the training loss is decreasing.  If the values from 80 to 90% are not higher
    # than the values from 90 to 100, we can signal to decrease the learning rate

    # Don't do this unless there are more than 4000 iterations

    # if max_step > 20:
    #     range_80_to_90 = (train_steps >= 0.8 * max_step) * (train_steps < 0.9*max_step)
    #     # range_80_to_90 = numpy.ma.mask_or(train_steps >= 0.8 * max_step, train_steps < 0.9*max_step)
    #     range_90_to_100 = train_steps >= 0.9 * max_step

    #     loss_80_to_90 = train_loss[range_80_to_90]
    #     mean_80_to_90 = numpy.mean(loss_80_to_90)
    #     std_80_to_90 = numpy.std(loss_80_to_90)

    #     loss_90_to_100 = train_loss[range_90_to_100]
    #     mean_90_to_100 = numpy.mean(loss_90_to_100)
    #     std_90_to_100 = numpy.std(loss_90_to_100)

    #     print(mean_80_to_90, std_80_to_90)
    #     print(mean_90_to_100, std_90_to_100)

    #     if numpy.abs(mean_90_to_100 - mean_80_to_90) < 0.01 * numpy.max([std_80_to_90, std_90_to_100]):
    #         return 1
    #     else:
    #         return 0

    # else:
    return 0

def generic_parser():
    # This parser only knows enough about the arguments to find the log path, number of iterations, etc
    
    ITERATIONS = None
    MINIBATCH_SIZE = None
    LOG_DIRECTORY = None
    LEARNING_RATE = None
    FILE=None

    parser = argparse.ArgumentParser(description="Configuration Flags")
    parser.add_argument('-i', '--iterations', type=int, default=ITERATIONS,
            help="Number of iterations to process [default: {}]".format(ITERATIONS))
    parser.add_argument('-mb', '--minibatch-size', type=int, default=MINIBATCH_SIZE,
            help="batch size[default: {}]".format(MINIBATCH_SIZE))
    parser.add_argument('-ld','--log-directory', default=LOG_DIRECTORY,
            help='Prefix (directory + file prefix) for snapshots of weights [default: {}]'.format(LOG_DIRECTORY))
    parser.add_argument('-lr','--learning-rate', default=LOG_DIRECTORY,
            help='Prefix (directory + file prefix) for snapshots of weights [default: {}]'.format(LOG_DIRECTORY))
    parser.add_argument('-f','--file', default=FILE,
            help='input file [default: {}]'.format(FILE))
    parser.add_argument('-cd', '--checkpoint-directory', type=str, default=None,
            help='Prefix for the weights, also where inference files will be stored')

    print("Trying to parse the following args for the logdir:")
    print(dag.current_job.args)

    args, unknown = parser.parse_known_args(dag.current_job.args.split(" "))


    return args, unknown


def postprocess_training():

    # First, get this current job:
    current_job = dag.current_job

    # Let's parse the arguments to get the logdir:
    args, unknown = generic_parser()

    print(args)
    print(unknown)

    # We should be able to see the log dir.
    print("Attempting to scrape tensorboard information from {}".format(args.log_directory))

    train_steps, train_loss, test_steps, test_loss = tabulate_events(args.log_directory)

    value = quantify_overtraining(
        minibatch_size=args.minibatch_size,
        train_loss=train_loss, 
        test_loss=test_loss, 
        train_steps=train_steps, 
        test_steps=test_steps)

    if "2D" in dag.current_job.application:
        dimension = "2D"
    else:
        dimension = "3D"

    print(dimension)

    if value == 0:

        print("Should spawn another identical job")

        next_job = spawn_training_job(
            num_nodes = dag.current_job.num_nodes, 
            wall_time_minutes = dag.current_job.wall_time_minutes, 
            name = dag.current_job.name + "C", 
            workflow = dag.current_job.workflow, 
            dimension = dimension, 
            args=dag.current_job.args
        )

        dag.add_dependency(dag.current_job, next_job)

    elif value == -1:
        print("Aborting this path")
    
    # elif value == 1:

    #     args.learning_rate = str(float(args.learning_rate)*0.5)


    #     new_args = " ".join(unknown)
    #     for key in args:
    #         new_args += "--{key} {value}".format(key=key, value=args[key])

    #     print("Should spawn another job, lower LR")

    #     next_job = spawn_training_job(
    #         num_nodes = dag.current_job.num_nodes, 
    #         wall_time_minutes = dag.current_job.wall_time_minutes, 
    #         name = dag.current_job.name + "-", 
    #         workflow = dag.current_job.workflow, 
    #         dimension = dimension, 
    #         args=new_args
    #     )

    #     dag.add_dependency(dag.current_job, next_job)


    #     print("Decrease learning rate")
    elif value == 2:

        # Inference files:
        inference_files = [
        "/lus/theta-fs0/projects/datascience/cadams/wire_pixel_preprocessed_files_split/val_event_id_1_of_8.root",
        "/lus/theta-fs0/projects/datascience/cadams/wire_pixel_preprocessed_files_split/val_event_id_2_of_8.root",
        "/lus/theta-fs0/projects/datascience/cadams/wire_pixel_preprocessed_files_split/val_event_id_3_of_8.root",
        "/lus/theta-fs0/projects/datascience/cadams/wire_pixel_preprocessed_files_split/val_event_id_4_of_8.root",
        "/lus/theta-fs0/projects/datascience/cadams/wire_pixel_preprocessed_files_split/val_event_id_5_of_8.root",
        "/lus/theta-fs0/projects/datascience/cadams/wire_pixel_preprocessed_files_split/val_event_id_6_of_8.root",
        "/lus/theta-fs0/projects/datascience/cadams/wire_pixel_preprocessed_files_split/val_event_id_7_of_8.root",
        "/lus/theta-fs0/projects/datascience/cadams/wire_pixel_preprocessed_files_split/val_event_id_8_of_8.root",
        ]

        for i, _file in enumerate(inference_files):
            basename = os.path.basename(_file)
            basename = basename.replace('.root', '_out.root')
            out_file = args.checkpoint_directory + basename

            args.file = _file
            args.minibatch_size = 1
            args.iterations = 10

            new_args = " ".join(unknown)
            for key in vars(args):
                new_args += "--{key} {value} ".format(key=key.replace("_", "-"), value=getattr(args,key))

            # Add the output file
            new_args += "--output-file {}".format(out_file)

            print(new_args)

            next_job = spawn_inference_job(
                num_nodes = dag.current_job.num_nodes, 
                wall_time_minutes = dag.current_job.wall_time_minutes, 
                name = dag.current_job.name + "I{}".format(i), 
                workflow = dag.current_job.workflow, 
                dimension = dimension, 
                args=new_args
            )

            dag.add_dependency(dag.current_job, next_job)

            break

        print("Spawn inference jobs")


if __name__ == '__main__':
    postprocess_training()











