#!/usr/bin/env python

# This script is to begin a 1-off job launch

# It's meant for testing and debugging.

from balsam.launcher import dag

from utils import spawn_training_job


def main():

    # We need to spawn a job, so lets build up some kwargs
    
    num_nodes=1
    wall_time_minutes=10
    name="test_job_flow_2D"
    workflow="sparse_eventID_test_inference"
    dimension="2D"

    batch_size = num_nodes * 2 * 8

    # The rest of this is kwargs

    job = spawn_training_job(
        num_nodes=num_nodes, 
        wall_time_minutes=wall_time_minutes,
        name=name, 
        workflow=workflow, 
        dimension=dimension, 
        args=None,

        learning_rate=0.01,
        checkpoint_iteration=25,
        n_initial_filters=2,
        sparse=True,
        distributed=True,
        file="/lus/theta-fs0/projects/datascience/cadams/wire_pixel_preprocessed_files_split/train_event_id.root",
        aux_file="/lus/theta-fs0/projects/datascience/cadams/wire_pixel_preprocessed_files_split/test_event_id.root",
        label_mode="split",
        minibatch_size=batch_size,
        aux_minibatch_size=batch_size,
        iterations=26,
        compute_mode="GPU",
        log_directory="/home/cadams/Cooley/DeepLearnPhysics/SparseEventIDLogs/{}/{}/{}".format(
            dimension, name, workflow),
        checkpoint_directory="/lus/theta-fs0/projects/datascience/cadams/SparseEventIDChpts/{}/{}/{}".format(
            dimension, name, workflow)
        )


    return

    # We need to spawn a job, so lets build up some kwargs
    
    num_nodes=1
    wall_time_minutes=60
    name="test_job_flow_3D"
    workflow="sparse_eventID_testing"
    dimension="3D"

    batch_size = num_nodes * 2 * 8

    # The rest of this is kwargs

    job = spawn_training_job(
        num_nodes=num_nodes, 
        wall_time_minutes=wall_time_minutes,
        name=name, 
        workflow=workflow, 
        dimension=dimension, 
        args=None,

        learning_rate=0.01,
        checkpoint_iteration=100,
        n_initial_filters=2,
        sparse=True,
        distributed=True,
        file="/lus/theta-fs0/projects/datascience/cadams/wire_pixel_preprocessed_files_split/train_event_id.root",
        aux_file="/lus/theta-fs0/projects/datascience/cadams/wire_pixel_preprocessed_files_split/test_event_id.root",
        label_mode="split",
        minibatch_size=batch_size,
        aux_minibatch_size=batch_size,
        iterations=2000,
        compute_mode="GPU",
        log_directory="/home/cadams/Cooley/DeepLearnPhysics/SparseEventIDLogs/{}/{}/{}".format(
            dimension, name, workflow),
        checkpoint_directory="/lus/theta-fs0/projects/datascience/cadams/SparseEventIDChpts/{}/{}/{}".format(
            dimension, name, workflow)
        )

    print (job)

if __name__ == '__main__':
    main()
