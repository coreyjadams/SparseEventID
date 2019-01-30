#!/bin/bash

# This script adds balsam jobs for this folder's python scripts.

balsam app --name 'event-ID-2D-train' \
--desc 'Run 2D eventID training inside a pytorch singularity image' \
--preprocess /home/cadams/Cooley/DeepLearnPhysics/PointNetEventID/balsam/preprocess_train.py \
--postprocess /home/cadams/Cooley/DeepLearnPhysics/PointNetEventID/balsam/postprocess_train.py \
--exec 'singularity exec --nv -B /lus:/lus /home/cadams/public/centos-cuda-torch-mpich-root python /home/cadams/Cooley/DeepLearnPhysics/PointNetEventID/bin/resnet.py train'

balsam app --name 'event-ID-2D-inference' \
--desc 'Run 2D eventID inference inside a pytorch singularity image' \
--preprocess /home/cadams/Cooley/DeepLearnPhysics/PointNetEventID/balsam/preprocess_inference.py \
--postprocess /home/cadams/Cooley/DeepLearnPhysics/PointNetEventID/balsam/postprocess_inference.py \
--exec 'singularity exec --nv -B /lus:/lus /home/cadams/public/centos-cuda-torch-mpich-root python /home/cadams/Cooley/DeepLearnPhysics/PointNetEventID/bin/resnet.py inference'

balsam app --name 'event-ID-3D-train' \
--desc 'Run 3D eventID training inside a pytorch singularity image' \
--preprocess /home/cadams/Cooley/DeepLearnPhysics/PointNetEventID/balsam/preprocess_train.py \
--postprocess /home/cadams/Cooley/DeepLearnPhysics/PointNetEventID/balsam/postprocess_train.py \
--exec 'singularity exec --nv -B /lus:/lus /home/cadams/public/centos-cuda-torch-mpich-root python /home/cadams/Cooley/DeepLearnPhysics/PointNetEventID/bin/resnet3d.py train'

balsam app --name 'event-ID-3D-inference' \
--desc 'Run 3D eventID inference inside a pytorch singularity image' \
--preprocess /home/cadams/Cooley/DeepLearnPhysics/PointNetEventID/balsam/preprocess_inference.py \
--postprocess /home/cadams/Cooley/DeepLearnPhysics/PointNetEventID/balsam/postprocess_inference.py \
--exec 'singularity exec --nv -B /lus:/lus /home/cadams/public/centos-cuda-torch-mpich-root python /home/cadams/Cooley/DeepLearnPhysics/PointNetEventID/bin/resnet3d.py inference'