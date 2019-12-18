# Sparse Event ID

This repository contains a set of models that work on sparse data in pytorch. There are several dependencies for some models (Like Submanifold sparse convolutions)

In general, all models share the same tools for IO, distributed training, saving and restoring, and doing evaluations of test, train, and inference steps.



### Submanifold sparse convolution ResNet

Implementation of a standard resnet architecture with submanifold sparse convolutions. Available for 2D and 3D.  In 2D, its possible to use the multiplane archicture with shared weights across planes, which merges downstream.

(Coming eventually)
### Dynamic Graph Convolutional Neural Network

### Point Net

# Dependencies
 - The IO model is based on larcv3, so you need larcv3 data files to run this.  An open dataset is availalbe: https://osf.io/kbn5a/
 - The sparse convolution requires the submanifold sparse convolution functions, from facebook.
   - In turn, this has a dependency on sparsehash.
 - Pytorch is the neural network framework
 - To run in a distrubuted way, horovod and mpi4py are required. 

