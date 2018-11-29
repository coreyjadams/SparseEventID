# Sparse Event ID

This repository contains a set of models that work on sparse data in pytorch. There are several dependencies for some models (Like Submanifold sparse convolutions)

In general, all models share the same tools for IO, distributed training, saving and restoring, and doing evaluations of test, train, and inference steps.

The models included here are (initially):

### PointNet

Implementation of PointNet for event identification in neutrino events in  lartpcs.  Implementation is available for both 2D and 3D networks.


### Submanifold sparse convolution ResNet

Implementation of a standard resnet architecture with submanifold sparse convolutions. Available for 2D and 3D.  In 2D, its possible to use the multiplane archicture with shared weights across planes, which merges downstream.

(Coming eventually)
### Dynamic Graph Convolutional Neural Network

# Dependencies
 - The IO model is based on larcv, so you need larcv data files to run this.  
 - The sparse convolution requires the submanifold sparse convolution functions, from facebook.
 - Pytorch is the neural network framework
 - To run in a distrubuted way, horovod and mpi4py are required. 

# Running

Run 
```
bin/pointnet.py train [options]
```

to execute the point net algorithm.  Similarly, you can run pointnet_seg, dgcnn, dgcnn_seg.