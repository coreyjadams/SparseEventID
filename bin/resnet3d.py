#!/usr/bin/env python
import os,sys,signal
import time

import numpy

# Add the local folder to the import path:
network_dir = os.path.dirname(os.path.abspath(__file__))
network_dir = os.path.dirname(network_dir)
sys.path.insert(0,network_dir)


# import the necessary
from src.utils import flags
# from src.networks import resnet3d
from src.networks import sparseresnet3d


def main():

    FLAGS = flags.resnet3D()
    FLAGS.parse_args()
    # FLAGS.dump_config()

    


    

    if FLAGS.MODE is None:
        raise Exception()

    if FLAGS.DISTRIBUTED:
        from src.utils import distributed_trainer

        trainer = distributed_trainer.distributed_trainer()
    else:
        from src.utils import trainercore
        trainer = trainercore.trainercore()
        
    if FLAGS.MODE == 'train' or FLAGS.MODE == 'inference':
        if FLAGS.SPARSE:
            net = sparseresnet3d.ResNet
        else:
            net = resnet.ResNet
        FLAGS.set_net(net)
        trainer.initialize()
        trainer.batch_process()

    if FLAGS.MODE == 'iotest':
        trainer.initialize(io_only=True)

        time.sleep(0.1)

        times = []

        for i in range(FLAGS.ITERATIONS):
            start = time.time()
            mb = trainer.fetch_next_batch()
            end = time.time()
            if not FLAGS.DISTRIBUTED:
                print(i, ": Time to fetch a minibatch of data: {}".format(end - start))
                times.append(end - start)
            else:
                if trainer._rank == 0:
                    print(i, ": Time to fetch a minibatch of data: {}".format(end - start))
                    times.append(end - start)
            time.sleep(0.5)

        if not FLAGS.DISTRIBUTED:
            print ("Average time to fetch a minibatch of data: {} +- {} seconds.".format(numpy.array(times).mean(), numpy.array(times).std()))
        else:
            if trainer._rank == 0: 
                print ("Average time to fetch a minibatch of data: {} +- {} seconds, (median: {}).".format(numpy.array(times).mean(), numpy.array(times).std(), numpy.median(numpy.array(times))))

    if not FLAGS.DISTRIBUTED:
        trainer.stop()

if __name__ == '__main__':
    main()
