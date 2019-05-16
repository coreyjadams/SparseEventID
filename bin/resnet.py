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


def main():

    FLAGS = flags.resnet()
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
            from src.networks import sparseresnet
            net = sparseresnet.ResNet
        else:
            from src.networks import resnet
            net = resnet.ResNet
        FLAGS.set_net(net)
        trainer.initialize()
        trainer.batch_process()

    if FLAGS.MODE == 'iotest':
        trainer.initialize(io_only=True)

        label_stats = numpy.zeros((36,))

        print("running")
        time.sleep(0.1)
        for i in range(FLAGS.ITERATIONS):
            start = time.time()
            mb = trainer.fetch_next_batch()
            label_stats += numpy.sum(mb['label'], axis=0)

            end = time.time()
            if not FLAGS.DISTRIBUTED:
                print(i, ": Time to fetch a minibatch of data: {}".format(end - start))
            else:
                if trainer._rank == 0:
                    print(i, ": Time to fetch a minibatch of data: {}".format(end - start))
            # time.sleep(0.5)
        print(label_stats)


    trainer.stop()

if __name__ == '__main__':
    main()  