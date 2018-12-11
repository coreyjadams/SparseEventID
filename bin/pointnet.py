#!/usr/bin/env python
import os,sys,signal
import time

import numpy

# Add the local folder to the import path:
network_dir = os.path.dirname(os.path.abspath(__file__))
network_dir = os.path.dirname(network_dir)
sys.path.insert(0,network_dir)

# import the necessary
from python.flags import FLAGS


def main():
    FLAGS.parse_args()
    # FLAGS.dump_config()



    if FLAGS.MODE is None:
        raise Exception()

    if FLAGS.DISTRIBUTED:
        from network import distributed_trainer

        trainer = distributed_trainer.distributed_trainer()
    else:
        from network import trainercore
        trainer = trainercore.trainercore()
        
    if FLAGS.MODE == 'train':
        trainer.initialize()
        trainer.batch_process()

    if FLAGS.MODE == 'iotest':
        trainer.initialize(io_only=True)

        print("running")
        time.sleep(0.1)
        for i in range(FLAGS.ITERATIONS):
            start = time.time()
            mb = trainer.fetch_next_batch()
            print(mb['image'].shape)
            print(mb.keys)
            # print(mb['image'][0])
            print("Number of non zero elements: ", numpy.count_nonzero(mb['image']))
            end = time.time()
            if not FLAGS.DISTRIBUTED:
                print(i, ": Time to fetch a minibatch of data: {}".format(end - start))
            else:
                if trainer._rank == 0:
                    print(i, ": Time to fetch a minibatch of data: {}".format(end - start))
            # time.sleep(0.5)

    trainer.stop()

if __name__ == '__main__':
    main()