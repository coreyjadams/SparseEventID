#!/usr/bin/env python
import os,sys
import time


# Add the local folder to the import path:
network_dir = os.path.dirname(os.path.abspath(__file__))
network_dir = os.path.dirname(network_dir)
sys.path.insert(0,network_dir)

# import the necessary
from network.flags import FLAGS

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

        for i in range(FLAGS.ITERATIONS):
            start = time.time()
            _ = trainer.fetch_next_batch()
            end = time.time()
            if not FLAGS.DISTRIBUTED:
                print("Time to fetch a minibatch of data: {}".format(end - start))
            else:
                if trainer._rank == 0:
                    print("Time to fetch a minibatch of data: {}".format(end - start))

if __name__ == '__main__':
  main()