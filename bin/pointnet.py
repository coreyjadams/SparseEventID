#!/usr/bin/env python


# Add the local folder to the import path:
import os,sys
network_dir = os.path.dirname(os.path.abspath(__file__))
print(network_dir)
network_dir = os.path.dirname(network_dir)
print(network_dir)
sys.path.insert(0,network_dir)

# import the necessary
from network import flags

def main():
    f = flags.FLAGS
    f.parse_args()  
    f.dump_config()

    print(f.COMPUTE_MODE)

    if f.DISTRIBUTED:
        from network import distributed_trainer
        trainer = distributed_trainer.distributed_trainer(f)
    else:
        from network import trainercore
        trainer = trainercore.trainercore(f)
    
    trainer.initialize()
    trainer.batch_process()

if __name__ == '__main__':
  main()