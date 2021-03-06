from enum import Enum

from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

class DistributedMode(Enum):
    DDP     = 0
    horovod = 1

@dataclass
class Framework:
    name: str = MISSING

class DataFormat(Enum):
    channels_first = 0
    channels_last = 1


@dataclass
class Tensorflow(Framework):
    name:                         str = "tensorflow"
    inter_op_parallelism_threads: int = 2
    intra_op_parallelism_threads: int = 24
    data_format:                  DataFormat = DataFormat.channels_first

@dataclass
class Torch(Framework):
    name:             str             = "torch"
    distributed_mode: DistributedMode = DistributedMode.DDP

cs = ConfigStore.instance()
cs.store(group="framework", name="tensorflow", node=Tensorflow)
cs.store(group="framework", name="torch", node=Torch)
