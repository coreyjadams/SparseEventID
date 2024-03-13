from enum import Enum

from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

class DistributedMode(Enum):
    DDP       = 0
    horovod   = 1
    DeepSpeed = 2

class DataMode(Enum):
    dense  = 0
    sparse = 1
    graph  = 2

@dataclass
class Framework:
    name:      str = MISSING
    mode: DataMode = DataMode.sparse

@dataclass
class Lightning(Framework):
    name:             str             = "lightning"
    distributed_mode: DistributedMode = DistributedMode.DDP
    oversubscribe:                int = 1

cs = ConfigStore.instance()
cs.store(group="framework", name="lightning", node=Lightning)
