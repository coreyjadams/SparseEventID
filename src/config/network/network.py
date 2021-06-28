from enum import Enum

from dataclasses import dataclass
from hydra.core.config_store import ConfigStore


class DataFormat(enum):
    channels_first = 0
    channels_last  = 1
    sparse         = 2
    graph          = 3

@dataclass 
class PointNet:
    name:   str = "pointnet"
    data_format: DataFormat = DataFormat.graph


cs = ConfigStore.instance()
cs.store(name="pointnet", node=pointnet)