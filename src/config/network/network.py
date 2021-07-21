from enum import Enum

from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


class DataFormat(enum):
    channels_first = 0
    channels_last  = 1
    sparse         = 2
    graph          = 3

@dataclass
class Network:
    name: str = MISSING

@dataclass
class PointNet(Network):
    name:        str        = "pointnet"
    data_format: DataFormat = DataFormat.graph

@dataclass
class ResNet(Network):
    name:        str        = "resnet"
    data_format: DataFormat = DataFormat.sparse
    n_initial_filters:    int  = MISSING
    res_blocks_per_layer: int  = MISSING
    network_depth:        int  = MISSING
    batch_norm:           bool = MISSING
    leaky_relu:           bool = MISSING
    depth_pre_merge:      int  = MISSING


@dataclass:
class DGCNN(Network):
    name:        str:       = "dgcnn"
    data_format: DataFormat = DataFormat.graph
    k:           int        = MISSING
    emb_dims:    int        = MISSING
    dropout:     float      = MISSING

cs = ConfigStore.instance()
cs.store(group="network", name="pointnet", node=PointNet)
cs.store(group="network", name="resnet",   node=ResNet)
cs.store(group="network", name="dgcnn",   node=DGCNN)
