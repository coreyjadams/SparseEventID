from enum import Enum

from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from typing import Tuple, Any, List
from omegaconf import MISSING


class GrowthRate(Enum):
    multiplicative = 0
    additive       = 1

class DownSampling(Enum):
    convolutional = 0
    pooling       = 1

class Norm(Enum):
    none  = 0
    batch = 1
    layer = 2
    group = 3

@dataclass
class Repr:
    depth:             int   = 3
    n_initial_filters: int   = 32
    n_output_filters:  int   = 128


@dataclass
class ConvRepresentation(Repr):
    normalization:        Norm         = Norm.batch
    bias:                 bool         = True
    blocks_per_layer:     int          = 4
    residual:             bool         = True
    filter_size:          int          = 3
    growth_rate:          GrowthRate   = GrowthRate.additive
    downsampling:         DownSampling = DownSampling.convolutional


@dataclass
class MLPConfig():
    layers:     List[int] = field(default_factory=lambda: [16,])
    bias:            bool = True

@dataclass
class GraphRepresentation(Repr):
    mlp_config: MLPConfig = field(default_factory= lambda : MLPConfig(layers=[32,32]))
    graph_layer:      str = "GINConv"



@dataclass
class MLP:
    layers: Tuple[int] = field(default_factory=list)

@dataclass
class YoloHead:
    layers: Tuple[int] = field(default_factory=list)

cs = ConfigStore.instance()
cs.store(group="encoder", name="convnet", node=ConvRepresentation)
cs.store(group="encoder", name="gnn",     node=GraphRepresentation)
cs.store(group="head", name="mlp",        node=MLP)
cs.store(group="head", name="yolo",       node=YoloHead)
