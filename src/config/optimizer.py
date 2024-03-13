
from enum import Enum

from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from typing import List, Any
from omegaconf import MISSING

class LossBalanceScheme(Enum):
    none  = 0
    even  = 1
    focal = 2

class OptimizerKind(Enum):
    adam     = 0
    rmsprop  = 1
    sgd      = 2
    adagrad  = 3
    adadelta = 4
    lars     = 5
    lamb     = 6
    novograd = 7

@dataclass
class LRScheduleConfig:
    name:                 str = ""
    peak_learning_rate: float = 3e-3

@dataclass
class OneCycleConfig(LRScheduleConfig):
    name:                 str = "one_cycle"
    min_learning_rate:  float = 1e-5
    decay_floor:        float = 1e-5
    decay_epochs:         int = 5

@dataclass
class WarmupFlatDecayConfig(LRScheduleConfig):
    name:                 str = "standard"
    decay_floor:        float = 1e-3
    decay_epochs:         int = 5

@dataclass
class FlatLR(LRScheduleConfig):
    name:                 str = "flat"

@dataclass
class Optimizer:
    lr_schedule:          LRScheduleConfig = field(default_factory= lambda:WarmupFlatDecayConfig())
    loss_balance_scheme: LossBalanceScheme = LossBalanceScheme.focal
    name:                    OptimizerKind = OptimizerKind.adam
    gradient_accumulation:             int = 1
    weight_decay:                    float = 1e-6

cs = ConfigStore.instance()

cs.store(group="lr_schedule", name="flat",      node=FlatLR)
cs.store(group="lr_schedule", name="one_cycle", node=OneCycleConfig)
cs.store(group="lr_schedule", name="standard",  node=WarmupFlatDecayConfig)
cs.store(name="optimizer", node=Optimizer)
