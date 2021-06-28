
from enum import Enum

from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

class LossBalanceScheme(Enum):
    none  = 0
    focal = 1

class OptimizerKind(Enum):
    adam    = 0
    rmsprop = 1
    sgd     = 2




@dataclass 
class Optimizer:
    learning_rate:         float             =  0.0003
    loss_balance_scheme:   LossBalanceScheme = LossBalanceScheme.focal
    name:                  OptimizerKind     = OptimizerKind.adam
    gradient_accumulation: int               = 1
    

cs = ConfigStore.instance()
cs.store(name="optimizer", node=Optimizer)
