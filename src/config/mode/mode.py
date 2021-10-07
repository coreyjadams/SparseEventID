from enum import Enum

from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from .optimizer import Optimizer

class ModeKind(Enum):
    training  = 0
    iotest    = 1
    inference = 2

@dataclass
class Mode:
    name: ModeKind = ModeKind.training

class LearningSchedule(Enum):
    flat = 0
    1cycle = 1
    triangle = 2
    decay = 3
    expincrease = 4



@dataclass
class Train(Mode):
    checkpoint_iteration: int =  500
    summary_iteration:  int = 1
    no_summary_images: bool = False
    logging_iteration: int = 1
    optimizer: Optimizer = Optimizer()

@dataclass
class Inference(Mode):
    start_index: int = 0


cs = ConfigStore.instance()
cs.store(group="mode", name="train", node=Train)
cs.store(group="mode", name="inference", node=Inference)
