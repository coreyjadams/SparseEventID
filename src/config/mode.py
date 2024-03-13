from enum import Enum

from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from . optimizer import Optimizer

class ModeKind(Enum):
    train     = 0
    iotest    = 1
    inference = 2
    # visualize = 3

@dataclass
class Mode:
    name:             ModeKind = ModeKind.train
    no_summary_images:    bool = True
    weights_location:     str  = ""
    restore_encoder_only: bool = True

@dataclass
class Train(Mode):
    checkpoint_iteration:   int = 500
    summary_iteration:      int = 1
    logging_iteration:      int = 1
    optimizer:        Optimizer = field( default_factory= lambda : Optimizer() )
    quantization_aware:    bool = False
    weight_decay:         float = 0.00

@dataclass
class Inference(Mode):
    name:         ModeKind = ModeKind.inference
    start_index:       int = 0
    summary_iteration: int = 1
    logging_iteration: int = 1

@dataclass
class IOTest(Mode):
    name:   ModeKind = ModeKind.iotest
    start_index: int = 0

# @dataclass
# class Visualize(Mode):
#     name:    ModeKind = ModeKind.visualize
#     start_index:  int = 0
#     save_images: bool = False


cs = ConfigStore.instance()
cs.store(group="mode", name="train",     node=Train)
cs.store(group="mode", name="inference", node=Inference)
cs.store(group="mode", name="iotest",    node=IOTest)
# cs.store(group="mode", name="visualize", node=Visualize)
