from enum import Enum

from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from .network   import Network
from .mode      import Mode
from .framework import Framework
from .data      import Data

class ComputeMode(Enum):
    CPU   = 0
    CUDA  = 1
    XPU   = 2

class Precision(Enum):
    float32  = 0
    mixed    = 1
    bfloat16 = 2
    float16  = 3

@dataclass
class Run:
    distributed:        bool        = True
    compute_mode:       ComputeMode = ComputeMode.GPU
    length:             int         = MISSING
    minibatch_size:     int         = MISSING
    id:                 int         = MISSING
    precision:          Precision   = Precision.float32
    profile:            bool        = False
    world_size:         int         = 1


cs = ConfigStore.instance()

cs.store(group="run", name="base_run", node=Run)

cs.store(
    name="disable_hydra_logging",
    group="hydra/job_logging",
    node={"version": 1, "disable_existing_loggers": False, "root": {"handlers": []}},
)



defaults = [
    {"run"       : "base_run"},
    {"mode"      : "train"},
    {"data"      : "mc_tl208"},
    {"framework" : "lightning"},
    {"encoder"   : "convnet"}
]

@dataclass
class LearnRepresentation:
    defaults: List[Any] = field(default_factory=lambda: defaults)


    run:        Run       = MISSING
    mode:       Mode      = MISSING
    data:       Data      = MISSING
    framework:  Framework = MISSING
    encoder:    Representation = MISSING
    head:       ClassificationHead = field(default_factory= lambda : ClassificationHead())
    output_dir: str       = "output/"
    name:       str       = "simclr"

@dataclass
class DetectVertex:
    defaults: List[Any] = field(default_factory=lambda: defaults)


    run:        Run       = MISSING
    mode:       Mode      = MISSING
    data:       Data      = MISSING
    framework:  Framework = MISSING
    encoder:    Representation = MISSING
    head:       YoloHead  = field(default_factory= lambda : YoloHead())
    output_dir: str       = "output/"
    name:       str       = "yolo"



@dataclass
class SupervisedClassification:
    defaults: List[Any] = field(default_factory=lambda: defaults)


    run:        Run       = MISSING
    mode:       Mode      = MISSING
    data:       Data      = MISSING
    framework:  Framework = MISSING
    encoder:    Representation = MISSING
    head:       ClassificationHead = field(default_factory= lambda : ClassificationHead())
    output_dir: str       = "output/"
    name:       str       = "supervised_eventID"

@dataclass
class Config:
    defaults: List = field(
        default_factory=lambda: [
            {"hydra/job_logging": "disable_hydra_logging"},
        ]
    )
    network:    network.Network     = MISSING
    framework:  framework.Framework = MISSING
    dataset:    dataset.Dataset     = MISSING


cs.store(name="representation",            node=LearnRepresentation)
cs.store(name="supervised_classification", node=SupervisedClassification)
cs.store(name="detect_vertex",             node=DetectVertex)
