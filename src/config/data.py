from enum import Enum

from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from typing import Tuple, Any
from omegaconf import MISSING


class LabelType(Enum):
    Classification = 0
    Segmentation   = 1


class AccessMode(Enum):
    serial_access = 0
    random_blocks = 1
    random_events = 2

class Detector(Enum):
    dune2d = 0
    dune3d = 1

@dataclass
class Data:
    name:        str = ""
    label:      bool = True
    vertex:     bool = False
    mode: AccessMode = AccessMode.random_events
    seed:        int = -1
    train:       str = ""
    test:        str = ""
    val:         str = ""
    image_key:   str = ""
    active: Tuple[str] = field(default_factory=list)
    normalize:  bool = True 
    transform1: bool = False
    transform2: bool = False
    dimension:   int = 3
    images:      int = 1
    mc:         bool = True



# @dataclass
# class Dataset:
#     dimension:      int        = 3
#     imgaes:         int        = 1
#     label:          LabelType  = LabelType.Classification
#     access_mode:    AccessMode = AccessMode.random_blocks
#     data_directory: str        = MISSING
#     train_file:     str        = ""
#     test_file:      str        = ""
#     val_file:       str        = ""

@dataclass
class dune2d(Data):
    train:          str        = "/data/datasets/DUNE/pixsim_small/train.h5"
    test:           str        = "/data/datasets/DUNE/pixsim_small/test.h5"
    val:            str        = "/data/datasets/DUNE/pixsim_small/test.h5"
    dimension:      int        = 2
    images:         int        = 3
    image_key:      str        = "dunevoxels"
    detector:       Detector   = Detector.dune2d

# @dataclass
# class dune3d(Dataset):
#     data_directory: str        = MISSING
#     file:           str        = "test.h5"
#     aux_file:       str        = "test.h5"
#     dimension:      int        = 3
#     detector:       Detector   = Detector.dune3d
    

# @dataclass
# class next_new(Dataset):
#     data_directory: str        = MISSING
#     file:           str        = "NEXT_White_Tl_train.h5"
#     aux_file:       str        = "NEXT_White_Tl_test.h5"
#     dimension:      int        = 3


cs = ConfigStore.instance()
cs.store(group="data", name="dune2d",   node=dune2d)
# cs.store(group="dataset", name="dune3d",   node=dune3d)
# cs.store(group="dataset", name="next_new", node=next_new)