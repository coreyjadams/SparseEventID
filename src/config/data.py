from enum import Enum

from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


class LabelType(Enum):
    Classification = 0
    Segmentation   = 1


class RandomAccessMode(enum):
    serial_access = 0
    random_blocks = 1
    random_events = 2

@dataclass
class Data:
    name:        str = ""
    label:      bool = True
    vertex:     bool = False
    mode: RandomMode = RandomMode.random_events
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
    images       int = 1



@dataclass
class Dataset:
    dimension:      int        = 3
    imgaes:         int        = 1
    label:          LabelType  = LabelType.Classification
    access_mode:    RandomAccessMode = RandomAccessMode.random_blocks
    data_directory: str        = MISSING
    train_file:     str        = ""
    test_file:      str        = ""
    val_file:       str        = ""

@dataclass
class dune2d(Dataset):
    data_directory: str        = MISSING
    file:           str        = "test.h5"
    aux_file:       str        = "test.h5"
    dimension:      int        = 2
    images:         int        = 3

@dataclass
class dune3d(Dataset):
    data_directory: str        = MISSING
    file:           str        = "test.h5"
    aux_file:       str        = "test.h5"
    dimension:      int        = 3

@dataclass
class next_new(Dataset):
    data_directory: str        = MISSING
    file:           str        = "NEXT_White_Tl_train.h5"
    aux_file:       str        = "NEXT_White_Tl_test.h5"
    dimension:      int        = 3


cs = ConfigStore.instance()
cs.store(group="dataset", name="dune2d",   node=dune2d)
cs.store(group="dataset", name="dune3d",   node=dune3d)
cs.store(group="dataset", name="next_new", node=next_new)