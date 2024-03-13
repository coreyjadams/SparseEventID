import torch

from torch.utils import data

class TorchLarcvDataset(data.IterableDataset):

    def __init__(self, larcv_dataset):
        super(TorchLarcvDataset).__init__()
        self.ds = larcv_dataset

    def __iter__(self):
        for batch in self.ds:
            yield batch

def create_torch_larcv_dataloader(larcv_ds):

    ids =  TorchLarcvDataset(larcv_ds)

    torch_dl = data.DataLoader(ids, 
        num_workers    = 0, 
        batch_size    = None, 
        batch_sampler = None,
        pin_memory    = True)

    return torch_dl