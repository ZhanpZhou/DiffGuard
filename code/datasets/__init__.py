"""This package includes all the modules related to data loading and preprocessing

 To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
 You need to implement four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point from data loader.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.

Now you can use the dataset class by specifying flag '--dataset_mode dummy'.
See our template dataset class 'template_dataset.py' for more details.
"""
import importlib
from random import sample
from .base_dataset import BaseDataset
import torch
from torch.utils.data import DataLoader


def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "datasets." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataset(opt, data_type='train'):
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from datasets import create_dataset
        >>> dataset = create_dataset(opt)
    """
    data_loader = CustomDatasetDataLoader(opt, data_type)
    dataset = data_loader.load_data()
    return dataset


class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt, data_type, rank=0, **kwargs):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        if data_type == 'train':
            dataset_class = find_dataset_using_name(opt.dataset_mode)
            batch_size = opt.batch_size // opt.world_size if opt.parallel_mode == 'distributed' else opt.batch_size
            self.dataset = dataset_class(opt,data_type)
            opt.dataset = self
            shuffle = not opt.serial_batches and opt.parallel_mode != 'distributed'
            drop_last = (opt.drop_last == 1)
        else:
            dataset_class = find_dataset_using_name(opt.v_dataset_mode)
            batch_size = opt.v_batch_size
            self.dataset = dataset_class(opt,data_type)
            opt.v_dataset = self
            shuffle = False
            drop_last = False

        if data_type == 'train' and opt.parallel_mode == 'distributed':
            self.sampler = torch.utils.data.distributed.DistributedSampler(self.dataset,
                                                                    num_replicas=len(opt.gpu_ids),
                                                                    rank=rank)
        else:
            self.sampler = None
        
        print("dataset [%s] was created" % type(self.dataset).__name__)

        collate_fn = self.dataset.get_collate_fn() if hasattr(self.dataset, 'collate_fn') else None
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=self.sampler,
            num_workers=int(opt.num_threads),
            collate_fn=collate_fn,
            pin_memory=False,
            drop_last=drop_last)

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return len(self.dataset)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            yield data
