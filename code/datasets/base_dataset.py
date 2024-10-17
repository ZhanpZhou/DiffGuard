"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.

It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod
from imgaug import augmenters as iaa
from torchvision.transforms.functional import normalize
import torch

class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt, task):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        assert task in ['train', 'valid', 'test']
        self.task = task
        
        if self.task == 'train':
            dataset_id = opt.dataset_id
        else:
            dataset_id = opt.v_dataset_id

        datasets = getattr(self.opt, self.task+'_datasets')[dataset_id]
        self.datasets = []
        for dataset in datasets:
            try:
                o = eval(dataset)
                assert isinstance(o, str)
            except:
                o = dataset
            self.datasets.append(o)
            
        try:
            self.mean, self.std = self.norm_mean_std(self.opt.data_norm_type)
        except:
            self.mean, self.std = None, None

    @staticmethod
    def modify_commandline_options(parser):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass

    def norm_mean_std(self, data_norm_type):
        if data_norm_type == 'imagenet':
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        elif data_norm_type == 'mmdetection':
            mean = [123.675 / 255, 116.28 / 255, 103.53 / 255]
            std = [58.395 / 255, 57.12 / 255, 57.375 / 255]
        elif data_norm_type == 'gray':
            mean = [0.5]
            std = [0.5]
        elif data_norm_type == 'activitynet':
            mean = [0.4477, 0.4209, 0.3906]
            std = [0.2767, 0.2695, 0.2714]
        elif data_norm_type == 'kinetics':
            mean = [0.4345, 0.4051, 0.3775]
            std = [0.2768, 0.2713, 0.2737]
        elif data_norm_type == 'competition':
            mean = [0.456, 0.456, 0.456]
            std = [0.224, 0.224, 0.224]
        else:
            mean = None
            std = None
        return mean, std

    def get_class_label(self, label:int, head_index=0) -> int:
        l = str(label)
        for i, s in enumerate(self.opt.classes[head_index]):
            if l in s:
                return i
        return None

    @abstractmethod
    def prepare_dataset(self):
        pass

    def prepare_new_epoch(self):        
        if self.task == 'train':
            self.sample_list = []
            if self.opt.sample_strategy == 'original':
                for k, v in self.samples.items():
                    for sample in v:
                        self.sample_list.append((sample, k))
            elif self.opt.sample_strategy == 'resample':
                max_class_num = max(len(v) for k, v in self.samples.items())
                for k, v in self.samples.items():
                    if len(v) > 0:
                        repeat = max_class_num // len(v)
                        candidates = [(sample, k) for sample in v]
                        self.sample_list.extend(candidates * repeat)
                        self.sample_list.extend(random.sample(candidates,  max_class_num % len(v)))
            elif self.opt.sample_strategy == 'resample_posmax':
                max_class_num = max(len(v) for k, v in self.samples.items() if k not in (0, '0', 'normal'))
                for k, v in self.samples.items():
                    if len(v) > 0:
                        repeat = max_class_num // len(v)
                        candidates = [(sample, k) for sample in v]
                        self.sample_list.extend(candidates * repeat)
                        self.sample_list.extend(random.sample(candidates,  max_class_num % len(v)))
            elif self.opt.sample_strategy == 'resample_ratio':
                assert len(self.opt.classes[0]) == len(self.opt.resample_ratios)
                max_unit_num = 0
                for i, k in enumerate(self.samples.keys()):
                    v = self.samples[k]
                    if len(v) > 0:
                        max_unit_num = max(max_unit_num, int(len(v) / self.opt.resample_ratios[i]))
                for i, k in enumerate(self.samples.keys()):
                    v = self.samples[k]
                    if len(v) > 0:
                        total = int(max_unit_num * self.opt.resample_ratios[i])
                        repeat = total // len(v)
                        candidates = [(sample, k) for sample in v]
                        self.sample_list.extend(candidates * repeat)
                        self.sample_list.extend(random.sample(candidates,  total % len(v)))
            elif self.opt.sample_strategy == 'weighted':
                assert len(self.opt.classes[0]) == len(self.opt.sample_weights)
                for i, k in enumerate(self.samples.keys()):
                    v = self.samples[k]
                    if len(v) > 0:
                        total = int(len(v) * self.opt.sample_weights[i])
                        repeat = total // len(v)
                        candidates = [(sample, k) for sample in v]
                        self.sample_list.extend(candidates * repeat)
                        self.sample_list.extend(random.sample(candidates,  total % len(v)))
                    
            random.shuffle(self.sample_list)

    def get_collate_fn(self):
        return self.collate_fn

    def norm_data(self, tensor, inplace=True, data_norm_type=None):
        """Normalize a tensor image with mean and standard deviation.

        .. note::
            This transform acts out of place by default, i.e., it does not mutates the input tensor.

        Args:
            tensor (Tensor): Tensor image of size (C, ...) to be normalized.
            mean (sequence): Sequence of means for each channel.
            std (sequence): Sequence of standard deviations for each channel.
            inplace(bool,optional): Bool to make this operation inplace.

        Returns:
            Tensor: Normalized Tensor image.
        """

        if not inplace:
            tensor = tensor.clone()

        if data_norm_type is None:
            data_norm_type = self.opt.data_norm_type
            mean, std = self.mean, self.std
        else:
            mean, std = self.norm_mean_std(data_norm_type)

        if data_norm_type == 'original':
            return tensor

        # if data_norm_type in ['imagenet', 'gray', 'normal', 'activitynet', 'kinetics', 'competition']:
        #     tensor.div_(255.0)
        
        dtype = tensor.dtype
        mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
        std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
        if tensor.ndim == 3:
            tensor.sub_(mean[:, None, None]).div_(std[:, None, None])
        elif tensor.ndim == 4:
            tensor.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])
        else:
            raise AssertionError('invalid number of tensor dims')
        return tensor

    def norm_data_in_batch(self, tensor, inplace=True, data_norm_type=None):
        """Normalize tensor images with mean and standard deviation.

        .. note::
            This transform acts out of place by default, i.e., it does not mutates the input tensor.

        Args:
            tensor (Tensor): Tensor image of size (B, C, ...) to be normalized.
            mean (sequence): Sequence of means for each channel.
            std (sequence): Sequence of standard deviations for each channel.
            inplace(bool,optional): Bool to make this operation inplace.

        Returns:
            Tensor: Normalized Tensor image.
        """

        if not inplace:
            tensor = tensor.clone()

        if data_norm_type is None:
            data_norm_type = self.opt.data_norm_type
            mean, std = self.mean, self.std
        else:
            mean, std = self.norm_mean_std(data_norm_type)

        if data_norm_type == 'original':
            return tensor

        if data_norm_type in ['imagenet', 'gray', 'normal', 'activitynet', 'kinetics', 'competition']:
            tensor.div_(255.0)
        
        dtype = tensor.dtype
        mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
        std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
        if tensor.ndim == 4:
            tensor.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
        elif tensor.ndim == 5:
            tensor.sub_(mean[None, :, None, None, None]).div_(std[None, :, None, None, None])
        else:
            raise AssertionError('invalid number of tensor dims')
        return tensor

    def cal_target(self, model):
        return model.instance_ys[0], model.instance_scores[0], model.instance_preds[0]
