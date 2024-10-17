import json
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import torch
from torchvision import transforms
from .base_dataset import BaseDataset
import util.augmentation as aug

class CTSliceDiffusionDataset(BaseDataset):
    def __init__(self, opt, task):
        super().__init__(opt, task)
        self.prepare_dataset()
        self.prepare_new_epoch()

    def get_class_label(self, record) -> int:
        # class label for the scan
        for i, l in enumerate(self.opt.classes[0]):
            if record['label'] in l:
                return i

    def prepare_dataset(self):
        all_samples = []
        for dataset in self.datasets:
            with open(dataset, 'r') as f:
                records = json.load(f)
            for record in records:
                class_label = self.get_class_label(record)
                if class_label is None: 
                    continue
                if self.opt.positive_only and class_label == 0:
                    continue
                all_samples.append((record, class_label))

        if self.task == 'train':
            self.samples = dict()
            for c in range(len(self.opt.classes[0])):
                self.samples[c] = []
            for sample in all_samples:
                self.samples[sample[-1]].append(sample)
            if self.opt.positive_only:
                self.samples[0] = []
            for k, v in self.samples.items():
                print('sample num of class {}:'.format(k), len(v))
        else:
            self.sample_list = []
            for sample in all_samples:
                if self.opt.positive_only:
                    if sample[-1] != 0:
                        self.sample_list.append((sample, sample[-1]))
                else:
                    self.sample_list.append((sample, sample[-1]))
            if self.opt.test_samples:
                self.sample_list = [self.sample_list[idx] for idx in self.opt.test_samples]

    def _load_sample(self, sample):
        record, class_label = sample
        image_path = record['image_path']
        gray = np.load(image_path)

        mask_value_interval = 1. / (len(self.opt.classes[0]))
        saliency_mask_value = mask_value_interval

        mask_path = record['mask_path']
        if self.opt.condition == 'tumor':
            if mask_path is not None:
                mask = np.load(mask_path).astype(np.float64)
                mask[mask > 0] = (class_label + 1) * mask_value_interval
            else:
                mask = np.zeros_like(gray, dtype=np.float64)
        elif self.opt.condition == 'anytumor':
            if mask_path is not None:
                mask = np.load(mask_path).astype(np.float64)
                mask[mask > 0] = 1
            else:
                mask = np.zeros_like(gray, dtype=np.float64)
        elif self.opt.condition == 'saliency':
            mask = np.zeros_like(gray, dtype=np.float64)
            mask[gray > -200] = saliency_mask_value
        elif self.opt.condition == 'saliency_tumor':
            if mask_path is not None:
                mask = np.load(mask_path).astype(np.float64)
                mask[mask > 0] = (class_label + 1) * mask_value_interval
                mask = np.where(np.logical_and(mask < 0.01, gray > -200), saliency_mask_value, mask)
            else:
                mask = np.zeros_like(gray, dtype=np.float64)
        elif self.opt.condition == 'saliency_anytumor':
            if mask_path is not None:
                mask = np.load(mask_path).astype(np.float64)
                mask[mask > 0] = 1
                mask = np.where(np.logical_and(mask < 0.01, gray > -200), 0.5, mask)
            else:
                mask = np.zeros_like(gray, dtype=np.float64)
        else:
            mask = np.zeros_like(gray, dtype=np.float64)

        if self.task == 'train':
            image = aug.CTWindowing(self.opt, jitter=self.opt.sample_wwl, return_PIL=False)(gray)[:, :, 0]
            augmentation1 = A.Compose([
                A.HorizontalFlip(), 
                A.Rotate(5),
                A.RandomResizedCrop(self.opt.input_h, self.opt.input_w, scale=(0.8, 1.0), ratio=(0.8, 1.2)),
                ToTensorV2()
            ])
            augmented = augmentation1(image=image, mask=mask)
            tensor = augmented['image']
            tensor = tensor.float() / 255.
            mask = augmented['mask'].float()
            
        else:
            augmentation = transforms.Compose([
                aug.CTWindowing(self.opt, jitter=False),
                transforms.Resize((self.opt.input_h, self.opt.input_w)),
                transforms.Grayscale(1),
                transforms.ToTensor(),
            ])
            tensor = augmentation(gray)
            if mask is not None:
                mask = cv2.resize(mask, (self.opt.input_w, self.opt.input_h), interpolation=cv2.INTER_NEAREST)
                mask = torch.FloatTensor(mask)

        if self.opt.visualize:
            vis_img = aug.CTWindowing(self.opt, jitter=False)(gray)
        else:
            vis_img = None

        if mask is not None:
            mask = self.norm_data(mask.unsqueeze(0))

        if self.opt.input_type == 'mask':
            tensor = mask
        else:
            tensor = self.norm_data(tensor)

        if self.opt.out_channel > tensor.shape[0]:
            # generate image and mask together
            tensor = torch.cat((tensor, mask), dim=0)

        if hasattr(self.opt, 'noise_channel'):
            if self.opt.in_channel == self.opt.noise_channel:
                # no condition
                mask = None

        return tensor, vis_img, mask

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        sample, class_label = self.sample_list[idx]
        data, vis_image, mask = self._load_sample(sample)
        record = sample[-4]
        mode = sample[-3]
        slice_id = sample[-2]
        return data, vis_image, mask, class_label, '{}_{}_{}'.format(record['id'], mode, slice_id)

    @staticmethod
    def collate_fn(data):
        images = []
        instance_labels = []
        instance_masklabels = []
        vis_images = []
        sample_id_list = []
        
        for batch in data:
            image, vis_image, mask, label, id = batch
            images.append(image)
            vis_images.append(vis_image)
            instance_labels.append(label)
            instance_masklabels.append(mask)
            sample_id_list.append(id)

        collate_data = {
            'input': [torch.stack(images, dim=0)],
            'x_cond': torch.stack(instance_masklabels, dim=0) if instance_masklabels[0] is not None else None,
            'mask': None,
            'instance_label': torch.LongTensor(instance_labels),
            'vis_image': [vis_images],
            'id': sample_id_list
        }

        return collate_data