import json
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from .base_dataset import BaseDataset
import util.augmentation as aug

class CTSliceDataset(BaseDataset):
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
        all_generated_samples = {c: [] for c in range(len(self.opt.classes[0]))}
        all_real_samples = {c: [] for c in range(len(self.opt.classes[0]))}
        for dataset in self.datasets:
            with open(dataset, 'r') as f:
                records = json.load(f)
            for record in records:
                class_label = self.get_class_label(record)
                if class_label is None: 
                    continue
                if self.opt.positive_only and class_label == 0:
                    continue
                if 'generated' in dataset:
                    is_synthetic_sample = True 
                    all_generated_samples[class_label].append((record, class_label, is_synthetic_sample))
                else:
                    is_synthetic_sample = False
                    all_real_samples[class_label].append((record, class_label, is_synthetic_sample))

        for k, v in all_real_samples.items():
            print('total real sample num of class {}:'.format(k), len(v))
        for k, v in all_generated_samples.items():
            print('total synthetic sample num of class {}:'.format(k), len(v))

        if self.task == 'train':
            self.samples = dict()
            self.samples[0] = all_generated_samples[0][:self.opt.gen_neg_num] if self.opt.gen_neg_num >= 0 else all_generated_samples[0]
            for c in range(1, len(self.opt.classes[0])):
                self.samples[c] = all_generated_samples[c][:self.opt.gen_num_by_class] if self.opt.gen_num_by_class >= 0 else all_generated_samples[c]
            
            real_samples = all_real_samples[0][:self.opt.real_neg_num] if self.opt.real_neg_num >= 0 else all_real_samples[0]
            self.samples[0].extend(real_samples)
            for c in range(1, len(self.opt.classes[0])):
                real_samples = all_real_samples[c][:self.opt.real_num_by_class] if self.opt.real_num_by_class >= 0 else all_real_samples[c]
                self.samples[c].extend(real_samples)

            for k, v in self.samples.items():
                print('sample num of class {}:'.format(k), len(v))

        else:
            self.sample_list = []
            for c, l in all_generated_samples.items():
                for sample in l:
                    self.sample_list.append((sample, c))
            for c, l in all_real_samples.items():
                for sample in l:
                    self.sample_list.append((sample, c))
            if self.opt.test_samples:
                self.sample_list = [self.sample_list[idx] for idx in self.opt.test_samples]
            print('test sample num:', len(self.sample_list))

    def _load_sample(self, sample):
        record, class_label, is_synthesis_sample = sample
        image_path = record['image_path']
        mask_path = record['mask_path']
        if is_synthesis_sample:
            image = cv2.imread(image_path)
            if mask_path is not None:
                mask = np.load(mask_path).astype(np.uint8)
            else:
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            if self.opt.class_mode == 'multi':
                mask[mask == 1] = 0
                mask = np.where(mask > 1, mask-1, mask)
            else:
                mask = (mask >= 2).astype(np.uint8)
        else:
            gray = np.load(image_path)
            image = aug.CTWindowing(self.opt, jitter=(self.task == 'train' and self.opt.sample_wwl), return_PIL=False)(gray)
            if mask_path is not None:
                mask = np.load(mask_path).astype(np.uint8)
            else:
                mask = np.zeros_like(gray, dtype=np.uint8)
            if self.opt.class_mode == 'multi':
                mask = class_label * mask

        if self.task == 'train':
            augmentation1 = A.Compose([
                A.HorizontalFlip(), 
                A.Rotate(5),
                A.RandomResizedCrop(self.opt.input_h, self.opt.input_w, scale=(0.8, 1.0), ratio=(0.8, 1.2)),
                ToTensorV2()
            ])
            augmented = augmentation1(image=image, mask=mask)
            tensor = augmented['image']
            tensor = tensor.float() / 255.
            tensor = tensor[0:self.opt.in_channel]
            mask = augmented['mask'].long()
        else:
            image_pil = Image.fromarray(image)
            augmentation = transforms.Compose([
                transforms.Resize((self.opt.input_h, self.opt.input_w)),
                transforms.Grayscale(self.opt.in_channel),
                transforms.ToTensor(),
            ])
            tensor = augmentation(image_pil)
            if mask is not None:
                mask = cv2.resize(mask, (self.opt.input_w, self.opt.input_h), interpolation=cv2.INTER_NEAREST)
                mask = torch.LongTensor(mask)

        if self.opt.visualize:
            if is_synthesis_sample:
                vis_img = image
            else:
                vis_img = aug.CTWindowing(self.opt, jitter=False)(gray)
        else:
            vis_img = None

        tensor = self.norm_data(tensor)
        return tensor, vis_img, mask, is_synthesis_sample

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        sample, class_label = self.sample_list[idx]
        data, vis_image, mask, is_synthesis_sample = self._load_sample(sample)
        record = sample[0]
        return data, vis_image, mask, class_label, record['id'], is_synthesis_sample

    @staticmethod
    def collate_fn(data):
        images = []
        instance_labels = []
        instance_masklabels = []
        vis_images = []
        sample_id_list = []
        
        for batch in data:
            image, vis_image, mask, label, id, is_synthesis_sample = batch
            images.append(image)
            vis_images.append(vis_image)
            instance_labels.append(label)
            instance_masklabels.append(mask)
            sample_id_list.append(id)

        collate_data = {
            'input': [torch.stack(images, dim=0)],
            'vis_image': [vis_images],
            'instance_label': [torch.LongTensor(instance_labels)] * 2,
            'instance_masklabel': torch.stack(instance_masklabels, dim=0),
            'id': sample_id_list
        }

        return collate_data