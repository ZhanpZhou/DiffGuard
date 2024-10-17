import copy
import glob
import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from .base_dataset import BaseDataset
import util.augmentation as aug

def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    return cv_img

class GeneratedDataset(BaseDataset):
    def __init__(self, opt, task):
        super().__init__(opt, task)
        self.prepare_dataset()
        self.prepare_new_epoch()

    def prepare_dataset(self):
        all_samples = []
        files = glob.glob(os.path.join(self.opt.data_dir, '*.png'))

        for image_path in files:
            sample_idx = image_path.split('/')[-1][:-4].split('(loss')[0]
            all_samples.append((image_path, sample_idx, 1))

        if self.task == 'train':
            self.samples = dict()
            for c in range(len(self.opt.classes[0])):
                self.samples[c] = []
            for sample in all_samples:
                self.samples[sample[-1]].append(sample)
            for k, v in self.samples.items():
                print('sample num of class {}:'.format(k), len(v))
        else:
            self.sample_list = []
            for sample in all_samples:
                self.sample_list.append((sample, sample[-1]))            
            if self.opt.test_samples:
                self.sample_list = [self.sample_list[idx] for idx in self.opt.test_samples]

    def _load_sample(self, sample):
        image_path, _, class_label = sample
        try:
            image = cv_imread(image_path)
        except:
            print(image_path); exit()
        rownum, colnum = image.shape[0] // self.opt.input_h, image.shape[1] // self.opt.input_w
        if rownum == 2:
            # no guidance
            gen_img = image[0:self.opt.input_h, -self.opt.input_w:]
            gen_mask = image[self.opt.input_h:2*self.opt.input_h, -self.opt.input_w:]
        elif colnum == 2:
            # mask guidance
            gen_img = image[:, -self.opt.input_w:]
            gen_mask = image[:, self.opt.input_w:2*self.opt.input_w]
        else:
            # single image
            gen_img = image
            gen_mask = np.zeros_like(gen_img)

        augmentation = transforms.Compose([
            aug.CTWindowing(self.opt, jitter=False),
            transforms.Resize((self.opt.input_h, self.opt.input_w)),
            transforms.Grayscale(1),
            transforms.ToTensor(),
        ])
        tensor = augmentation(gen_img)

        mask = cv2.resize(gen_mask, (self.opt.input_w, self.opt.input_h), interpolation=cv2.INTER_NEAREST)
        mask = torch.LongTensor(mask)
        
        if self.opt.visualize:
            vis_img = aug.CTWindowing(self.opt, jitter=False)(gen_img)
        else:
            vis_img = None

        tensor = self.norm_data(tensor)
        return tensor, vis_img, mask

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        sample, class_label = self.sample_list[idx]
        data, vis_image, mask = self._load_sample(sample)
        return data, vis_image, mask, class_label, sample[0]

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
            'instance_label': [torch.LongTensor(instance_labels)] * 2,
            'instance_masklabel': torch.stack(instance_masklabels, dim=0).unsqueeze(1),
            'id': sample_id_list
        }

        return collate_data