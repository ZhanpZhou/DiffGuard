import json
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from .base_dataset import BaseDataset
import util.augmentation as aug

class CTSliceMIADataset(BaseDataset):
    def __init__(self, opt, task):
        super().__init__(opt, task)
        self.prepare_dataset()
        self.prepare_new_epoch()
        shadow_output_name = list(filter(lambda x: opt.shadow_model in x, os.listdir(opt.shadow_output_dir)))[0]
        self.shadow_output_dir = os.path.join(opt.shadow_output_dir, shadow_output_name)

    def prepare_dataset(self):
        all_samples = []
        for dataset in self.datasets:
            with open(dataset, 'r') as f:
                records = json.load(f)
            for record in records:
                try:
                    disease_label = ['normal', 'thymoma', 'benign cysts', 'germ cell tumor', 'neurogenic tumor', 'thymic carcinoma'].index(record['label'])
                except:
                    continue
                if disease_label < 1:
                    continue
                if 'generated' in dataset:
                    is_synthetic_sample = True
                    label = record['MIA_label']
                else:
                    is_synthetic_sample = False
                    label = 1 if 'internal_train' in dataset else 0
                all_samples.append((record, disease_label, label, is_synthetic_sample))

        if self.task == 'train':
            self.samples = {c:[] for c in (0,1)}
            for sample in all_samples:
                self.samples[sample[-2]].append(sample)
            for k, v in self.samples.items():
                print('sample num of class {}:'.format(k), len(v))
        else:
            self.sample_list = [(sample, sample[-2]) for sample in all_samples]
            if self.opt.test_samples:
                self.sample_list = [self.sample_list[idx] for idx in self.opt.test_samples]
            print('test sample num:', len(self.sample_list))

    def _load_sample(self, sample):
        record, disease_label, label, is_synthesis_sample = sample
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
            id = record['id']
            if 'nnUNet' in self.shadow_output_dir:
                output = sitk.ReadImage(os.path.join(self.shadow_output_dir, 'former_latter_prediction', f'Mediastinum_id={id}_mode={mode}.nii.gz'))
                output = sitk.GetArrayFromImage(output)
                output = output[0]
            else:
                output = np.load(os.path.join(self.shadow_output_dir, 'maskpred', f'{id}.npy'))
        else:
            gray = np.load(image_path)
            image = aug.CTWindowing(self.opt, jitter=(self.task == 'train' and self.opt.sample_wwl), return_PIL=False)(gray)
            if mask_path is not None:
                mask = np.load(mask_path).astype(np.uint8)
            else:
                mask = np.zeros_like(gray, dtype=np.uint8)
            if self.opt.class_mode == 'multi':
                mask = disease_label * mask
            id = record['id']
            if 'nnUNet' in self.shadow_output_dir:
                mode = self.opt.modality
                output = sitk.ReadImage(os.path.join(self.shadow_output_dir, '2d_prediction2D', f'Mediastinum_id={id}_mode={mode}.nii.gz'))
                output = sitk.GetArrayFromImage(output)
                output = output[0]
            else:
                output = np.load(os.path.join(self.shadow_output_dir, 'maskpred', f'{id}.npy'))

        output = cv2.resize(output.astype(np.uint8), (mask.shape[1], mask.shape[0]))
        if self.opt.input_strategy == 'onehot':
            mask_one_hot = F.one_hot(torch.LongTensor(mask), num_classes=6).numpy()
            output_one_hot = F.one_hot(torch.LongTensor(output), num_classes=6).numpy()
            concat_mask = np.concatenate((mask_one_hot, output_one_hot), axis=-1)
        elif self.opt.input_strategy == 'onehot_foreground':
            mask_one_hot = F.one_hot(torch.LongTensor(mask), num_classes=6).numpy()[:, :, 1:]
            output_one_hot = F.one_hot(torch.LongTensor(output), num_classes=6).numpy()[:, :, 1:]
            concat_mask = np.concatenate((mask_one_hot, output_one_hot), axis=-1)
        elif self.opt.input_strategy == 'map':
            concat_mask = np.stack((mask, output), axis=-1)
            concat_mask = concat_mask / 5.
        elif self.opt.input_strategy == 'oneclass':
            concat_mask = np.stack((mask, output), axis=-1)
            concat_mask[concat_mask > 0] = 1

        if self.task == 'train':
            augmentation1 = A.Compose([
                A.Resize(height=256, width=256),
                A.RandomCrop(height=self.opt.input_size, width=self.opt.input_size),
                ToTensorV2()
            ])
            augmented = augmentation1(image=image, mask=concat_mask)
            concat_mask = augmented['mask'].float()
            data = concat_mask * 2 - 1 # (-1, 1)
            data_list = [data]
            id_list = [record['id']]
        else:
            concat_mask = cv2.resize(concat_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
            concat_mask = torch.FloatTensor(concat_mask) * 2 - 1 # (-1, 1)
            data_list = []
            id_list = []
            margin = (256-self.opt.input_size) // 2

            left_upper = concat_mask[:self.opt.input_size, :self.opt.input_size]; data_list.append(left_upper); id_list.append('{}_left_upper'.format(record['id']))
            upper = concat_mask[:self.opt.input_size, margin:margin+self.opt.input_size]; data_list.append(upper); id_list.append('{}_upper'.format(record['id']))
            right_upper = concat_mask[:self.opt.input_size, -self.opt.input_size:]; data_list.append(right_upper); id_list.append('{}_right_upper'.format(record['id']))
            left = concat_mask[margin:margin+self.opt.input_size, :self.opt.input_size]; data_list.append(left); id_list.append('{}_left'.format(record['id']))
            center = concat_mask[margin:margin+self.opt.input_size, margin:margin+self.opt.input_size]; data_list.append(center); id_list.append('{}_center'.format(record['id']))
            right = concat_mask[margin:margin+self.opt.input_size, -self.opt.input_size:]; data_list.append(right); id_list.append('{}_right'.format(record['id']))
            left_lower = concat_mask[-self.opt.input_size:, :self.opt.input_size]; data_list.append(left_lower); id_list.append('{}_left_lower'.format(record['id']))
            lower = concat_mask[-self.opt.input_size:, margin:margin+self.opt.input_size]; data_list.append(lower); id_list.append('{}_lower'.format(record['id']))
            right_lower = concat_mask[-self.opt.input_size:, -self.opt.input_size:]; data_list.append(right_lower); id_list.append('{}_right_lower'.format(record['id']))

        return data_list, label, id_list

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        sample, class_label = self.sample_list[idx]
        data_list, class_label, id_list = self._load_sample(sample)
        return data_list, class_label, id_list

    @staticmethod
    def collate_fn(data):
        all_data = []
        instance_labels = []
        sample_id_list = []
        
        for batch in data:
            data_list, label, id_list = batch
            all_data.extend(data_list)
            instance_labels.extend([label] * len(data_list))
            sample_id_list.extend(id_list)

        collate_data = {
            'input': [torch.stack(all_data, dim=0).permute(0, 3, 1, 2)],
            'instance_label': [torch.LongTensor(instance_labels)] * 2,
            'id': sample_id_list
        }

        return collate_data