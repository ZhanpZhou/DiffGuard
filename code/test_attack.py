import os
from options.base_options import TestOptions
import torch
from datasets import create_dataset
from models import create_model
from util.visualizer import Visualizer

if __name__ == '__main__':
    opt = TestOptions('MIA').parse()

    # CT modality
    opt.modality = 'CE' # 'CE' or 'PL'

    # 1. nnU-Net as shadow model
    shadow_backbone = 'nnUNet'
    shadow_checkpoints_dir = '../nnUNetFrame/DATASET/nnUNet_raw_data_base/nnUNet_raw_data'
    # Test the models trained on different training datasets
    opt.shadow_model = name = 'real'
    opt.shadow_model = name = 'synthetic500'
    opt.shadow_model = name = 'synthetic1000'
    opt.shadow_model = name = 'synthetic2000'
    opt.shadow_model = name = 'synthetic6000'
    opt.shadow_model = name = 'synthetic10000'

    # 2. TransUNet or U-Net as shadow model
    shadow_backbone = 'transunet'
    # shadow_backbone = 'unet'
    shadow_checkpoints_dir = './checkpoints/segmentation'
    opt.shadow_output_dir = f'./vis_{shadow_backbone.lower()}'
    # Test the models trained on different training datasets
    opt.shadow_model = list(filter(lambda x:f'modality={opt.modality}' in x and f'shadow_model={shadow_backbone.lower()}' in x.lower() and 'real_neg_num=99999,real_num_by_class=99999,gen_neg_num=0,gen_num_by_class=0' in x, os.listdir(shadow_checkpoints_dir)))[0]; name='real'
    opt.shadow_model = list(filter(lambda x:f'modality={opt.modality}' in x and f'shadow_model={shadow_backbone.lower()}' in x.lower() and 'real_neg_num=0,real_num_by_class=0,gen_neg_num=500,gen_num_by_class=500' in x, os.listdir(shadow_checkpoints_dir)))[0]; name='synthetic500'
    opt.shadow_model = list(filter(lambda x:f'modality={opt.modality}' in x and f'shadow_model={shadow_backbone.lower()}' in x.lower() and 'real_neg_num=0,real_num_by_class=0,gen_neg_num=1000,gen_num_by_class=1000' in x, os.listdir(shadow_checkpoints_dir)))[0]; name='synthetic1000'
    opt.shadow_model = list(filter(lambda x:f'modality={opt.modality}' in x and f'shadow_model={shadow_backbone.lower()}' in x.lower() and 'real_neg_num=0,real_num_by_class=0,gen_neg_num=2000,gen_num_by_class=2000' in x, os.listdir(shadow_checkpoints_dir)))[0]; name='synthetic2000'
    opt.shadow_model = list(filter(lambda x:f'modality={opt.modality}' in x and f'shadow_model={shadow_backbone.lower()}' in x.lower() and 'real_neg_num=0,real_num_by_class=0,gen_neg_num=6000,gen_num_by_class=6000' in x, os.listdir(shadow_checkpoints_dir)))[0]; name='synthetic6000'
    opt.shadow_model = list(filter(lambda x:f'modality={opt.modality}' in x and f'shadow_model={shadow_backbone.lower()}' in x.lower() and 'real_neg_num=0,real_num_by_class=0,gen_neg_num=10000,gen_num_by_class=10000' in x, os.listdir(shadow_checkpoints_dir)))[0]; name='synthetic10000'

    opt.name = list(filter(lambda x:f'modality={opt.modality}' in x and f'shadow_model={shadow_backbone.lower()}' in x.lower(), os.listdir(opt.checkpoints_dir)))[0]
    opt.load_dir = os.path.join(opt.checkpoints_dir, opt.name)
    opt.l_state = 'test'
    opt.load_suffix = 'epoch500'
    visualizer = Visualizer(opt, opt.l_state)     
    model = create_model(opt)
    model.setup(opt)
    model.save_dir = opt.load_dir
    if not os.path.exists(model.save_dir):
        os.makedirs(model.save_dir)

    opt.test_datasets = [[f'./data/{opt.modality}_internal_train.json', f'./data/{opt.modality}_internal_test.json', f'./data/{opt.modality}_external_test.json']]

    for i, datasets in enumerate(opt.test_datasets):
        opt.v_dataset_id = i
        t_dataset = create_dataset(opt, 'test')
        print('The number of test samples = %d' % len(t_dataset))
        model.test_dataset = t_dataset.dataset
        with torch.no_grad():
            model.test(t_dataset, visualizer, -1, f'victim={shadow_backbone.lower()}_{name}')