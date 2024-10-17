import os
from os.path import join
from options.base_options import TestOptions
import torch
from datasets import create_dataset
from models import create_model
from util.visualizer import Visualizer

if __name__ == '__main__':
    opt = TestOptions('seg').parse()

    # CT modality
    opt.modality = 'CE' # 'CE' or 'PL'

    # TransUNet
    opt.backbone = 'TransUNet'; opt.arch = 'R50-ViT-B_16'; opt.data_norm_type = 'imagenet'; opt.in_channel = 3
    # U-Net
    opt.backbone = 'unet'; opt.arch = ''; opt.data_norm_type = 'gray'; opt.in_channel = 1

    # Test the models trained on different training datasets
    # opt.name = list(filter(lambda x:f'modality={opt.modality}' in x and f'backbone={opt.backbone}'.lower() in x.lower() and 'real_neg_num=99999,real_num_by_class=99999,gen_neg_num=0,gen_num_by_class=0' in x, os.listdir(opt.checkpoints_dir)))[0]; opt.load_suffix = 'iter300000'
    # opt.name = list(filter(lambda x:f'modality={opt.modality}' in x and f'backbone={opt.backbone}'.lower() in x.lower() and 'real_neg_num=0,real_num_by_class=0,gen_neg_num=500,gen_num_by_class=500' in x, os.listdir(opt.checkpoints_dir)))[0]; opt.load_suffix = 'iter300000'
    # opt.name = list(filter(lambda x:f'modality={opt.modality}' in x and f'backbone={opt.backbone}'.lower() in x.lower() and 'real_neg_num=0,real_num_by_class=0,gen_neg_num=1000,gen_num_by_class=1000' in x, os.listdir(opt.checkpoints_dir)))[0]; opt.load_suffix = 'iter300000'
    # opt.name = list(filter(lambda x:f'modality={opt.modality}' in x and f'backbone={opt.backbone}'.lower() in x.lower() and 'real_neg_num=0,real_num_by_class=0,gen_neg_num=2000,gen_num_by_class=2000' in x, os.listdir(opt.checkpoints_dir)))[0]; opt.load_suffix = 'iter300000'
    # opt.name = list(filter(lambda x:f'modality={opt.modality}' in x and f'backbone={opt.backbone}'.lower() in x.lower() and 'real_neg_num=0,real_num_by_class=0,gen_neg_num=6000,gen_num_by_class=6000' in x, os.listdir(opt.checkpoints_dir)))[0]; opt.load_suffix = 'iter300000'
    opt.name = list(filter(lambda x:f'modality={opt.modality}' in x and f'backbone={opt.backbone}'.lower() in x.lower() and 'real_neg_num=0,real_num_by_class=0,gen_neg_num=10000,gen_num_by_class=10000' in x, os.listdir(opt.checkpoints_dir)))[0]; opt.load_suffix = 'iter300000'

    opt.visualize = 1
    opt.l_state = 'test'
    opt.load_dir = os.path.join(opt.checkpoints_dir, opt.name)
    model = create_model(opt)
    model.setup(opt)
    model.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

    model.vis_dir = opt.vis_dir = join(f'./vis_{opt.backbone.lower()}', opt.name+'_'+opt.load_suffix)
    if not os.path.exists(model.vis_dir):
        os.makedirs(model.vis_dir)

    opt.test_datasets = [[f'./data/{opt.modality}_internal_test_generated_50_per_cls.json']]

    for i, datasets in enumerate(opt.test_datasets):
        opt.v_dataset_id = i
        t_dataset = create_dataset(opt, 'test')
        print('The number of test samples = %d' % len(t_dataset))
        model.test_dataset = t_dataset.dataset
        visualizer = Visualizer(opt, opt.l_state)
        with torch.no_grad():
            model.test(t_dataset, visualizer, -1, 'optimal_test')