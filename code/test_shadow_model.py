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
    
    opt.checkpoints_dir = './checkpoints/shadow'
    opt.load_suffix = 'iter300000'
    opt.visualize = 1
    opt.l_state = 'test'
    opt.name = list(filter(lambda x: f'backbone={opt.backbone}' in x and f'modality={opt.modality}' in x , os.listdir(opt.checkpoints_dir)))[0]; 
    opt.load_dir = os.path.join(opt.checkpoints_dir, opt.name)
    model = create_model(opt)
    model.setup(opt)
    model.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    visualizer = Visualizer(opt, opt.l_state)

    model.vis_dir = opt.vis_dir = join('./shadow_vis', opt.name+'_'+opt.load_suffix)
    if not os.path.exists(model.vis_dir):
        os.makedirs(model.vis_dir)

    opt.test_datasets = [['./data/shadow_model_private_data.json', './data/shadow_model_public_data.json']]
    for i, datasets in enumerate(opt.test_datasets):
        opt.v_dataset_id = i
        t_dataset = create_dataset(opt, 'test')
        print('The number of test samples = %d' % len(t_dataset))
        model.test_dataset = t_dataset.dataset
        with torch.no_grad():
            model.test(t_dataset, visualizer, -1, 'optimal_test')