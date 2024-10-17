import os
from os.path import join
from options.base_options import TestOptions
import torch
from datasets import create_dataset
from models import create_model
from util.visualizer import Visualizer

if __name__ == '__main__':
    opt = TestOptions('rec').parse()  # get test options
    
    # CT modality
    opt.modality = 'CE'
    opt.modality = 'PL'

    # Synthetic data
    save_name = f'{opt.modality}_internal_train_generated_10000_per_cls'; opt.test_datasets = [[f'./data/{opt.modality}_internal_train_generated_10000_per_cls.json']]
    save_name = f'{opt.modality}_internal_test_generated_10000_per_cls'; opt.test_datasets = [[f'./data/{opt.modality}_internal_test_generated_10000_per_cls.json']]

    opt.checkpoints_dir = './checkpoints/reconstruction'
    opt.name = list(filter(lambda x:f'modality={opt.modality}' in x, os.listdir(opt.checkpoints_dir)))[0]
    opt.load_suffix = 'epoch200'
    opt.visualize = 1
    opt.l_state = 'test'
    opt.load_dir = os.path.join(opt.checkpoints_dir, opt.name)
    model = create_model(opt)
    model.setup(opt)
    model.save_dir = opt.load_dir

    model.vis_dir = opt.vis_dir = join(f'./feature', save_name)
    if not os.path.exists(model.vis_dir):
        os.makedirs(model.vis_dir)

    for i, datasets in enumerate(opt.test_datasets):
        opt.v_dataset_id = i
        t_dataset = create_dataset(opt, 'test')
        print('The number of test samples = %d' % len(t_dataset))
        model.test_dataset = t_dataset.dataset
        visualizer = Visualizer(opt, opt.l_state)
        with torch.no_grad():
            model.test(t_dataset, visualizer, -1, 'optimal_test')