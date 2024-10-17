import os
from options.base_options import TestOptions
import torch
from datasets import create_dataset
from models import create_model
from util.visualizer import Visualizer

if __name__ == '__main__':
    opt = TestOptions('rec').parse() 

    # CT modality
    opt.modality = 'CE' # 'CE' or 'PL'


    # DiffGuard trained on internal train set or internal test set
    opt.data_dir = f'./synthetic data/{opt.modality}_internal_train_generated/'    
    # opt.data_dir = f'./synthetic data/{opt.modality}_internal_test_generated/'    
    
    opt.name = list(filter(lambda x:f'modality={opt.modality}' in x, os.listdir(opt.checkpoints_dir)))[0]
    opt.checkpoints_dir = './checkpoints/reconstruction'
    opt.load_dir = os.path.join(opt.checkpoints_dir, opt.name)
    opt.load_suffix = 'epoch200'
    model = create_model(opt)
    model.setup(opt)
    model.save_dir = os.path.join(opt.data_dir)
    
    opt.visualize = 0
    opt.l_state = 'test'
    opt.v_dataset_mode = 'generated'
    opt.v_dataset_id = 0
    t_dataset = create_dataset(opt, 'test')
    print('The number of test samples = %d' % len(t_dataset))
    model.test_dataset = t_dataset.dataset
    visualizer = Visualizer(opt, opt.l_state)
    with torch.no_grad():
        model.test(t_dataset, visualizer, -1, 'optimal_test')
