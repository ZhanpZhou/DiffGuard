import os
from options.base_options import TestOptions
import torch
from models import create_model

if __name__ == '__main__':
    opt = TestOptions('DiffGuard').parse()  # get test options
    
    # CT modality
    opt.modality = 'CE' # 'CE' or 'PL'

    # Training data
    opt.data_dir = f'./synthetic data/{opt.modality}_internal_train_generated/'    
    # opt.data_dir = f'./synthetic data/{opt.modality}_internal_test_generated/'

    # Generate CT images with mediastinum neoplasms or normal control
    generate_normal_control = False

    # Number of samples to synthesize
    synthesis_num = 50000

    if generate_normal_control:
        opt.in_channel = opt.out_channel = opt.noise_channel = 1
    else:
        opt.in_channel = opt.out_channel = opt.noise_channel = 2

    if not os.path.exists(opt.data_dir):
        os.makedirs(opt.data_dir)
    opt.visualize = 1
    opt.l_state = 'test'
    opt.save_num = 1
    opt.checkpoints_dir = './checkpoints/DiffGuard'
    opt.load_dir = os.path.join(opt.checkpoints_dir, opt.name)
    opt.name = list(filter(lambda x:f'modality={opt.modality}' in x and f'in_channel={opt.in_channel}' in x, os.listdir(opt.checkpoints_dir)))[0]
    model = create_model(opt)
    model.setup(opt)
    model.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

    with torch.no_grad():
        model.sample(synthesis_num, save_intermediate=False)