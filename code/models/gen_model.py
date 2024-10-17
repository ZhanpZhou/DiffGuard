import json
import os

import cv2
import numpy as np
import torch

from .base_model import BaseModel
from .model_option import model_dict
from optim import get_reg_criterion
from util import mkdir


class GenModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser):
        parser = BaseModel.modify_commandline_options(parser)
        parser.add_argument('--backbone', type=str, default='unet')
        parser.add_argument('--in_channel', type=int, default=1, help='# of input image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--out_channel', type=int, default=1, help='# of output image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--losses', type=str, default='L1,1')
        parser.set_defaults(greater_is_better=0)

        return parser

    def __init__(self, opt):
        super().__init__(opt)

        self.net_names = ['gen']
        model = model_dict[self.opt.method_name]['name']
        param_dict = dict()
        for param_name in model_dict[self.opt.method_name]['params']:
            try:
                param_dict[param_name] = getattr(self.opt, param_name)
            except:
                pass
        self.net_gen = model(**param_dict)

        self.opt.losses = [(loss.split(',')[0], float(loss.split(',')[1])) for loss in self.opt.losses.split('-')]

        self.opt.valid_metric = 'loss'
        self.opt.scheduler_metric = 'loss'
        self.best_m_value = 10

        self.buffer_g_loss = []
        self.buffer_g_ids = []

        self.vis_dir = os.path.join(opt.vis_dir, opt.remark)
        mkdir(self.vis_dir)

    def _define_metrics(self, l_state, dataset_mode):
        super()._define_metrics(l_state, dataset_mode)
        setattr(self, l_state+'_loss_names', ['reconstr'])
        setattr(self, l_state+'_s_metric_names', ['loss'])
        setattr(self, l_state+'_g_metric_names', ['loss'])
        setattr(self, l_state+'_t_metric_names', [])
        
    def get_parameters(self):
        parameter_list = [{"params": self.net_gen.parameters(), "lr_mult": 1, 'decay_mult': 1}]
        return parameter_list

    def set_input(self, data):
        self.input = data['input']
        self.target = data['target'] if 'target' in data.keys() else self.input[0]
        self.input_size = len(self.target)
        self.input_id = data['id'] if isinstance(data['id'], list) else [data['id']+'_{}'.format(i+1) for i in range(self.input_size)]

    def forward(self):
        self.output, self.hiddens, self.encoder_features = self.net_gen(*tuple(self.input))

    def cal_loss(self):
        self.loss_reconstr = torch.zeros((self.input_size))
        for loss_type, loss_weight in self.opt.losses:
            criterion = get_reg_criterion(loss_type, self.opt)
            loss = loss_weight * criterion(self.output.cpu(), self.target)
            loss = loss.view(loss.shape[0], -1)
            self.loss_reconstr += torch.mean(loss, dim=-1)
        self.instance_losses = self.loss_reconstr.detach().cpu().tolist()
        self.loss_reconstr = torch.mean(self.loss_reconstr)
        
    def cal_s_metric(self):
        self.s_metric_loss = self.loss_reconstr.detach().cpu().item()

    def cal_g_metric(self):
        self.g_metric_loss = np.mean(self.buffer_g_loss)
        
    def cal_t_metric(self):
        pass

    def stat_info(self):
        super().stat_info()
        self.buffer_g_loss.extend(self.instance_losses)
        self.buffer_g_ids.extend(self.input_id)

        if self.opt.visualize and self.opt.l_state != 'train':
            self.visualize()

    def save_stat_info(self, epoch):
        super().save_stat_info(epoch)
        records = []
        f = open(os.path.join(self.save_dir, '{}_stat_info_{}.json'.format(epoch, self.opt.v_dataset_id)), 'w')
        for i in range(len(self.buffer_g_ids)):
            record = {
                    'sample_idx': self.buffer_g_ids[i], 
                    'loss': float(self.buffer_g_loss[i])
                }
            records.append(record)
        json.dump(records, f)
        f.close()

    def visualize(self):
        self.save_hiddens()
        self.save_features()

    def save_hiddens(self):
        save_dir = os.path.join(self.vis_dir, 'hidden')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for i in range(len(self.hiddens)):
            hidden = self.hiddens[i].detach().cpu().numpy()
            np.save(os.path.join(save_dir, '{}.npy'.format(self.input_id[i])), hidden)

    def save_features(self):
        if self.encoder_features is not None:
            for layer_name, features in self.encoder_features:
                save_dir = os.path.join(self.vis_dir, layer_name)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                for j in range(len(self.input_id)):
                    feature = features[j].detach().cpu().numpy()
                    np.save(os.path.join(save_dir, '{}.npy'.format(self.input_id[j])), feature)