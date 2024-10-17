import json
import os
import numpy as np
import torch
from .base_model import BaseModel
from .model_option import model_dict
from optim import get_seg_criterion
from util import mkdir

class SegModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser):
        parser = BaseModel.modify_commandline_options(parser)

        # model
        parser.add_argument('--backbone', type=str, default='TransUNet')
        parser.add_argument('--arch', type=str, default='R50-ViT-B_16')
        parser.add_argument('--in_channel', type=int, default=3, help='input channels of each backbone')
        parser.add_argument('--out_channel', type=int, default=6, help='output channels of each backbone')

        # optimization
        parser.add_argument('--pretrained', type=int, default=0)
        parser.add_argument('--loss_type', type=str, default='dice_ce')
        parser.set_defaults(valid_metric='instance_dice', scheduler_metric='instance_dice')

        return parser

    def __init__(self, opt):
        super().__init__(opt)

        self.opt.class_num = 2 if self.opt.class_mode == 'single' else len(self.opt.classes[0])
        self.net_names = ['seg']
        model = model_dict[self.opt.method_name]['name']
        param_dict = dict()
        for param_name in model_dict[self.opt.method_name]['params']:
            try:
                param_dict[param_name] = getattr(self.opt, param_name)
            except:
                pass
        self.net_seg = model(pretrained=(opt.l_state=='train' and opt.pretrained), **param_dict)

        self.buffer_g_instance_dices = []
        self.buffer_g_ids = []

        # visualization
        self.vis_dir = os.path.join(opt.vis_dir, opt.remark)
        mkdir(self.vis_dir)

    def _define_metrics(self, l_state, dataset_mode):
        setattr(self, l_state+'_loss_names', ['seg'])
        setattr(self, l_state+'_s_metric_names', ['instance_dice'])
        setattr(self, l_state+'_g_metric_names', ['instance_dice'])
        setattr(self, l_state+'_t_metric_names', [])
    
    def get_parameters(self):
        parameter_list = [{"params": self.net_seg.parameters(), "lr_mult": 1, 'decay_mult': 1}]
        return parameter_list

    def set_input(self, data):
        self.input = data['input']
        self.instance_masklabels = data['instance_masklabel']
        self.input_size = len(self.instance_masklabels[0])
        self.input_id = data['id'] if isinstance(data['id'], list) else [data['id']+'_{}'.format(i+1) for i in range(self.input_size)]

    def forward(self):
        item_num = len(self.input)
        for i in range(item_num):
            self.input[i] = self.input[i].cuda()
        self.ys, self.hiddens, self.encoder_features = self.net_seg(*tuple(self.input))
        if self.opt.out_channel == 1:
            self.instance_maskscores = torch.sigmoid(self.ys)
            self.instance_maskpreds = (self.instance_maskscores > 0.5).detach()[:, 0, :, :]
        else:
            self.instance_maskscores = torch.softmax(self.ys, dim=1)
            self.instance_maskpreds = self.instance_maskscores.argmax(dim=1).detach()

    def cal_loss(self):
        self.loss_seg = 0
        criterion = get_seg_criterion(self.opt.loss_type, self.opt)
        self.loss_seg += criterion(self.ys, self.instance_masklabels.cuda())

    def stat_info(self):
        super().stat_info()
        self.buffer_g_ids.extend(self.input_id)

    def save_stat_info(self, epoch):
        records = []
        f = open(os.path.join(self.save_dir, '{}_stat_info_{}.json'.format(epoch, self.opt.v_dataset_id)), 'w')
        for i in range(len(self.buffer_g_ids)):
            sample_idx = self.buffer_g_ids[i]
            record = {'sample_idx': sample_idx,
                        'instance_dice': self.buffer_g_instance_dices[i]}
            records.append(record)
        json.dump(records, f)
        f.close()

    def visualize(self):
        save_dir = os.path.join(self.vis_dir, 'maskpred')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for i in range(len(self.instance_maskpreds)):
            maskpred = self.instance_maskpreds[i].detach().cpu().numpy()
            np.save(os.path.join(save_dir, '{}.npy'.format(self.input_id[i])), maskpred)