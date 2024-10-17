import json
import os
import torch
import torch.nn.functional as F
from .base_model import BaseModel
from .model_option import model_dict
from optim import get_cls_criterion


class ClsModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser):
        parser = BaseModel.modify_commandline_options(parser)

        # model
        parser.add_argument('--backbones', type=str, default='resnet')
        parser.add_argument('--archs', type=str, default='50')
        parser.add_argument('--share_weight', type=int, default=0, help='share backbone in cls_net')
        parser.add_argument('--in_channels', type=str, default='12', help='input channels of each backbone')
        parser.add_argument('--heads', type=str, default='0_2', help='head of cls_net')
        parser.add_argument('--frozen_stages', type=int, default=0)
        parser.add_argument('--output_layers', type=int, default=1)
        parser.add_argument('--recall_thresholds', type=str, default=None)

        # optimization
        parser.add_argument('--imagenet_pretrain', type=int, default=0)
        parser.add_argument('--loss_type', type=str, default='ce')
        parser.add_argument('--loss_weights', type=str, default='0,1')
        parser.set_defaults(valid_metric='instance_auc', scheduler_metric='instance_auc')

        return parser

    def __init__(self, opt):
        super().__init__(opt)

        self.opt.class_num = len(self.opt.classes)
        self.net_names = ['cls']
        model = model_dict[self.opt.method_name]['name']
        param_dict = dict()
        for param_name in model_dict[self.opt.method_name]['params']:
            try:
                param_dict[param_name] = getattr(self.opt, param_name)
            except:
                pass
        self.net_cls = model(pretrained=(opt.l_state=='train' and opt.imagenet_pretrain), **param_dict)

        self.buffer_g_instance_scores = [[] for i in range(1+len(self.opt.heads))]
        self.buffer_g_instance_preds = [[] for i in range(1+len(self.opt.heads))]
        self.buffer_g_instance_labels = [[] for i in range(1+len(self.opt.heads))]
        self.buffer_g_ids = []

    def _define_metrics(self, l_state, dataset_mode):
        setattr(self, l_state+'_loss_names', ['c'])
        setattr(self, l_state+'_s_metric_names', ['instance_auc', 'instance_accuracy'])
        setattr(self, l_state+'_g_metric_names', ['instance_auc', 'instance_accuracy'])
        setattr(self, l_state+'_t_metric_names', ['instance_cmatrix'])
    
    def get_parameters(self):
        parameter_list = [{"params": self.net_cls.parameters(), "lr_mult": 1, 'decay_mult': 1}]
        return parameter_list

    def set_input(self, data):
        self.input = data['input']
        self.instance_labels = data['instance_label']
        self.input_size = len(self.instance_labels[0])
        self.input_id = data['id'] if isinstance(data['id'], list) else [data['id']+'_{}'.format(i+1) for i in range(self.input_size)]
            
    def forward(self):
        item_num = len(self.input)
        for i in range(item_num):
            self.input[i] = self.input[i].cuda()
        self.instance_ys = self.net_cls(*tuple(self.input))
        self.instance_scores = []
        self.instance_preds = []

        for i, y in enumerate(self.instance_ys):
            instance_out = F.softmax(y, dim=1)
            instance_scores = instance_out.detach().cpu()
            self.instance_scores.append(instance_scores)
            if instance_scores.shape[1] == 2 and self.opt.recall_thresholds is not None:
                instance_preds = instance_scores[:, 1] > self.opt.recall_thresholds[i+1]
                instance_preds = instance_preds.long()
            else:
                instance_preds = torch.argmax(instance_out, dim=1)
            self.instance_preds.append(instance_preds)
                
        # calculate the final classification prediction
        dataset = getattr(self, self.opt.l_state+'_dataset')
        target_y, target_score, target_pred = dataset.cal_target(self)
        self.instance_ys.insert(0, target_y)
        self.instance_scores.insert(0, target_score)
        self.instance_preds.insert(0, target_pred)

    def cal_loss(self):
        self.loss_c = 0
        criterion = get_cls_criterion(self.opt.loss_type, self.opt)
        for i, y in enumerate(self.instance_ys):
            self.loss_c += self.opt.loss_weights[i] * criterion(y.cuda(), self.instance_labels[i].long().cuda())

    def stat_info(self):
        super().stat_info()
        
        for i in range(1+len(self.opt.heads)):
            self.buffer_g_instance_scores[i].extend(
                self.instance_scores[i].cpu().tolist())
            self.buffer_g_instance_labels[i].extend(
                self.instance_labels[i].cpu().long().view(-1).tolist())
            self.buffer_g_instance_preds[i].extend(self.instance_preds[i].view(-1).tolist())
        self.buffer_g_ids.extend(self.input_id)

    def save_stat_info(self, epoch):
        super().save_stat_info(epoch)
        records = []
        f = open(os.path.join(self.save_dir, '{}_stat_info_{}.json'.format(epoch, self.opt.v_dataset_id)), 'w')
        for i in range(len(self.buffer_g_ids)):
            sample_idx = self.buffer_g_ids[i]
            instance_labels = [self.buffer_g_instance_labels[j][i] for j in range(1+len(self.opt.heads))]
            instance_scores = [self.buffer_g_instance_scores[j][i] for j in range(1+len(self.opt.heads))]
            record = {'sample_idx': sample_idx,
                        'instance_labels': instance_labels,
                        'instance_scores': instance_scores}
            records.append(record)
        json.dump(records, f)
        f.close()

    def visualize(self):
        pass