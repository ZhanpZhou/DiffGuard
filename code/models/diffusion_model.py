import json
import math
import os
import cv2
import torch
import tqdm
from .base_model import BaseModel
from .model_option import model_dict
from optim import get_reg_criterion
from util import mkdir, stitch


class DiffusionModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser):
        parser = BaseModel.modify_commandline_options(parser)
        parser.add_argument('--backbone', type=str, default='guided_diffusion_unet')
        parser.add_argument('--arch', type=str, default='guided_diffusion_unet')
        parser.add_argument('--in_channel', type=int, default=2, help='# of input image channels')
        parser.add_argument('--out_channel', type=int, default=2, help='# of output image channels')
        parser.add_argument('--noise_channel', type=int, default=2, help='# of input noise channels')
        parser.add_argument('--channel_mults', default=(1,2,4,8))
        parser.add_argument('--use_cfg', type=bool, default=False)
        parser.add_argument('--w_guide', type=float, default=0.1)
        parser.add_argument('--p_uncond', type=float, default=0.1)
        parser.add_argument('--losses', type=str, default='L2,1')
        parser.add_argument('--save_num', type=int, default=1)
        parser.set_defaults(greater_is_better=0)

        return parser

    def __init__(self, opt):
        super().__init__(opt)

        self.opt.input_height = self.opt.input_width = self.opt.load_size
        self.net_names = ['gen']
        model = model_dict[self.opt.method_name]['name']
        self.opt.num_classes = len(self.opt.classes[0]) if self.opt.use_cfg else 0
        param_dict = dict()
        for param_name in model_dict[self.opt.method_name]['params']:
            param_dict[param_name] = getattr(self.opt, param_name)
        self.net_gen = model(**param_dict)

        self.opt.losses = [(loss.split(',')[0], float(loss.split(',')[1])) for loss in self.opt.losses.split('-')]

        self.opt.valid_metric = 'instance_mse'
        self.opt.scheduler_metric = 'instance_mse'
        self.best_m_value = 10

        self.buffer_g_instance_mses = []
        self.buffer_g_ids = []

        if self.opt.visualize and self.opt.l_state != 'train':
            name = opt.name.split(',')[0]
            self.vis_dir = os.path.join(opt.vis_dir, name+',remark='+opt.remark)
            mkdir(self.vis_dir)

    def _define_metrics(self, l_state, dataset_mode):
        super()._define_metrics(l_state, dataset_mode)
        setattr(self, l_state+'_loss_names', ['reconstr'])
        setattr(self, l_state+'_s_metric_names', ['instance_mse'])
        setattr(self, l_state+'_g_metric_names', ['instance_mse'])
        setattr(self, l_state+'_t_metric_names', [])
        
    def get_parameters(self):
        parameter_list = [{"params": self.net_gen.parameters(), "lr_mult": 1, 'decay_mult': 1}]
        return parameter_list

    def set_input(self, data):
        self.x_0 = data['input'][0]
        self.x_cond = data['x_cond'] 
        self.mask = data['mask']
        self.y = data['instance_label'] + 1 if self.opt.use_cfg else None
        self.input_size = len(self.x_0)
        self.input_id = data['id'] if isinstance(data['id'], list) else [data['id']+'_{}'.format(i+1) for i in range(self.input_size)]

    def forward(self):
        x_0 = self.x_0.cuda(0) if self.x_0 is not None else None
        x_cond = self.x_cond.cuda(0) if self.x_cond is not None else None
        mask = self.mask.cuda(0) if self.mask is not None else None
        y = self.y.cuda(0) if self.y is not None else None
        if not (self.opt.l_state == 'test' and self.opt.visualize):
            self.instance_targets, self.instance_preds = self.net_gen(x_0=x_0, x_cond=x_cond, mask=mask, y=y)
        else:
            self.instance_targets = x_0
            self.instance_preds, self.step_outputs = self.net_gen(x_0=x_0, x_cond=x_cond, mask=mask, y=y, restoration=True, 
                num_samples=self.input_size, save_num=self.opt.save_num)

    def sample(self, num_samples, class_id=None, save_intermediate=False):
        self.x_0 = self.x_cond = self.mask = None
        num_epochs = math.ceil(num_samples / self.opt.v_batch_size)
        for epoch in tqdm.tqdm(range(num_epochs)):
            self.input_size = num_samples - epoch * self.opt.v_batch_size if epoch == num_epochs - 1 else self.opt.v_batch_size
            y = torch.ones((self.input_size,), dtype=torch.long) * (class_id + 1) if self.opt.use_cfg else None
            self.instance_preds, self.step_outputs = self.net_gen(x_0=None, x_cond=None, mask=None, y=y, restoration=True, 
                num_samples=self.input_size, save_num=self.opt.save_num)
            self.input_id = ['sampled_{}'.format(i) for i in range(1+epoch*self.opt.v_batch_size, 1+epoch*self.opt.v_batch_size+self.input_size)]
            # self.visualize()
            self.save_samples(save_intermediate)

    def cal_loss(self):
        self.loss_reconstr = 0
        for loss_type, loss_weight in self.opt.losses:
            criterion = get_reg_criterion(loss_type, self.opt)
            loss = loss_weight * criterion(self.instance_preds, self.instance_targets)
            loss = torch.mean(loss)
            # self.loss_reconstr += loss
            self.loss_reconstr = loss
        
    def cal_t_metric(self):
        pass

    def stat_info(self):
        super().stat_info()
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
                    'mse': float(self.buffer_g_instance_mses[i])
                }
            records.append(record)
        json.dump(records, f)
        f.close()

    def visualize(self):
        for i in range(self.input_size):
            all_images_by_channel = [[] for _ in range(self.opt.out_channel)]
            if self.x_cond is not None:
                if self.x_0 is not None:
                    all_images_by_channel[0].append(self.x_0[i]*0.5+0.5)
                all_images_by_channel[0].append(self.x_cond[i]*0.5+0.5)

            for output in self.step_outputs:
                out = output[i]*0.5+0.5
                for j in range(self.opt.out_channel):
                    all_images_by_channel[j].append(out[j])

            image = stitch(all_images_by_channel, factor=1)

            save_name = os.path.join(self.vis_dir, '{}.png'.format(self.input_id[i]))
            cv2.imwrite(save_name, image)

    def postprocess_output(self, output):
        out = output*0.5+0.5
        images = []
        for i in range(self.input_size):
            all_images_by_channel = [[] for _ in range(self.opt.out_channel)]
            if self.x_cond is not None:
                if self.x_0 is not None:
                    all_images_by_channel[0].append(self.x_0[i]*0.5+0.5)
                all_images_by_channel[0].append(self.x_cond[i]*0.5+0.5)
            
            for j in range(self.opt.out_channel):
                all_images_by_channel[j].append(out[i][j])

            image = stitch(all_images_by_channel, factor=1)
            images.append(image)
        return images

    def save_samples(self, save_intermediate):
        if save_intermediate:
            for i in range(self.input_size):
                save_dir = os.path.join(self.vis_dir, f'{self.input_id[i]}')
                os.makedirs(save_dir)
            for j, (step, outputs) in enumerate(self.step_outputs):
                images = self.postprocess_output(outputs)
                for i in range(self.input_size):
                    save_name = os.path.join(self.vis_dir, f'{self.input_id[i]}', f'step{step}.png')
                    cv2.imwrite(save_name, images[i])
        else:
            images = self.postprocess_output(self.instance_preds)
            for i in range(self.input_size):
                save_name = os.path.join(self.vis_dir, f'{self.input_id[i]}.png')
                cv2.imwrite(save_name, images[i])