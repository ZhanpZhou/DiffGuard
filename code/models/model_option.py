from .networks.cls_net import ClsNet
from .networks.gen_net import GenNet
from .networks.DiffGuard import DiffGuard

model_dict = {
    'cls': {'name': ClsNet,
            'params': ['backbones', 'archs', 'share_weight', 'heads', 'in_channels', 'frozen_stages', 'output_layers', 'layer_norm_type', 'activation_type']
            },
    'gen': {'name': GenNet,
            'params': ['backbone', 'arch', 'in_channel', 'out_channel', 'input_h', 'input_w']
            },
    'DiffGuard':{'name': DiffGuard,
            'params': ['backbone', 'arch', 'in_channel', 'out_channel', 'channel_mults', 'input_h', 'input_w', 'noise_channel', 'num_classes', 'w_guide', 'p_uncond']
            }
    }
