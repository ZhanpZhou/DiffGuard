import torch.nn as nn
from .backbone.resnet import resnet
from .backbone.unet import unet
from .backbone.TransUNet import transunet
from .backbone.guided_diffusion import guided_diffusion

BACKBONES = {
    'resnet': resnet, 
    'unet': unet,
    'TransUNet': transunet,
    'guided_diffusion_unet': guided_diffusion,
    }

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('ConvTranspose') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        nn.init.zeros_(m.bias)
