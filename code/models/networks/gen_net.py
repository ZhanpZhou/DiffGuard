import torch.nn as nn
from . import BACKBONES


class GenNet(nn.Module):

    def __init__(self, backbone, in_channel, out_channel, **kwargs):
        super(GenNet, self).__init__()
        self.backbone, hidden_dim = BACKBONES[backbone](in_channel=in_channel, out_channel=out_channel, **kwargs)
 
    def forward(self, input, *args):
        return self.backbone(input)