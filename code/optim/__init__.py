import torch.nn as nn
from .dice import DiceLoss, DiceCELoss

def get_cls_criterion(loss_type, opt):
    if opt.class_num > 1:
        if loss_type == 'ce':
            return nn.CrossEntropyLoss()
        else:
            raise ValueError("not support criterion named: {}".format(loss_type))
    else:
        if loss_type == 'ce':
            return nn.BCEWithLogitsLoss()
        else:
            raise ValueError("not support criterion named: {}".format(loss_type))

def get_reg_criterion(loss_type, opt):
    if loss_type.lower() == 'l1':
        return nn.L1Loss(reduction='none')
    elif loss_type.lower() in ('l2', 'mse'):
        return nn.MSELoss(reduction='none')

def get_seg_criterion(loss_type, opt):
    if loss_type.lower() == 'ce':
        return nn.CrossEntropyLoss()
    elif loss_type.lower() == 'dice':
        return DiceLoss(opt.class_num)
    elif loss_type.lower() == 'dice_ce':
        return DiceCELoss(opt.class_num)
    else:
        raise ValueError("not support criterion named: {}".format(loss_type))
