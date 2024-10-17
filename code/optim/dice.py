import torch
import torch.nn as nn

def dice_coeff(input, target, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon), [(2 * inter + epsilon) / (sets_sum + epsilon)]
    else:
        # compute and average metric for each batch element
        dice_acc = 0
        all_dices = []
        for i in range(input.shape[0]):
            dice, _ = dice_coeff(input[i, ...], target[i, ...])
            dice_acc += dice
            all_dices.append(dice.item())
        
        return dice_acc / input.shape[0], all_dices

def multiclass_dice_coeff(input, target, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice_acc = 0
    all_dices = []
    for channel in range(input.shape[1]):
        average_dice, dices = dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)
        dice_acc += average_dice
        all_dices.extend(dices)

    return dice_acc / input.shape[1], all_dices

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=True):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

class DiceCELoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceCELoss, self).__init__()
        self.dice_loss = DiceLoss(n_classes)
        self.ce_loss = nn.CrossEntropyLoss()
        self.n_classes = n_classes

    def forward(self, inputs, target, weight=None, softmax=True):
        loss = 0.5 * (self.dice_loss(inputs, target, weight, softmax) + self.ce_loss(inputs, target))
        return loss

