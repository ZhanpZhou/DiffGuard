import cv2
import time
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
from optim.dice import dice_coeff, multiclass_dice_coeff

class Meter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class TimeMeter(Meter):
    def __init__(self):
        Meter.__init__(self)

    def start(self):
        self.start_time = time.time()

    def record(self, n = 1):
        spent_time = time.time() - self.start_time
        self.update(spent_time, n)

# def accuracy(output, target, topk=(1,)):
#     if output.shape[1] == 1:
#         output = (output.view(-1) > 0.5).long()
#         correct = output.eq(target.view(-1))
#         return [torch.sum(correct).float()/correct.shape[0] * 100]

#     """Computes the precision@k for the specified values of k"""
#     maxk = max(topk)
#     batch_size = target.size(0)

#     _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t()
#     correct = pred.eq(target.view(1, -1).expand_as(pred))

#     res = []
#     for k in topk:
#         correct_k = correct[:k].view(-1).float().sum(0)
#         res.append(correct_k.mul_(100.0 / batch_size))
#     return res

def accuracy(pred, target):
    correct = (pred.cpu().long() == target)
    return 100.0 * correct.sum() / len(correct)

def accuracy_from_matrix(c_matrix):
    total = np.sum(c_matrix)
    correct = np.trace(c_matrix)
    return 100.0 * correct / total

def precision(c_matrix, ti):
    pre = c_matrix[ti,ti] / np.sum(c_matrix[:,ti])
    return pre

def recall(c_matrix, ti):
    recall = c_matrix[ti,ti] / np.sum(c_matrix[ti])
    return recall

def f_score(c_matrix, ti):
    pre = c_matrix[ti, ti] / np.sum(c_matrix[:, ti])
    recall = c_matrix[ti, ti] / np.sum(c_matrix[ti])
    score = 2 * pre * recall / (pre + recall)
    return score

def comfusion_matrix(preds, labels, c_num):
    if c_num == 1: # single output
        c_num = 2 
    confuse_m = np.zeros((c_num, c_num))
    for i in range(len(labels)):
        label = int(labels[i])
        pred = int(preds[i]) 
        confuse_m[label,pred] += 1
    return confuse_m
    
# def comfusion_matrix(output, target, c_num, thred = 0.5):
#     confuse_m = np.zeros((c_num, c_num))
#     if len(output.shape) == 1:
#         pred = output
#     else:
#         _, pred = torch.max(output, dim=1)
#
#     for m in range(pred.shape[0]):
#         label = target[m].item()
#         label = int(label)
#
#         if len(output.shape) == 1:
#             if output[m] > thred:
#                 pre = 1
#             else:
#                 pre = 0
#
#         elif output.shape[1] == 1:
#             if output[m][0] > thred:
#                 pre = 1
#             else:
#                 pre = 0
#         else:
#             pre = pred[m].item()
#
#
#         confuse_m[label][pre] += 1
#     return confuse_m
#
def auc_score(y_true, y_scores):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    if isinstance(y_scores, torch.Tensor):
        y_scores = y_scores.detach().cpu().numpy()

    if isinstance(y_true, list):
        y_true = np.array(y_true)

    if isinstance(y_scores, list):
        y_scores = np.array(y_scores)
    
    if y_scores.shape[1] == 2:
        # binary classification
        try:
            auc = roc_auc_score(y_true, y_scores[:, 1]) 
        except:
            return 1.0
    else:
        # multi-class AUC in 'macro' mode
        label_mask = np.zeros_like(y_scores, dtype=np.uint8)
        for i, label in enumerate(y_true):
            label_mask[i, label] = 1
        auc = roc_auc_score(label_mask, y_scores)

    return auc

def Dice_Similarity_Coefficient(mask_preds, mask_trues, num_classes=1):
    all_dice_scores = []
    mask_preds_one_hot = F.one_hot(mask_preds, num_classes).detach().permute(0, 3, 1, 2).float()
    if num_classes == 1:
        # compute the Dice score
        dice_score, dice_scores = dice_coeff(mask_preds_one_hot, mask_trues, reduce_batch_first=False)
        all_dice_scores.extend(dice_scores)
    else:
        tensor_list = []
        for i in range(num_classes):
            temp_prob = mask_trues == i 
            tensor_list.append(temp_prob.unsqueeze(1))
        mask_trues = torch.cat(tensor_list, dim=1).float()
        # compute the Dice score, ignoring background
        dice_score, dice_scores = multiclass_dice_coeff(mask_preds_one_hot[:, 1:, ...], mask_trues[:, 1:, ...], reduce_batch_first=False)
        all_dice_scores.extend(dice_scores)
    
    return np.mean(all_dice_scores), all_dice_scores
