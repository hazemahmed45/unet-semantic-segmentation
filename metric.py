import numpy as np
import torch
from torch._C import dtype
from dataset import SaltDataset


def IOU(predicted_mask, mask):
    epsilon = 1e-13
    batch_size = predicted_mask.shape[0]
    # intersection = torch.logical_and(predicted_mask, mask)
    intersection = predicted_mask.bool() & mask.bool()
    # union = torch.logical_or(predicted_mask, mask)
    union = predicted_mask.bool() | mask.bool()
    # print(torch.sum(torch.sum(intersection, dim=2), dim=2).T,
    #       torch.sum(torch.sum(union, dim=2), dim=2).T)
    return torch.sum(torch.sum(intersection, dim=2), dim=2)/(torch.sum(torch.sum(union, dim=2), dim=2)+epsilon)


def get_iou_score(outputs, labels):
    batch_size = labels.shape[0]
    A = labels.squeeze().bool().view((batch_size, 101, 101))
    pred = torch.where(outputs < 0., torch.zeros_like(
        outputs), torch.ones_like(outputs))
    B = pred.squeeze().bool().view((batch_size, 101, 101))
    intersection = (A & B).float().sum((1, 2))
    union = (A | B).float().sum((1, 2))
    # print(intersection, union)
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou

# def AveragePrecisonIOU(IOUs_value):
#     print(IOUs_value)
#     threshholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
#     for thres in threshholds:
#         print(torch.tensor(IOUs_value > thres).float())

#     return


# data = SaltDataset('train.csv', 'train')
# img, mask1 = data[0]
# img, mask2 = data[1]
# img, mask3 = data[2]

# _, mask4 = data[3]
# _, mask5 = data[4]
# _, mask6 = data[5]
# mask = torch.stack([mask1, mask2, mask3])
# mask_ = torch.stack([mask4, mask5, mask6])
# print(mask.shape)
# get_iou_score(mask, mask_)
# IOU(mask, mask_)
