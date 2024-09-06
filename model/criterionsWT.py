import torch
import logging
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

def expand_target(x, n_class,mode='softmax'):
    """
        Converts NxDxHxW label image to NxCxDxHxW, where each label is stored in a separate channel
        :param input: 4D input image (NxDxHxW)
        :param C: number of channels/labels
        :return: 5D output image (NxCxDxHxW)
        """
    assert x.dim() == 4
    shape = list(x.size())
    shape.insert(1, n_class)
    shape = tuple(shape)
    xx = torch.zeros(shape)
    if mode.lower() == 'softmax':
        xx[:, 1, :, :, :] = (x == 1)
        #xx[:, 2, :, :, :] = (x == 2)
        #xx[:, 3, :, :, :] = (x == 3)
    if mode.lower() == 'sigmoid':
        xx[:, 0, :, :, :] = (x == 1)
        #xx[:, 1, :, :, :] = (x == 2)
        #xx[:, 2, :, :, :] = (x == 3)
    return xx.to(x.device)

def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.reshape(C, -1)

def Dice(output, target, eps=1e-5):
    #dice liss
    target = target.float()
    num = 2 * (output * target).sum()
    den = output.sum() + target.sum() + eps

    return (1.0 - num/den)


def softmax_dice(output, target):
    '''
    The dice loss for using softmax activation function
    :param output: (b, num_class, d, h, w)
    :param target: (b, d, h, w)
    :return: softmax dice loss
    '''
    loss1 = Dice(output[:, 1, ...], (target == 1).float())
    loss2 = Dice(output[:, 2, ...], (target == 2).float())
    loss3 = Dice(output[:, 3, ...], (target == 4).float())

    return loss1 + loss2 + loss3, 1-loss1.data, 1-loss2.data, 1-loss3.data


def softmax_dice2(output,target):
    '''
    The dice loss for using softmax activation function
    :param output: (b, num_class, d, h, w)
    :param target: (b, d, h, w)
    :return: softmax dice loss
    '''
    #target = (target>0).float()
    #loss0 = Dice(output1[:, 0, ...], (target == 0).float())
    #loss1 = Dice(output1[:, 1, ...], (target >= 1).float())
    #loss_1 = 0.5*loss0 + 0.5*loss1
    loss00 = Dice(output[:, 0, ...], (target == 0 ).float())
    loss01 = Dice(output[:, 1, ...], (target == 1 ).float())
    #loss02 = Dice(output[:, 2, ...], (target == 2 ).float())
    #loss_2 = 0.2*loss00+0.2*loss01+loss02
    #loss03 = Dice(output[:, 3, ...], (target == 3 ).float())
    loss = loss00 + loss01
   
    return loss,loss00.data, loss01.data


def sigmoid_dice(output, target):
    '''
    The dice loss for using sigmoid activation function
    :param output: (b, num_class-1, d, h, w)
    :param target: (b, d, h, w)
    :return:
    '''
    loss1 = Dice(output[:, 0, ...], (target == 1).float())
    loss2 = Dice(output[:, 1, ...], (target == 2).float())
    loss3 = Dice(output[:, 2, ...], (target == 4).float())

    return loss1 + loss2 + loss3, 1-loss1.data, 1-loss2.data, 1-loss3.data


def Generalized_dice(output, target, eps=1e-5, weight_type='square'):
    if target.dim() == 4:  #(b, h, w, d)
        #target[target == 4] = 3  #transfer label 4 to 3
        target = target.long()
        target = expand_target(target, n_class=output.size()[1])  #extend target from (b, h, w, d) to (b, c, h, w, d)

    output = flatten(output)[1:, ...]  # transpose [N,4，H,W,D] -> [4，N,H,W,D] -> [3, N*H*W*D] voxels
    target = flatten(target)[1:, ...]  # [class, N*H*W*D]

    target_sum = target.sum(-1)  # sub_class_voxels [3,1] -> 3个voxels
    if weight_type == 'square':
        class_weights = 1. / (target_sum * target_sum + eps)
    elif weight_type == 'identity':
        class_weights = 1. / (target_sum + eps)
    elif weight_type == 'sqrt':
        class_weights = 1. / (torch.sqrt(target_sum) + eps)
    else:
        raise ValueError('Check out the weight_type :', weight_type)

    # print(class_weights)
    intersect = (output * target).sum(-1)
    intersect_sum = (intersect * class_weights).sum()
    denominator = (output + target).sum(-1)
    denominator_sum = (denominator * class_weights).sum() + eps

    loss1 = 2*intersect[0] / (denominator[0] + eps)
    #loss2 = 2*intersect[1] / (denominator[1] + eps)
    #loss3 = 2*intersect[2] / (denominator[2] + eps)

    return 1 - 2. * intersect_sum / denominator_sum, loss1

def cross_entropy_3D(output, target):
    n,c,h,w,s = output.size()
    log_p = F.log_softmax(output,dim=1)
    log_p = log_p.transpose(1,2).transpose(2,3).transpose(3,4).contiguous().view(-1,c)
    target = target.long()
    target = target.view(target.numel())
    loss = F.nll_loss(log_p,target)
    return loss

def CE_DICE_GDL(output,target):
    loss_CE = cross_entropy_3D(output, target)
    loss_GDL,loss1 = Generalized_dice(output, target, eps=1e-5, weight_type='square')
    loss_dice,loss_dice_0,loss_dice_WT = softmax_dice2(output, target)
    return loss_CE+loss_GDL+loss_dice, loss_GDL.data, loss_dice_WT.data
 



def Dual_focal_loss(output, target):
    loss1 = Dice(output[:, 1, ...], (target == 1).float())
    loss2 = Dice(output[:, 2, ...], (target == 2).float())
    loss3 = Dice(output[:, 3, ...], (target == 4).float())
    
    if target.dim() == 4:  #(b, h, w, d)
        target[target == 4] = 3  #transfer label 4 to 3
        target = expand_target(target, n_class=output.size()[1])  #extend target from (b, h, w, d) to (b, c, h, w, d)

    target = target.permute(1, 0, 2, 3, 4).contiguous()
    output = output.permute(1, 0, 2, 3, 4).contiguous()
    target = target.view(4, -1)
    output = output.view(4, -1)
    log = 1-(target - output)**2

    return -(F.log_softmax((1-(target - output)**2), 0)).mean(), 1-loss1.data, 1-loss2.data, 1-loss3.data
