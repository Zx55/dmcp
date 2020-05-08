# -*- coding:utf-8  -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothCELoss(nn.Module):
    def __init__(self, smooth_ratio, num_classes):
        super(LabelSmoothCELoss, self).__init__()

        self.smooth_ratio = smooth_ratio
        self.val = smooth_ratio / num_classes
        self.log_soft = nn.LogSoftmax(dim=1)

    def forward(self, x, label):
        one_hot = torch.zeros_like(x)
        one_hot.fill_(self.val)
        y = label.to(torch.long).view(-1, 1)
        one_hot.scatter_(1, y, 1 - self.smooth_ratio + self.val)

        loss = -torch.sum(self.log_soft(x) * one_hot.detach()) / x.size(0)
        return loss


def KL(temperature):
    def kl_loss(student_outputs, teacher_outputs):
        loss = nn.KLDivLoss(size_average=False, reduce=False)(
            F.log_softmax(student_outputs / temperature, dim=1),
            F.softmax(teacher_outputs.detach() / temperature, dim=1)) \
                * (temperature * temperature)
        return torch.mean(torch.sum(loss, dim=-1))
    return kl_loss


def flop_loss(config, model):
    input_size = config.dataset.input_size
    loss_type = config.arch.floss_type
    loss_weight = config.arch.flop_loss_weight
    target_flops = config.arch.target_flops

    e_flops = model.module.expected_flops(input_size, input_size)
    if loss_type == 'l2':
        loss = torch.pow(e_flops - float(target_flops), 2)
    elif loss_type == 'inverted_log_l1':
        loss = -torch.log(1 / (torch.abs(e_flops - target_flops) + 1e-5))
    elif loss_type == 'log_l1':
        # piecewise log function
        ratio = 1.0
        if abs(e_flops.item() - target_flops) > 200:
            ratio = 0.1

        if e_flops < target_flops * 0.95:
            loss = torch.log(ratio * torch.abs(e_flops - target_flops))
        elif target_flops * 0.95 <= e_flops < target_flops:
            loss = e_flops * 0
        else:
            loss = torch.log(ratio * torch.abs(e_flops - (target_flops * 0.95)))
    elif loss_type == 'l1':
        loss = torch.abs(e_flops - float(target_flops))
    else:
        raise NotImplementedError

    return loss_weight * loss, e_flops
