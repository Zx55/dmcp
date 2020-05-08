# -*- coding:utf-8  -*-

import torch
import math


class WarmUpCosineLRScheduler:
    """
    update lr every step
    """
    def __init__(self, optimizer, T_max, eta_min, base_lr, warmup_lr, warmup_steps, last_iter=-1):
        # Attach optimizer
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        # Initialize step and base learning rates
        if last_iter == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.last_iter = last_iter

        assert warmup_steps < T_max
        self.T_max = T_max
        self.eta_min = eta_min

        # warmup settings
        self.base_lr = base_lr
        self.warmup_lr = warmup_lr
        self.warmup_step = warmup_steps
        self.warmup_k = None

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def get_lr(self):
        return list(map(lambda group: group['lr'], self.optimizer.param_groups))

    def step(self, this_iter=None):
        if this_iter is None:
            this_iter = self.last_iter + 1
        self.last_iter = this_iter

        # get lr during warmup stage
        if self.warmup_step > 0 and this_iter < self.warmup_step:
            if self.warmup_k is None:
                self.warmup_k = (self.warmup_lr - self.base_lr) / self.warmup_step
            scale = (self.warmup_k * this_iter + self.base_lr) / self.base_lr
        # get lr during cosine annealing
        else:
            step_ratio = (this_iter - self.warmup_step) / (self.T_max - self.warmup_step)
            scale = self.eta_min + (self.warmup_lr - self.eta_min) * (1 + math.cos(math.pi * step_ratio)) / 2
            scale /= self.base_lr

        values = [scale * lr for lr in self.base_lrs]
        for param_group, lr in zip(self.optimizer.param_groups, values):
            param_group['lr'] = lr
