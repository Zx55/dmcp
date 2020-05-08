# -*- coding:utf-8  -*-

# -*- coding:utf-8  -*-

import random
import torch.nn as nn
import torch.nn.functional as F


def make_divisible(x, divisor=8, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8.

    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor

    new_v = max(min_value, int(x + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * x:
        new_v += divisor

    return int(new_v)


class USModule:
    """
    Base class for universally slimmable layers.

    :param us_switch: two boolean indicating if input channel and output
                    channel is slimmable, e.g., [True, True]
    """
    def __init__(self, in_ch, out_ch, group, us_switch, expand):
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.group = group
        self.us_switch = us_switch
        self.expand = expand

        # number of channel at current width
        self.cur_in_ch = in_ch
        self.cur_out_ch = out_ch
        self.cur_group = group

    def set_width(self, width):
        self.set_input_width(width)
        self.set_output_width(width)

    def set_input_width(self, width=None, specific_ch=None):
        if not self.us_switch[0]:
            return

        if width is not None:
            self.cur_in_ch = int(make_divisible(self.in_ch * width / self.expand))

        if specific_ch is not None:
            self.cur_in_ch = specific_ch

        if self.group != 1:
            self.cur_group = self.cur_in_ch

    def set_output_width(self, width=None, specific_ch=None):
        if not self.us_switch[1]:
            return

        if width is not None:
            self.cur_out_ch = int(make_divisible(self.out_ch * width / self.expand))

        if specific_ch is not None:
            self.cur_out_ch = specific_ch

    def random_sample_output_width(self, candidate):
        width = candidate[random.randint(0, len(candidate) - 1)]
        self.set_output_width(width)
        return width, self.cur_out_ch


class USConv2d(USModule, nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True, us_switch=None, expand=1.0):
        us_switch = [True, True] if us_switch is None else us_switch
        super(USConv2d, self).__init__(in_channels, out_channels, groups, us_switch, expand)
        super(USModule, self).__init__(in_channels, out_channels, kernel_size, stride, padding,
                                       dilation, groups, bias)

    def forward(self, x):
        weight = self.weight[:self.cur_out_ch, :self.cur_in_ch, :, :]
        bias = None if self.bias is None else self.bias[:self.cur_out_ch]
        return F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.cur_group)


class USLinear(USModule, nn.Linear):
    def __init__(self, input_features, output_features, bias=True, us_switch=None, expand=1.0):
        us_switch = [True, True] if us_switch is None else us_switch
        super(USLinear, self).__init__(input_features, output_features, None, us_switch, expand)
        super(USModule, self).__init__(input_features, output_features, bias=bias)

    def forward(self, x):
        weight = self.weight[:self.cur_out_ch, :self.cur_in_ch]
        bias = None if self.bias is None else self.bias[:self.cur_out_ch]
        return F.linear(x, weight, bias)


class USBatchNorm2d(USModule, nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, us_switch=None, expand=1.0):
        us_switch = [False, True] if us_switch is None else us_switch
        super(USBatchNorm2d, self).__init__(None, num_features, None, us_switch, expand)
        super(USModule, self).__init__(num_features, eps, momentum, affine, track_running_stats=True)

    def forward(self, x):
        weight = self.weight[:self.cur_out_ch] if self.affine else self.weight
        bias = self.bias[:self.cur_out_ch] if self.affine else self.bias

        running_mean = self.running_mean[:self.cur_out_ch]
        running_var = self.running_var[:self.cur_out_ch]

        return F.batch_norm(x, running_mean, running_var, weight, bias, self.training, self.momentum, self.eps)
