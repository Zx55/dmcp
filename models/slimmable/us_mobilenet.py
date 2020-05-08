# -*- coding:utf-8  -*-

import torch.nn as nn
from models.slimmable.us_ops import USModule, USConv2d, USBatchNorm2d, make_divisible


class USInvertedResidual(nn.Module):
    def __init__(self, inplanes, outplanes, stride, t, expand=1.0):
        super(USInvertedResidual, self).__init__()

        assert stride in [1, 2]
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.stride = stride
        self.t = t
        hidden_dim = int(inplanes * t)

        self.hidden_dim = hidden_dim
        self.expand = expand

        self.relu = nn.ReLU6(inplace=True)

        if t != 1:
            self.conv1 = USConv2d(inplanes, hidden_dim, 1, bias=False, expand=expand)
            self.bn1 = USBatchNorm2d(hidden_dim, expand=expand)
        # depth-wise
        self.conv2 = USConv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim,
                              bias=False, expand=expand)
        self.bn2 = USBatchNorm2d(hidden_dim, expand=expand)
        # point-wise
        self.conv3 = USConv2d(hidden_dim, outplanes, 1, bias=False, expand=expand)
        self.bn3 = USBatchNorm2d(outplanes, expand=expand)

    def forward(self, x):
        residual = x

        if self.t != 1:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
        else:
            out = x

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.stride == 1 and self.inplanes == self.outplanes:
            out += residual

        return out

    def non_uniform_set_width(self, ch_in, candidate):
        if self.t == 1:
            ch_out = ch_in
        else:
            self.conv1.set_input_width(specific_ch=ch_in)
            _, ch_out = self.conv1.random_sample_output_width(candidate)
            self.bn1.set_output_width(specific_ch=ch_out)

        self.conv2.set_input_width(specific_ch=ch_out)
        self.conv2.set_output_width(specific_ch=ch_out)
        self.bn2.set_output_width(specific_ch=ch_out)

        self.conv3.set_input_width(specific_ch=ch_out)
        if self.stride == 1 and self.inplanes == self.outplanes:
            self.conv3.set_output_width(specific_ch=ch_in)
            self.bn3.set_output_width(specific_ch=ch_in)
            ch_out = ch_in
        else:
            _, ch_out = self.conv3.random_sample_output_width(candidate)
            self.bn3.set_output_width(specific_ch=ch_out)

        return ch_out


class USMobileNetV2(nn.Module):
    block = USInvertedResidual

    def __init__(self, num_classes=1000, input_size=224, max_width=1.0):
        super(USMobileNetV2, self).__init__()
        self.num_classes = num_classes
        self.max_width = max_width
        self.last_channel = make_divisible(1280 * max_width) \
            if max_width > 1.0 else 1280
        ch_out = make_divisible(32 * max_width)

        self.relu = nn.ReLU6(inplace=True)
        self.conv1 = USConv2d(3, ch_out, 3, 2, 1, bias=False,
                              us_switch=[False, True], expand=max_width)
        self.bn1 = USBatchNorm2d(ch_out, expand=max_width)
        self.blocks, blocks_out = self._make_blocks()
        self.conv_last = USConv2d(blocks_out, self.last_channel, 1,
                                  bias=False, expand=max_width)
        self.bn_last = USBatchNorm2d(self.last_channel, expand=max_width)
        self.avgpool = nn.AvgPool2d(input_size // 32)
        self.fc = USConv2d(self.last_channel, self.num_classes, 1,
                           us_switch=[True, False], expand=max_width)

        self._init_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.blocks(x)
        x = self.conv_last(x)
        x = self.bn_last(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = self.fc(x)
        return x.view(x.size(0), -1)

    def uniform_set_width(self, width):
        for m in self.modules():
            if isinstance(m, USModule):
                m.set_width(width)

    def non_uniform_set_width(self, candidate):
        _, ch_out = self.conv1.random_sample_output_width(candidate)
        self.bn1.set_output_width(specific_ch=ch_out)

        for m in self.modules():
            if isinstance(m, USInvertedResidual):
                ch_out = m.non_uniform_set_width(ch_out, candidate)

        self.conv_last.set_input_width(specific_ch=ch_out)
        _, ch_out = self.conv_last.random_sample_output_width(candidate)
        self.bn_last.set_output_width(specific_ch=ch_out)
        self.fc.set_input_width(specific_ch=ch_out)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_blocks(self):
        blocks_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # for cifar-10
        if self.num_classes == 10:
            blocks_setting[2] = [6, 24, 2, 1]

        blocks = []
        ch_in, ch_out = make_divisible(32 * self.max_width), 0
        for t, c, n, s in blocks_setting:
            ch_out = make_divisible(c * self.max_width)
            for i in range(n):
                blocks.append(self.block(
                    ch_in, ch_out, s if i == 0 else 1, t, expand=self.max_width))
                ch_in = ch_out

        return nn.Sequential(*blocks), ch_out


def us_mobilenet_v2(num_classes=1000, input_size=224, max_width=1.0):
    return USMobileNetV2(num_classes, input_size, max_width)
