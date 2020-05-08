# -*- coding:utf-8  -*-

import numpy as np
import os
import torch.nn as nn


class AdaptiveInvertedResidual(nn.Module):
    def __init__(self, residual_settings, stride, t):
        super(AdaptiveInvertedResidual, self).__init__()

        assert stride in [1, 2]
        self.stride = stride
        self.t = t
        self.relu = nn.ReLU6(inplace=True)

        if t != 1:
            conv1_in_ch, conv1_out_ch = residual_settings['conv1']
            self.conv1 = nn.Conv2d(conv1_in_ch, conv1_out_ch, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(conv1_out_ch)

        conv2_in_ch, conv2_out_ch = residual_settings['conv2']
        self.conv2 = nn.Conv2d(conv2_in_ch, conv2_out_ch, 3, stride, 1, groups=conv2_in_ch, bias=False)
        self.bn2 = nn.BatchNorm2d(conv2_out_ch)

        conv3_in_ch, conv3_out_ch = residual_settings['conv3']
        self.conv3 = nn.Conv2d(conv3_in_ch, conv3_out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(conv3_out_ch)

        self.inplanes = conv1_in_ch if t != 1 else conv2_in_ch
        self.outplanes = conv3_out_ch

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


class AdaptiveMobileNetV2(nn.Module):
    def __init__(self, ch_cfg, num_classes=1000, input_size=224):
        super(AdaptiveMobileNetV2, self).__init__()

        channels = np.load(os.path.join(ch_cfg, 'sample.npy'), allow_pickle=True).item()
        self.num_classes = num_classes
        self.relu = nn.ReLU6(inplace=True)

        conv1_out_ch = channels['conv1']
        self.conv1 = nn.Conv2d(3, conv1_out_ch, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(conv1_out_ch)

        self.blocks = self._make_blocks(channels)

        conv_last_in_ch, conv_last_out_ch = channels['conv_last']
        self.conv_last = nn.Conv2d(conv_last_in_ch, conv_last_out_ch, 1,
                                   bias=False)
        self.bn_last = nn.BatchNorm2d(conv_last_out_ch)
        self.avgpool = nn.AvgPool2d(input_size // 32)
        self.fc = nn.Conv2d(conv_last_out_ch, self.num_classes, 1)

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

    def _make_blocks(self, residual_settings):
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
        cnt = 0
        for t, c, n, s in blocks_setting:
            for i in range(n):
                blocks.append(AdaptiveInvertedResidual(
                    residual_settings[str(cnt)], s if i == 0 else 1, t))
                cnt += 1

        return nn.Sequential(*blocks)


def adaptive_mobilenet_v2(ch_cfg, num_classes=1000, input_size=224):
    return AdaptiveMobileNetV2(ch_cfg, num_classes, input_size)
