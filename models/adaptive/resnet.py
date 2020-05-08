# -*- coding:utf-8  -*-

import math
import numpy as np
import os
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class AdaptiveBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, bottleneck_settings, stride=1, downsample=None):
        super(AdaptiveBasicBlock, self).__init__()
        conv1_in_ch, conv1_out_ch = bottleneck_settings['conv1']
        self.conv1 = conv3x3(conv1_in_ch, conv1_out_ch, stride)
        self.bn1 = nn.BatchNorm2d(conv1_out_ch)

        conv2_in_ch, conv2_out_ch = bottleneck_settings['conv2']
        self.conv2 = conv3x3(conv2_in_ch, conv2_out_ch)
        self.bn2 = nn.BatchNorm2d(conv2_out_ch)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class AdaptiveBottleneck(nn.Module):
    expansion = 4

    def __init__(self, bottleneck_settings, stride=1, downsample=None):
        super(AdaptiveBottleneck, self).__init__()
        conv1_in_ch, conv1_out_ch = bottleneck_settings['conv1']
        self.conv1 = nn.Conv2d(conv1_in_ch, conv1_out_ch, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(conv1_out_ch)

        conv2_in_ch, conv2_out_ch = bottleneck_settings['conv2']
        self.conv2 = nn.Conv2d(conv2_in_ch, conv2_out_ch, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(conv2_out_ch)

        conv3_in_ch, conv3_out_ch = bottleneck_settings['conv3']
        self.conv3 = nn.Conv2d(conv3_in_ch, conv3_out_ch, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(conv3_out_ch)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class AdaptiveResNet(nn.Module):
    def __init__(self, ch_cfg, block, layers, num_classes=1000, input_size=224):
        super(AdaptiveResNet, self).__init__()

        channels = np.load(os.path.join(ch_cfg, 'sample.npy'), allow_pickle=True).item()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, channels['conv1'], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(channels['conv1'])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], 1, channels['layer1'])
        self.layer2 = self._make_layer(block, 128, layers[1], 2, channels['layer2'])
        self.layer3 = self._make_layer(block, 256, layers[2], 2, channels['layer3'])
        self.layer4 = self._make_layer(block, 512, layers[3], 2, channels['layer4'])

        self.avgpool = nn.AvgPool2d(input_size // 32, stride=1)
        self.fc = nn.Linear(channels['fc'], num_classes)

        self._init_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride, bottleneck_settings):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            in_ch, _ = bottleneck_settings['0']['conv1']
            if 'conv3' in bottleneck_settings['0'].keys():
                _, out_ch = bottleneck_settings['0']['conv3']
            else:
                # basic block
                _, out_ch = bottleneck_settings['0']['conv2']

            downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

        layers = [block(bottleneck_settings['0'], stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(bottleneck_settings[str(i)]))

        return nn.Sequential(*layers)


def adaptive_res18(ch_cfg, num_classes=1000, input_size=224):
    return AdaptiveResNet(ch_cfg, AdaptiveBasicBlock, [2, 2, 2, 2], num_classes, input_size)


def adaptive_res50(ch_cfg, num_classes=1000, input_size=224):
    return AdaptiveResNet(ch_cfg, AdaptiveBottleneck, [3, 4, 6, 3], num_classes, input_size)
