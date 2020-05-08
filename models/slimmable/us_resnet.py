# -*- coding:utf-8  -*-

import math
import torch.nn as nn
import torch.nn.functional as F
from models.slimmable.us_ops import USConv2d, USBatchNorm2d, USModule


def conv3x3(in_planes, out_planes, stride=1, expand=1.0):
    return USConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                    padding=1, bias=False, expand=expand)


class USBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, expand=1.0):
        super(USBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, expand=expand)
        self.bn1 = USBatchNorm2d(planes, expand=expand)

        self.conv2 = conv3x3(planes, planes, expand=expand)
        self.bn2 = USBatchNorm2d(planes, expand=expand)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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

    def non_uniform_set_width(self, ch_in, candidate):
        self.conv1.set_input_width(specific_ch=ch_in)
        _, ch_out = self.conv1.random_sample_output_width(candidate)
        self.bn1.set_output_width(specific_ch=ch_out)

        self.conv2.set_input_width(specific_ch=ch_out)
        if self.downsample is None:
            self.conv2.set_output_width(specific_ch=ch_in)
            self.bn2.set_output_width(specific_ch=ch_in)
            ch_out = ch_in
        else:
            _, ch_out = self.conv2.random_sample_output_width(candidate)
            self.bn2.set_output_width(specific_ch=ch_out)
            self.downsample[0].set_input_width(specific_ch=ch_in)
            self.downsample[0].set_output_width(specific_ch=ch_out)
            self.downsample[1].set_output_width(specific_ch=ch_out)

        return ch_out


class USBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, expand=1.0):
        super(USBottleneck, self).__init__()

        self.conv1 = USConv2d(inplanes, planes, kernel_size=1,
                              bias=False, expand=expand)
        self.bn1 = USBatchNorm2d(planes, expand=expand)
        self.conv2 = USConv2d(planes, planes, kernel_size=3, stride=stride,
                              padding=1, bias=False, expand=expand)
        self.bn2 = USBatchNorm2d(planes, expand=expand)
        self.conv3 = USConv2d(planes, planes * 4, kernel_size=1,
                              bias=False, expand=expand)
        self.bn3 = USBatchNorm2d(planes * 4, expand=expand)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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

    def non_uniform_set_width(self, ch_in, candidate):
        self.conv1.set_input_width(specific_ch=ch_in)
        _, ch_out = self.conv1.random_sample_output_width(candidate)
        self.bn1.set_output_width(specific_ch=ch_out)

        self.conv2.set_input_width(specific_ch=ch_out)
        _, ch_out = self.conv2.random_sample_output_width(candidate)
        self.bn2.set_output_width(specific_ch=ch_out)

        self.conv3.set_input_width(specific_ch=ch_out)
        if self.downsample is None:
            self.conv3.set_output_width(specific_ch=ch_in)
            self.bn3.set_output_width(specific_ch=ch_in)
            ch_out = ch_in
        else:
            _, ch_out = self.conv3.random_sample_output_width(candidate)
            self.bn3.set_output_width(specific_ch=ch_out)
            self.downsample[0].set_input_width(specific_ch=ch_in)
            self.downsample[0].set_output_width(specific_ch=ch_out)
            self.downsample[1].set_output_width(specific_ch=ch_out)

        return ch_out


class USResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, input_size=224, max_width=1.0):
        super(USResNet, self).__init__()
        self.inplanes = int(64 * max_width)
        self.max_width = max_width
        self.input_size = input_size

        self.conv1 = USConv2d(3, int(64 * self.max_width), kernel_size=7, stride=2, padding=3,
                              bias=False, us_switch=[False, True], expand=max_width)
        self.bn1 = USBatchNorm2d(self.inplanes, expand=max_width)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, int(64 * max_width), layers[0])
        self.layer2 = self._make_layer(block, int(128 * max_width), layers[1], stride=2)
        self.layer3 = self._make_layer(block, int(256 * max_width), layers[2], stride=2)
        self.layer4 = self._make_layer(block, int(512 * max_width), layers[3], stride=2)

        self.avgpool = nn.AvgPool2d(self.input_size // 32, stride=1)
        self.fc = USConv2d(int(512 * block.expansion * max_width), num_classes, kernel_size=1,
                           us_switch=[True, False], expand=max_width)

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

        x = self.fc(x)
        x = x.view(x.size(0), -1)
        return x

    def uniform_set_width(self, width):
        for m in self.modules():
            if isinstance(m, USModule):
                m.set_width(width)

    def non_uniform_set_width(self, candidate):
        _, ch_out = self.conv1.random_sample_output_width(candidate)
        self.bn1.set_output_width(specific_ch=ch_out)

        for m in self.modules():
            if isinstance(m, (USBasicBlock, USBottleneck)):
                ch_out = m.non_uniform_set_width(ch_out, candidate)

        self.fc.set_input_width(specific_ch=ch_out)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                USConv2d(self.inplanes, planes * block.expansion, kernel_size=1,
                         stride=stride, bias=False, expand=self.max_width),
                USBatchNorm2d(planes * block.expansion, expand=self.max_width),
            )

        layers = [block(self.inplanes, planes, stride, downsample, expand=self.max_width)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, expand=self.max_width))

        return nn.Sequential(*layers)


def us_resnet18(num_classes=1000, input_size=224, max_width=1.0):
    return USResNet(USBasicBlock, [2, 2, 2, 2], num_classes, input_size, max_width)


def us_resnet50(num_classes=1000, input_size=224, max_width=1.0):
    return USResNet(USBottleneck, [3, 4, 6, 3], num_classes, input_size, max_width)
