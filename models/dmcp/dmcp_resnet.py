# -*- coding:utf-8  -*-

from models.dmcp.alpha_op import AlphaLayer
from models.dmcp.utils import conv_compute_flops
from models.slimmable.us_resnet import USBasicBlock, USBottleneck, USResNet


Alpha = None


class DMCPBasicBlock(USBasicBlock):
    def __init__(self, inplanes, planes, stride=1, downsample=None, expand=1.0):
        super(DMCPBasicBlock, self).__init__(inplanes, planes, stride,
                                             downsample, expand=expand)

        global Alpha
        self.alpha1 = Alpha(planes)
        self.alpha2 = Alpha(planes) if downsample is not None else None
        self.alpha_training = False

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        if self.alpha_training:
            out = self.alpha1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.alpha_training and self.alpha2 is not None:
            out = self.alpha2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
            if self.alpha_training:
                residual = self.alpha2(residual)
        out += residual
        return self.relu(out)

    def expected_flops(self, in_alpha, in_height, in_width):
        if self.downsample is None and self.alpha2 is None:
            self.alpha2 = in_alpha
        e_in_channel = in_alpha.expected_channel()

        # conv1
        e_conv1_out = self.alpha1.expected_channel()
        e_conv1_flops, out_height, out_width = conv_compute_flops(
            self.conv1, in_height, in_width, e_in_channel, e_conv1_out)

        # conv2
        e_conv2_out = self.alpha2.expected_channel()
        e_conv2_flops, out_height, out_width = conv_compute_flops(
            self.conv2, out_height, out_width, e_conv1_out, e_conv2_out)

        e_flops = e_conv1_flops + e_conv2_flops
        # downsample
        if self.downsample is not None:
            e_downsample_flops, out_height, out_width = conv_compute_flops(
                self.downsample[0], in_height, in_width, e_in_channel, e_conv2_out)
            e_flops += e_downsample_flops

        return e_flops, out_height, out_width

    def direct_sampling(self, ch_in):
        self.conv1.set_input_width(specific_ch=ch_in)
        ch_out = self.alpha1.direct_sampling()
        self.conv1.set_output_width(specific_ch=ch_out)
        self.bn1.set_output_width(specific_ch=ch_out)

        self.conv2.set_input_width(specific_ch=ch_out)
        if self.downsample is None:
            self.conv2.set_output_width(specific_ch=ch_in)
            self.bn2.set_output_width(specific_ch=ch_in)
            ch_out = ch_in
        else:
            ch_out = self.alpha2.direct_sampling()
            self.conv2.set_output_width(specific_ch=ch_out)
            self.bn2.set_output_width(specific_ch=ch_out)
            self.downsample[0].set_input_width(specific_ch=ch_in)
            self.downsample[0].set_output_width(specific_ch=ch_out)
            self.downsample[1].set_output_width(specific_ch=ch_out)

        return ch_out

    def expected_sampling(self, ch_in, e_ch_in):
        self.conv1.set_input_width(specific_ch=e_ch_in)
        ch_out, e_ch = self.alpha1.expected_sampling()
        self.conv1.set_output_width(specific_ch=e_ch)
        self.bn1.set_output_width(specific_ch=e_ch)

        self.conv2.set_input_width(specific_ch=e_ch)
        if self.downsample is None:
            self.conv2.set_output_width(specific_ch=e_ch_in)
            self.bn2.set_output_width(specific_ch=e_ch_in)
            ch_out = ch_in
            e_ch = e_ch_in
        else:
            ch_out, e_ch = self.alpha2.expected_sampling()
            self.conv2.set_output_width(specific_ch=e_ch)
            self.bn2.set_output_width(specific_ch=e_ch)
            self.downsample[0].set_input_width(specific_ch=e_ch_in)
            self.downsample[0].set_output_width(specific_ch=e_ch)
            self.downsample[1].set_output_width(specific_ch=e_ch)

        return ch_out, e_ch


class DMCPBottleneck(USBottleneck):
    def __init__(self, inplanes, planes, stride=1, downsample=None, expand=1.0):
        super(DMCPBottleneck, self).__init__(inplanes, planes, stride,
                                             downsample, expand=expand)
        global Alpha
        self.inplanes = inplanes
        self.alpha1 = Alpha(planes)
        self.alpha2 = Alpha(planes)
        self.alpha3 = Alpha(planes * self.expansion) \
            if downsample is not None else None

        self.alpha_training = False

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        if self.alpha_training:
            out = self.alpha1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.alpha_training:
            out = self.alpha2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.alpha_training and self.alpha3 is not None:
            out = self.alpha3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
            if self.alpha_training:
                residual = self.alpha3(residual)
        out += residual
        return self.relu(out)

    def expected_flops(self, in_alpha, in_height, in_width):
        if self.downsample is None and self.alpha3 is None:
            self.alpha3 = in_alpha
        e_in_channel = in_alpha.expected_channel()

        # conv1
        e_conv1_out = self.alpha1.expected_channel()
        e_conv1_flops, out_height, out_width = conv_compute_flops(
            self.conv1, in_height, in_width, e_in_channel, e_conv1_out)

        # conv2
        e_conv2_out = self.alpha2.expected_channel()
        e_conv2_flops, out_height, out_width = conv_compute_flops(
            self.conv2, out_height, out_width, e_conv1_out, e_conv2_out)

        # conv3
        e_conv3_out = self.alpha3.expected_channel()
        e_conv3_flops, out_height, out_width = conv_compute_flops(
            self.conv3, out_height, out_width, e_conv2_out, e_conv3_out)

        e_flops = e_conv1_flops + e_conv2_flops + e_conv3_flops
        # downsample
        if self.downsample is not None:
            e_downsample_flops, out_height, out_width = conv_compute_flops(
                self.downsample[0], in_height, in_width, e_in_channel, e_conv3_out)
            e_flops += e_downsample_flops

        return e_flops, out_height, out_width

    def direct_sampling(self, ch_in):
        self.conv1.set_input_width(specific_ch=ch_in)
        ch_out = self.alpha1.direct_sampling()
        self.conv1.set_output_width(specific_ch=ch_out)
        self.bn1.set_output_width(specific_ch=ch_out)

        self.conv2.set_input_width(specific_ch=ch_out)
        ch_out = self.alpha1.direct_sampling()
        self.conv2.set_output_width(specific_ch=ch_out)
        self.bn2.set_output_width(specific_ch=ch_out)

        self.conv3.set_input_width(specific_ch=ch_out)
        if self.downsample is None:
            self.conv3.set_output_width(specific_ch=ch_in)
            self.bn3.set_output_width(specific_ch=ch_in)
            ch_out = ch_in
        else:
            ch_out = self.alpha3.direct_sampling()
            self.conv3.set_output_width(specific_ch=ch_out)
            self.bn3.set_output_width(specific_ch=ch_out)
            self.downsample[0].set_input_width(specific_ch=ch_in)
            self.downsample[0].set_output_width(specific_ch=ch_out)
            self.downsample[1].set_output_width(specific_ch=ch_out)

        return ch_out

    def expected_sampling(self, ch_in, e_ch_in):
        self.conv1.set_input_width(specific_ch=e_ch_in)
        ch_out, e_ch = self.alpha1.expected_sampling()
        self.conv1.set_output_width(specific_ch=e_ch)
        self.bn1.set_output_width(specific_ch=e_ch)

        self.conv2.set_input_width(specific_ch=e_ch)
        ch_out, e_ch = self.alpha2.expected_sampling()
        self.conv2.set_output_width(specific_ch=e_ch)
        self.bn2.set_output_width(specific_ch=e_ch)

        self.conv3.set_input_width(specific_ch=e_ch)
        if self.downsample is None:
            self.conv3.set_output_width(specific_ch=e_ch_in)
            self.bn3.set_output_width(specific_ch=e_ch_in)
            ch_out = ch_in
            e_ch = e_ch_in
        else:
            ch_out, e_ch = self.alpha3.expected_sampling()
            self.conv3.set_output_width(specific_ch=e_ch)
            self.bn3.set_output_width(specific_ch=e_ch)
            self.downsample[0].set_input_width(specific_ch=e_ch_in)
            self.downsample[0].set_output_width(specific_ch=e_ch)
            self.downsample[1].set_output_width(specific_ch=e_ch)

        return ch_out, e_ch


class DMCPResNet(USResNet):
    def __init__(self, block, layers, num_classes=1000, input_size=224,
                 width=None, prob_type='exp'):
        if width is None:
            width = [0.1, 1.0, 0.025]
        min_width, max_width, width_offset = width

        def alpha(channels):
            return AlphaLayer(channels, min_width, max_width, width_offset, prob_type)
        global Alpha
        Alpha = alpha

        super(DMCPResNet, self).__init__(block, layers, num_classes,
                                         input_size, max_width)

        self.alpha1 = Alpha(int(64 * max_width))
        self.alpha_training = False
        self.expected_flops(input_size, input_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        if self.alpha_training:
            x = self.alpha1(x)
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

    def expected_flops(self, in_height, in_width):
        # conv1
        total_flops, out_height, out_width = conv_compute_flops(
            self.conv1, in_height, in_width, e_out_ch=self.alpha1.expected_channel())
        cur_alpha = self.alpha1

        # max_pool
        out_height, out_width = out_height // 2, out_width // 2
        # bottlenecks
        for m in self.modules():
            if isinstance(m, (DMCPBasicBlock, DMCPBottleneck)):
                flops, out_height, out_width = m.expected_flops(
                    cur_alpha, out_height, out_width)
                total_flops += flops
                if isinstance(m, DMCPBottleneck):
                    cur_alpha = m.alpha3
                else:
                    cur_alpha = m.alpha2

        # fc
        flops, out_height, out_width = conv_compute_flops(
            self.fc, 1, 1, e_in_ch=cur_alpha.expected_channel())
        total_flops += flops

        return total_flops / 1e6

    # override
    def parameters(self, recurse=True):
        params = []
        for n, m in self.named_parameters():
            if n.find('alpha') > -1:
                continue
            params.append(m)
        return iter(params)

    def arch_parameters(self):
        params = []
        for n, m in self.named_parameters():
            if n.find('alpha') > -1:
                params.append(m)
        return iter(params)

    def direct_sampling(self):
        ch_out = self.alpha1.direct_sampling()
        self.conv1.set_output_width(specific_ch=ch_out)
        self.bn1.set_output_width(specific_ch=ch_out)

        for m in self.modules():
            if isinstance(m, (DMCPBasicBlock, DMCPBottleneck)):
                ch_out = m.direct_sampling(ch_out)

        self.fc.set_input_width(specific_ch=ch_out)

    def expected_sampling(self):
        ch_out, e_ch = self.alpha1.expected_sampling()
        self.conv1.set_output_width(specific_ch=e_ch)
        self.bn1.set_output_width(specific_ch=e_ch)

        for m in self.modules():
            if isinstance(m, (DMCPBasicBlock, DMCPBottleneck)):
                ch_out, e_ch = m.expected_sampling(ch_out, e_ch)

        self.fc.set_input_width(specific_ch=e_ch)

    def set_alpha_training(self, training):
        self.alpha_training = training
        for m in self.modules():
            if isinstance(m, (DMCPBasicBlock, DMCPBottleneck)):
                m.alpha_training = training


def dmcp_resnet18(num_classes=1000, input_size=224, width=None, prob_type='exp'):
    if width is None:
        width = [0.1, 1.0, 0.1]
    return DMCPResNet(DMCPBasicBlock, [2, 2, 2, 2], num_classes,
                      input_size, width, prob_type)


def dmcp_resnet50(num_classes=1000, input_size=224, width=None, prob_type='exp'):
    if width is None:
        width = [0.1, 1.0, 0.1]
    return DMCPResNet(DMCPBottleneck, [3, 4, 6, 3], num_classes,
                      input_size, width, prob_type)
