# -*- coding:utf-8  -*-

from models.dmcp.alpha_op import AlphaLayer
from models.dmcp.utils import conv_compute_flops
from models.slimmable.us_mobilenet import USInvertedResidual, USMobileNetV2, make_divisible


Alpha = None


class DMCPInvertedResidual(USInvertedResidual):
    def __init__(self, inplanes, outplanes, stride, t, expand):
        super(DMCPInvertedResidual, self).__init__(inplanes, outplanes, stride, t, expand)

        global Alpha
        self.alpha1 = Alpha(inplanes * t) if t != 1 else None
        self.alpha2 = Alpha(outplanes) if not (stride == 1 and inplanes == outplanes) else None
        self.alpha_training = False

    def forward(self, x):
        residual = x

        if self.t != 1:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            if self.alpha_training:
                out = self.alpha1(out)
        else:
            out = x

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        if self.alpha_training:
            out = self.alpha1(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.alpha_training:
            out = self.alpha2(out)

        if self.stride == 1 and self.inplanes == self.outplanes:
            out += residual

        return out

    def expected_flops(self, input_alpha, in_height, in_width):
        if self.stride == 1 and self.inplanes == self.outplanes and self.alpha2 is None:
            self.alpha2 = input_alpha

        if self.t == 1 and self.alpha1 is None:
            self.alpha1 = input_alpha

        e_in_channel = input_alpha.expected_channel()
        if not self.t == 1:
            e_conv1_out = self.alpha1.expected_channel()
            e_conv1_flops, out_height, out_width = conv_compute_flops(
                self.conv1, in_height, in_width, e_in_channel, e_conv1_out)
        else:
            e_conv1_out = e_in_channel
            e_conv1_flops = 0
            out_height = in_height
            out_width = in_width
        e_conv3_out = self.alpha2.expected_channel()

        e_conv2_flops, out_height, out_width = conv_compute_flops(
            self.conv2, out_height, out_width, e_conv1_out, e_conv1_out)
        e_conv3_flops, out_height, out_width = conv_compute_flops(
            self.conv3, out_height, out_width, e_conv1_out, e_conv3_out)
        e_flops = e_conv1_flops + e_conv2_flops + e_conv3_flops

        return e_flops, out_height, out_width

    def direct_sampling(self, ch_in):
        if self.t == 1:
            ch_out = ch_in
        else:
            self.conv1.set_input_width(specific_ch=ch_in)
            ch_out = self.alpha1.direct_sampling()
            self.conv1.set_output_width(specific_ch=ch_out)
            self.bn1.set_output_width(specific_ch=ch_out)

        self.conv2.set_input_width(specific_ch=ch_out)
        self.conv2.set_output_width(specific_ch=ch_out)
        self.bn2.set_output_width(specific_ch=ch_out)

        self.conv3.set_input_width(specific_ch=ch_out)
        if self.stride == 1 and self.inplanes == self.outplanes:
            ch_out = ch_in
        else:
            ch_out = self.alpha2.direct_sampling()
        self.conv3.set_output_width(specific_ch=ch_out)
        self.bn3.set_output_width(specific_ch=ch_out)

        return ch_out

    def expected_sampling(self, ch_in, e_ch_in):
        if self.t == 1:
            ch_out, e_ch = ch_in, e_ch_in
        else:
            self.conv1.set_input_width(specific_ch=e_ch_in)
            ch_out, e_ch = self.alpha1.expected_sampling()
            self.conv1.set_output_width(specific_ch=e_ch)
            self.bn1.set_output_width(specific_ch=e_ch)

        self.conv2.set_input_width(specific_ch=e_ch)
        self.conv2.set_output_width(specific_ch=e_ch)
        self.bn2.set_output_width(specific_ch=e_ch)

        self.conv3.set_input_width(specific_ch=e_ch)
        if self.stride == 1 and self.inplanes == self.outplanes:
            ch_out, e_ch = ch_in, e_ch_in
        else:
            ch_out, e_ch = self.alpha2.expected_sampling()
        self.conv3.set_output_width(specific_ch=e_ch)
        self.bn3.set_output_width(specific_ch=e_ch)

        return ch_out, e_ch


class DMCPMobileNetV2(USMobileNetV2):
    block = DMCPInvertedResidual

    def __init__(self, num_classes, input_size, width=None, prob_type='exp'):
        if width is None:
            width = [0.1, 1.0, 0.025]
        min_width, max_width, width_offset = width

        def alpha(channels):
            return AlphaLayer(channels, min_width, max_width, width_offset, prob_type)
        global Alpha
        Alpha = alpha

        super(DMCPMobileNetV2, self).__init__(num_classes, input_size, max_width)

        self.alpha1 = Alpha(make_divisible(32 * max_width))
        self.alpha_last = Alpha(self.last_channel)
        self.alpha_training = False

        self.expected_flops(input_size, input_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        if self.alpha_training:
            x = self.alpha1(x)
        x = self.relu(x)

        x = self.blocks(x)
        x = self.conv_last(x)
        x = self.bn_last(x)
        if self.alpha_training:
            x = self.alpha_last(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = self.fc(x)
        return x.view(x.size(0), -1)

    def expected_flops(self, in_height, in_width):
        total_flops, out_height, out_width = conv_compute_flops(
            self.conv1, in_height, in_width, e_out_ch=self.alpha1.expected_channel())
        cur_alpha = self.alpha1

        for m in self.modules():
            if isinstance(m, DMCPInvertedResidual):
                flops, out_height, out_width = m.expected_flops(cur_alpha, out_height, out_width)
                total_flops += flops
                cur_alpha = m.alpha2

        # fc
        flops, _, _ = conv_compute_flops(
            self.conv_last, out_height, out_width, cur_alpha.expected_channel(),
            self.alpha_last.expected_channel())
        total_flops += flops
        flops, _, _ = conv_compute_flops(
            self.fc, 1, 1, e_in_ch=self.alpha_last.expected_channel())
        total_flops += flops

        return total_flops / 1e6

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
            if isinstance(m, DMCPInvertedResidual):
                ch_out = m.direct_sampling(ch_out)

        self.conv_last.set_input_width(specific_ch=ch_out)
        ch_out = self.alpha_last.direct_sampling()
        self.conv_last.set_output_width(specific_ch=ch_out)
        self.bn_last.set_output_width(specific_ch=ch_out)
        self.fc.set_input_width(specific_ch=ch_out)

    def expected_sampling(self):
        ch_out, e_ch = self.alpha1.expected_sampling()
        self.conv1.set_output_width(specific_ch=e_ch)
        self.bn1.set_output_width(specific_ch=e_ch)

        for m in self.modules():
            if isinstance(m, DMCPInvertedResidual):
                ch_out, e_ch = m.expected_sampling(ch_out, e_ch)

        self.conv_last.set_input_width(specific_ch=e_ch)
        ch_out, e_ch = self.alpha_last.expected_sampling()
        self.conv_last.set_output_width(specific_ch=e_ch)
        self.bn_last.set_output_width(specific_ch=e_ch)
        self.fc.set_input_width(specific_ch=e_ch)

    def set_alpha_training(self, training):
        self.alpha_training = training
        for m in self.modules():
            if isinstance(m, DMCPInvertedResidual):
                m.alpha_training = training


def dmcp_mobilenet_v2(num_classes=1000, input_size=224, width=None, prob_type='exp'):
        if width is None:
            width = [0.1, 1.5, 0.1]
        return DMCPMobileNetV2(num_classes, input_size, width, prob_type)
