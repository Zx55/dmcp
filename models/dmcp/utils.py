# -*- coding:utf-8  -*-

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from models.slimmable.us_ops import USModule
import numpy as np
import os
import utils.distributed as dist
from utils.meter import calc_model_flops


def conv_compute_flops(conv_layer, in_height, in_width, e_in_ch=None, e_out_ch=None):
    out_channel = conv_layer.out_channels
    in_channel = conv_layer.in_channels
    groups = conv_layer.groups

    if e_out_ch is not None:
        out_channel = e_out_ch
    if e_in_ch is not None:
        if groups == in_channel:
            groups = e_in_ch.detach().cpu().item()
        else:
            assert groups == 1, 'Unknown group number'
        in_channel = e_in_ch

    padding_height, padding_width = conv_layer.padding
    stride_height, stride_width = conv_layer.stride
    kernel_height, kernel_width = conv_layer.kernel_size
    assert conv_layer.dilation == (1, 1)  # not support deformable conv

    bias_ops = 1 if conv_layer.bias is not None else 0

    kernel_ops = kernel_height * kernel_width * (in_channel / groups)
    output_height = (in_height + padding_height * 2 - kernel_height) // stride_height + 1
    output_width = (in_width + padding_width * 2 - kernel_width) // stride_width + 1
    flops = (kernel_ops + bias_ops) * output_height * output_width * out_channel

    return flops, output_height, output_width


def dump_flops_stats(iteration, config, model):
    sample_flops = []
    num_sample = config.arch.num_flops_stats_sample

    for _ in range(num_sample):
        model.module.direct_sampling()
        cur_flops = calc_model_flops(model, config.dataset.input_size)
        sample_flops.append(cur_flops)

    if dist.is_master():
        save_folder = os.path.join(config.save_path, 'flops_stats')
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        plt.hist(sample_flops, 50, density=True, facecolor='g', alpha=0.75)
        plt.axvline(x=np.mean(sample_flops), color='r', linestyle='--')
        pp = PdfPages(os.path.join(save_folder, 'flops_stats_{}.pdf'.format(iteration)))
        plt.savefig(pp, format='pdf')
        pp.close()
        plt.gcf().clear()


def compute_mean_channel(iteration, config, model):
    mean_chs = []
    offset = []
    num_sample = config.arch.num_flops_stats_sample

    for n, m in model.named_modules():
        if n.find('alpha') > -1:
            mean_ch = 0
            for i in range(num_sample):
                mean_ch += m.direct_sampling()
            mean_chs.append(round(mean_ch / num_sample, 3))
            offset.append(m.channels - mean_chs[-1])

    if dist.is_master():
        ind = np.arange(len(mean_chs))
        width = 0.35

        p1 = plt.bar(ind, mean_chs, width)
        p2 = plt.bar(ind, offset, width, bottom=mean_chs)

        plt.ylabel('#channel')
        plt.xticks(ind, ind)
        plt.yticks(np.arange(0, max([mean_chs[i] + offset[i] for i in range(len(mean_chs))]), 100))
        plt.legend((p1[0], p2[0]), ('expected', 'max'))

        save_folder = os.path.join(config.save_path, 'mean_chs')
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        pp = PdfPages(os.path.join(save_folder, 'mean_chs_{}.pdf'.format(iteration)))
        plt.savefig(pp, format='pdf')
        pp.close()
        plt.gcf().clear()


def layer_flops_distribution(config, model):
    num_sample = config.arch.num_flops_stats_sample
    repo = {}

    for _ in range(num_sample):
        cur_flops = config.arch.target_flops * 10
        while cur_flops > config.arch.target_flops * 1.05 or cur_flops < config.arch.target_flops * 0.95:
            model.module.direct_sampling()
            cur_flops = calc_model_flops(model, config.dataset.input_size)
            for n, m in model.named_modules():
                if isinstance(m, USModule):
                    if n not in repo.keys():
                        repo[n] = []
                    repo[n].append(m.cur_out_ch)

    if dist.is_master():
        root_dir = os.path.join(config.save_path, 'layer_flops_distribution')
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        for n in repo.keys():
            save_path = os.path.join(root_dir, n + '.pdf')
            plt.hist(repo[n], 50, density=True, facecolor='g', alpha=0.75)
            pp = PdfPages(save_path)
            plt.savefig(pp, format='pdf')
            pp.close()
            plt.gcf().clear()


def sample_model(config, model):
    num_sample = config.arch.num_model_sample
    root_dir = os.path.join(config.save_path, 'model_sample')

    if dist.is_master():
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)

    for i in range(num_sample + 1):
        cur_flops = config.arch.target_flops * 10
        while cur_flops > config.arch.target_flops * 1.01 or cur_flops < config.arch.target_flops * 0.99:
            model.module.direct_sampling()
            cur_flops = calc_model_flops(model, config.dataset.input_size)

        if dist.is_master():
            if i == num_sample:
                sample_dir = os.path.join(root_dir, 'expected_ch')
                model.module.expected_sampling()
            else:
                sample_dir = os.path.join(root_dir, 'sample_{}'.format(i + 1))
            if not os.path.exists(sample_dir):
                os.makedirs(sample_dir)

            save_path = os.path.join(sample_dir, 'sample.npy')
            if config.model.type.find('MobileNetV2') > -1:
                dump_mbv2_setting(model.module, save_path)
            elif config.model.type.find('ResNet') > -1:
                dump_resnet_setting(model.module, save_path)
            else:
                raise ValueError('Unknown model: {}'.format(config.model.type))

    dist.barrier()


def dump_mbv2_setting(model, path):
    from models.dmcp.dmcp_mobilenet import DMCPInvertedResidual, USInvertedResidual

    def dump_residual_setting(residual, ch_dict):
        if not residual.t == 1:
            ch_dict['conv1'] = (residual.conv1.cur_in_ch, residual.conv1.cur_out_ch)
        ch_dict['conv2'] = (residual.conv2.cur_in_ch, residual.conv2.cur_out_ch)
        ch_dict['conv3'] = (residual.conv3.cur_in_ch, residual.conv3.cur_out_ch)

    ch_dict = {}
    ch_dict['conv1'] = model.conv1.cur_out_ch
    ch_dict['conv_last'] = (model.conv_last.cur_in_ch, model.conv_last.cur_out_ch)

    for n, m in model.named_modules():
        if isinstance(m, (USInvertedResidual, DMCPInvertedResidual)):
            ch_dict[n.split('.')[1]] = {}
            dump_residual_setting(m, ch_dict[n.split('.')[1]])

    np.save(path, ch_dict)


def dump_resnet_setting(model, path):
    from .dmcp_resnet import DMCPBottleneck, DMCPBasicBlock, USBottleneck, USBasicBlock

    def dump_bottleneck_setting(bottleneck, ch_dict):
        ch_dict['conv1'] = (bottleneck.conv1.cur_in_ch, bottleneck.conv1.cur_out_ch)
        ch_dict['conv2'] = (bottleneck.conv2.cur_in_ch, bottleneck.conv2.cur_out_ch)
        ch_dict['conv3'] = (bottleneck.conv3.cur_in_ch, bottleneck.conv3.cur_out_ch)

    def dump_basic_block_setting(block, ch_dict):
        ch_dict['conv1'] = (block.conv1.cur_in_ch, block.conv1.cur_out_ch)
        ch_dict['conv2'] = (block.conv2.cur_in_ch, block.conv2.cur_out_ch)

    ch_dict = {}
    ch_dict['conv1'] = model.conv1.cur_out_ch
    ch_dict['fc'] = model.fc.cur_in_ch

    for n, m in model.named_modules():
        if isinstance(m, (USBottleneck, DMCPBottleneck)):
            if not n.split('.')[0] in ch_dict.keys():
                ch_dict[n.split('.')[0]] = {}
            ch_dict[n.split('.')[0]][n.split('.')[1]] = {}
            dump_bottleneck_setting(m, ch_dict[n.split('.')[0]][n.split('.')[1]])
        elif isinstance(m, (USBasicBlock, DMCPBasicBlock)):
            if not n.split('.')[0] in ch_dict.keys():
                ch_dict[n.split('.')[0]] = {}
            ch_dict[n.split('.')[0]][n.split('.')[1]] = {}
            dump_basic_block_setting(m, ch_dict[n.split('.')[0]][n.split('.')[1]])

    np.save(path, ch_dict)
