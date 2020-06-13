# -*- coding:utf-8  -*-

from easydict import EasyDict
import logging
import numpy as np
import os
import random
from tensorboardX import SummaryWriter
import time
import torch
import torch.nn as nn
import yaml
import models.dmcp as dmcp
import models.adaptive as adaptive
from models.dmcp.alpha_op import AlphaLayer
from runner import USRunner, DMCPRunner, NormalRunner
from utils.lr_scheduler import WarmUpCosineLRScheduler
import utils.data as data
import utils.distributed as dist


def init(config):
    random_seed = config.random_seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    dist.init_dist(config.distributed.enable)


def check_dist_init(config, logger):
    # check distributed initialization
    if config.distributed.enable:
        import os
        # for slurm
        try:
            node_id = int(os.environ['SLURM_NODEID'])
        except KeyError:
            return

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        gpu_id = dist.gpu_id

        logger.info('World: {}/Node: {}/Rank: {}/GpuId: {} initialized.'
                    .format(world_size, node_id, rank, gpu_id))


def get_args(parser):
    args = parser.parse_args()

    assert args.mode in ['train', 'eval', 'sample', 'calc_flops', 'calc_params']

    return args


def get_config(args):
    """
    load experiment config
    """
    with open(args.config) as f:
        config = yaml.load(f)
    config = EasyDict(config)

    config.arch.target_flops = args.flops
    config.dataset.path = args.data
    if config.model.type.find('Adaptive') > -1:
        assert args.chcfg is not None, "error: miss channel config"
        config.model.kwargs.ch_cfg = args.chcfg

    return config


def get_logger(config, name='global_logger'):
    save_dir = config.model.type + '_'
    if config.get('arch', False):
        save_dir += str(config.arch.target_flops) + '_'
    save_dir = time.strftime(save_dir + '%m%d%H')
    save_dir = os.path.join(config.save_path, save_dir)

    if dist.is_master():
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    else:
        while not os.path.exists(save_dir):
            time.sleep(1)
    config.save_path = save_dir

    events_dir = config.save_path + '/events'
    if dist.is_master():
        if not os.path.exists(events_dir):
            os.makedirs(events_dir)
    else:
        while not os.path.exists(events_dir):
            time.sleep(1)

    tb_logger = SummaryWriter(config.save_path + '/events')
    logger = logging.getLogger(name)
    formatter = logging.Formatter('[%(asctime)s][%(filename)15s][line:%(lineno)4d][%(levelname)8s] %(message)s')
    fh = logging.FileHandler(config.save_path + '/log.txt')
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)

    return tb_logger, logger


def get_data_loader(config):
    if config.dataset.type == 'CIFAR10':
        dataset = data.get_cifar10(config.dataset)
    elif config.dataset.type == 'ImageNet':
        dataset = data.get_image_net(config.dataset)
    else:
        raise KeyError('invalid dataset type')

    train_loader, val_loader = data.get_loader(config.dataset, config.dataset.batch_size,
                                               config.distributed.enable, *dataset)

    max_iter = len(train_loader) * config.training.epoch
    config.lr_scheduler.max_iter = max_iter
    if config.get('arch_lr_scheduler', None) is not None:
        config.arch_lr_scheduler.max_iter = max_iter
        config.arch.start_train = max_iter // 2
        config.arch_lr_scheduler.warmup_steps = max_iter // 2

    return train_loader, val_loader


def get_param_group(model):
    param_group_no_wd = []
    names_no_wd = []
    param_group_normal = []
    arch_parameters = []

    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                param_group_no_wd.append(m.bias)
                names_no_wd.append(name + '.bias')
        elif isinstance(m, nn.Linear):
            if m.bias is not None:
                param_group_no_wd.append(m.bias)
                names_no_wd.append(name + '.bias')
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            if m.weight is not None:
                param_group_no_wd.append(m.weight)
                names_no_wd.append(name + '.weight')
            if m.bias is not None:
                param_group_no_wd.append(m.bias)
                names_no_wd.append(name + '.bias')
        elif isinstance(m, AlphaLayer):
            # exclude architecture parameters
            arch_parameters.append(name + '.alpha')

    for name, p in model.named_parameters():
        if (name not in names_no_wd) and (name not in arch_parameters):
            param_group_normal.append(p)

    return [{'params': param_group_normal}, {'params': param_group_no_wd, 'weight_decay': 0.0}]


def get_optimizer(model, config, checkpoint=None):
    if config.optimizer.no_wd:
        param_group = get_param_group(model)
        optimizer = torch.optim.SGD(
            param_group, config.lr_scheduler.base_lr, momentum=config.optimizer.momentum,
            weight_decay=config.optimizer.weight_decay, nesterov=config.optimizer.nesterov)
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), config.lr_scheduler.base_lr, momentum=config.optimizer.momentum,
            weight_decay=config.optimizer.weight_decay, nesterov=config.optimizer.nesterov)

    if checkpoint is not None:
        if checkpoint.get('optimizer', None) is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])

    if config.get('arch_lr_scheduler', False):
        arch_optimizer = torch.optim.SGD(
            model.module.arch_parameters(), config.arch_lr_scheduler.base_lr,
            momentum=config.optimizer.momentum, weight_decay=0,
            nesterov=config.optimizer.nesterov)

        if checkpoint is not None:
            if checkpoint.get('arch_optimizer', None) is not None:
                arch_optimizer.load_state_dict(checkpoint['arch_optimizer'])
        return optimizer, arch_optimizer

    return optimizer


def get_lr_scheduler(optimizer, config, last_iter=-1):
    return WarmUpCosineLRScheduler(
        optimizer, config.max_iter, config.min_lr, config.base_lr,
        config.warmup_lr, config.warmup_steps, last_iter=last_iter)


def get_checkpoint(config):
    if config.recover.enable:
        return torch.load(config.recover.checkpoint, map_location='cpu')
    else:
        return None


def get_model(config, checkpoint=None):
    # differentiable markov model
    if config.model.type == 'DMCPResNet18':
        model = dmcp.dmcp_resnet18(**config.model.kwargs)
    elif config.model.type == 'DMCPResNet50':
        model = dmcp.dmcp_resnet50(**config.model.kwargs)
    elif config.model.type == 'DMCPMobileNetV2':
        model = dmcp.dmcp_mobilenet_v2(**config.model.kwargs)
    # adaptive model (train pruned model from scratch)
    elif config.model.type == 'AdaptiveResNet18':
        model = adaptive.adaptive_res18(**config.model.kwargs)
    elif config.model.type == 'AdaptiveResNet50':
        model = adaptive.adaptive_res50(**config.model.kwargs)
    elif config.model.type == 'AdaptiveMobileNetV2':
        model = adaptive.adaptive_mobilenet_v2(**config.model.kwargs)
    else:
        raise NotImplementedError

    if config.distributed.enable:
        gpu_id = dist.gpu_id
        wrapper = nn.parallel.DistributedDataParallel(model.cuda(), [gpu_id], gpu_id)
    else:
        wrapper = nn.parallel.DataParallel(model).cuda()

    if checkpoint is not None:
        wrapper.load_state_dict(checkpoint['model'])
        wrapper = wrapper.cuda()

    if config.model.runner.type == 'USRunner':
        runner = USRunner(config, wrapper)
    elif config.model.runner.type == 'DMCPRunner':
        runner = DMCPRunner(config, wrapper)
    elif config.model.runner.type == 'NormalRunner':
        runner = NormalRunner(config, wrapper)
    else:
        raise NotImplementedError

    if checkpoint is not None:
        runner.load(checkpoint)

    return runner


@dist.master
def get_model_flops(config, model):
    from utils.meter import calc_adaptive_model_flops, calc_model_flops

    input_size = config.dataset.input_size
    if config.model.type.find('Adaptive') > -1:
        flops = calc_adaptive_model_flops(model, input_size)
    else:
        flops = calc_model_flops(model, input_size)

    return flops


@dist.master
def get_model_parameters(model):
    from utils.meter import calc_model_parameters

    return calc_model_parameters(model)
