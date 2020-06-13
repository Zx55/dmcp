# -*- coding:utf-8  -*-

import argparse
import utils.distributed as dist
import utils.tools as tools


parser = argparse.ArgumentParser(description='DMCP Implementation')
parser.add_argument('-C', '--config', required=True)
parser.add_argument('-M', '--mode', default='eval')
parser.add_argument('-F', '--flops', required=True)
parser.add_argument('-D', '--data', required=True)
parser.add_argument('--chcfg', default=None)


def train(config, runner, loaders, checkpoint, tb_logger):
    # load optimizer and scheduler
    optimizer = tools.get_optimizer(runner.get_model(), config, checkpoint)
    if config.get('arch_lr_scheduler', False):
        assert len(optimizer) == 2

        lr_scheduler = tools.get_lr_scheduler(optimizer[0], config.lr_scheduler)
        arch_lr_scheduler = tools.get_lr_scheduler(optimizer[1], config.arch_lr_scheduler)
        lr_scheduler = (lr_scheduler, arch_lr_scheduler)
    else:
        lr_scheduler = tools.get_lr_scheduler(optimizer, config.lr_scheduler)

    # train and calibrate
    train_loader, val_loader = loaders
    runner.train(train_loader, val_loader, optimizer, lr_scheduler, tb_logger)
    runner.infer(val_loader, train_loader=train_loader)


def evaluate(runner, loaders):
    train_loader, val_loader = loaders
    runner.infer(val_loader, train_loader=train_loader)


def main():
    args = tools.get_args(parser)
    config = tools.get_config(args)
    tools.init(config)
    tb_logger, logger = tools.get_logger(config)
    tools.check_dist_init(config, logger)

    checkpoint = tools.get_checkpoint(config)
    runner = tools.get_model(config, checkpoint)
    loaders = tools.get_data_loader(config)

    if dist.is_master():
        logger.info(config)

    if args.mode == 'train':
        train(config, runner, loaders, checkpoint, tb_logger)
    elif args.mode == 'evaluate':
        evaluate(runner, loaders)
    elif args.mode == 'calc_flops':
        if dist.is_master():
            flops = tools.get_model_flops(config, runner.get_model())
            logger.info('flops: {}'.format(flops))
    elif args.mode == 'calc_params':
        if dist.is_master():
            params = tools.get_model_parameters(runner.get_model())
            logger.info('params: {}'.format(params))
    else:
        assert checkpoint is not None
        from models.dmcp.utils import sample_model
        sample_model(config, runner.get_model())

    if dist.is_master():
        logger.info('Done')


if __name__ == '__main__':
    main()
