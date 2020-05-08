# -*- coding:utf-8  -*-

import time
import models.dmcp.utils as dmcp_utils
import os
import random
import torch
import utils.distributed as dist
from utils.loss_fn import flop_loss
from utils.meter import AverageMeter, accuracy
from .us_runner import USRunner


class DMCPRunner(USRunner):
    def __init__(self, config, model):
        super(DMCPRunner, self).__init__(config, model)

    def train(self, train_loader, val_loader, optimizer, lr_scheduler, tb_logger):
        print_freq = self.config.logging.print_freq
        batch_time = AverageMeter(print_freq)
        data_time = AverageMeter(print_freq)
        loss_meter = [AverageMeter(print_freq) for _ in range(3)]
        top1_meter = [AverageMeter(print_freq) for _ in range(3)]
        top5_meter = [AverageMeter(print_freq) for _ in range(3)]
        # track stats of architecture parameter
        arch_loss_meter = AverageMeter(print_freq)
        floss_meter = AverageMeter(print_freq)
        eflops_meter = AverageMeter(print_freq)
        arch_top1_meter = AverageMeter(print_freq)
        meters = [
            top1_meter, top5_meter, loss_meter, arch_loss_meter,
            floss_meter, eflops_meter, arch_top1_meter, data_time
        ]
        criterions = self._get_criterion()

        self._sample_width()
        end = time.time()
        for e in range(self.cur_epoch, self.config.training.epoch):
            # train
            self.model.train()

            if self.config.distributed.enable:
                train_loader.sampler.set_epoch(e)
            for batch_idx, (x, y) in enumerate(train_loader):
                x, y = x.cuda(), y.cuda()
                self._train_one_batch(x, y, optimizer, lr_scheduler, meters, criterions, end)
                batch_time.update(time.time() - end)
                end = time.time()

                cur_lr = lr_scheduler[0].get_lr()[0]
                cur_arch_lr = lr_scheduler[1].get_lr()[0]
                # logging
                self._logging(tb_logger, e, batch_idx, len(train_loader),
                              meters + [batch_time], [cur_lr, cur_arch_lr])

            # validation
            self.validate(val_loader, train_loader, self.config.validation.width,
                          tb_logger=tb_logger)
            self.save(optimizer, e)

        # sample model
        self._info('sampling model...')
        dmcp_utils.sample_model(self.config, self.model)
        self._info('draw layer flops distribution...')
        dmcp_utils.layer_flops_distribution(self.config, self.model)

    def validate(self, val_loader, train_loader=None, val_width=None, tb_logger=None):
        super(DMCPRunner, self).validate(val_loader, train_loader, val_width, tb_logger)

        if self.cur_step >= self.config.arch.start_train:
            dmcp_utils.dump_flops_stats(self.cur_step, self.config, self.model)
            dmcp_utils.compute_mean_channel(self.cur_step, self.config, self.model)

    def infer(self, test_loader, train_loader=None):
        self.model.module.set_alpha_training(True)
        super(DMCPRunner, self).infer(test_loader, train_loader)

    @dist.master
    def save(self, optimizer=None, epoch=None, best_top1=None):
        chk_dir = os.path.join(self.config.save_path, 'checkpoints')
        if not os.path.exists(chk_dir):
            os.makedirs(chk_dir)
        name = time.strftime('%m%d_%H%M.pth')
        name = os.path.join(chk_dir, name)

        state = {'model': self.model.state_dict()}
        if optimizer is not None:
            optimizer, arch_optimizer = optimizer
            state['optimizer'] = optimizer.state_dict()
            state['arch_optimizer'] = arch_optimizer.state_dict()
        if epoch is not None:
            state['epoch'] = epoch
            state['cur_step'] = self.cur_step
        if best_top1 is not None:
            state['best_top1'] = best_top1

        torch.save(state, name)
        self._info('model saved at {}'.format(name))
        return name

    def _set_width(self, idx, top1, top5, loss, width=None):
        if self.model.training and self.cur_step >= self.config.arch.start_train:
            all_type = self.config.arch.sample_type
            tp = all_type[idx]
            assert tp in ['min', 'max', 'scheduled_random',
                          'non_uni_random', 'arch_random']

            max_width = self.config.training.sandwich.max_width
            min_width = self.config.training.sandwich.min_width
            if tp == 'min':
                self.model.module.uniform_set_width(min_width)
                return top1[0], top5[0], loss[0]
            elif tp == 'max':
                self.model.module.uniform_set_width(max_width)
                return top1[1], top5[1], loss[1]
            elif tp == 'scheduled_random':
                if not hasattr(self, 'cur_sample_prob'):
                    self.cur_sample_prob = 1.0
                if self.cur_step % 10 == 0:
                    self.cur_sample_prob *= 0.9999
                if random.uniform(0, 1) < self.cur_sample_prob:
                    self.model.module.non_uniform_set_width(self.sample_width)
                else:
                    self.model.module.direct_sampling()
            elif tp == 'non_uni_random':
                self.model.module.non_uniform_set_width(self.sample_width)
            elif tp == 'arch_random':
                self.model.module.direct_sampling()
            else:
                raise NotImplementedError
            return top1[2], top5[2], loss[2]
        else:
            return super(DMCPRunner, self)._set_width(idx, top1, top5, loss, width)

    def _train_one_batch(self, x, y, optimizer, lr_scheduler, meters, criterions, end):
        lr_scheduler, arch_lr_scheduler = lr_scheduler
        optimizer, arch_optimizer = optimizer
        top1_meter, top5_meter, loss_meter, arch_loss_meter, \
            floss_meter, eflops_meter, arch_top1_meter, data_time = meters
        criterion, _ = criterions

        self.model.module.set_alpha_training(False)
        super(DMCPRunner, self)._train_one_batch(
            x, y, optimizer, lr_scheduler, [top1_meter, top5_meter, loss_meter, data_time],
            criterions, end)

        arch_lr_scheduler.step(self.cur_step)
        world_size = dist.get_world_size()

        # train architecture params
        if self.cur_step >= self.config.arch.start_train \
                and self.cur_step % self.config.arch.train_freq == 0:
            self._set_width(0, top1_meter, top5_meter, loss_meter)
            self.model.module.set_alpha_training(True)

            self.model.zero_grad()
            arch_out = self.model(x)
            arch_loss = criterion(arch_out, y)
            arch_loss /= world_size
            floss, eflops = flop_loss(self.config, self.model)
            floss /= world_size

            arch_top1 = accuracy(arch_out, y, top_k=(1,))[0]
            reduced_arch_loss = dist.all_reduce(arch_loss.clone())
            reduced_floss = dist.all_reduce(floss.clone())
            reduced_eflops = dist.all_reduce(eflops.clone(), div=True)
            reduced_arch_top1 = dist.all_reduce(arch_top1.clone(), div=True)

            arch_loss_meter.update(reduced_arch_loss.item())
            floss_meter.update(reduced_floss.item())
            eflops_meter.update(reduced_eflops.item())
            arch_top1_meter.update(reduced_arch_top1.item())

            floss.backward()
            arch_loss.backward()
            dist.average_gradient(self.model.module.arch_parameters())
            arch_optimizer.step()

    def _logging(self, tb_logger, epoch_idx, batch_idx, total_batch, meters, cur_lr):
        cur_lr, cur_arch_lr = cur_lr
        top1_meter, top5_meter, loss_meter, arch_loss_meter, floss_meter, \
            eflops_meter, arch_top1_meter, data_time, batch_time = meters

        super(DMCPRunner, self)._logging(
            tb_logger, epoch_idx, batch_idx, total_batch,
            [top1_meter, top5_meter, loss_meter, data_time, batch_time], cur_lr)

        print_freq = self.config.logging.print_freq
        if self.cur_step % print_freq == 0 and dist.is_master() \
                and self.cur_step >= self.config.arch.start_train:
            tb_logger.add_scalar('arc_loss', arch_loss_meter.avg, self.cur_step)
            tb_logger.add_scalar('flops_loss', floss_meter.avg, self.cur_step)
            tb_logger.add_scalar('eflops', eflops_meter.avg, self.cur_step)
            tb_logger.add_scalar('arc_top1', arch_top1_meter.avg, self.cur_step)

            self._info('expected_flops {:.2f} flops_loss {:.4f}, arch_task_loss {:.4f}, '
                       'arch_top1 {:.2f}, arch_lr {:.4f}'
                       .format(eflops_meter.avg, floss_meter.avg, arch_loss_meter.avg,
                               arch_top1_meter.avg, cur_arch_lr))
            tb_logger.add_scalar('expected_flops', eflops_meter.avg, self.cur_step)
