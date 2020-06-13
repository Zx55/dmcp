# -*- coding:utf-8  -*-

import logging
import os
import time
import torch
import torch.nn as nn
import utils.distributed as dist
from utils.loss_fn import LabelSmoothCELoss
from utils.meter import AverageMeter, accuracy, \
    calc_adaptive_model_flops, calc_model_parameters


class NormalRunner:
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.logger = logging.getLogger('global_logger')

        self.cur_epoch = 0
        self.cur_step = 0
        self.best_top1 = 0

    def train(self, train_loader, val_loader, optimizer, lr_scheduler, tb_logger):
        print_freq = self.config.logging.print_freq

        flops = calc_adaptive_model_flops(self.model, self.config.dataset.input_size)
        params = calc_model_parameters(self.model)
        self._info('flops: {}, params: {}'.format(flops, params))

        # meters
        batch_time = AverageMeter(print_freq)
        data_time = AverageMeter(print_freq)
        loss_meter = AverageMeter(print_freq)
        top1_meter = AverageMeter(print_freq)
        top5_meter = AverageMeter(print_freq)
        meters = [top1_meter, top5_meter, loss_meter, data_time]
        criterion = self._get_criterion()

        end = time.time()
        for e in range(self.cur_epoch, self.config.training.epoch):
            # train
            if self.config.distributed.enable:
                train_loader.sampler.set_epoch(e)
            for batch_idx, (x, y) in enumerate(train_loader):
                self.model.train()
                x, y = x.cuda(), y.cuda()
                self._train_one_batch(x, y, optimizer, lr_scheduler, meters, [criterion], end)
                batch_time.update(time.time() - end)
                end = time.time()
                cur_lr = lr_scheduler.get_lr()[0]
                self._logging(tb_logger, e, batch_idx, len(train_loader), meters + [batch_time], cur_lr)

                # validation
                if self.cur_step >= self.config.validation.start_val and self.cur_step % self.config.validation.val_freq == 0:
                    best_top1 = self.best_top1
                    self.validate(val_loader, tb_logger=tb_logger)
                    save_file = self.save(optimizer, e, best_top1=self.best_top1)

                    if self.best_top1 > best_top1:
                        from shutil import copyfile
                        best_file_dir = os.path.join(self.config.save_path, 'best')
                        if not os.path.exists(best_file_dir):
                            os.makedirs(best_file_dir)
                        best_file = os.path.join(best_file_dir, 'best.pth')
                        copyfile(save_file, best_file)

    def validate(self, val_loader, tb_logger=None):
        batch_time = AverageMeter(0)
        loss_meter = AverageMeter(0)
        top1_meter = AverageMeter(0)
        top5_meter = AverageMeter(0)

        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        end = time.time()

        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(val_loader):
                x, y = x.cuda(), y.cuda()
                num = x.size(0)

                out = self.model(x)
                loss = criterion(out, y)
                top1, top5 = accuracy(out, y, top_k=(1, 5))

                loss_meter.update(loss.item(), num)
                top1_meter.update(top1.item(), num)
                top5_meter.update(top5.item(), num)

                batch_time.update(time.time() - end)
                end = time.time()

                if batch_idx % self.config.logging.print_freq == 0:
                    self._info('Test: [{0}/{1}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})'
                               .format(batch_idx, len(val_loader), batch_time=batch_time))

        total_num = torch.tensor([loss_meter.count]).cuda()
        loss_sum = torch.tensor([loss_meter.avg * loss_meter.count]).cuda()
        top1_sum = torch.tensor([top1_meter.avg * top1_meter.count]).cuda()
        top5_sum = torch.tensor([top5_meter.avg * top5_meter.count]).cuda()

        dist.all_reduce(total_num)
        dist.all_reduce(loss_sum)
        dist.all_reduce(top1_sum)
        dist.all_reduce(top5_sum)

        val_loss = loss_sum.item() / total_num.item()
        val_top1 = top1_sum.item() / total_num.item()
        val_top5 = top5_sum.item() / total_num.item()

        self._info('Prec@1 {:.3f}\tPrec@5 {:.3f}\tLoss {:.3f}\ttotal_num={}'
                   .format(val_top1, val_top5, val_loss, loss_meter.count))

        if dist.is_master():
            if val_top1 > self.best_top1:
                self.best_top1 = val_top1

            if tb_logger is not None:
                tb_logger.add_scalar('loss_val', val_loss, self.cur_step)
                tb_logger.add_scalar('acc1_val', val_top1, self.cur_step)
                tb_logger.add_scalar('acc5_val', val_top5, self.cur_step)

    def infer(self, test_loader, train_loader=None):
        self.validate(test_loader)

    @dist.master
    def save(self, optimizer=None, epoch=None, best_top1=None):
        chk_dir = os.path.join(self.config.save_path, 'checkpoints')
        if not os.path.exists(chk_dir):
            os.makedirs(chk_dir)
        name = time.strftime('%m%d_%H%M.pth')
        name = os.path.join(chk_dir, name)

        state = {'model': self.model.state_dict()}
        if optimizer is not None:
            state['optimizer'] = optimizer.state_dict()
        if epoch is not None:
            state['epoch'] = epoch
            state['cur_step'] = self.cur_step
        if best_top1 is not None:
            state['best_top1'] = best_top1

        torch.save(state, name)
        self._info('model saved at {}'.format(name))
        return name

    def load(self, checkpoint):
        if checkpoint.get('cur_step', None) is not None:
            self.cur_step = checkpoint['cur_step']
        if checkpoint.get('epoch', None) is not None:
            self.cur_epoch = checkpoint['epoch'] + 1

    def get_model(self):
        return self.model

    @dist.master
    def _info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def _get_criterion(self):
        if self.config.training.label_smooth != 'None':
            label_smooth = self.config.training.label_smooth
            criterion = LabelSmoothCELoss(label_smooth, 1000)
            self._info('using label_smooth: {}'.format(label_smooth))
        else:
            criterion = nn.CrossEntropyLoss()

        return criterion

    def _train_one_batch(self, x, y, optimizer, lr_scheduler, meters, criterions, end):
        top1_meter, top5_meter, loss_meter, data_time = meters
        criterion = criterions[0]
        world_size = dist.get_world_size()

        lr_scheduler.step(self.cur_step)
        self.cur_step += 1
        data_time.update(time.time() - end)

        self.model.zero_grad()
        out = self.model(x)
        loss = criterion(out, y)
        loss /= world_size

        top1, top5 = accuracy(out, y, top_k=(1, 5))
        reduced_loss = dist.all_reduce(loss.clone())
        reduced_top1 = dist.all_reduce(top1.clone(), div=True)
        reduced_top5 = dist.all_reduce(top5.clone(), div=True)

        loss_meter.update(reduced_loss.item())
        top1_meter.update(reduced_top1.item())
        top5_meter.update(reduced_top5.item())

        loss.backward()
        dist.average_gradient(self.model.parameters())
        optimizer.step()

    def _logging(self, tb_logger, epoch_idx, batch_idx, total_batch, meters, cur_lr):
        print_freq = self.config.logging.print_freq
        top1_meter, top5_meter, loss_meter, data_time, batch_time = meters

        if self.cur_step % print_freq == 0 and dist.is_master():
            tb_logger.add_scalar('lr', cur_lr, self.cur_step)
            tb_logger.add_scalar('acc1_train', top1_meter.avg, self.cur_step)
            tb_logger.add_scalar('acc5_train', top5_meter.avg, self.cur_step)
            tb_logger.add_scalar('loss_train', loss_meter.avg, self.cur_step)
            self._info('-' * 80)
            self._info('Epoch: [{0}/{1}]\tIter: [{2}/{3}]\t'
                       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                       'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                       'LR {lr:.4f}'.format(
                epoch_idx, self.config.training.epoch, batch_idx, total_batch,
                batch_time=batch_time, data_time=data_time, lr=cur_lr))
            self._info('Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                       'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                       'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                       .format(loss=loss_meter, top1=top1_meter, top5=top5_meter))
