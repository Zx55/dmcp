# -*- coding:utf-8  -*-

import time
import torch
import torch.nn as nn
import utils.distributed as dist
from utils.loss_fn import KL
from utils.meter import AverageMeter, accuracy
from models.slimmable.us_ops import USBatchNorm2d
from .normal_runner import NormalRunner


class USRunner(NormalRunner):
    def __init__(self, config, model):
        super(USRunner, self).__init__(config, model)
        self.sample_width = None

    def train(self, train_loader, val_loader, optimizer, lr_scheduler, tb_logger):
        print_freq = self.config.logging.print_freq

        # meters
        batch_time = AverageMeter(print_freq)
        data_time = AverageMeter(print_freq)
        # track stats of min width, max width and random width
        loss_meter = [AverageMeter(print_freq) for _ in range(3)]
        top1_meter = [AverageMeter(print_freq) for _ in range(3)]
        top5_meter = [AverageMeter(print_freq) for _ in range(3)]
        meters = [top1_meter, top5_meter, loss_meter, data_time]
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
                cur_lr = lr_scheduler.get_lr()[0]
                self._logging(tb_logger, e, batch_idx, len(train_loader), meters + [batch_time], cur_lr)

            # validation
            self.validate(val_loader, train_loader, self.config.validation.width,
                          tb_logger=tb_logger)
            self.save(optimizer, lr_scheduler, e)

    def validate(self, val_loader, train_loader=None, val_width=None, tb_logger=None):
        assert train_loader is not None
        assert val_width is not None

        batch_time = AverageMeter(0)
        loss_meter = [AverageMeter(0) for _ in range(len(val_width))]
        top1_meter = [AverageMeter(0) for _ in range(len(val_width))]
        top5_meter = [AverageMeter(0) for _ in range(len(val_width))]
        val_loss, val_top1, val_top5 = [], [], []

        # switch to evaluate mode
        self.model.eval()

        criterion = nn.CrossEntropyLoss()
        end = time.time()

        with torch.no_grad():
            for idx, width in enumerate(val_width):
                top1_m, top5_m, loss_m = self._set_width(idx, top1_meter, top5_meter, loss_meter, width=width)

                self._info('-' * 80)
                self._info('Evaluating [{}/{}]@{}'.format(idx + 1, len(val_width), width))

                self.calibrate(train_loader)
                for j, (x, y) in enumerate(val_loader):
                    x, y = x.cuda(), y.cuda()
                    num = x.size(0)

                    out = self.model(x)
                    loss = criterion(out, y)
                    top1, top5 = accuracy(out.data, y, top_k=(1, 5))

                    loss_m.update(loss.item(), num)
                    top1_m.update(top1.item(), num)
                    top5_m.update(top5.item(), num)

                    # measure elapsed time
                    batch_time.update(time.time() - end)
                    end = time.time()

                    if j % self.config.logging.print_freq == 0:
                        self._info('Test: [{0}/{1}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})'
                                   .format(j, len(val_loader), batch_time=batch_time))

                total_num = torch.tensor([loss_m.count]).cuda()
                loss_sum = torch.tensor([loss_m.avg * loss_m.count]).cuda()
                top1_sum = torch.tensor([top1_m.avg * top1_m.count]).cuda()
                top5_sum = torch.tensor([top5_m.avg * top5_m.count]).cuda()

                dist.all_reduce(total_num)
                dist.all_reduce(loss_sum)
                dist.all_reduce(top1_sum)
                dist.all_reduce(top5_sum)

                val_loss.append(loss_sum.item() / total_num.item())
                val_top1.append(top1_sum.item() / total_num.item())
                val_top5.append(top5_sum.item() / total_num.item())

                self._info('Prec@1 {:.3f}\tPrec@5 {:.3f}\tLoss {:.3f}\ttotal_num={}'
                           .format(val_top1[-1], val_top5[-1], val_loss[-1], loss_m.count))

            if dist.is_master() and tb_logger is not None:
                for i in range(len(val_loss)):
                    tb_logger.add_scalar('loss_val@{}'.format(val_width[i]), val_loss[i], self.cur_step)
                    tb_logger.add_scalar('acc1_val@{}'.format(val_width[i]), val_top1[i], self.cur_step)
                    tb_logger.add_scalar('acc5_val@{}'.format(val_width[i]), val_top5[i], self.cur_step)

    def infer(self, test_loader, train_loader=None):
        self.validate(test_loader, train_loader, self.config.evaluation.width)

    def calibrate(self, train_loader):
        self.model.eval()

        momentum_bk = None
        for m in self.model.module.modules():
            if isinstance(m, USBatchNorm2d):
                m.reset_running_stats()
                m.training = True
                if momentum_bk is None:
                    momentum_bk = m.momentum
                m.momentum = 1.0

        with torch.no_grad():
            for batch_idx, (x, _) in enumerate(train_loader):
                if batch_idx == self.config.validation.calibration.num_batch:
                    break

                x = x.cuda()
                self.model(x)

        for m in self.model.module.modules():
            if isinstance(m, USBatchNorm2d):
                m.momentum = momentum_bk
                m.training = False

    def _set_width(self, idx, top1, top5, loss, width=None):
        max_width = self.config.training.sandwich.max_width
        min_width = self.config.training.sandwich.min_width

        if self.model.training:
            if idx == 1:
                self.model.module.uniform_set_width(min_width)
                return top1[0], top5[0], loss[0]
            elif idx == 0:
                self.model.module.uniform_set_width(max_width)
                return top1[1], top5[1], loss[1]
            else:
                self.model.module.non_uniform_set_width(self.sample_width)
                return top1[2], top5[2], loss[2]
        else:
            assert width is not None
            self.model.module.uniform_set_width(width)
            return top1[idx], top5[idx], loss[idx]

    def _sample_width(self):
        # calculate all width based on offset
        if self.sample_width is None:
            max_width = self.config.training.sandwich.max_width
            min_width = self.config.training.sandwich.min_width
            num_sample = self.config.training.sandwich.num_sample

            offset = self.config.training.sandwich.width_offset
            num_offset = int((max_width - min_width) / offset + 1e-4)
            self.sample_width = [round(min_width + i * offset, 3) for i in range(1, num_offset)]
            assert len(self.sample_width) >= num_sample - 2
            self.sample_width = [max_width, min_width] + self.sample_width

    def _get_criterion(self):
        criterion = super(USRunner, self)._get_criterion()
        distill_loss = KL(self.config.training.distillation.temperature)
        return criterion, distill_loss

    def _train_one_batch(self, x, y, optimizer, lr_scheduler, meters, criterions, end):
        top1_meter, top5_meter, loss_meter, data_time = meters
        criterion, distill_loss = criterions
        world_size = dist.get_world_size()
        max_width = self.config.training.sandwich.max_width

        lr_scheduler.step(self.cur_step)
        self.cur_step += 1
        data_time.update(time.time() - end)

        self.model.zero_grad()

        max_pred = None
        for idx in range(self.config.training.sandwich.num_sample):
            # sandwich rule
            top1_m, top5_m, loss_m = self._set_width(idx, top1_meter, top5_meter, loss_meter)

            out = self.model(x)
            if self.config.training.distillation.enable:
                if idx == 0:
                    max_pred = out.detach()
                    loss = criterion(out, y)
                else:
                    loss = self.config.training.distillation.loss_weight * \
                           distill_loss(out, max_pred)
                    if self.config.training.distillation.hard_label:
                        loss += criterion(out, y)
            else:
                loss = criterion(out, y)
            loss /= world_size

            top1, top5 = accuracy(out, y, top_k=(1, 5))
            reduced_loss = dist.all_reduce(loss.clone())
            reduced_top1 = dist.all_reduce(top1.clone(), div=True)
            reduced_top5 = dist.all_reduce(top5.clone(), div=True)

            loss_m.update(reduced_loss.item())
            top1_m.update(reduced_top1.item())
            top5_m.update(reduced_top5.item())

            loss.backward()

        dist.average_gradient(self.model.parameters())
        optimizer.step()

    def _logging(self, tb_logger, epoch_idx, batch_idx, total_batch, meters, cur_lr):
        print_freq = self.config.logging.print_freq
        top1_meter, top5_meter, loss_meter, data_time, batch_time = meters

        if self.cur_step % print_freq == 0 and dist.is_master():
            tb_logger.add_scalar('lr', cur_lr, self.cur_step)
            self._info('-' * 80)
            self._info('Epoch: [{0}/{1}]\tIter: [{2}/{3}]\t'
                       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                       'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                       'LR {lr:.4f}'.format(
                epoch_idx, self.config.training.epoch, batch_idx, total_batch,
                batch_time=batch_time, data_time=data_time, lr=cur_lr))

            titles = ['min_width', 'max_width', 'random_width']
            for idx in range(3):
                tb_logger.add_scalar('loss_train@{}'.format(titles[idx]), loss_meter[idx].avg, self.cur_step)
                tb_logger.add_scalar('acc1_train@{}'.format(titles[idx]), top1_meter[idx].avg, self.cur_step)
                tb_logger.add_scalar('acc5_train@{}'.format(titles[idx]), top5_meter[idx].avg, self.cur_step)
                self._info('{title}\t'
                           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                           'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                           'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                           .format(title=titles[idx], loss=loss_meter[idx],
                                   top1=top1_meter[idx], top5=top5_meter[idx]))
