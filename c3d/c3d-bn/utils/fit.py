import torch
import numpy as np
from matplotlib import pyplot as plt
import math

from tqdm import tqdm, trange
from fastvision.utils.checkpoints import SaveModel

class Fit():

    def __init__(self, model, device, optimizer, scheduler, loss, metric, end_epoch, start_epoch=0, train_loader=None,
                 val_loader=None, test_loader=None):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.loss = loss
        self.metric = metric
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scheduler = scheduler

    def find_lr(self, beta=0.98):

        avg_loss = 0
        avg_metric = 0

        record_losses = []
        record_metrics = []
        record_lrs = []

        mininum_loss = 0

        self.model.train()
        losses, metrics, group_lrs = self._run_epoch(1, self.train_loader, mode='train')

        for batch_idx, (loss, metric, lr) in enumerate(zip(losses, metrics, group_lrs)):
            batch_idx += 1

            avg_loss = beta * avg_loss + (1 - beta) * loss
            smoothed_loss = avg_loss / (1 - beta ** batch_idx)

            avg_metric = beta * avg_metric + (1 - beta) * metric
            smoothed_metric = avg_metric / (1 - beta ** batch_idx)

            if batch_idx > 1 and smoothed_loss > 4 * mininum_loss:
                break

            if smoothed_loss < mininum_loss or batch_idx == 1:
                mininum_loss = smoothed_loss

            record_metrics.append(smoothed_metric)
            record_losses.append(smoothed_loss)
            record_lrs.append(math.log10(lr[0]))

        record_lrs = np.array(record_lrs)
        record_losses = np.array(record_losses)
        record_metrics = np.array(record_metrics)

        print(record_losses)

        plt.plot(record_lrs, record_losses)
        plt.savefig('./loss.png')

    def trainEpoches(self):
        train_losses = []
        train_metrics = []
        train_lrs = []

        val_losses = []
        val_metrics = []

        for epoch in range(self.start_epoch, self.end_epoch):

            self.model.train()
            losses, metrics, group_lrs = self._run_epoch(epoch, self.train_loader, mode='train')
            train_losses.append(sum(losses) / len(losses))
            train_metrics.append(sum(metrics) / len(metrics))
            train_lrs.extend(group_lrs)
            print(f'Train Loss : {sum(losses) / len(losses)}, Train Metric : {sum(metrics) / len(metrics)}')

            if self.val_loader:
                with torch.no_grad():
                    self.model.eval()
                    losses, metrics, _ = self._run_epoch(epoch, self.val_loader, mode='val')
                    val_losses.append(sum(losses) / len(losses))
                    val_metrics.append(sum(metrics) / len(metrics))
                    print(f'Val Loss : {sum(losses) / len(losses)}, Val Metric : {sum(metrics) / len(metrics)}')

            ckpt = {
                'model': self.model,
                'optimizer': self.optimizer.state_dict()
            }

            SaveModel(ckpt, 'last.pth', weights_only=True)

        with open('train_log.txt', 'a') as f:
            f.write(f'{train_losses}\n')
            f.write(f'{train_metrics}\n')
            f.write(f'{train_lrs}\n')
            f.write(f'{val_losses}\n')
            f.write(f'{val_metrics}\n')

    def _run_epoch(self, epoch, data_loader, mode='train'):
        assert data_loader, 'data_loader can not be None'

        total_loss = []
        total_metric = []
        group_lrs = []

        with tqdm(data_loader) as t:
            for batch_idx, (images, labels) in enumerate(t):
                if self.device.type == 'cuda':
                    images = images.cuda(non_blocking=True)
                    labels = labels.cuda(non_blocking=True)

                pred = self.model(images)

                metric = self.metric(pred, labels)

                if mode == 'train':
                    self.optimizer.zero_grad()

                loss = self.loss(pred, labels)

                if mode == 'train':
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()

                t.set_description(f"{mode} Epoch {epoch + 1}")
                t.set_postfix(batch=batch_idx + 1, loss=loss.item(), metric=metric.item(), lrs=[group['lr'] for group in self.optimizer.state_dict()['param_groups']])

                total_loss.append(loss.item())
                total_metric.append(metric.item())

                group_lrs.append([group['lr'] for group in self.optimizer.state_dict()['param_groups']])

        return total_loss, total_metric, group_lrs