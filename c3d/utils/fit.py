import torch
import numpy as np

from tqdm import tqdm, trange
from fastvision.utils.checkpoints import SaveModel

class Fit():

    def __init__(self, model, device, optimizer, scheduler, loss, metric, end_epoch, start_epoch=0, train_loader=None, val_loader=None, test_loader=None):
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

    def run_epoches(self):

        for epoch in range(self.start_epoch, self.end_epoch):
            self._train(epoch)

            if self.val_loader:
                self._val()

            ckpt = {
                    'model': self.model,
                    'optimizer': self.optimizer.state_dict()
                }

            SaveModel(ckpt, 'last.pth', weights_only=True)

        if self.test_loader:
            self._test()

    def _train(self, epoch):
        assert self.train_loader, 'train_loader can not be None'

        running_loss = 0
        running_metric = 0

        self.model.train()
        with tqdm(self.train_loader) as t:
            for batch_idx, (images, labels) in enumerate(t):
                if self.device.type == 'cuda':
                    images = images.cuda(non_blocking=True)
                    labels = labels.cuda(non_blocking=True)

                pred = self.model(images)

                metric = self.metric(pred, labels)

                self.optimizer.zero_grad()
                loss = self.loss(pred, labels)

                loss.backward()
                self.optimizer.step()

                t.set_description(f"Epoch {epoch + 1}")
                t.set_postfix(batch=batch_idx + 1, loss=loss.item(), metric=metric.item())

                running_loss += loss.item()
                running_metric += metric.item()

            self.scheduler.step()

        print(f'Train Loss : {running_loss / len(self.train_loader)}, Train Metric : {running_metric / len(self.train_loader)}')

    @torch.no_grad()
    def _val(self):

        running_loss = 0
        running_metric = 0

        self.model.eval()
        with tqdm(self.val_loader) as t:
            for batch_idx, (images, labels) in enumerate(t):

                if self.device.type == 'cuda':
                    images = images.cuda(non_blocking=True)
                    labels = labels.cuda(non_blocking=True)

                pred = self.model(images)

                metric = self.metric(pred, labels)

                loss = self.loss(pred, labels)

                t.set_description(f"Validation")
                t.set_postfix(batch=batch_idx + 1, loss=loss.item(), metric=metric.item())

                running_loss += loss.item()
                running_metric += metric.item()

        print(f'Val Loss : {running_loss / len(self.val_loader)}, Val Metric : {running_metric / len(self.val_loader)}')

    def _test(self):

        pass