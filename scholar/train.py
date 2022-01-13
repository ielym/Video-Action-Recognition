import os
import yaml

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from fastvision.videoRecognition.models import resnet50_3d
from fastvision.utils.checkpoints import LoadStatedict, SqueezeModel
from fastvision.loss import CrossEntropyLoss
from fastvision.utils.sheduler import CosineLR, LinearLR, ExponentialLR
from fastvision.metrics import Accuracy

from data_gen import create_dataloader
from utils.fit import Fit

def dataloader_fn(args, device):
    data_dict = yaml.safe_load(open(args.data_yaml, 'r'))

    shard_idx = args.local_rank
    shards_dict = eval(os.environ['FASTVISON_SHARDS'])
    print(shard_idx, shards_dict, type(shard_idx), type(shards_dict))

    num_classes = data_dict['num_classes']
    category_names = data_dict['categories']
    assert (num_classes == len(category_names)), f"num_classes {num_classes} must equal len(category_names) {len(category_names)}"

    train_dir = os.path.join(data_dict['data_root'], data_dict['train_dir'])
    train_loader = create_dataloader(prefix='train', data_dir=train_dir, batch_size=args.batch_size, frames=args.frames, input_size=args.input_size, num_workers=args.num_workers, device=device, cache=args.cache_dir, use_cache=args.use_data_cache, DistributedDataParallel=args.DistributedDataParallel, shuffle=True, pin_memory=True, drop_last=False)

    val_dir = os.path.join(data_dict['data_root'], data_dict['val_dir'])
    val_loader = create_dataloader(prefix='val', data_dir=val_dir, batch_size=args.batch_size, frames=args.frames, input_size=args.input_size, num_workers=args.num_workers, device=device, cache=args.cache_dir, use_cache=args.use_data_cache, DistributedDataParallel=args.DistributedDataParallel, shuffle=True, pin_memory=True, drop_last=False)

    # show_dataset(prefix='train', data_dir=train_dir, category_names=category_names, num_workers=num_workers, cache=cache, use_cache=use_cache)

    args.num_classes = num_classes
    args.category_names = category_names

    return train_loader, val_loader, data_dict

def loss_fn(device):
    loss = CrossEntropyLoss(reduction='mean')
    if device.type == 'cuda':
        loss = loss.cuda()
    return loss

def optimizer_fn(model, lr, weight_decay):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optimizer

def model_fn(args, device):

    model = resnet50_3d(in_channels=args.in_channels, num_classes=args.num_classes)

    if args.pretrained_weights:
        model = LoadStatedict(model=model, weights=args.pretrained_weights, device=device, strict=False)

    if device.type == 'cuda':
        print('Model : using cuda')
        model = model.cuda()

    if device.type == 'cuda' and args.DataParallel and not args.DistributedDataParallel:
        print('Model : using DataParallel')
        model = nn.DataParallel(model)

    if device.type == 'cuda' and args.DistributedDataParallel:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
        print('Model : using DistributedDataParallel')

    if device.type == 'cuda' and args.DistributedDataParallel and args.SyncBatchNorm:
        print('Model : using SyncBatchNorm')
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = SqueezeModel(model, 'all', True)
    model = SqueezeModel(model, ['classifier'], True)

    model.half().float()
    return model

def Train(args, device):

    # ======================= Data Loader ============================
    train_loader, val_loader, data_dict = dataloader_fn(args, device=device)

    # ======================= Model ============================
    model = model_fn(args, device=device)

    # # ======================= Loss ============================
    loss = loss_fn(device)

    # # ======================= metrics ============================
    metric = Accuracy()

    # ======================= Optimizer ============================
    optimizer = optimizer_fn(model=model, lr=1, weight_decay=args.weight_decay) # here lr have to set to 1
    scheduler = CosineLR(optimizer=optimizer, steps=args.epochs * len(train_loader), initial_lr=args.initial_lr, last_lr=args.last_lr)
    # scheduler = ExponentialLR(optimizer=optimizer, steps=1 * len(train_loader), initial_lr=2e-6, last_lr=1e-4)

    est = Fit(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                loss=loss,
                metric=metric,

                start_epoch=0,
                end_epoch=args.epochs,

                device = device,

                train_loader=train_loader,
                val_loader=val_loader,
        )

    # est.find_lr()

    est.trainEpoches()