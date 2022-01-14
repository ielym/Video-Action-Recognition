# -*- coding: utf-8 -*-
import sys
sys.path.append(r'../')
sys.path.append(r'../../')
sys.path.append(r'../../../')

import argparse
import os
import random
import multiprocessing
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch import distributed as dist

from fastvision.utils.seed import set_random_seeds
from fastvision.utils.device import set_device

from train import Train
from inference import Inference

# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 run.py

torch.distributed.init_process_group(backend="nccl")
parser = argparse.ArgumentParser(description='FastVision')

parser.add_argument('--mode', default='Train', choices=['Train', 'Test'], type=str, help='')
parser.add_argument('--local_rank', default=-1, type=int)

# Data generation
parser.add_argument('--data_yaml', default=r'./data/ucf101.yaml', type=str, help='')
parser.add_argument('--batch_size', default=32, help='')
parser.add_argument('--in_channels', default=3, type=int, help='')
parser.add_argument('--frames', default=8, type=int, help='input frames')
parser.add_argument('--input_size', default=224, type=int, help='')
parser.add_argument('--num_workers', default=0.4, type=float, help='')

# Train
parser.add_argument('--device', default=[0, 1], type=list, help='[] empty for CPU, or a list like [0] or [0, 2, 3] for GPU')
parser.add_argument('--epochs', default=40000, type=int, help='')
parser.add_argument('--initial_lr', default=1e-4, type=float, help='')
parser.add_argument('--last_lr', default=1e-7, type=float, help='')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='')
parser.add_argument('--seed', default=49, type=int, help='0/1/2/... or None')
parser.add_argument('--DataParallel', default=True, type=bool, help='')
parser.add_argument('--DistributedDataParallel', default=True, type=bool, help='')
parser.add_argument('--SyncBatchNorm', default=True, type=bool, help='')
parser.add_argument('--pretrained_weights', default='/app/zoos/resnet50_3d.pth', type=str, help='')


# Inference
parser.add_argument('--inference_weights', default=r'P:\PythonWorkSpace\last.pth', type=str, help='')
parser.add_argument('--img_path', default=r'S:\datasets\voc2012\val\images', type=str, help='absolute path of a img or a folder')

# cache
parser.add_argument('--cache_dir', default='./cache', type=str, help='')
parser.add_argument('--use_data_cache', default=True, type=bool, help='')


args, unknown = parser.parse_known_args()
torch.cuda.set_device(args.local_rank)

def main(args):

    set_random_seeds(args.seed)

    device = set_device(args.device)

    if args.num_workers >= 0 and args.num_workers <= 1:
        args.num_workers = min(int(multiprocessing.cpu_count() * args.num_workers), int(multiprocessing.cpu_count()))
        print(f'Num Workers : {args.num_workers}')
    else:
        raise Exception(f"num_works must be 0 or in range [0, 1]")


    if args.mode == 'Train':
        args.training = True
        Train(args=args, device=device)
    elif args.mode == 'Test':
        args.training = False
        Inference(args=args, device=device)

if __name__ == '__main__':
    main(args)
