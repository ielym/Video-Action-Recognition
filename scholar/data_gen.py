import yaml
import os
from glob import glob
import tqdm
import math
import multiprocessing
import time
import numpy as np
import cv2

import torch
from torch.utils.data import DataLoader, Dataset, distributed
import random

from fastvision.detection.tools import xyxy2xywhn, xyxy2xywh, xywh2xyxy
from fastvision.detection.plot import draw_box_label

from fastvision.datasets.common.video_sampler import randomClipSampling, averageSampling, randomSampling, consecutiveSampling
# from fastvision.datasets.common.augmentation import Augmentation, HorizontalFlip, VerticalFlip, Normalization, Resize, Padding, CenterCrop, RandomCrop, BGR2RGB

from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)
from albumentations.pytorch import ToTensorV2

class BaseDataset(Dataset):
    def __init__(self, samples, frames, input_size):

        self.samples = samples

        self.frames = frames

        self.input_size = input_size
        if isinstance(input_size, int):
            self.input_height = input_size
            self.input_width = input_size
        else:
            self.input_height = input_size[0]
            self.input_width = input_size[1]

        self.augmentation = Compose([
			RandomResizedCrop(height=self.input_height, width=self.input_width, p=1.0),
			# Transpose(p=0.5),
			HorizontalFlip(p=0.5),
			# VerticalFlip(p=0.5),
			# ShiftScaleRotate(p=0.5),
			# HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
			# RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=0.5),
			Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
			# CoarseDropout(p=0.5),
			# Cutout(p=0.5, max_h_size=20, max_w_size=20),
			ToTensorV2(p=1.0),
		], p=1.)

    def __len__(self):
        return len(self.samples)

    def load_video(self, video_path):

        # cap = cv2.VideoCapture(video_path)
        # sampling_frames = randomClipSampling(cap, clips=self.frames, frames_per_clip=1)

        cache_path = os.path.join('./cache/cache_data', f'{os.path.basename(video_path)}.npy')
        sampling_frames = np.load(cache_path)
        return sampling_frames

    def __getitem__(self, idx):
        all_stime = time.time()

        sample = self.samples[idx]

        video_path = sample[0]
        category_idx = sample[1]

        # ======================================== process image ========================================
        load_file_stime = time.time()
        ori_frames = self.load_video(video_path)
        load_file_etime = time.time()

        aug_stime = time.time()
        frames = []
        seed = random.choice(range(1000))
        for frame_idx in range(len(ori_frames)):
            random.seed(seed)
            frames.append(self.augmentation(image=cv2.cvtColor(ori_frames[frame_idx, ...], cv2.COLOR_BGR2RGB))['image'])
        frames = torch.stack(frames, dim=1).float()
        aug_etime = time.time()

        # ======================================== process label ========================================
        labels = torch.tensor(category_idx)
        # print(f'all : {time.time() - all_stime}, load : {load_file_etime - load_file_stime}, aug : {aug_etime - aug_stime}')

        return frames, labels

def load_samples(data_dir, prefix, num_workers, cache, use_cache):

    if use_cache:
        with open(os.path.join(cache, f'{prefix}.txt'), 'r') as f:
            samples = eval(f.read())

        print(f'Use {prefix} data from cache {cache} {prefix}.txt')
        return samples

    videos_dir = os.path.join(data_dir, 'videos')
    labels_path = os.path.join(data_dir, 'labels.txt')

    with open(labels_path, 'r') as f:
        lines = f.readlines()

    samples = []
    for line in tqdm.tqdm(lines, desc=f'Extract {prefix} dataset '):
        video_name, category_idx = line.strip().split()
        cap = cv2.VideoCapture(os.path.join(videos_dir, video_name))
        if cap.get(cv2.CAP_PROP_FRAME_COUNT) >= 128:
            samples.append((os.path.join(videos_dir, video_name), int(category_idx)))

    if cache:
        with open(os.path.join(cache, f'{prefix}.txt'), 'w') as f:
            f.write(str(samples))
        print(f'Save {prefix} data to cache {cache} {prefix}.txt')

    return samples

def save_2_npy(video_path):
    cap = cv2.VideoCapture(video_path)
    sampling_frames = consecutiveSampling(cap, frames=64)

    frames = []
    for frame in sampling_frames:
        ori_height, ori_width = frame.shape[:2]
        ratio = 256 / min(ori_height, ori_width)
        target_height, target_width = int(ori_height * ratio), int(ori_width * ratio)
        frame = cv2.resize(frame, (target_width, target_height))
        frames.append(np.expand_dims(frame, 0))
    frames = np.concatenate(frames, 0).astype(np.uint8)  # (16, 112, 112, 3)

    cache_path = os.path.join('./cache/cache_data', f'{os.path.basename(video_path)}.npy')
    np.save(cache_path, frames)

def preprocess(samples):
    pool = multiprocessing.Pool(4)

    # ------------- tqdm with multiprocessing -------------
    pbar = tqdm.tqdm(total=len(samples))
    pbar.set_description(f'Process : ')
    update_tqdm = lambda *args: pbar.update()
    # -----------------------------------------------------

    for sample in samples:
        video_path = sample[0]
        pool.apply_async(save_2_npy, args=(video_path, ), callback=update_tqdm)

    pool.close()
    pool.join()
    pbar.close()

def create_dataloader(prefix, data_dir, batch_size, frames, input_size, device, num_workers=0, cache='./cache', use_cache=False, DistributedDataParallel=False, shuffle=True, pin_memory=True, drop_last=False):

    samples = load_samples(data_dir, prefix, num_workers, cache, use_cache)
    # preprocess(samples)

    dataset = BaseDataset(samples, frames, input_size)

    loader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=shuffle and not DistributedDataParallel,
                pin_memory=pin_memory,
                # persistent_workers=True,
                prefetch_factor = 2,
                sampler=distributed.DistributedSampler(dataset, shuffle=shuffle),
                drop_last=drop_last,
                num_workers=num_workers if device.type != 'cpu' else 0,
            )

    return loader


