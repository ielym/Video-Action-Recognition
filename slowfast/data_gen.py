import yaml
import os
from glob import glob
import tqdm
import math
import multiprocessing

import numpy as np
import cv2

import torch
from torch.utils.data import DataLoader, Dataset, distributed

from fastvision.detection.tools import xyxy2xywhn, xyxy2xywh, xywh2xyxy
from fastvision.detection.plot import draw_box_label

from fastvision.datasets.common.video_sampler import randomClipSampling, averageSampling, randomSampling, consecutiveSampling
from fastvision.datasets.common.augmentation import Augmentation, HorizontalFlip, VerticalFlip, Normalization, Resize, Padding, CenterCrop, RandomCrop, BGR2RGB


class BaseDataset(Dataset):
    def __init__(self, samples, frames, tau, alpha, beta, input_size):

        self.samples = samples

        self.frames = frames
        self.tau = tau
        self.alpha = alpha
        self.beta = beta

        self.input_size = input_size
        if isinstance(input_size, int):
            self.input_height = input_size
            self.input_width = input_size
        else:
            self.input_height = input_size[0]
            self.input_width = input_size[1]

        self.augmentation = Augmentation([
                                Resize(size=256, p=1.0),
                                RandomCrop(size=(self.input_height, self.input_width), p=1.0),
                                HorizontalFlip(p=0.5),
                                BGR2RGB(p=1.0),
                                Normalization(p=1.0),
                            ], mode='classification')

    def __len__(self):
        return len(self.samples)

    def load_video(self, video_path):

        cap = cv2.VideoCapture(video_path)

        sampling_frames = consecutiveSampling(cap, frames=self.frames * self.tau)

        return sampling_frames

    def __getitem__(self, idx):
        sample = self.samples[idx]

        video_path = sample[0]
        category_idx = sample[1]

        # ======================================== process image ========================================
        ori_frames = self.load_video(video_path)

        frames = []
        self.augmentation.lock_prob()
        for frame_idx in range(len(ori_frames)):
            frames.append(np.expand_dims(self.augmentation(ori_frames[frame_idx, ...]), 0))
        self.augmentation.unlock_prob()

        frames = np.concatenate(frames, 0) # (64, 224, 224, 3)
        frames = frames.transpose([3, 0, 1, 2])
        frames = np.ascontiguousarray(frames)
        frames = frames.astype(np.float32)
        frames = torch.from_numpy(frames)

        print(frames.size)

        # ======================================== process label ========================================
        labels = torch.tensor(category_idx)


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
        samples.append((os.path.join(videos_dir, video_name), int(category_idx)))

    if cache:
        with open(os.path.join(cache, f'{prefix}.txt'), 'w') as f:
            f.write(str(samples))
        print(f'Save {prefix} data to cache {cache} {prefix}.txt')

    return samples

def create_dataloader(prefix, data_dir, batch_size, frames, tau, alpha, beta, input_size, device, num_workers=0, cache='./cache', use_cache=False, shuffle=True, pin_memory=True, drop_last=False):

    samples = load_samples(data_dir, prefix, num_workers, cache, use_cache)

    dataset = BaseDataset(samples, frames, tau, alpha, beta, input_size)

    loader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                pin_memory=pin_memory,
                # sampler=distributed.DistributedSampler(datasets, shuffle=shuffle),
                drop_last=drop_last,
                num_workers=num_workers if device.type != 'cpu' else 0,
            )

    return loader


