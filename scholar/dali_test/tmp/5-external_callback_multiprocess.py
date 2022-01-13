from nvidia.dali import pipeline_def, Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali.plugin.pytorch import LastBatchPolicy

import os
from glob import glob
import numpy as np
from random import shuffle
import random
import cv2
import time

# ========================================================
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import DataLoader, Dataset, distributed
# ========================================================

class ExternalInputCallable:
    def __init__(self, image_dir, batch_size):
        self.batch_size = batch_size

        self.samples = glob(os.path.join(image_dir, '*.jpg'))
        self.full_iterations = len(self.samples) // batch_size

    def __call__(self, sample_info):
        sample_idx = sample_info.idx_in_epoch

        if sample_info.iteration >= self.full_iterations:
            raise StopIteration()

        file_path = self.samples[sample_idx]

        # with open(file_path, 'rb') as f:
        #     encoded_img = np.frombuffer(f.read(), dtype=np.uint8)
        encoded_img = np.fromfile(file_path, dtype=np.uint8)

        return encoded_img

@pipeline_def
def callable_pipeline(external_data):
    # jpegs = fn.external_source(source=external_data, num_outputs=1, batch=False, parallel=True)
    jpegs = fn.external_source(source=external_data, batch=False, parallel=True)

    images = fn.decoders.image(jpegs, device="mixed")

    images = fn.resize(images, resize_shorter=256, interp_type=types.INTERP_LINEAR)
    images = fn.crop_mirror_normalize(
        images,
        crop_pos_x=fn.random.uniform(range=(0.0, 1.0)),
        crop_pos_y=fn.random.uniform(range=(0.0, 1.0)),
        mirror=fn.random.coin_flip(),
        dtype=types.FLOAT,
        crop=(224, 224),
        mean=[0., 0., 0.],
        std=[1., 1., 1.],
    )

    return images


class BaseDataset(Dataset):
    def __init__(self, images_dir):

        self.samples = glob(os.path.join(images_dir, '*.jpg'))

        self.augmentation = Compose([
            RandomResizedCrop(height=224, width=224, p=1.0),
            HorizontalFlip(p=0.5),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = cv2.imread(sample)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.augmentation(image=img)['image']
        return img


if __name__ == '__main__':

    # image_dir = r"/app/datasets/dali/dog"
    image_dir = r"/app/datasets/voc2012/train/images"
    num_threads = 2
    gpus = [0]  # number of GPUs
    device_id = 0
    num_gpus = len(gpus)
    batch_size = 8  # batch size per GPU
    epochs = 5

    # ======================== Nvidia DALI =============================
    stime = time.time()
    external_data = ExternalInputCallable(image_dir, batch_size)
    pipe = callable_pipeline(external_data=external_data, batch_size=batch_size, num_threads=num_threads, device_id=0, py_num_workers=4, py_start_method='spawn')
    dataloader = DALIGenericIterator(pipelines=pipe, output_map=['data'])

    for e in range(epochs):
        for i, data in enumerate(dataloader):
            images = data[0]['data']
            print("epoch: {}, batch {}, image size: {}, device : {}".format(e, i, images.size(), images.device))
        dataloader.reset()
    print(f'NVIDIA DALI : {time.time() - stime}')
    input('continue')

    # ======================== Nvidia DALI =============================
    stime = time.time()
    dataset = BaseDataset(image_dir)
    loader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=True,
                # persistent_workers=True,
                prefetch_factor = 1,
                # sampler=distributed.DistributedSampler(dataset, shuffle=shuffle),
                drop_last=False,
                num_workers=num_threads,
            )
    for e in range(epochs):
        for i, images in enumerate(loader):
            images = images.cuda(non_blocking=True)
            print("epoch: {}, batch {}, image size: {}, device : {}".format(e, i, images.size(), images.device))

    print(f'Pytorch DataLoader : {time.time() - stime}')




