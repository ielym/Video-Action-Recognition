from nvidia.dali import pipeline_def, Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali.plugin.pytorch import DALIClassificationIterator as PyTorchIterator
from nvidia.dali.plugin.pytorch import LastBatchPolicy

import os
from glob import glob
import numpy as np
from random import shuffle
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

def ExternalSourcePipeline(batch_size, num_threads, device_id, external_data):
    pipe = Pipeline(batch_size, num_threads, device_id)
    with pipe:
        jpegs = fn.external_source(source=external_data)
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

        output = fn.cast(images, dtype=types.UINT8)
        pipe.set_outputs(output)
    return pipe

class ExternalInputIterator(object):
    def __init__(self, batch_size, image_dir, device_id, num_gpus):
        self.batch_size = batch_size
        self.images_dir = image_dir

        # with open(self.images_dir + "file_list.txt", 'r') as f:
        #     self.files = [line.rstrip() for line in f if line is not '']

        self.samples = glob(os.path.join(self.images_dir, '*.jpg'))

        # whole data set size
        self.data_set_len = len(self.samples)

        # based on the device_id and total number of GPUs - world size
        # get proper shard
        self.samples = self.samples[self.data_set_len * device_id // num_gpus:self.data_set_len * (device_id + 1) // num_gpus]
        self.n = len(self.samples)

    def __iter__(self):
        self.i = 0
        shuffle(self.samples)
        return self

    def __next__(self):
        datas = []

        if self.i >= self.n:
            self.__iter__()
            raise StopIteration

        for _ in range(self.batch_size):

            file_name = self.samples[self.i % self.n]

            # img = cv2.imread(file_name)
            # datas.append(img.astype(np.uint8))  # we can use numpy
            datas.append(np.fromfile(file_name, dtype = np.uint8))
            # labels.append(torch.tensor([int(label)], dtype = torch.uint8)) # or PyTorch's native tensors
            self.i += 1
        return datas

    def __len__(self):
        return self.data_set_len

    next = __next__


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

    image_dir = r"/app/datasets/dali/dog"
    num_threads = 2
    gpus = [0]  # number of GPUs
    device_id = 0
    num_gpus = len(gpus)
    BATCH_SIZE = 1  # batch size per GPU
    epochs = 200

    # ======================== Nvidia DALI =============================
    stime = time.time()
    eii = ExternalInputIterator(batch_size=BATCH_SIZE, image_dir=image_dir, device_id=device_id, num_gpus=num_gpus)
    pipe = ExternalSourcePipeline(batch_size=BATCH_SIZE, num_threads=num_threads, device_id=device_id, external_data=eii)

    dali_iter = DALIGenericIterator(pipelines=pipe, output_map=['data'])

    for e in range(epochs):
        for i, data in enumerate(dali_iter):
            images = data[0]['data']
            print("epoch: {}, batch {}, image size: {}, device : {}".format(e, i, images.size(), images.device))
        dali_iter.reset()
    print(f'NVIDIA DALI : {time.time() - stime}')

    # ======================== Nvidia DALI =============================
    stime = time.time()
    dataset = BaseDataset(image_dir)
    loader = DataLoader(
                dataset=dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                pin_memory=True,
                # persistent_workers=True,
                prefetch_factor = 1,
                # sampler=distributed.DistributedSampler(dataset, shuffle=shuffle),
                drop_last=False,
                num_workers=num_threads,
            )
    for e in range(epochs):
        for i, (images, ) in enumerate(loader):
            images = images.cuda(non_blocking=True)
            # print("epoch: {}, batch {}, image size: {}, device : {}".format(e, i, images.size(), images.device))

    print(f'Pytorch DataLoader : {time.time() - stime}')




