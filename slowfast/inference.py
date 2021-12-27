import os
from glob import glob
import yaml
import cv2

import numpy as np
import torch
import torch.nn as nn

from fastvision.utils.checkpoints import LoadStatedict
from fastvision.videoRecognition.models import c3d, c3d_bn
from fastvision.datasets.common import randomClipSampling
from fastvision.datasets.common.augmentation import Augmentation, HorizontalFlip, VerticalFlip, Normalization, Resize, Padding, CenterCrop, RandomCrop, BGR2RGB

def dataloader_fn(args):
    data_dict = yaml.safe_load(open(args.data_yaml, 'r'))

    num_classes = data_dict['num_classes']
    category_names = data_dict['categories']
    assert (num_classes == len(category_names)), f"num_classes {num_classes} must equal len(category_names) {len(category_names)}"

    if os.path.isdir(args.data_path):
        base_names = os.listdir(args.data_path)
        file_names = [os.path.join(args.data_path, name) for name in base_names]

    else:
        file_names = [args.data_path]

    args.num_classes = num_classes
    args.category_names = category_names

    return file_names

def model_fn(args, device):

    # model = c3d(in_channels=args.in_channels, num_classes=args.num_classes, including_top=True)
    model = c3d_bn(in_channels=args.in_channels, num_classes=args.num_classes, including_top=True)

    if args.inference_weights:
        model = LoadStatedict(model=model, weights=args.inference_weights, device=device, strict=True)

    if device.type == 'cuda':
        print('Model : using cuda')
        model = model.cuda()

    if device.type == 'cuda' and args.DataParallel:
        print('Model : using DataParallel')
        model = nn.DataParallel(model)

    if device.type == 'cuda' and args.DistributedDataParallel and args.SyncBatchNorm:
        print('Model : using SyncBatchNorm')
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)



    model.half().float()
    return model

def preprocess(video_path, input_size):

    if isinstance(input_size, int):
        input_height = input_size
        input_width = input_size
    else:
        input_height = input_size[0]
        input_width = input_size[1]

    augmentation = Augmentation([
        Resize(size=(128, 171), p=1.0),
        RandomCrop(size=(input_height, input_width), p=1.0),
        # HorizontalFlip(p=0.5),
        BGR2RGB(p=1.0),
        Normalization(p=1.0),
    ], mode='classification')

    cap = cv2.VideoCapture(video_path)

    ori_frames = randomClipSampling(cap, clips=16, frames_per_clip=1)

    frames = []
    augmentation.lock_prob()
    for frame_idx in range(len(ori_frames)):
        frames.append(np.expand_dims(augmentation(ori_frames[frame_idx, ...]), 0))
    augmentation.unlock_prob()

    frames = np.concatenate(frames, 0)  # (16, 112, 112, 3)

    frames = frames.transpose([3, 0, 1, 2])
    frames = np.ascontiguousarray(frames)
    frames = frames.astype(np.float32)
    frames = torch.from_numpy(frames)

    return frames

@torch.no_grad()
def Inference(args, device):

    file_names = dataloader_fn(args)

    category_id_name_map = {k : v for k, v in enumerate(args.category_names)}


    # ======================= Model ============================
    model = model_fn(args, device=device)
    model.eval()

    np.random.shuffle(file_names)
    for file_name in file_names:

        frames = preprocess(file_name, args.input_size)

        results = model(frames.unsqueeze(0))
        scores = torch.softmax(results, dim=1).cpu().numpy()

        category = np.argmax(scores, axis=1)[0]
        score = scores[..., category][0]

        cap = cv2.VideoCapture(file_name)
        ret, frame = cap.read()
        while ret:
            cv2.putText(frame, category_id_name_map[category], (30, 50), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 2)

            cv2.imshow('img', frame)
            cv2.waitKey(10)
            ret, frame = cap.read()



