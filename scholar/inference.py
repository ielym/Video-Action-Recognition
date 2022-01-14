import os
from glob import glob
import yaml
import cv2
import argparse
import tqdm

import numpy as np
import torch
import torch.nn as nn

from fastvision.utils.device import set_device
from fastvision.utils.checkpoints import LoadStatedict
from fastvision.videoRecognition.models import resnet50_3d
from fastvision.datasets.common import randomClipSampling, consecutiveSampling
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

    model = resnet50_3d(in_channels=args.in_channels, num_classes=args.num_classes)

    if args.inference_weights:
        model = LoadStatedict(model=model, weights=args.inference_weights, device=device, strict=True)

    model = model.cuda()

    # model.half().float()
    return model

def preprocess(video_path, input_size):

    if isinstance(input_size, int):
        input_height = input_size
        input_width = input_size
    else:
        input_height = input_size[0]
        input_width = input_size[1]

    augmentation = Augmentation([
        Resize(size=256, resize_by='shorter', p=1.0),
        RandomCrop(size=(input_height, input_width), p=1.0),
        # HorizontalFlip(p=0.5),
        BGR2RGB(p=1.0),
    ], mode='classification')

    cap = cv2.VideoCapture(video_path)

    ori_frames = randomClipSampling(cap, clips=8, frames_per_clip=1)
    # ori_frames = consecutiveSampling(cap, frames=8)

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

    # ===================================== Process Labels
    with open(r'S:\datasets\ucf101\train\labels.txt', 'r') as f:
        lines = f.readlines()
    labels = {}
    for line in lines:
        line = line.strip()
        category_name, category_id = line.split()
        labels[category_name] = int(category_id)


    file_names = dataloader_fn(args)
    category_id_name_map = {k : v for k, v in enumerate(args.category_names)}


    # ======================= Model ============================
    model = model_fn(args, device=device)
    model.eval()

    np.random.seed(0)
    np.random.shuffle(file_names)

    total_sample = 0
    match = 0
    for file_name in tqdm.tqdm(file_names):

        frames = preprocess(file_name, args.input_size) / 255.

        results = model(frames.unsqueeze(0).cuda())
        scores = torch.softmax(results, dim=1).cpu().numpy()

        category = np.argmax(scores, axis=1)[0]
        score = scores[..., category][0]

        base_name = os.path.basename(file_name)
        gt_category_id = labels[base_name]

        total_sample += 1
        if category == gt_category_id:
            match += 1

    print(match / total_sample)

        # cap = cv2.VideoCapture(file_name)
        # ret, frame = cap.read()
        # while ret:
        #     cv2.putText(frame, category_id_name_map[category], (30, 50), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 2)
        #     cv2.putText(frame, category_id_name_map[gt_category_id], (30, 80), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), 2)
        #
        #     cv2.imshow('img', frame)
        #     cv2.waitKey(10)
        #     ret, frame = cap.read()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='FastVision')

    parser.add_argument('--in_channels', default=3, type=int, help='')
    parser.add_argument('--num_classes', default=101, type=int)
    parser.add_argument('--device', default=[0], type=list, help='[] empty for CPU, or a list like [0] or [0, 2, 3] for GPU')
    parser.add_argument('--inference_weights', default=r'S:\last.pth', type=str, help='')
    parser.add_argument('--data_yaml', default=r'./data/ucf101.yaml', type=str, help='')
    parser.add_argument('--data_path', default=r'S:\datasets\ucf101\train\videos', type=str, help='absolute path of a video or a folder')
    parser.add_argument('--input_size', default=224, type=int, help='')

    args, unknown = parser.parse_known_args()

    device = set_device(args.device)

    # pretrained = torch.load(args.inference_weights)
    # print(pretrained.keys())

    Inference(args, device)

