import sys
sys.path.append(r'../')
sys.path.append(r'../../')
sys.path.append(r'../../../')
sys.path.append(r'../../../../')

import numpy as np
import multiprocessing
import tqdm
import cv2
import os

from fastvision.datasets.common.video_sampler import randomClipSampling, averageSampling, randomSampling, consecutiveSampling

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

def load_samples(data_dir, prefix, cache):

    videos_dir = os.path.join(data_dir, 'videos')
    labels_path = os.path.join(data_dir, 'labels.txt')

    with open(labels_path, 'r') as f:
        lines = f.readlines()

    samples = []
    for line in tqdm.tqdm(lines, desc=f'Extract {prefix} dataset '):
        video_name, category_idx = line.strip().split()
        cap = cv2.VideoCapture(os.path.join(videos_dir, video_name))
        if cap.get(cv2.CAP_PROP_FRAME_COUNT) >= 32:
            samples.append((os.path.join(videos_dir, video_name), int(category_idx)))

    if cache:
        with open(os.path.join(cache, f'{prefix}.txt'), 'w') as f:
            f.write(str(samples))
        print(f'Save {prefix} data to cache {cache} {prefix}.txt')

    return samples

def preprocess(samples, num_works):
    pool = multiprocessing.Pool(num_works)

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

if __name__ == '__main__':
    data_dir = r'/app/datasets/kinetics400/train'
    cache_dir = r'./cache'
    num_works = 8

    if not os.path.exists('../cache/cache_data'):
        os.makedirs('../cache/cache_data')

    samples = load_samples(data_dir, 'train', cache_dir)
    preprocess(samples, num_works)
