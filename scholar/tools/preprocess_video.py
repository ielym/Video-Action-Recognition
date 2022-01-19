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

def check_video(video_path, category_idx, samples):
    cap = cv2.VideoCapture(video_path)

    if cap.get(cv2.CAP_PROP_FRAME_COUNT) >= 16:
        samples.append((video_path, int(category_idx)))

def load_samples(data_dir, prefix, cache_dir, num_works):

    videos_dir = os.path.join(data_dir, 'videos')
    labels_path = os.path.join(data_dir, 'labels.txt')

    with open(labels_path, 'r') as f:
        lines = f.readlines()

    pool = multiprocessing.Pool(num_works)

    # ------------- tqdm with multiprocessing -------------
    pbar = tqdm.tqdm(total=len(lines))
    pbar.set_description(f'Load and Check Samples : ')
    update_tqdm = lambda *args: pbar.update()
    # -----------------------------------------------------

    mgr = multiprocessing.Manager()
    ret_samples = mgr.list()

    for line in lines:
        video_name, category_idx = line.strip().split()
        video_path = os.path.join(videos_dir, video_name)

        pool.apply_async(check_video, args=(video_path, category_idx, ret_samples), callback=update_tqdm)

    pool.close()
    pool.join()
    pbar.close()

    return ret_samples

def save_2_npy(video_path, category_idx):
    cap = cv2.VideoCapture(video_path)
    sampling_frames = randomClipSampling(cap, clips=8, frames_per_clip=1)

    frames = []
    for frame in sampling_frames:
        ori_height, ori_width = frame.shape[:2]
        ratio = 256 / min(ori_height, ori_width)
        target_height, target_width = int(ori_height * ratio), int(ori_width * ratio)
        frame = cv2.resize(frame, (target_width, target_height))
        frames.append(np.expand_dims(frame, 0))
    frames = np.concatenate(frames, 0).astype(np.uint8)  # (16, 112, 112, 3)

    cache_data_path = os.path.join('/app/datasets/cache', f'{os.path.basename(video_path)}.npy')
    np.save(cache_data_path, frames)

    cache_label_path = os.path.join('/app/datasets/cache', f'label-{os.path.basename(video_path)}.npy')
    np.save(cache_label_path, np.array(category_idx, dtype=np.uint8))


def preprocess(samples, num_works):
    pool = multiprocessing.Pool(num_works)

    # ------------- tqdm with multiprocessing -------------
    pbar = tqdm.tqdm(total=len(samples))
    pbar.set_description(f'Process : ')
    update_tqdm = lambda *args: pbar.update()
    # -----------------------------------------------------

    for sample in samples:
        video_path = sample[0]
        category_idx = sample[1]
        pool.apply_async(save_2_npy, args=(video_path, category_idx, ), callback=update_tqdm)

    pool.close()
    pool.join()
    pbar.close()

if __name__ == '__main__':
    data_dir = r'/app/datasets/ucf101/train'
    cache_dir = r'../cache'
    num_works = 32

    if not os.path.exists('/app/datasets/cache'):
        os.makedirs('/app/datasets/cache')

    samples = load_samples(data_dir, 'train', cache_dir, num_works)
    preprocess(samples, num_works)

    with open('../cache/dali_train.txt', 'w') as f:
        for sample in tqdm.tqdm(samples):
            video_path, category_idx = sample
            base_name = os.path.basename(video_path)
            npy_path = os.path.join(r'/app/datasets/cache/', f'{base_name}.npy')
            label_path = os.path.join(r'/app/datasets/cache/', f'label-{base_name}.npy')
            f.write(f'{npy_path} {label_path}\n')