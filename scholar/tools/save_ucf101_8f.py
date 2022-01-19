import sys
sys.path.append(r'../')
sys.path.append(r'../../')
sys.path.append(r'../../../')
sys.path.append(r'../../../../')

import numpy as np
import cv2
import multiprocessing
import tqdm
import os
from fastvision.datasets import randomClipSampling


def write_video(source_path, target_path, category_idx, return_list):
    try:
        cap = cv2.VideoCapture(source_path)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if total_frames > 8:
            frames = randomClipSampling(cap, clips=8, frames_per_clip=1)

            ori_height, ori_width = frames[0].shape[:2]
            ratio = 256 / min(ori_height, ori_width)
            target_height, target_width = int(ratio * ori_height), int(ratio * ori_width)
            target_height = (target_height // 32) * 32
            target_width = (target_width // 32) * 32

            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(target_path, fourcc, 1, (target_width, target_height))

            for frame in frames:
                frame = cv2.resize(frame, (target_width, target_height))
                out.write(frame)

            cap.release()
            out.release()
            return_list.append((os.path.basename(target_path), category_idx))
    except Exception as e:
        print(e)


if __name__ == '__main__':

    data_dir = r'/app/datasets/kinetics400/train'
    labels_path = os.path.join(data_dir, 'labels.txt')

    target_dir = r'/app/datasets/kinetics400/train8f'
    target_label_path = os.path.join(target_dir, 'labels.txt')

    with open(labels_path, 'r') as f:
        lines = f.readlines()


    pool = multiprocessing.Pool(32)
    mgr = multiprocessing.Manager()
    return_list = mgr.list()

    # ------------- tqdm with multiprocessing -------------
    pbar = tqdm.tqdm(total=len(lines))
    pbar.set_description(f'Process : ')
    update_tqdm = lambda *args: pbar.update()
    # -----------------------------------------------------


    try:
        for line in lines:
        # for line in tqdm.tqdm(lines):
            line = line.strip()
            video_name, category_idx = line.split()

            source_path = os.path.join(data_dir, 'videos', video_name)
            target_path = os.path.join(target_dir, 'videos', video_name.replace('.mp4', '.avi'))
            pool.apply_async(write_video, args=(source_path, target_path, category_idx, return_list,), callback=update_tqdm)
            # write_video(source_path, target_path, category_idx, return_list)

    except Exception as e:
        print(e)

    pool.close()
    pool.join()
    pbar.close()

    with open(target_label_path, 'w') as f:
        for line in return_list:
            video_name, category_idx = line[0], line[1]
            f.write(f'{video_name} {category_idx}\n')