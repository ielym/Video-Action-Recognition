from nvidia.dali import pipeline_def, Pipeline
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import nvidia.dali.types as types

import numpy as np
import cv2
import time
import os
from glob import glob

@pipeline_def
def video_pipe(file_list, sequence_length, stride, shard_id, num_shards, initial_fill):
    videos, labels = fn.readers.video(device="gpu", file_list=file_list, sequence_length=sequence_length, pad_sequences=True, stride=stride, shard_id=shard_id, num_shards=num_shards, random_shuffle=True, initial_fill=initial_fill)
    return videos, labels

def trans_fastvision_2_dali(label_path, data_dir, cache_dir):
    with open(label_path, 'r') as f:
        lines = f.readlines()

    with open(os.path.join(cache_dir, os.path.basename(label_path)), 'w') as f:
        for line in lines:
            line = line.strip()
            file_name, category_idx = line.split()
            f.write(f'{os.path.join(data_dir, file_name)} {category_idx}')

    return os.path.join(cache_dir, os.path.basename(label_path))


def create_dataloader(prefix, data_dir, batch_size, frames, input_size, device, num_workers=0, cache='./cache'):

    sequence_length = 8
    initial_prefetch_size = 16
    stride = 8
    shard_id = 0

    fastvision_labels_path = os.path.join(data_dir, 'lables.txt')
    file_list_path = trans_fastvision_2_dali(fastvision_labels_path, data_dir, cache_dir=cache)

    pipe = video_pipe(batch_size=batch_size, num_threads=2, device_id=0, filenames=video_files)
    pipe = video_pipe(file_list=file_list_path, sequence_length=sequence_length, stride=stride, shard_id=shard_id, seed=123456)

    dali_iter = DALIGenericIterator(pipelines=pipe, output_map=['frames', 'labels'])

    stime = time.time()
    for batch_idx, data in enumerate(dali_iter):
        frames = data[0]["frames"]
        labels = data[0]["labels"]
        # print(frames.size(), labels.size(), labels)
        print("batch {}, frames size: {}, labels : {}, device : {}".format(batch_idx, frames.size(), labels.cpu().numpy().tolist(), frames.device))
    dali_iter.reset()
    print(time.time() - stime)


    return dali_iter


