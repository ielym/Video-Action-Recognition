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
def video_pipe(files, labels):
    # videos, labels = fn.readers.video(device="gpu", file_list=filenames, sequence_length=sequence_length, pad_sequences=True, stride=8, shard_id=0, num_shards=1, random_shuffle=True, initial_fill=initial_prefetch_size)
    frames = fn.readers.numpy(device="gpu", files=files, seed=2020)
    label = fn.readers.numpy(device="cpu", files=labels, seed=2020)
    # frame = fn.resize(frame, resize_shorter=256, interp_type=types.INTERP_LINEAR)
    frames = fn.random_resized_crop(frames.gpu(), size=(224, 224), random_area=[1.0, 1.5], device="gpu")

    return frames, label.gpu()


if __name__ == '__main__':

    numpy_files = r"/app/src/cache/dali_train.txt"

    file_label_path = []
    with open(numpy_files, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        file_path, label_path = line.split()
        file_label_path.append((file_path, label_path))
    file_label_path = np.array(file_label_path).reshape([-1, 2])

    files = file_label_path[:, 0].tolist()
    labels = file_label_path[:, 1].tolist()

    initial_prefetch_size = 16
    n_iter = 1000
    batch_size = 16  # batch size per GPU

    pipe = video_pipe(batch_size=batch_size, num_threads=2, device_id=0, files=files, labels=labels, seed=123456)
    dali_iter = DALIGenericIterator(pipelines=pipe, output_map=['frames', 'labels'])

    stime = time.time()
    for batch_idx, data in enumerate(dali_iter):
        frames = data[0]["frames"]
        labels = data[0]["labels"]
        print("batch {}, frames size: {}, labels : {}, device : {} {}".format(batch_idx, frames.size(), labels.cpu().numpy().tolist(), frames.device, labels.device))
        # print("batch {}, frames size: {}, device : {}".format(batch_idx, frames.size(), frames.device))
        # frames = frames.cpu().numpy()[0, ...] # (8, 240, 320, 3)
        # np.save(f'{batch_idx}.npy', frames)
        # input('continue')

    dali_iter.reset()
    print(time.time() - stime)