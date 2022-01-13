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
def video_pipe(filenames):
    videos, labels = fn.readers.video(device="gpu", file_list=filenames, sequence_length=sequence_length, pad_sequences=True, stride=8, shard_id=0, num_shards=1, random_shuffle=True, initial_fill=initial_prefetch_size)
    return videos, labels


if __name__ == '__main__':

    # videos_dir = r"/app/datasets/kinetics400/train/videos/"
    # videos_dir = r"/app/datasets/ucf101/train/videos/"
    # video_files = r"/app/datasets/dali_videos/file.txt"
    video_files = r"/app/src/cache/dali_train.txt"
    # video_files = glob(os.path.join(videos_dir, '*.avi'))

    sequence_length = 8
    initial_prefetch_size = 16
    n_iter = 1000
    batch_size = 16  # batch size per GPU

    pipe = video_pipe(batch_size=batch_size, num_threads=2, device_id=0, filenames=video_files, seed=123456)

    dali_iter = DALIGenericIterator(pipelines=pipe, output_map=['frames', 'labels'])

    stime = time.time()
    for batch_idx, data in enumerate(dali_iter):
        frames = data[0]["frames"]
        labels = data[0]["labels"]
        # print(frames.size(), labels.size(), labels)
        print("batch {}, frames size: {}, labels : {}, device : {}".format(batch_idx, frames.size(), labels.cpu().numpy().tolist(), frames.device))
        # frames = frames.cpu().numpy()[0, ...] # (8, 240, 320, 3)
        # np.save(f'{batch_idx}.npy', frames)
        # input('continue')


    dali_iter.reset()
    print(time.time() - stime)